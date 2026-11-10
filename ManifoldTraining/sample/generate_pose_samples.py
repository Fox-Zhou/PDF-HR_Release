#!/usr/bin/env python3
"""Generate joint-only and root-plus-joint pose-distance samples."""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from batch_scheduler import (
    batch_already_complete,
    batch_path,
    plan_missing_batches,
    save_array_no_overwrite,
)
from faiss_backend import build_nn_backend
from root_features import ROOT_FEATURE_LAYOUT, convert_root_features


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FEATURE_DIM = 97
POSE_FEATURE_DIM = 32
ROOT_TILT_DIM = 3
JOINT_DIM = 29
OUTPUT_DIM = POSE_FEATURE_DIM + 2
JOINT_DISTANCE_NORMALIZATION = JOINT_DIM
COMBINED_DISTANCE_NORMALIZATION = 1 + JOINT_DIM
ROOT_TILT_SLICE = slice(0, 3)
SOURCE_ROOT_QUAT_SLICE = slice(0, 4)
SOURCE_JOINT_SLICE = slice(10, 39)
POSE_JOINT_SLICE = slice(3, 32)
SAMPLE_RATIOS = {"on": 0.05, "near": 0.75, "interp": 0.20, "off": 0.0}
DEFAULT_CACHE_CHUNK = 1_000_000


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pose-only PDF-HR samples from a BONES-SEED NPY dataset. "
            "Each row is [pose(32), root+29dof distance, 29dof distance]."
        )
    )
    parser.add_argument(
        "--dataset-file",
        "--dataset_file",
        dest="dataset_file",
        type=Path,
        required=True,
        help="Raw merged dataset (N, 97), or yaw-invariant pose cache (N, 32).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory for output NPY batches.",
    )
    parser.add_argument(
        "--cache-dir",
        "--index_cache_dir",
        dest="cache_dir",
        type=Path,
        default=None,
        help="Pose-cache and FAISS-index directory. Defaults to <output-dir>/.cache.",
    )
    parser.add_argument(
        "--split-name",
        "--split_name",
        dest="split_name",
        default=None,
        help="Output subdirectory name. Defaults to dataset filename stem.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "assets" / "g1skel29" / "g1_29dof.urdf",
        help="G1 URDF used for 29 joint limits.",
    )
    parser.add_argument("--samples-per-batch", "--samples_per_batch", dest="samples_per_batch", type=int, default=1_000_000)
    parser.add_argument("--batch-num", "--batch_num", dest="batch_num", type=int, default=100)
    parser.add_argument("--batch-start", "--batch_start", dest="batch_start", type=int, default=0)
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=500_000)
    parser.add_argument("--cache-chunk-size", type=int, default=DEFAULT_CACHE_CHUNK)
    parser.add_argument(
        "--rerank-candidates",
        "--candidate-k",
        dest="rerank_candidates",
        type=int,
        default=1000,
        help="Number of coarse tilt-vector/joint FAISS candidates reranked exactly (default: 1000).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=512,
        help="Queries processed together during coarse retrieval and exact reranking.",
    )
    parser.add_argument(
        "--rerank-chunk-size",
        type=int,
        default=256,
        help="GPU exact-rerank candidate chunk size (default: 256).",
    )
    parser.add_argument(
        "--faiss-device",
        choices=("auto", "gpu", "cuda", "cpu"),
        default="auto",
        help="FAISS search device. Auto prefers FAISS GPU independently of PyTorch CUDA.",
    )
    return parser.parse_args()


def init_distributed() -> DistributedInfo:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed ranks: rank={rank}, world_size={world_size}")

    return DistributedInfo(rank, local_rank, world_size)


def wait_for_file(path: Path, timeout_seconds: float = 7200.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for cache file: {path}")
        time.sleep(0.5)


def source_cache_key(path: Path, split_name: str) -> str:
    stat = path.stat()
    return f"{split_name}_{stat.st_size}_{stat.st_mtime_ns}"


def normalize_tilt_vector(tilt: np.ndarray) -> np.ndarray:
    """Validate and normalize body-frame gravity vectors."""
    tilt = np.ascontiguousarray(tilt, dtype=np.float32)
    if not np.isfinite(tilt).all():
        raise ValueError("Root tilt vector contains NaN or infinity")
    norms = np.linalg.norm(tilt, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Root tilt vector contains a near-zero value")
    return np.ascontiguousarray(tilt / norms, dtype=np.float32)


def sphere_exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Exponential map from tangent vectors onto the unit sphere."""
    angle = np.linalg.norm(tangent, axis=1, keepdims=True)
    scale = np.ones_like(angle)
    nonzero = angle[:, 0] > 1e-8
    scale[nonzero] = np.sin(angle[nonzero]) / angle[nonzero]
    result = np.cos(angle) * base + scale * tangent
    return normalize_tilt_vector(result)


def sphere_slerp(first: np.ndarray, second: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Shortest unit-sphere interpolation with a stable antipodal fallback."""
    first = normalize_tilt_vector(first)
    second = normalize_tilt_vector(second)
    dot = np.clip(np.sum(first * second, axis=1, keepdims=True), -1.0, 1.0)
    result = np.empty_like(first)

    regular = np.abs(dot[:, 0]) < 1.0 - 1e-5
    if np.any(regular):
        theta = np.arccos(dot[regular])
        denominator = np.sin(theta)
        result[regular] = (
            np.sin((1.0 - alpha[regular]) * theta) / denominator * first[regular]
            + np.sin(alpha[regular] * theta) / denominator * second[regular]
        )

    parallel = dot[:, 0] >= 1.0 - 1e-5
    if np.any(parallel):
        result[parallel] = (
            (1.0 - alpha[parallel]) * first[parallel]
            + alpha[parallel] * second[parallel]
        )

    antipodal = ~(regular | parallel)
    if np.any(antipodal):
        values = first[antipodal]
        axes = np.zeros_like(values)
        least_aligned = np.argmin(np.abs(values), axis=1)
        axes[np.arange(len(values)), least_aligned] = 1.0
        tangent = np.cross(values, axes)
        tangent = normalize_tilt_vector(tangent)
        angle = np.pi * alpha[antipodal]
        result[antipodal] = np.cos(angle) * values + np.sin(angle) * tangent

    return normalize_tilt_vector(result)


def pose_cache_is_valid(cache_path: Path, expected_rows: int) -> bool:
    if not cache_path.is_file():
        return False
    try:
        pose = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        valid = pose.shape == (expected_rows, POSE_FEATURE_DIM) and pose.dtype == np.float32
        del pose
        return bool(valid)
    except (OSError, ValueError):
        return False


def build_pose_cache(
    dataset_path: Path,
    cache_path: Path,
    chunk_size: int,
    info: DistributedInfo,
) -> Path:
    source = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.dtype != np.float32:
        raise ValueError(f"Dataset must be a 2D float32 NPY, got {source.shape} {source.dtype}")
    rows, columns = map(int, source.shape)
    if columns not in {SOURCE_FEATURE_DIM, POSE_FEATURE_DIM}:
        raise ValueError(f"Dataset must have 97 or 32 columns, got {source.shape}")
    if rows <= 0 or chunk_size <= 0:
        raise ValueError("Dataset and cache chunk size must be positive")

    if columns == POSE_FEATURE_DIM:
        del source
        return dataset_path

    del source
    if info.rank == 0 and not pose_cache_is_valid(cache_path, rows):
        source = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(
            f"{cache_path.stem}.rank0-{os.getpid()}.tmp.npy"
        )
        pose = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(rows, POSE_FEATURE_DIM)
        )
        for start in tqdm(range(0, rows, chunk_size), desc="Extracting pose-only cache"):
            end = min(start + chunk_size, rows)
            # Intentionally access only root pose and joint-position columns.
            pose[start:end, ROOT_TILT_SLICE] = convert_root_features(
                source[start:end, SOURCE_ROOT_QUAT_SLICE]
            )[0]
            pose[start:end, POSE_JOINT_SLICE] = source[start:end, SOURCE_JOINT_SLICE]
        pose.flush()
        del pose, source
        os.replace(temporary, cache_path)
    if info.world_size > 1:
        wait_for_file(cache_path)
    if not pose_cache_is_valid(cache_path, rows):
        raise RuntimeError(f"Pose cache is missing or invalid: {cache_path}")
    return cache_path


def read_urdf_joint_limits(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    root = ET.parse(path).getroot()
    limits: list[tuple[float, float]] = []

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for joint in root.iter():
        if local_name(joint.tag) != "joint" or joint.attrib.get("type") not in {
            "revolute",
            "prismatic",
        }:
            continue
        limit = next((x for x in joint if local_name(x.tag) == "limit"), None)
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"Joint {joint.attrib.get('name')} has no finite limits in {path}")
        lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        limits.append((min(lower, upper), max(lower, upper)))
    result = np.asarray(limits, dtype=np.float32)
    if result.shape != (JOINT_DIM, 2):
        raise ValueError(f"Expected 29 limited joints in URDF, got {result.shape}: {path}")
    return result


def observed_joint_limits(pose_db: np.ndarray, chunk_size: int) -> np.ndarray:
    low = np.full(JOINT_DIM, np.inf, dtype=np.float32)
    high = np.full(JOINT_DIM, -np.inf, dtype=np.float32)
    for start in tqdm(range(0, len(pose_db), chunk_size), desc="Scanning joint limits"):
        joints = pose_db[start : start + chunk_size, POSE_JOINT_SLICE]
        low = np.minimum(low, np.min(joints, axis=0))
        high = np.maximum(high, np.max(joints, axis=0))
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Pose database contains non-finite joint positions")
    return np.column_stack((low, high)).astype(np.float32)


class PoseSampler:
    def __init__(
        self,
        pose_db: np.ndarray,
        joint_limits: np.ndarray,
        nn_backend,
        rng: np.random.Generator,
    ) -> None:
        if pose_db.ndim != 2 or pose_db.shape[1] != POSE_FEATURE_DIM:
            raise ValueError(f"Pose database must have shape (N, 32), got {pose_db.shape}")
        if joint_limits.shape != (JOINT_DIM, 2):
            raise ValueError(f"Joint limits must have shape (29, 2), got {joint_limits.shape}")
        self.db = pose_db
        self.limits = joint_limits.astype(np.float32, copy=False)
        self.nn = nn_backend
        self.rng = rng

    def sample_db(self, count: int) -> np.ndarray:
        indices = self.rng.integers(0, len(self.db), size=count)
        return np.ascontiguousarray(self.db[indices], dtype=np.float32)

    def sample_near(self, count: int) -> np.ndarray:
        base = self.sample_db(count)
        # The projected root direction has two effective tangent dimensions.
        direction = self.rng.standard_normal((count, ROOT_TILT_DIM + JOINT_DIM)).astype(np.float32)
        base_tilt = base[:, ROOT_TILT_SLICE]
        direction[:, ROOT_TILT_SLICE] -= (
            np.sum(direction[:, ROOT_TILT_SLICE] * base_tilt, axis=1, keepdims=True)
            * base_tilt
        )
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
        epsilon = (
            np.abs(self.rng.standard_normal((count, 1))).astype(np.float32) * 4.0 + 0.2
        )
        delta = direction * epsilon

        result = np.empty_like(base)
        result[:, ROOT_TILT_SLICE] = sphere_exp(
            base_tilt, delta[:, ROOT_TILT_SLICE]
        )
        result[:, POSE_JOINT_SLICE] = np.clip(
            base[:, POSE_JOINT_SLICE] + delta[:, ROOT_TILT_DIM:],
            self.limits[:, 0],
            self.limits[:, 1],
        )
        return result

    def sample_interp(self, count: int) -> np.ndarray:
        first = self.sample_db(count)
        second = self.sample_db(count)
        alpha = self.rng.random((count, 1), dtype=np.float32)
        result = np.empty_like(first)
        result[:, ROOT_TILT_SLICE] = sphere_slerp(
            first[:, ROOT_TILT_SLICE], second[:, ROOT_TILT_SLICE], alpha
        )
        result[:, POSE_JOINT_SLICE] = (
            (1.0 - alpha) * first[:, POSE_JOINT_SLICE]
            + alpha * second[:, POSE_JOINT_SLICE]
        )
        return result

    def sample_off(self, count: int) -> np.ndarray:
        result = np.empty((count, POSE_FEATURE_DIM), dtype=np.float32)
        # A normalized isotropic Gaussian is uniform on S^2 and has no seam.
        tilt = self.rng.standard_normal((count, ROOT_TILT_DIM)).astype(np.float32)
        result[:, ROOT_TILT_SLICE] = normalize_tilt_vector(tilt)
        result[:, POSE_JOINT_SLICE] = self.rng.random(
            (count, JOINT_DIM), dtype=np.float32
        ) * (self.limits[:, 1] - self.limits[:, 0]) + self.limits[:, 0]
        return result

    def sample_chunk(
        self, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_on = int(count * SAMPLE_RATIOS["on"])
        n_interp = int(count * SAMPLE_RATIOS["interp"])
        n_off = 0
        # Assign integer-rounding residue to near samples, never to off-manifold.
        n_near = count - n_on - n_interp

        poses: list[np.ndarray] = []
        joint_distances: list[np.ndarray] = []
        combined_distances: list[np.ndarray] = []
        if n_on:
            poses.append(self.sample_db(n_on))
            joint_distances.append(np.zeros(n_on, dtype=np.float32))
            combined_distances.append(np.zeros(n_on, dtype=np.float32))
        for number, generator in (
            (n_near, self.sample_near),
            (n_interp, self.sample_interp),
            (n_off, self.sample_off),
        ):
            if number:
                queries = generator(number)
                poses.append(queries)
                joint_distance, combined_distance = self.nn.query(queries)
                joint_distances.append(joint_distance)
                combined_distances.append(combined_distance)

        pose = np.concatenate(poses, axis=0)
        joint_distance = np.concatenate(joint_distances, axis=0)
        combined_distance = np.concatenate(combined_distances, axis=0)
        permutation = self.rng.permutation(count)
        return (
            pose[permutation],
            joint_distance[permutation],
            combined_distance[permutation],
        )


def run(args: argparse.Namespace) -> None:
    info = init_distributed()
    dataset_path = args.dataset_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if args.samples_per_batch <= 0 or args.chunk_size <= 0:
        raise ValueError("samples-per-batch and chunk-size must be positive")

    split_name = args.split_name or dataset_path.stem
    batch_dir = output_dir / split_name
    missing_batches, batch_indices = plan_missing_batches(
        batch_dir=batch_dir,
        batch_start=args.batch_start,
        batch_num=args.batch_num,
        rows=args.samples_per_batch,
        columns=OUTPUT_DIM,
        rank=info.rank,
        world_size=info.world_size,
    )
    print(
        f"[rank {info.rank}/{info.world_size}] requested={args.batch_num}, "
        f"missing={len(missing_batches)}, assigned={batch_indices}"
    )
    if not missing_batches:
        print(f"[rank {info.rank}] all requested pose batches already exist; nothing to do")
        return
    if not batch_indices:
        print(f"[rank {info.rank}] no missing pose batch assigned; nothing to do")
        return

    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else output_dir / ".cache"
    )
    key = source_cache_key(dataset_path, split_name)
    pose_cache_path = cache_dir / f"{key}.{ROOT_FEATURE_LAYOUT}.pose.npy"
    pose_path = build_pose_cache(
        dataset_path, pose_cache_path, args.cache_chunk_size, info
    )
    pose_db = np.load(pose_path, mmap_mode="r", allow_pickle=False)

    if args.urdf is not None:
        limits = read_urdf_joint_limits(args.urdf.expanduser().resolve())
        limit_source = str(args.urdf)
    else:
        limits = observed_joint_limits(pose_db, args.cache_chunk_size)
        limit_source = "observed pose database min/max"

    metric_cache_key = f"{key}_{ROOT_FEATURE_LAYOUT}_joint32_l1"
    backend = build_nn_backend(
        pose_db,
        index_cache_dir=cache_dir,
        cache_key=metric_cache_key,
        device=args.faiss_device,
        device_id=info.local_rank,
        rank=info.rank,
        world_size=info.world_size,
        add_chunk_size=args.cache_chunk_size,
        rerank_candidates=args.rerank_candidates,
        query_batch_size=args.query_batch_size,
        rerank_chunk_size=args.rerank_chunk_size,
    )
    print(
        f"[rank {info.rank}/{info.world_size}] pose_db={pose_db.shape}, "
        f"FAISS={backend.device_description}, batches={batch_indices}, limits={limit_source}"
    )
    for batch_index in tqdm(batch_indices, desc=f"rank {info.rank} sampling"):
        output_path = batch_path(batch_dir, batch_index)
        if batch_already_complete(
            output_path, args.samples_per_batch, OUTPUT_DIM
        ):
            continue
        sampler = PoseSampler(
            pose_db,
            limits,
            backend,
            np.random.default_rng(RANDOM_SEED + batch_index),
        )
        batch = np.empty((args.samples_per_batch, OUTPUT_DIM), dtype=np.float32)
        offset = 0
        while offset < args.samples_per_batch:
            count = min(args.chunk_size, args.samples_per_batch - offset)
            pose, joint_distance, combined_distance = sampler.sample_chunk(count)
            batch[offset : offset + count, :POSE_FEATURE_DIM] = pose
            batch[offset : offset + count, POSE_FEATURE_DIM] = (
                combined_distance / COMBINED_DISTANCE_NORMALIZATION
            )
            batch[offset : offset + count, POSE_FEATURE_DIM + 1] = (
                joint_distance / JOINT_DISTANCE_NORMALIZATION
            )
            offset += count
        save_array_no_overwrite(output_path, batch, info.rank)


if __name__ == "__main__":
    run(parse_args())
