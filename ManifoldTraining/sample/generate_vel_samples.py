#!/usr/bin/env python3
"""Generate fixed-pose, noisy-velocity samples with two distance metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

from batch_scheduler import (
    batch_already_complete,
    batch_path,
    plan_missing_batches,
    publish_temporary_batch,
    temporary_batch_path,
)
from root_features import ROOT_FEATURE_LAYOUT, convert_root_features

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FEATURE_DIM = 97
POSE_DIM = 32
VELOCITY_DIM = 32
POSE_COARSE_DIM = 3 + 29
POSE_VEL_DIM = POSE_DIM + VELOCITY_DIM
OUTPUT_DIM = POSE_VEL_DIM + 2
JOINT_DISTANCE_NORMALIZATION = 29
COMBINED_DISTANCE_NORMALIZATION = 3 + 29

SOURCE_ROOT_QUAT_SLICE = slice(0, 4)
SOURCE_ROOT_VEL_SLICE = slice(4, 7)
SOURCE_JOINT_POS_SLICE = slice(10, 39)
SOURCE_JOINT_VEL_SLICE = slice(39, 68)

CACHE_ROOT_TILT_SLICE = slice(0, 3)
CACHE_JOINT_POS_SLICE = slice(3, 32)
CACHE_ROOT_VEL_SLICE = slice(32, 35)
CACHE_JOINT_VEL_SLICE = slice(35, 64)
CACHE_VELOCITY_SLICE = slice(32, 64)

DEFAULT_CACHE_CHUNK = 1_000_000
GAUSSIAN_RATIO = 0.70
SWAP_RATIO = 0.30
RANDOM_RATIO = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate samples [fixed pose(32), noisy velocity(32), root+29dof "
            "distance(1), 29dof distance(1)] from a BONES-SEED dataset."
        )
    )
    parser.add_argument(
        "--dataset-file",
        "--dataset_file",
        dest="dataset_file",
        type=Path,
        required=True,
        help="Raw merged dataset (N, 97), or yaw-invariant pose-velocity cache (N, 64).",
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
        help="Pose-velocity cache and FAISS-index directory. Defaults to <output-dir>/.cache.",
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
        help="G1 URDF supplying the 29 joint velocity limits.",
    )
    parser.add_argument("--samples-per-batch", type=int, default=1_000_000)
    parser.add_argument("--batch-num", type=int, default=100)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--cache-chunk-size", type=int, default=DEFAULT_CACHE_CHUNK)
    parser.add_argument(
        "--rerank-candidates",
        "--candidate-k",
        dest="rerank_candidates",
        type=int,
        default=1000,
        help=(
            "Number of coarse pose-L1 FAISS candidates whose velocities are "
            "compared with the exact root-XYZ-L2/joint-L1 metric (default: 1000)."
        ),
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
        "--noise-scale",
        type=float,
        default=3.0,
        help="Scale of the half-normal velocity-noise magnitude (default: 2.0).",
    )
    parser.add_argument(
        "--noise-min",
        type=float,
        default=0.0,
        help="Minimum velocity-noise magnitude (default: 0.0).",
    )
    parser.add_argument(
        "--faiss-device",
        choices=("auto", "gpu", "cuda", "cpu"),
        default="auto",
        help="FAISS search device. Auto prefers a FAISS GPU.",
    )
    return parser.parse_args()


def distributed_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed ranks: rank={rank}, world_size={world_size}")
    return rank, local_rank, world_size


def source_cache_key(path: Path, split_name: str) -> str:
    stat = path.stat()
    return f"{split_name}_{stat.st_size}_{stat.st_mtime_ns}"


def wait_for_file(path: Path, timeout_seconds: float = 7200.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for cache file: {path}")
        time.sleep(0.5)


def read_urdf_joint_velocity_limits(path: Path) -> np.ndarray:
    """Read positive velocity limits in the URDF's 29-joint order."""
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    root = ET.parse(path).getroot()
    limits: list[float] = []

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for joint in root.iter():
        if local_name(joint.tag) != "joint" or joint.attrib.get("type") not in {
            "revolute",
            "prismatic",
        }:
            continue
        limit = next((x for x in joint if local_name(x.tag) == "limit"), None)
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(
                f"Joint {joint.attrib.get('name')} has no velocity limit in {path}"
            )
        velocity = abs(float(limit.attrib["velocity"]))
        if not np.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(
                f"Joint {joint.attrib.get('name')} has invalid velocity limit "
                f"{velocity} in {path}"
            )
        limits.append(velocity)

    result = np.asarray(limits, dtype=np.float32)
    if result.shape != (29,):
        raise ValueError(
            f"Expected 29 limited joints in URDF, got {result.shape}: {path}"
        )
    return result


def bounds_are_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        bounds = np.load(path, allow_pickle=False)
        valid = (
            bounds.shape == (2, VELOCITY_DIM)
            and bounds.dtype == np.float32
            and np.isfinite(bounds).all()
            and np.all(bounds[0] <= bounds[1])
        )
        del bounds
        return bool(valid)
    except (OSError, ValueError):
        return False


def build_velocity_bounds(
    database: np.ndarray,
    path: Path,
    chunk_size: int,
    rank: int,
    world_size: int,
) -> np.ndarray:
    if rank == 0 and not bounds_are_valid(path):
        minimum = np.full(VELOCITY_DIM, np.inf, dtype=np.float32)
        maximum = np.full(VELOCITY_DIM, -np.inf, dtype=np.float32)
        for start in tqdm(
            range(0, len(database), chunk_size), desc="Computing velocity bounds"
        ):
            end = min(start + chunk_size, len(database))
            values = np.asarray(
                database[start:end, CACHE_VELOCITY_SLICE], dtype=np.float32
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Velocity database contains NaN/inf in [{start}:{end}]")
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
        bounds = np.stack((minimum, maximum)).astype(np.float32, copy=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
        np.save(temporary, bounds, allow_pickle=False)
        os.replace(temporary, path)
    if world_size > 1:
        wait_for_file(path)
    if not bounds_are_valid(path):
        raise ValueError(f"Velocity bounds are missing or invalid: {path}")
    return np.load(path, allow_pickle=False)


def cache_is_valid(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = data.shape == (expected_rows, POSE_VEL_DIM) and data.dtype == np.float32
        del data
        return bool(valid)
    except (OSError, ValueError):
        return False


def build_pose_velocity_cache(
    dataset_path: Path,
    cache_path: Path,
    chunk_size: int,
    rank: int,
    world_size: int,
) -> Path:
    source = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.dtype != np.float32:
        raise ValueError(f"Dataset must be a 2D float32 NPY, got {source.shape} {source.dtype}")
    rows, columns = map(int, source.shape)
    if columns not in {SOURCE_FEATURE_DIM, POSE_VEL_DIM}:
        raise ValueError(f"Dataset must have 97 or 64 columns, got {source.shape}")
    if rows <= 0 or chunk_size <= 0:
        raise ValueError("Dataset and cache chunk size must be positive")
    if columns == POSE_VEL_DIM:
        del source
        return dataset_path
    del source

    if rank == 0 and not cache_is_valid(cache_path, rows):
        source = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(
            f"{cache_path.stem}.rank0-{os.getpid()}.tmp.npy"
        )
        cache = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(rows, POSE_VEL_DIM)
        )
        for start in tqdm(range(0, rows, chunk_size), desc="Extracting pose-velocity cache"):
            end = min(start + chunk_size, rows)
            # Only pose and velocity columns are accessed. Both acceleration
            # Root/joint acceleration groups are intentionally ignored.
            root_tilt, root_velocity_body = convert_root_features(
                source[start:end, SOURCE_ROOT_QUAT_SLICE],
                source[start:end, SOURCE_ROOT_VEL_SLICE],
            )
            cache[start:end, CACHE_ROOT_TILT_SLICE] = root_tilt
            cache[start:end, CACHE_JOINT_POS_SLICE] = source[
                start:end, SOURCE_JOINT_POS_SLICE
            ]
            cache[start:end, CACHE_ROOT_VEL_SLICE] = root_velocity_body
            cache[start:end, CACHE_JOINT_VEL_SLICE] = source[
                start:end, SOURCE_JOINT_VEL_SLICE
            ]
        cache.flush()
        del cache, source
        os.replace(temporary, cache_path)
    if world_size > 1:
        wait_for_file(cache_path)
    if not cache_is_valid(cache_path, rows):
        raise RuntimeError(f"Pose-velocity cache is missing or invalid: {cache_path}")
    return cache_path


def pose_to_coarse_features(pose: np.ndarray) -> np.ndarray:
    """Map root tilt + joints to normalized tilt + 29 joints for FAISS L1."""
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != POSE_DIM:
        raise ValueError(f"Pose must have shape (B, {POSE_DIM}), got {pose.shape}")
    tilt = np.ascontiguousarray(pose[:, :3], dtype=np.float32)
    norms = np.linalg.norm(tilt, axis=1, keepdims=True)
    if not np.isfinite(pose).all() or np.any(norms < 1e-8):
        raise ValueError("Pose contains NaN, infinity, or an invalid root tilt vector")
    features = np.ascontiguousarray(pose, dtype=np.float32).copy()
    features[:, :3] = tilt / norms
    return np.ascontiguousarray(features, dtype=np.float32)


class VelocityFaissDistance:
    """Retrieve by coarse pose L1, then minimize exact velocity distance."""

    def __init__(
        self,
        pose_velocity_db: np.ndarray,
        cache_path: Path,
        device: str,
        device_id: int,
        rank: int,
        world_size: int,
        add_chunk_size: int,
        rerank_candidates: int,
        query_batch_size: int,
        rerank_chunk_size: int,
    ) -> None:
        import faiss

        if pose_velocity_db.shape[1] != POSE_VEL_DIM:
            raise ValueError(f"Pose-velocity database must have 64 columns: {pose_velocity_db.shape}")
        if add_chunk_size <= 0 or query_batch_size <= 0 or rerank_candidates <= 0:
            raise ValueError("FAISS chunk sizes must be positive")
        self.faiss = faiss
        self.database = pose_velocity_db
        self.count = int(len(pose_velocity_db))
        self.rerank_candidates = min(int(rerank_candidates), self.count)
        self.query_batch_size = int(query_batch_size)
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cpu_index = self._load_or_build_cpu_index(
            rank, world_size, add_chunk_size
        )
        self.index, self.device_description = self._select_index(
            cpu_index, device, device_id
        )
        self._gpu_reranker = None
        if self.device_description.startswith("gpu:"):
            from gpu_rerank import GpuExactReranker

            self._gpu_reranker = GpuExactReranker(
                self.database[:, CACHE_VELOCITY_SLICE],
                device_id,
                rerank_chunk_size,
                add_chunk_size,
            )

    def _validate_index(self, index) -> None:
        if int(index.d) != POSE_COARSE_DIM or int(index.ntotal) != self.count:
            raise ValueError(
                f"Cached pose-candidate index mismatch: index=(N={index.ntotal}, "
                f"D={index.d}), expected=(N={self.count}, D={POSE_COARSE_DIM})"
            )

    def _build_cpu_index(self, chunk_size: int):
        index = self.faiss.IndexFlat(POSE_COARSE_DIM, self.faiss.METRIC_L1)
        for start in range(0, self.count, chunk_size):
            end = min(start + chunk_size, self.count)
            pose_features = pose_to_coarse_features(
                self.database[start:end, :POSE_DIM]
            )
            index.add(pose_features)
        return index

    def _load_or_build_cpu_index(self, rank: int, world_size: int, chunk_size: int):
        if world_size > 1:
            if rank == 0 and not self.cache_path.exists():
                index = self._build_cpu_index(chunk_size)
                temporary = self.cache_path.with_name(
                    f"{self.cache_path.name}.rank0-{os.getpid()}.tmp"
                )
                self.faiss.write_index(index, temporary.as_posix())
                os.replace(temporary, self.cache_path)
            wait_for_file(self.cache_path)
        elif not self.cache_path.exists():
            index = self._build_cpu_index(chunk_size)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.faiss.write_index(index, self.cache_path.as_posix())
            return index

        index = self.faiss.read_index(self.cache_path.as_posix())
        self._validate_index(index)
        return index

    def _select_index(self, cpu_index, device: str, device_id: int):
        requested = device.lower()
        use_gpu = requested in {"gpu", "cuda"} or (
            requested == "auto" and self.faiss.get_num_gpus() > 0
        )
        if not use_gpu:
            return cpu_index, "cpu"
        if self.faiss.get_num_gpus() <= device_id:
            if requested == "auto":
                return cpu_index, "cpu"
            raise RuntimeError(
                f"FAISS GPU {device_id} unavailable; count={self.faiss.get_num_gpus()}"
            )
        import torch

        torch.cuda.set_device(device_id)
        resources = self.faiss.StandardGpuResources()
        gpu_index = self.faiss.index_cpu_to_gpu(resources, device_id, cpu_index)
        import faiss.contrib.torch_utils  # noqa: F401; enables CUDA Tensor I/O

        self._gpu_resources = resources
        return gpu_index, f"gpu:{device_id}"

    def _rerank(
        self, velocity: np.ndarray, candidate_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(
            self.database[candidate_indices, CACHE_VELOCITY_SLICE], dtype=np.float32
        )
        root_l2 = np.linalg.norm(
            candidates[:, :, :3] - velocity[:, None, :3], axis=2
        )
        joint_l1 = np.sum(
            np.abs(candidates[:, :, 3:] - velocity[:, None, 3:]), axis=2
        )
        return (
            np.min(joint_l1, axis=1).astype(np.float32),
            np.min(root_l2 + joint_l1, axis=1).astype(np.float32),
        )

    def query(
        self, pose: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(pose, dtype=np.float32)
        velocity = np.asarray(velocity, dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM:
            raise ValueError(f"Pose queries must have shape (B, {POSE_DIM}): {pose.shape}")
        if velocity.ndim != 2 or velocity.shape[1] != VELOCITY_DIM:
            raise ValueError(f"Velocity queries must have shape (B, 32): {velocity.shape}")
        if len(pose) != len(velocity):
            raise ValueError("Pose and velocity query batch sizes must match")
        if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
            raise ValueError("Pose or velocity queries contain NaN or infinity")
        joint_result = np.empty(len(velocity), dtype=np.float32)
        combined_result = np.empty(len(velocity), dtype=np.float32)
        for start in range(0, len(velocity), self.query_batch_size):
            end = min(start + self.query_batch_size, len(velocity))
            coarse_pose = pose_to_coarse_features(pose[start:end])
            if self._gpu_reranker is None:
                _, candidate_indices = self.index.search(
                    coarse_pose, self.rerank_candidates
                )
                joint_distance, combined_distance = self._rerank(
                    velocity[start:end], candidate_indices
                )
                joint_result[start:end] = joint_distance
                combined_result[start:end] = combined_distance
            else:
                coarse_cuda = self._gpu_reranker.as_cuda(coarse_pose)
                velocity_cuda = self._gpu_reranker.as_cuda(velocity[start:end])
                _, candidate_indices = self.index.search(
                    coarse_cuda, self.rerank_candidates
                )
                joint_distance = self._gpu_reranker.vector_joint_distance(
                    velocity_cuda, candidate_indices
                )
                combined_distance = self._gpu_reranker.vector_distance(
                    velocity_cuda, candidate_indices
                )
                joint_result[start:end] = joint_distance.cpu().numpy()
                combined_result[start:end] = combined_distance.cpu().numpy()
        return (
            np.maximum(joint_result, 0.0),
            np.maximum(combined_result, 0.0),
        )


def sample_chunk(
    database: np.ndarray,
    backend: VelocityFaissDistance,
    count: int,
    rng: np.random.Generator,
    noise_scale: float,
    noise_min: float,
    velocity_bounds: np.ndarray,
) -> np.ndarray:
    if len(database) < 2:
        raise ValueError("Velocity swap sampling requires at least two database rows")
    if velocity_bounds.shape != (2, VELOCITY_DIM):
        raise ValueError(
            f"Velocity random bounds must have shape (2, {VELOCITY_DIM})"
        )
    indices = rng.integers(0, len(database), size=count)
    base = np.ascontiguousarray(database[indices], dtype=np.float32)
    pose = base[:, :POSE_DIM]
    base_velocity = base[:, CACHE_VELOCITY_SLICE]

    swap_count = int(count * SWAP_RATIO)
    random_count = 0
    # Assign integer-rounding residue to Gaussian near samples.
    gaussian_count = count - swap_count
    sampled_velocity = np.empty((count, VELOCITY_DIM), dtype=np.float32)

    if gaussian_count:
        direction = rng.standard_normal((gaussian_count, VELOCITY_DIM)).astype(
            np.float32
        )
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
        magnitude = (
            np.abs(rng.standard_normal((gaussian_count, 1))).astype(np.float32)
            * noise_scale
            + noise_min
        )
        sampled_velocity[:gaussian_count] = (
            base_velocity[:gaussian_count] + direction * magnitude
        )

    swap_end = gaussian_count + swap_count
    if swap_count:
        base_indices = indices[gaussian_count:swap_end]
        donor_indices = rng.integers(0, len(database) - 1, size=swap_count)
        donor_indices += donor_indices >= base_indices
        sampled_velocity[gaussian_count:swap_end] = np.asarray(
            database[donor_indices, CACHE_VELOCITY_SLICE], dtype=np.float32
        )

    if random_count:
        random_values = rng.random(
            (random_count, VELOCITY_DIM), dtype=np.float32
        )
        sampled_velocity[swap_end:] = (
            velocity_bounds[0]
            + random_values * (velocity_bounds[1] - velocity_bounds[0])
        )

    # Gaussian, swap, and random samples all obey the same limits. Root XYZ
    # uses observed dataset bounds; the 29 joints use symmetric URDF limits.
    np.clip(
        sampled_velocity,
        velocity_bounds[0],
        velocity_bounds[1],
        out=sampled_velocity,
    )

    joint_distance, combined_distance = backend.query(pose, sampled_velocity)

    samples = np.empty((count, OUTPUT_DIM), dtype=np.float32)
    samples[:, :POSE_DIM] = pose
    samples[:, POSE_DIM:POSE_VEL_DIM] = sampled_velocity
    samples[:, POSE_VEL_DIM] = (
        combined_distance / COMBINED_DISTANCE_NORMALIZATION
    )
    samples[:, POSE_VEL_DIM + 1] = (
        joint_distance / JOINT_DISTANCE_NORMALIZATION
    )
    return samples[rng.permutation(count)]


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = distributed_info()
    if args.samples_per_batch <= 0 or args.chunk_size <= 0:
        raise ValueError("samples-per-batch and chunk-size must be positive")
    if args.noise_scale < 0.0 or args.noise_min < 0.0:
        raise ValueError("noise-scale and noise-min must be non-negative")

    dataset_path = args.dataset_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    split_name = args.split_name or dataset_path.stem
    batch_dir = output_dir / split_name
    missing_batches, batch_indices = plan_missing_batches(
        batch_dir=batch_dir,
        batch_start=args.batch_start,
        batch_num=args.batch_num,
        rows=args.samples_per_batch,
        columns=OUTPUT_DIM,
        rank=rank,
        world_size=world_size,
    )
    print(
        f"[rank {rank}/{world_size}] requested={args.batch_num}, "
        f"missing={len(missing_batches)}, assigned={batch_indices}"
    )
    if not missing_batches:
        print(f"[rank {rank}] all requested velocity batches already exist; nothing to do")
        return
    if not batch_indices:
        print(f"[rank {rank}] no missing velocity batch assigned; nothing to do")
        return

    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else output_dir / ".cache"
    )
    key = source_cache_key(dataset_path, split_name)
    pose_velocity_cache = cache_dir / f"{key}.{ROOT_FEATURE_LAYOUT}.pose_vel.npy"
    cache_path = build_pose_velocity_cache(
        dataset_path,
        pose_velocity_cache,
        args.cache_chunk_size,
        rank,
        world_size,
    )
    database = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    bounds_path = cache_dir / f"{key}.velocity_bounds.npy"
    velocity_bounds = build_velocity_bounds(
        database,
        bounds_path,
        args.cache_chunk_size,
        rank,
        world_size,
    )
    joint_velocity_limits = read_urdf_joint_velocity_limits(
        args.urdf.expanduser().resolve()
    )
    velocity_bounds = np.asarray(velocity_bounds, dtype=np.float32).copy()
    velocity_bounds[0, 3:] = -joint_velocity_limits
    velocity_bounds[1, 3:] = joint_velocity_limits

    faiss_cache = cache_dir / f"{key}_{ROOT_FEATURE_LAYOUT}_pose32_l1.faiss"
    backend = VelocityFaissDistance(
        database,
        faiss_cache,
        args.faiss_device,
        local_rank,
        rank,
        world_size,
        args.cache_chunk_size,
        args.rerank_candidates,
        args.query_batch_size,
        args.rerank_chunk_size,
    )
    print(
        f"[rank {rank}/{world_size}] db={database.shape}, FAISS={backend.device_description}, "
        f"candidate_metric=pose32_tilt_l1, k={backend.rerank_candidates}, "
        f"batches={batch_indices}, "
        f"sampling=gaussian:{GAUSSIAN_RATIO:.0%}/swap:{SWAP_RATIO:.0%}/"
        f"random:{RANDOM_RATIO:.0%}, noise=abs(N)*{args.noise_scale}+{args.noise_min}"
    )

    for batch_index in tqdm(batch_indices, desc=f"rank {rank} velocity sampling"):
        output_path = batch_path(batch_dir, batch_index)
        if batch_already_complete(output_path, args.samples_per_batch, OUTPUT_DIM):
            continue
        rng = np.random.default_rng(RANDOM_SEED + batch_index)
        temporary_path = temporary_batch_path(output_path, rank)
        batch = None
        try:
            batch = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(args.samples_per_batch, OUTPUT_DIM),
            )
            offset = 0
            while offset < args.samples_per_batch:
                count = min(args.chunk_size, args.samples_per_batch - offset)
                batch[offset : offset + count] = sample_chunk(
                    database,
                    backend,
                    count,
                    rng,
                    args.noise_scale,
                    args.noise_min,
                    velocity_bounds,
                )
                offset += count
            batch.flush()
            del batch
            batch = None
            publish_temporary_batch(
                temporary_path, output_path, args.samples_per_batch, OUTPUT_DIM
            )
        finally:
            if batch is not None:
                del batch
            if temporary_path.exists():
                temporary_path.unlink()


if __name__ == "__main__":
    main()
