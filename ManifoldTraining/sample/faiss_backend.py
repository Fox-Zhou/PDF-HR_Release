"""Two-stage FAISS backend for tilt-vector pose distances."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Optional, Union

import numpy as np


ROOT_TILT_DIM = 3
JOINT_DIM = 29
POSE_DIM = ROOT_TILT_DIM + JOINT_DIM
COARSE_DIM = ROOT_TILT_DIM + JOINT_DIM


def _wait_for_file(path: Path, timeout_seconds: float = 7200.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for cache file: {path}")
        time.sleep(0.5)


def _cache_path(
    index_cache_dir: Optional[Union[str, Path]], cache_key: Optional[str]
) -> Optional[Path]:
    if index_cache_dir is None or cache_key is None:
        return None
    directory = Path(index_cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cache_key}.faiss"


def _normalize_tilt(tilt: np.ndarray) -> np.ndarray:
    tilt = np.ascontiguousarray(tilt, dtype=np.float32)
    norms = np.linalg.norm(tilt, axis=1, keepdims=True)
    if not np.isfinite(tilt).all() or np.any(norms < 1e-8):
        raise ValueError("Pose data contains invalid root tilt vectors")
    return tilt / norms


@dataclass(frozen=True)
class FaissPoseDistanceConfig:
    index_cache_dir: Optional[Union[str, Path]] = None
    cache_key: Optional[str] = None
    device: str = "auto"
    device_id: int = 0
    rank: int = 0
    world_size: int = 1
    add_chunk_size: int = 1_000_000
    rerank_candidates: int = 1000
    query_batch_size: int = 512
    rerank_chunk_size: int = 256


class FaissPoseDistance:
    """Coarse 32D L1 retrieval followed by exact mixed-distance reranking."""

    def __init__(self, pose_database: np.ndarray, config: FaissPoseDistanceConfig) -> None:
        import faiss

        if pose_database.ndim != 2 or pose_database.shape[1] != POSE_DIM:
            raise ValueError(f"pose_database must have shape (N, 32), got {pose_database.shape}")
        if pose_database.dtype != np.float32:
            raise ValueError(f"pose_database must be float32, got {pose_database.dtype}")
        if pose_database.shape[0] <= 0:
            raise ValueError("pose_database must not be empty")
        if (
            config.add_chunk_size <= 0
            or config.query_batch_size <= 0
            or config.rerank_chunk_size <= 0
        ):
            raise ValueError("FAISS and rerank chunk sizes must be positive")
        if config.rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")

        self.faiss = faiss
        self.database = pose_database
        self.count = int(pose_database.shape[0])
        self.rerank_candidates = min(int(config.rerank_candidates), self.count)
        self.query_batch_size = int(config.query_batch_size)
        self.cache_file = _cache_path(config.index_cache_dir, config.cache_key)

        cpu_index = self._load_or_build_cpu_index(config)
        self.index, self.device_description = self._select_search_index(
            cpu_index, config
        )
        self._gpu_reranker = None
        if self.device_description.startswith("gpu:"):
            from gpu_rerank import GpuExactReranker

            self._gpu_reranker = GpuExactReranker(
                self.database,
                config.device_id,
                config.rerank_chunk_size,
                config.add_chunk_size,
            )

    def _new_cpu_index(self):
        return self.faiss.IndexFlat(COARSE_DIM, self.faiss.METRIC_L1)

    def _validate_index(self, index, source: Path) -> None:
        if int(index.d) != COARSE_DIM or int(index.ntotal) != self.count:
            raise ValueError(
                f"Cached FAISS coarse index does not match database: {source}; "
                f"index=(N={index.ntotal}, D={index.d}), "
                f"expected=(N={self.count}, D={COARSE_DIM})"
            )

    def _coarse_features(self, poses: np.ndarray) -> np.ndarray:
        features = np.ascontiguousarray(poses, dtype=np.float32).copy()
        features[:, :ROOT_TILT_DIM] = _normalize_tilt(
            poses[:, :ROOT_TILT_DIM]
        )
        if not np.isfinite(features).all():
            raise ValueError("Coarse pose features contain NaN or infinity")
        return features

    def _build_cpu_index(self, add_chunk_size: int):
        index = self._new_cpu_index()
        for start in range(0, self.count, add_chunk_size):
            end = min(start + add_chunk_size, self.count)
            index.add(self._coarse_features(self.database[start:end]))
        return index

    def _load_or_build_cpu_index(self, config: FaissPoseDistanceConfig):
        if self.cache_file is None:
            return self._build_cpu_index(config.add_chunk_size)

        if config.world_size > 1:
            if config.rank == 0 and not self.cache_file.exists():
                index = self._build_cpu_index(config.add_chunk_size)
                temporary = self.cache_file.with_name(
                    f"{self.cache_file.name}.rank0-{os.getpid()}.tmp"
                )
                self.faiss.write_index(index, temporary.as_posix())
                os.replace(temporary, self.cache_file)
            _wait_for_file(self.cache_file)
            index = self.faiss.read_index(self.cache_file.as_posix())
            self._validate_index(index, self.cache_file)
            return index

        if self.cache_file.exists():
            index = self.faiss.read_index(self.cache_file.as_posix())
            self._validate_index(index, self.cache_file)
            return index

        index = self._build_cpu_index(config.add_chunk_size)
        self.faiss.write_index(index, self.cache_file.as_posix())
        return index

    def _select_search_index(self, cpu_index, config: FaissPoseDistanceConfig):
        requested = config.device.strip().lower()
        if requested not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError("FAISS device must be one of: auto, cpu, gpu, cuda")
        use_gpu = requested in {"gpu", "cuda"} or (
            requested == "auto" and self.faiss.get_num_gpus() > 0
        )
        if not use_gpu:
            return cpu_index, "cpu"
        if self.faiss.get_num_gpus() <= config.device_id:
            if requested == "auto":
                return cpu_index, "cpu"
            raise RuntimeError(
                f"FAISS GPU {config.device_id} is unavailable; count={self.faiss.get_num_gpus()}"
            )

        import torch

        torch.cuda.set_device(config.device_id)
        resources = self.faiss.StandardGpuResources()
        gpu_index = self.faiss.index_cpu_to_gpu(resources, config.device_id, cpu_index)
        import faiss.contrib.torch_utils  # noqa: F401; enables CUDA Tensor I/O

        self._gpu_resources = resources
        return gpu_index, f"gpu:{config.device_id}"

    def _rerank(self, queries: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        query_tilt = _normalize_tilt(queries[:, :ROOT_TILT_DIM])
        candidate_pose = np.asarray(self.database[candidate_indices], dtype=np.float32)

        dots = np.einsum(
            "bkd,bd->bk",
            candidate_pose[:, :, :ROOT_TILT_DIM],
            query_tilt,
            optimize=True,
        )
        root_geodesic = np.arccos(np.clip(dots, -1.0, 1.0))
        joint_l1 = np.sum(
            np.abs(
                candidate_pose[:, :, ROOT_TILT_DIM:]
                - queries[:, None, ROOT_TILT_DIM:]
            ),
            axis=2,
        )
        return np.min(root_geodesic + joint_l1, axis=1).astype(np.float32)

    def _rerank_joints(
        self, queries: np.ndarray, candidate_indices: np.ndarray
    ) -> np.ndarray:
        candidate_joints = np.asarray(
            self.database[candidate_indices, ROOT_TILT_DIM:], dtype=np.float32
        )
        joint_l1 = np.sum(
            np.abs(candidate_joints - queries[:, None, ROOT_TILT_DIM:]), axis=2
        )
        return np.min(joint_l1, axis=1).astype(np.float32)

    def query(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.ndim != 2 or queries.shape[1] != POSE_DIM:
            raise ValueError(f"queries must have shape (B, 32), got {queries.shape}")
        if not np.isfinite(queries).all():
            raise ValueError("Queries contain NaN or infinity")

        joint_results = np.empty(len(queries), dtype=np.float32)
        combined_results = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), self.query_batch_size):
            end = min(start + self.query_batch_size, len(queries))
            query_block = np.ascontiguousarray(queries[start:end])
            coarse_queries = self._coarse_features(query_block)
            if self._gpu_reranker is None:
                _, candidate_indices = self.index.search(
                    coarse_queries, self.rerank_candidates
                )
                combined_results[start:end] = self._rerank(
                    query_block, candidate_indices
                )
                joint_results[start:end] = self._rerank_joints(
                    query_block, candidate_indices
                )
            else:
                query_cuda = self._gpu_reranker.as_cuda(query_block)
                coarse_cuda = self._gpu_reranker.as_cuda(coarse_queries)
                _, candidate_indices = self.index.search(
                    coarse_cuda, self.rerank_candidates
                )
                combined_distance = self._gpu_reranker.pose_distance(
                    query_cuda, candidate_indices
                )
                joint_distance = self._gpu_reranker.pose_joint_distance(
                    query_cuda, candidate_indices
                )
                combined_results[start:end] = combined_distance.cpu().numpy()
                joint_results[start:end] = joint_distance.cpu().numpy()
        return (
            np.maximum(joint_results, 0.0),
            np.maximum(combined_results, 0.0),
        )


def build_nn_backend(
    database: np.ndarray,
    index_cache_dir: Optional[Union[str, Path]] = None,
    cache_key: Optional[str] = None,
    device: str = "auto",
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    add_chunk_size: int = 1_000_000,
    rerank_candidates: int = 1000,
    query_batch_size: int = 512,
    rerank_chunk_size: int = 256,
) -> FaissPoseDistance:
    return FaissPoseDistance(
        database,
        FaissPoseDistanceConfig(
            index_cache_dir=index_cache_dir,
            cache_key=cache_key,
            device=device,
            device_id=device_id,
            rank=rank,
            world_size=world_size,
            add_chunk_size=add_chunk_size,
            rerank_candidates=rerank_candidates,
            query_batch_size=query_batch_size,
            rerank_chunk_size=rerank_chunk_size,
        ),
    )
