"""TorchScript-compiled CUDA kernels for exact candidate reranking."""

from __future__ import annotations

import numpy as np
import torch


@torch.jit.script
def _pose_min_distance(
    queries: torch.Tensor,
    database: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_chunk_size: int,
) -> torch.Tensor:
    """Tilt-sphere geodesic + joint L1 minimum over candidate indices."""
    batch_size = queries.size(0)
    candidate_count = candidate_indices.size(1)
    best = torch.full(
        (batch_size,), float("inf"), dtype=queries.dtype, device=queries.device
    )
    query_tilt = queries[:, :3]
    query_tilt = query_tilt / torch.clamp(
        torch.sqrt(torch.sum(query_tilt * query_tilt, dim=1, keepdim=True)),
        min=1.0e-8,
    )
    for start in range(0, candidate_count, candidate_chunk_size):
        end = min(start + candidate_chunk_size, candidate_count)
        candidates = database[candidate_indices[:, start:end]]
        dots = torch.sum(
            candidates[:, :, :3] * query_tilt[:, None, :], dim=2
        )
        root_geodesic = torch.acos(torch.clamp(dots, min=-1.0, max=1.0))
        joint_l1 = torch.sum(
            torch.abs(candidates[:, :, 3:] - queries[:, None, 3:]), dim=2
        )
        best = torch.minimum(best, torch.min(root_geodesic + joint_l1, dim=1)[0])
    return best


@torch.jit.script
def _vector_min_distance(
    queries: torch.Tensor,
    database: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_chunk_size: int,
) -> torch.Tensor:
    """Root XYZ L2 + 29-joint L1 minimum over candidate indices."""
    batch_size = queries.size(0)
    candidate_count = candidate_indices.size(1)
    best = torch.full(
        (batch_size,), float("inf"), dtype=queries.dtype, device=queries.device
    )
    for start in range(0, candidate_count, candidate_chunk_size):
        end = min(start + candidate_chunk_size, candidate_count)
        candidates = database[candidate_indices[:, start:end]]
        root_delta = candidates[:, :, :3] - queries[:, None, :3]
        root_l2 = torch.sqrt(torch.sum(root_delta * root_delta, dim=2))
        joint_l1 = torch.sum(
            torch.abs(candidates[:, :, 3:] - queries[:, None, 3:]), dim=2
        )
        best = torch.minimum(best, torch.min(root_l2 + joint_l1, dim=1)[0])
    return best


@torch.jit.script
def _joint_min_distance(
    queries: torch.Tensor,
    database: torch.Tensor,
    candidate_indices: torch.Tensor,
    joint_start: int,
    candidate_chunk_size: int,
) -> torch.Tensor:
    """29-joint L1 minimum over candidate indices, ignoring root values."""
    batch_size = queries.size(0)
    candidate_count = candidate_indices.size(1)
    best = torch.full(
        (batch_size,), float("inf"), dtype=queries.dtype, device=queries.device
    )
    for start in range(0, candidate_count, candidate_chunk_size):
        end = min(start + candidate_chunk_size, candidate_count)
        candidates = database[candidate_indices[:, start:end]]
        joint_l1 = torch.sum(
            torch.abs(
                candidates[:, :, joint_start:]
                - queries[:, None, joint_start:]
            ),
            dim=2,
        )
        best = torch.minimum(best, torch.min(joint_l1, dim=1)[0])
    return best


class GpuExactReranker:
    """Own a GPU-resident metric database and run compiled exact reranking."""

    def __init__(
        self,
        database: np.ndarray,
        device_id: int,
        candidate_chunk_size: int,
        upload_chunk_size: int = 1_000_000,
    ) -> None:
        if candidate_chunk_size <= 0 or upload_chunk_size <= 0:
            raise ValueError("GPU rerank chunk sizes must be positive")
        self.device = torch.device("cuda", device_id)
        # faiss.contrib.torch_utils obtains its stream from the process-wide
        # current CUDA device. torchrun does not set that device automatically.
        torch.cuda.set_device(self.device)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.database = torch.empty(
            database.shape, dtype=torch.float32, device=self.device
        )
        for start in range(0, len(database), upload_chunk_size):
            end = min(start + upload_chunk_size, len(database))
            host = torch.from_numpy(
                np.ascontiguousarray(database[start:end], dtype=np.float32)
            )
            self.database[start:end].copy_(host)

    def as_cuda(self, array: np.ndarray) -> torch.Tensor:
        host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        return host.to(self.device, non_blocking=False)

    def pose_distance(
        self, queries: torch.Tensor, candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        return _pose_min_distance(
            queries, self.database, candidate_indices, self.candidate_chunk_size
        )

    def pose_joint_distance(
        self, queries: torch.Tensor, candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        return _joint_min_distance(
            queries, self.database, candidate_indices, 3, self.candidate_chunk_size
        )

    def vector_distance(
        self, queries: torch.Tensor, candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        return _vector_min_distance(
            queries, self.database, candidate_indices, self.candidate_chunk_size
        )

    def vector_joint_distance(
        self, queries: torch.Tensor, candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        return _joint_min_distance(
            queries, self.database, candidate_indices, 3, self.candidate_chunk_size
        )
