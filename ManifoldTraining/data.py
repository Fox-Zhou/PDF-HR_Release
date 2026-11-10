from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch

from PDFHR.field_specs import FieldSpec


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    rows: int


@dataclass(frozen=True)
class BatchRef:
    shard_index: int
    start: int
    end: int


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def discover_shards(
    sample_dir: Path, spec: FieldSpec, max_files: int = 0
) -> list[ShardInfo]:
    files = sorted(sample_dir.glob("batch_*.npy"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No batch_*.npy files found in {sample_dir}")

    shards: list[ShardInfo] = []
    for path in files:
        try:
            sample = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid NPY sample shard: {path}") from error
        try:
            if sample.ndim != 2 or sample.shape[1] != spec.sample_columns:
                raise ValueError(
                    f"{spec.name} sample must have shape (N, {spec.sample_columns}), "
                    f"got {sample.shape}: {path}"
                )
            if sample.dtype != np.float32:
                raise ValueError(
                    f"{spec.name} sample must be float32, got {sample.dtype}: {path}"
                )
            rows = int(sample.shape[0])
        finally:
            del sample
        if rows <= 0:
            raise ValueError(f"Sample shard has no rows: {path}")
        shards.append(ShardInfo(path=path, rows=rows))
    return shards


def _validation_key(path: Path, batch_index: int) -> float:
    digest = hashlib.sha256(f"{path.name}:{batch_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def build_batch_refs(
    shards: list[ShardInfo], batch_size: int, validation_fraction: float
) -> tuple[list[BatchRef], list[BatchRef], int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    train_refs: list[BatchRef] = []
    validation_refs: list[BatchRef] = []
    dropped_rows = 0
    for shard_index, shard in enumerate(shards):
        full_batches = shard.rows // batch_size
        dropped_rows += shard.rows - full_batches * batch_size
        for batch_index in range(full_batches):
            start = batch_index * batch_size
            ref = BatchRef(shard_index, start, start + batch_size)
            if (
                validation_fraction > 0
                and _validation_key(shard.path, batch_index) < validation_fraction
            ):
                validation_refs.append(ref)
            else:
                train_refs.append(ref)
    if not train_refs:
        raise ValueError("No full training batches remain; reduce batch_size")
    if validation_fraction > 0 and not validation_refs:
        validation_refs.append(train_refs.pop())
    if not train_refs:
        raise ValueError("Validation split consumed the only full batch")
    return train_refs, validation_refs, dropped_rows


class SampleStore:
    def __init__(
        self, shards: list[ShardInfo], spec: FieldSpec, *, root: bool
    ) -> None:
        self.shards = shards
        self.spec = spec
        self.root = bool(root)
        self.target_index = spec.target_index(self.root)
        self._arrays: dict[int, np.ndarray] = {}

    def array(self, shard_index: int) -> np.ndarray:
        array = self._arrays.get(shard_index)
        if array is None:
            array = np.load(
                self.shards[shard_index].path, mmap_mode="r", allow_pickle=False
            )
            self._arrays[shard_index] = array
        return array

    def load(self, ref: BatchRef) -> tuple[torch.Tensor, torch.Tensor]:
        rows = np.array(
            self.array(ref.shard_index)[ref.start : ref.end],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        if not np.isfinite(rows).all():
            path = self.shards[ref.shard_index].path
            raise ValueError(
                f"Sample batch contains NaN or infinity: "
                f"{path}[{ref.start}:{ref.end}]"
            )
        target = rows[:, self.target_index]
        if np.any(target < 0):
            raise ValueError(f"{self.spec.name} distance target must be non-negative")
        features = rows[:, : self.spec.input_dim]
        return torch.from_numpy(features), torch.from_numpy(target.copy())

    def close(self) -> None:
        self._arrays.clear()


def shuffled_rank_refs(
    refs: list[BatchRef],
    *,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
    max_batches_per_epoch: int,
) -> tuple[list[BatchRef], int]:
    shuffled = list(refs)
    random.Random(seed + epoch).shuffle(shuffled)
    if max_batches_per_epoch > 0:
        shuffled = shuffled[: max_batches_per_epoch * world_size]
    usable = len(shuffled) - len(shuffled) % world_size
    shuffled = shuffled[:usable]
    if not shuffled:
        raise ValueError(
            f"Not enough training batches ({len(refs)}) for world_size={world_size}"
        )
    return shuffled[rank::world_size], usable


def _select_stat_refs(
    refs: list[BatchRef], max_batches: int
) -> list[BatchRef]:
    if max_batches <= 0 or max_batches >= len(refs):
        return refs
    indices = np.linspace(0, len(refs) - 1, max_batches, dtype=np.int64)
    return [refs[int(index)] for index in indices]


def compute_feature_stats(
    store: SampleStore,
    refs: list[BatchRef],
    max_batches: int,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    selected = _select_stat_refs(refs, max_batches)
    sums = {
        group.name: np.zeros(group.width, dtype=np.float64)
        for group in store.spec.stat_groups
    }
    square_sums = {
        group.name: np.zeros(group.width, dtype=np.float64)
        for group in store.spec.stat_groups
    }
    total = 0
    for ref in selected:
        array = store.array(ref.shard_index)
        batch_rows = ref.end - ref.start
        for group in store.spec.stat_groups:
            values = np.asarray(
                array[ref.start : ref.end, group.value_slice], dtype=np.float64
            )
            if values.shape != (batch_rows, group.width):
                raise ValueError(
                    f"Statistics group {group.name} has unexpected shape {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Statistics group {group.name} contains NaN or infinity"
                )
            sums[group.name] += values.sum(axis=0, dtype=np.float64)
            square_sums[group.name] += np.square(values).sum(
                axis=0, dtype=np.float64
            )
        total += batch_rows
    if total <= 0:
        raise ValueError("No rows available for feature statistics")

    stats: dict[str, dict[str, np.ndarray]] = {}
    for group in store.spec.stat_groups:
        mean = sums[group.name] / total
        variance = np.maximum(
            square_sums[group.name] / total - np.square(mean), 1e-8
        )
        stats[group.name] = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
        }
    return stats, total
