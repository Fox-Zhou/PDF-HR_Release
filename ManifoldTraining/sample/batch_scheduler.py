#!/usr/bin/env python3
"""Shared no-overwrite batch planning and atomic publication utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


SCHEDULE_TIMEOUT_SECONDS = 7200.0
CHALLENGE_REFRESH_SECONDS = 0.25


def batch_path(batch_dir: Path, batch_index: int) -> Path:
    return batch_dir / f"batch_{batch_index:04d}.npy"


def validate_complete_batch(path: Path, rows: int, columns: int) -> None:
    """Validate the header of an existing, atomically published batch."""
    try:
        batch = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Existing batch is not a valid NPY file: {path}") from error
    try:
        if batch.shape != (rows, columns):
            raise ValueError(
                f"Existing batch has shape {batch.shape}, expected "
                f"({rows}, {columns}): {path}"
            )
        if batch.dtype != np.float32:
            raise ValueError(
                f"Existing batch has dtype {batch.dtype}, expected float32: {path}"
            )
    finally:
        del batch


def batch_already_complete(path: Path, rows: int, columns: int) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"Batch output exists but is not a file: {path}")
    validate_complete_batch(path, rows, columns)
    return True


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Coordination file must contain a JSON object: {path}")
    return value


def _schedule_key(
    batch_start: int,
    batch_num: int,
    rows: int,
    columns: int,
    world_size: int,
) -> str:
    config = f"{batch_start}:{batch_num}:{rows}:{columns}:{world_size}"
    return hashlib.sha256(config.encode("utf-8")).hexdigest()[:16]


def _scan_missing_batches(
    batch_dir: Path,
    batch_start: int,
    batch_num: int,
    rows: int,
    columns: int,
) -> list[int]:
    missing: list[int] = []
    for index in range(batch_start, batch_start + batch_num):
        path = batch_path(batch_dir, index)
        if not batch_already_complete(path, rows, columns):
            missing.append(index)
    return missing


def _wait_for_current_challenge(
    challenge_path: Path,
    schedule_key: str,
    process_started_ns: int,
    deadline: float,
) -> str:
    while time.monotonic() < deadline:
        try:
            stat = challenge_path.stat()
            # Rank 0 refreshes the current challenge until every worker has
            # acknowledged it. Requiring a post-start mtime rejects stale plans.
            if stat.st_mtime_ns >= process_started_ns:
                payload = _read_json(challenge_path)
                if payload.get("schedule_key") == schedule_key:
                    token = payload.get("token")
                    if isinstance(token, str) and token:
                        return token
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(CHALLENGE_REFRESH_SECONDS)
    raise TimeoutError(f"Timed out waiting for rank-0 challenge: {challenge_path}")


def plan_missing_batches(
    batch_dir: Path,
    batch_start: int,
    batch_num: int,
    rows: int,
    columns: int,
    rank: int,
    world_size: int,
) -> tuple[list[int], list[int]]:
    """Return (all missing batches, batches assigned to this rank)."""
    if batch_num <= 0 or rows <= 0 or columns <= 0:
        raise ValueError("batch_num, rows, and columns must be positive")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed ranks: rank={rank}, world_size={world_size}")

    batch_dir.mkdir(parents=True, exist_ok=True)
    if world_size == 1:
        missing = _scan_missing_batches(
            batch_dir, batch_start, batch_num, rows, columns
        )
        return missing, missing

    key = _schedule_key(batch_start, batch_num, rows, columns, world_size)
    challenge_path = batch_dir / f".batch_schedule_{key}.challenge.json"
    process_started_ns = time.time_ns()
    deadline = time.monotonic() + SCHEDULE_TIMEOUT_SECONDS

    if rank == 0:
        token = uuid.uuid4().hex
        challenge = {"schedule_key": key, "token": token}
        acknowledgement_paths = [
            batch_dir / f".batch_schedule_{key}.{token}.rank{worker}.ack"
            for worker in range(1, world_size)
        ]
        while time.monotonic() < deadline:
            _atomic_write_json(challenge_path, challenge)
            if all(path.is_file() for path in acknowledgement_paths):
                break
            time.sleep(CHALLENGE_REFRESH_SECONDS)
        else:
            missing_ranks = [
                worker
                for worker, path in enumerate(acknowledgement_paths, start=1)
                if not path.is_file()
            ]
            raise TimeoutError(
                f"Timed out waiting for batch-plan acknowledgements from ranks "
                f"{missing_ranks}"
            )

        try:
            missing = _scan_missing_batches(
                batch_dir, batch_start, batch_num, rows, columns
            )
            plan = {"schedule_key": key, "token": token, "missing": missing}
        except Exception as error:
            plan = {
                "schedule_key": key,
                "token": token,
                "error": f"{type(error).__name__}: {error}",
            }
        plan_path = batch_dir / f".batch_schedule_{key}.{token}.plan.json"
        _atomic_write_json(plan_path, plan)
    else:
        token = _wait_for_current_challenge(
            challenge_path, key, process_started_ns, deadline
        )
        acknowledgement_path = (
            batch_dir / f".batch_schedule_{key}.{token}.rank{rank}.ack"
        )
        _atomic_write_json(
            acknowledgement_path,
            {"schedule_key": key, "token": token, "rank": rank},
        )
        plan_path = batch_dir / f".batch_schedule_{key}.{token}.plan.json"

    while not plan_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for batch plan: {plan_path}")
        time.sleep(CHALLENGE_REFRESH_SECONDS)
    plan = _read_json(plan_path)
    if plan.get("schedule_key") != key or plan.get("token") != token:
        raise RuntimeError(f"Batch-plan identity mismatch: {plan_path}")
    if "error" in plan:
        raise RuntimeError(f"Rank 0 failed to create batch plan: {plan['error']}")
    missing = plan.get("missing")
    if not isinstance(missing, list) or not all(isinstance(x, int) for x in missing):
        raise ValueError(f"Invalid missing-batch list in {plan_path}")
    assigned = missing[rank::world_size]
    return missing, assigned


def temporary_batch_path(final_path: Path, rank: int) -> Path:
    return final_path.with_name(
        f".{final_path.stem}.rank{rank}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy"
    )


def publish_temporary_batch(
    temporary_path: Path,
    final_path: Path,
    rows: int,
    columns: int,
) -> bool:
    """Atomically publish without replacing an existing final batch."""
    validate_complete_batch(temporary_path, rows, columns)
    with temporary_path.open("rb") as stream:
        os.fsync(stream.fileno())
    try:
        os.link(temporary_path, final_path)
        published = True
    except FileExistsError:
        validate_complete_batch(final_path, rows, columns)
        published = False
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return published


def save_array_no_overwrite(path: Path, samples: np.ndarray, rank: int) -> bool:
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    rows, columns = map(int, samples.shape)
    if batch_already_complete(path, rows, columns):
        return False
    temporary = temporary_batch_path(path, rank)
    try:
        np.save(temporary, samples, allow_pickle=False)
        return publish_temporary_batch(temporary, path, rows, columns)
    finally:
        if temporary.exists():
            temporary.unlink()
