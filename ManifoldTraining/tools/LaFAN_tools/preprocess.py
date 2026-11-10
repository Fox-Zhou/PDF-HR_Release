#!/usr/bin/env python3
"""Preprocess retargeted LaFAN1 Unitree G1 CSV clips into NumPy matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Retargeted LaFAN1 G1 trajectories are published at 30 frames per second.
FPS = 30.0

ROOT_POSITION_SLICE = slice(0, 3)
ROOT_QUATERNION_SLICE = slice(3, 7)
JOINT_POSITION_SLICE = slice(7, 36)
SOURCE_COLUMN_COUNT = 36
JOINT_COUNT = 29


def normalize_quaternion_sequence(quat: np.ndarray) -> np.ndarray:
    """Normalize and sign-unwrap an xyzw quaternion sequence."""
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("root quaternion contains a near-zero value")
    quat = quat / norms
    # q and -q encode the same rotation. Select a continuous sequence so that
    # downstream relative rotations and interpolation do not see sign jumps.
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    return quat


def quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., :3] *= -1.0
    return result


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(lhs, -1, 0)
    rx, ry, rz, rw = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Return shortest-arc rotation vectors for normalized xyzw quaternions."""
    quat = quat.copy()
    quat[quat[:, 3] < 0.0] *= -1.0
    vector = quat[:, :3]
    vector_norm = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quat[:, 3], -1.0, 1.0))
    scale = np.empty_like(angle)
    small = vector_norm < 1e-8
    scale[small] = 2.0
    scale[~small] = angle[~small] / vector_norm[~small]
    return vector * scale[:, None]


def numerical_derivative(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) == 1:
        return np.zeros_like(values)
    if len(values) == 2:
        derivative = (values[1] - values[0]) / dt
        return np.stack((derivative, derivative))
    return np.gradient(values, dt, axis=0, edge_order=2)


def world_angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """Compute frame-aligned world-frame angular velocity from xyzw poses."""
    if len(quat) == 1:
        return np.zeros((1, 3), dtype=quat.dtype)
    # q_delta = q_next * inverse(q_current) is expressed in the world frame.
    delta = quaternion_multiply(quat[1:], quaternion_conjugate(quat[:-1]))
    interval_velocity = quaternion_to_rotvec(delta) / dt
    velocity = np.empty((len(quat), 3), dtype=quat.dtype)
    velocity[0] = interval_velocity[0]
    velocity[-1] = interval_velocity[-1]
    if len(quat) > 2:
        velocity[1:-1] = 0.5 * (interval_velocity[:-1] + interval_velocity[1:])
    return velocity


def load_csv(path: Path) -> np.ndarray:
    # LaFAN1 retargeting CSV files contain numeric rows without a header.
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[0] == 0:
        raise ValueError(f"empty motion: {path}")
    if data.shape[1] != SOURCE_COLUMN_COUNT:
        raise ValueError(
            f"expected {SOURCE_COLUMN_COUNT} columns, got {data.shape[1]}: {path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"motion contains NaN or infinity: {path}")
    return data


def preprocess_motion(input_path: Path, output_path: Path) -> None:
    data = load_csv(input_path)
    dt = 1.0 / FPS

    # Source root position is intentionally discarded. Source quaternions are
    # already xyzw and joint positions are already radians.
    root_quat = normalize_quaternion_sequence(
        data[:, ROOT_QUATERNION_SLICE].copy()
    )
    joint_pos = np.unwrap(data[:, JOINT_POSITION_SLICE], axis=0)
    if joint_pos.shape[1] != JOINT_COUNT:
        raise ValueError(f"expected {JOINT_COUNT} joints, got {joint_pos.shape[1]}")

    root_ang_vel = world_angular_velocity(root_quat, dt)
    joint_vel = numerical_derivative(joint_pos, dt)
    root_ang_acc = numerical_derivative(root_ang_vel, dt)
    joint_acc = numerical_derivative(joint_vel, dt)

    arrays = [
        root_quat,
        root_ang_vel,
        root_ang_acc,
        joint_pos,
        joint_vel,
        joint_acc,
    ]
    arrays = [np.ascontiguousarray(value, dtype=np.float32) for value in arrays]
    features = np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features, allow_pickle=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert retargeted LaFAN1 G1 CSV files to NPY matrices."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing CSV motions")
    parser.add_argument("output_dir", type=Path, help="Destination folder")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace already existing NPY files"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid file instead of processing the remaining files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if output_dir == input_dir or input_dir in output_dir.parents:
        print("error: output_dir must not be input_dir or one of its children", file=sys.stderr)
        return 2

    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        print(f"error: no CSV files found below {input_dir}", file=sys.stderr)
        return 2

    processed = skipped = failed = 0
    for index, input_path in enumerate(csv_paths, start=1):
        relative_path = input_path.relative_to(input_dir).with_suffix(".npy")
        output_path = output_dir / relative_path
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            preprocess_motion(input_path, output_path)
            processed += 1
        except Exception as error:  # keep a long batch useful when one clip is bad
            failed += 1
            print(f"[ERROR] {input_path}: {error}", file=sys.stderr)
            if args.strict:
                raise
        if index == 1 or index % 1000 == 0 or index == len(csv_paths):
            print(
                f"[{index}/{len(csv_paths)}] processed={processed} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
