#!/usr/bin/env python3
"""Preprocess BONES-SEED Unitree G1 CSV clips into NumPy matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# BONES-SEED G1 trajectories are published at 120 frames per second.
FPS = 120.0

ROOT_TRANSLATION_COLUMNS = [
    "root_translateX",
    "root_translateY",
    "root_translateZ",
]
ROOT_EULER_COLUMNS = ["root_rotateX", "root_rotateY", "root_rotateZ"]
JOINT_NAMES = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]
EXPECTED_COLUMNS = [
    "Frame",
    *ROOT_TRANSLATION_COLUMNS,
    *ROOT_EULER_COLUMNS,
    *JOINT_NAMES,
]


def euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert extrinsic XYZ / roll-pitch-yaw radians to xyzw quaternions."""
    half = 0.5 * euler
    cx, cy, cz = np.cos(half).T
    sx, sy, sz = np.sin(half).T
    quat = np.column_stack(
        (
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
    )
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # q and -q encode the same rotation. Select a continuous sequence so that
    # downstream interpolation and numerical processing do not see sign jumps.
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
    with path.open("r", encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
    if header != EXPECTED_COLUMNS:
        raise ValueError(
            f"unexpected CSV columns in {path}; expected {EXPECTED_COLUMNS}, got {header}"
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if data.shape[0] == 0:
        raise ValueError(f"empty motion: {path}")
    if data.shape[1] != len(EXPECTED_COLUMNS):
        raise ValueError(f"expected {len(EXPECTED_COLUMNS)} columns, got {data.shape[1]}")
    if not np.isfinite(data).all():
        raise ValueError(f"motion contains NaN or infinity: {path}")
    expected_frames = np.arange(data.shape[0], dtype=data.dtype)
    if not np.array_equal(data[:, 0], expected_frames):
        raise ValueError(f"Frame column must contain consecutive indices starting at zero: {path}")
    return data


def preprocess_motion(input_path: Path, output_path: Path) -> None:
    data = load_csv(input_path)
    dt = 1.0 / FPS

    root_quat = euler_xyz_to_quaternion(np.deg2rad(data[:, 4:7]))
    joint_pos = np.unwrap(np.deg2rad(data[:, 7:]), axis=0)

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
        description="Recursively convert BONES-SEED G1 CSV files to NumPy NPY matrices."
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
