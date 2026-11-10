"""Yaw-invariant root features derived from raw xyzw orientations."""

from __future__ import annotations

import numpy as np


ROOT_FEATURE_LAYOUT = "gravity_body_v2"


def convert_root_features(
    root_quaternion: np.ndarray,
    *world_vectors: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return body-frame gravity plus each world vector rotated into the body.

    ``root_quaternion`` is an ``(N, 4)`` body-to-world xyzw quaternion array.
    Every additional argument must be an ``(N, 3)`` world-frame vector array.
    The result is ``(root_tilt, *body_vectors)`` in contiguous float32 arrays.
    """
    quaternion = np.ascontiguousarray(root_quaternion, dtype=np.float32)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(
            f"Root quaternion must have shape (N, 4), got {quaternion.shape}"
        )
    if not np.isfinite(quaternion).all():
        raise ValueError("Root quaternion contains NaN or infinity")
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Root quaternion contains a near-zero value")
    quaternion = quaternion / norms

    x, y, z, w = quaternion.T
    # Physical gravity direction expressed in the body frame: R(q)^T [0, 0, -1].
    tilt = np.column_stack(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -1.0 + 2.0 * (x * x + y * y),
        )
    )
    tilt /= np.linalg.norm(tilt, axis=1, keepdims=True)
    result = [np.ascontiguousarray(tilt, dtype=np.float32)]

    inverse_qvec = -quaternion[:, :3]
    for vector in world_vectors:
        value = np.ascontiguousarray(vector, dtype=np.float32)
        if value.shape != (len(quaternion), 3):
            raise ValueError(
                f"Root world vector must have shape ({len(quaternion)}, 3), "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("Root world vector contains NaN or infinity")
        cross = 2.0 * np.cross(inverse_qvec, value)
        body = value + w[:, None] * cross + np.cross(inverse_qvec, cross)
        result.append(np.ascontiguousarray(body, dtype=np.float32))

    return tuple(result)
