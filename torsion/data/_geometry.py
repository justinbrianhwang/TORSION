"""Shared geometry helpers for dataset adapters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def quaternion_yaw(quaternion: Iterable[float]) -> float:
    """Return yaw from a ``[qw, qx, qy, qz]`` quaternion."""

    q = np.asarray(list(quaternion), dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite [qw, qx, qy, qz] sequence")
    qw, qx, qy, qz = q
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def rotate_xy(xy: Iterable[float], angle_rad: float) -> NDArray[np.float64]:
    """Rotate a 2D vector by ``angle_rad`` using a right-handed matrix."""

    vec = _vector2(xy, "xy")
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float64)


def global_to_ego_xy(
    global_xy: Iterable[float],
    ego_xy: Iterable[float],
    ego_yaw: float,
) -> NDArray[np.float64]:
    """Transform global ``xy`` to ego-local coordinates: ``x`` forward, ``y`` left."""

    delta = _vector2(global_xy, "global_xy") - _vector2(ego_xy, "ego_xy")
    return rotate_xy(delta, -float(ego_yaw))


def global_to_ego_velocity(
    velocity_xy: Iterable[float],
    ego_yaw: float,
) -> NDArray[np.float64]:
    """Rotate a global 2D velocity vector into the ego-local frame."""

    return rotate_xy(velocity_xy, -float(ego_yaw))


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)


def _vector2(value: Iterable[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr.copy()
