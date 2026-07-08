"""Pure kinematic safety metrics for object-set torsion runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.operators.object import ObjectSet, TargetSelector, select_targets


def min_actor_distance(
    ego_xy: tuple[float, float],
    objects: ObjectSet,
    *,
    target: TargetSelector = "all",
    track_ids: Any | None = None,
    clearance: bool = False,
    ego_width: float = 0.0,
    ego_length: float = 0.0,
) -> float:
    """Return the minimum ego-to-actor distance.

    By default this is center distance. With ``clearance=True`` it subtracts
    bounding-circle radii derived from ego and actor boxes.
    """

    mask = select_targets(objects, target=target, track_ids=track_ids)
    if not np.any(mask):
        return float("inf")

    ego = _xy_vector(ego_xy, "ego_xy")
    distances = np.linalg.norm(objects.xy[mask] - ego, axis=1)
    if clearance:
        distances = distances - _bbox_radius(ego_width, ego_length) - _bbox_radii(
            objects.w[mask], objects.l[mask]
        )
    return float(np.min(distances))


def min_ttc(
    ego_xy: tuple[float, float],
    ego_v: tuple[float, float],
    objects: ObjectSet,
    *,
    target: TargetSelector = "all",
    track_ids: Any | None = None,
    ego_width: float = 0.0,
    ego_length: float = 0.0,
    horizon_s: float | None = None,
) -> float:
    """Return minimum time-to-collision using constant-velocity kinematics.

    TTC uses a bounding-circle approximation from each 2D bbox footprint. It
    returns ``inf`` when no selected actor reaches the collision radius.
    """

    if horizon_s is not None and (not np.isfinite(horizon_s) or horizon_s <= 0.0):
        raise ValueError("horizon_s must be positive and finite when provided")

    mask = select_targets(objects, target=target, track_ids=track_ids)
    if not np.any(mask):
        return float("inf")

    ego = _xy_vector(ego_xy, "ego_xy")
    ego_velocity = _xy_vector(ego_v, "ego_v")
    ego_radius = _bbox_radius(ego_width, ego_length)

    rel_pos = objects.xy[mask] - ego
    rel_vel = objects.v[mask] - ego_velocity
    combined_radii = ego_radius + _bbox_radii(objects.w[mask], objects.l[mask])

    values = np.array(
        [_pair_ttc(pos, vel, radius) for pos, vel, radius in zip(rel_pos, rel_vel, combined_radii)],
        dtype=np.float64,
    )
    if horizon_s is not None:
        values = values[values <= horizon_s]
    if values.size == 0:
        return float("inf")
    return float(np.min(values))


def has_bbox_collision(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_width: float,
    ego_length: float,
    objects: ObjectSet,
    *,
    target: TargetSelector = "all",
    track_ids: Any | None = None,
) -> bool:
    """Return True when the ego 2D oriented bbox overlaps any selected actor."""

    return bool(
        colliding_track_ids(
            ego_x,
            ego_y,
            ego_yaw,
            ego_width,
            ego_length,
            objects,
            target=target,
            track_ids=track_ids,
        )
    )


def colliding_track_ids(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_width: float,
    ego_length: float,
    objects: ObjectSet,
    *,
    target: TargetSelector = "all",
    track_ids: Any | None = None,
) -> tuple[Any, ...]:
    """Return track IDs whose 2D oriented bboxes overlap the ego bbox."""

    ego_corners = oriented_box_corners(ego_x, ego_y, ego_width, ego_length, ego_yaw)
    mask = select_targets(objects, target=target, track_ids=track_ids)
    collisions: list[Any] = []

    for idx in np.flatnonzero(mask):
        actor_corners = oriented_box_corners(
            float(objects.x[idx]),
            float(objects.y[idx]),
            float(objects.w[idx]),
            float(objects.l[idx]),
            float(objects.yaw[idx]),
        )
        if _convex_quads_overlap(ego_corners, actor_corners):
            collisions.append(objects.track_id[idx])

    return tuple(collisions)


def oriented_box_corners(
    x: float,
    y: float,
    width: float,
    length: float,
    yaw: float,
) -> NDArray[np.float64]:
    """Return 2D corners for a yawed bbox whose length axis points along yaw."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    width = _positive_float(width, "width")
    length = _positive_float(length, "length")
    yaw = _finite_float(yaw, "yaw")

    half_l = 0.5 * length
    half_w = 0.5 * width
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([x, y], dtype=np.float64)


def _pair_ttc(
    rel_pos: NDArray[np.float64],
    rel_vel: NDArray[np.float64],
    radius: float,
) -> float:
    a = float(rel_vel @ rel_vel)
    c = float(rel_pos @ rel_pos) - radius * radius
    if c <= 0.0:
        return 0.0
    if a <= 1e-12:
        return float("inf")

    b = 2.0 * float(rel_pos @ rel_vel)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return float("inf")

    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    roots = ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))
    future_roots = [root for root in roots if root >= 0.0]
    if not future_roots:
        return float("inf")
    return float(min(future_roots))


def _convex_quads_overlap(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    for corners in (a, b):
        edges = np.roll(corners, shift=-1, axis=0) - corners
        axes = np.column_stack((-edges[:, 1], edges[:, 0]))
        norms = np.linalg.norm(axes, axis=1)
        axes = axes[norms > 0.0] / norms[norms > 0.0, None]

        for axis in axes:
            proj_a = a @ axis
            proj_b = b @ axis
            if float(np.max(proj_a)) < float(np.min(proj_b)):
                return False
            if float(np.max(proj_b)) < float(np.min(proj_a)):
                return False
    return True


def _bbox_radius(width: float, length: float) -> float:
    return float(_bbox_radii(np.array([width], dtype=np.float64), np.array([length], dtype=np.float64))[0])


def _bbox_radii(width: NDArray[np.float64], length: NDArray[np.float64]) -> NDArray[np.float64]:
    if np.any(width < 0.0) or np.any(length < 0.0):
        raise ValueError("bbox width and length must be non-negative")
    return 0.5 * np.sqrt(width * width + length * length)


def _xy_vector(value: tuple[float, float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr


def _finite_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive_float(value: float, name: str) -> float:
    out = _finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


# TODO(phase-2/3): Simulator-derived metrics such as route completion, lane/off-road
# violations, brake reaction delay, and recovery time require CARLA/nuPlan integration.
