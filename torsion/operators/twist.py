"""Canonical spatial and temporal TORSION twist fields.

The spatial field is a genuine local torsion: the rotation angle varies with
position,

    theta(p) = alpha * (||p-c|| / sigma) * exp(-0.5 * (||p-c|| / sigma)^2),

so it is not equivalent to a uniform rotation.  The Gaussian radial envelope
gives compact-ish support: near the pivot and far from the pivot the local
angle tends to zero, preserving the global frame while locally swirling the
representation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.operators.object import ObjectSet, TargetSelector, select_targets


def twist_angles(
    points: NDArray[np.float64] | Iterable[Iterable[float]] | Iterable[float],
    c: tuple[float, float] | NDArray[np.float64],
    alpha: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return the local twist angle ``theta(p)`` for each 2D point.

    Callers that attach vectors to points, such as object yaw or velocity, must
    rotate those vectors by this same local angle to keep the deformation
    coherent.
    """

    pts = _as_points(points, "points")
    center = _as_vector2(c, "c")
    alpha_value = _finite_float(alpha, "alpha")
    sigma_value = _positive_float(sigma, "sigma")

    rel = pts - center
    radius = np.linalg.norm(rel, axis=-1)
    scaled_radius = radius / sigma_value
    theta = alpha_value * scaled_radius * np.exp(-0.5 * scaled_radius * scaled_radius)
    return np.asarray(theta, dtype=np.float64)


def twist_points(
    points: NDArray[np.float64] | Iterable[Iterable[float]] | Iterable[float],
    c: tuple[float, float] | NDArray[np.float64],
    alpha: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Apply the local swirl ``T(p) = c + R(theta(p)) @ (p-c)`` to 2D points."""

    pts = _as_points(points, "points")
    center = _as_vector2(c, "c")
    theta = twist_angles(pts, center, alpha, sigma)
    return _rotate_about_center(pts, center, theta)


def twist_grid_inverse(
    coords: NDArray[np.float64] | Iterable[Iterable[Iterable[float]]],
    c: tuple[float, float] | NDArray[np.float64],
    alpha: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return source coordinates ``T^{-1}(p)`` for grid/field warping.

    The twist preserves radius around ``c``, so the inverse map uses the same
    radius-dependent local angle with the opposite sign.
    """

    grid = _as_points(coords, "coords")
    center = _as_vector2(c, "c")
    theta = -twist_angles(grid, center, alpha, sigma)
    return _rotate_about_center(grid, center, theta)


def temporal_curl_angle(t_offset: float, alpha: float, T_window: float) -> float:
    """Return the incremental heading twist inside a temporal fault window.

    ``t_offset`` is measured from the start of the active window.  The returned
    angle increases linearly from zero while ``0 <= t_offset < T_window`` and is
    zero outside that window.
    """

    offset = _finite_float(t_offset, "t_offset")
    alpha_value = _finite_float(alpha, "alpha")
    window = _positive_float(T_window, "T_window")
    if offset < 0.0 or offset >= window:
        return 0.0
    return float(alpha_value * offset / window)


def scene_swirl_torsion(
    objects: ObjectSet,
    *,
    pivot: tuple[float, float] | NDArray[np.float64] | None = None,
    alpha: float,
    sigma: float,
    target: TargetSelector = "all",
    track_ids: Iterable[Any] | Any | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
    bbox_size_atol: float = 0.0,
) -> ObjectSet:
    """Apply a coherent spatial swirl to selected object states.

    Position, yaw, and velocity are all driven by the same local angle
    ``theta(p_i)``.  Object count, class labels, track IDs, and bbox dimensions
    are preserved.
    """

    mask = select_targets(
        objects,
        target=target,
        track_ids=track_ids,
        ego_xy=ego_xy,
        front_axis=front_axis,
    )
    if not np.any(mask):
        return objects.copy()

    center = _resolve_pivot(objects, mask, pivot)
    selected_xy = objects.xy[mask]
    selected_theta = twist_angles(selected_xy, center, alpha, sigma)
    twisted_xy = twist_points(selected_xy, center, alpha, sigma)

    new_x = objects.x.copy()
    new_y = objects.y.copy()
    new_yaw = objects.yaw.copy()
    new_v = objects.v.copy()

    new_x[mask] = twisted_xy[:, 0]
    new_y[mask] = twisted_xy[:, 1]
    new_yaw[mask] = _wrap_to_pi(new_yaw[mask] + selected_theta)
    new_v[mask] = _rotate_vectors(new_v[mask], selected_theta)

    result = objects.replace(x=new_x, y=new_y, yaw=new_yaw, v=new_v)
    _assert_contract_preserved(objects, result, bbox_size_atol=bbox_size_atol)
    return result


def temporal_curl_torsion(
    objects: ObjectSet,
    *,
    target: TargetSelector = "all",
    alpha: float,
    T_window: float,
    t_offset: float,
    track_ids: Iterable[Any] | Any | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
    bbox_size_atol: float = 0.0,
) -> ObjectSet:
    """Rotate selected object velocities by the temporal curl angle.

    The constant-velocity predictor then sees a progressively curled target
    path across the active fault window.  The object-set contract is preserved.
    """

    mask = select_targets(
        objects,
        target=target,
        track_ids=track_ids,
        ego_xy=ego_xy,
        front_axis=front_axis,
    )
    theta = temporal_curl_angle(t_offset, alpha, T_window)
    if not np.any(mask) or abs(theta) <= 0.0:
        return objects.copy()

    new_v = objects.v.copy()
    new_v[mask] = _rotate_vectors(new_v[mask], np.full(int(mask.sum()), theta))

    result = objects.replace(v=new_v)
    _assert_contract_preserved(objects, result, bbox_size_atol=bbox_size_atol)
    return result


def _rotate_about_center(
    points: NDArray[np.float64],
    center: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    rel = points - center
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    out = np.empty_like(points, dtype=np.float64)
    out[..., 0] = center[0] + cos_t * rel[..., 0] - sin_t * rel[..., 1]
    out[..., 1] = center[1] + sin_t * rel[..., 0] + cos_t * rel[..., 1]
    return out


def _rotate_vectors(
    vectors: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    out = np.empty_like(vectors, dtype=np.float64)
    out[:, 0] = cos_t * vectors[:, 0] - sin_t * vectors[:, 1]
    out[:, 1] = sin_t * vectors[:, 0] + cos_t * vectors[:, 1]
    return out


def _resolve_pivot(
    objects: ObjectSet,
    mask: NDArray[np.bool_],
    pivot: tuple[float, float] | NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    if pivot is not None:
        return _as_vector2(pivot, "pivot")
    selected = objects.xy[mask]
    return np.mean(selected, axis=0, dtype=np.float64)


def _as_points(
    value: NDArray[np.float64] | Iterable[Iterable[float]] | Iterable[float],
    name: str,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (2,):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr.copy()
    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr.copy()


def _as_vector2(
    value: tuple[float, float] | NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr.copy()


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


def _wrap_to_pi(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _assert_contract_preserved(
    original: ObjectSet,
    result: ObjectSet,
    *,
    bbox_size_atol: float,
) -> None:
    if len(original) != len(result):
        raise AssertionError("object count changed")
    if not np.array_equal(original.cls, result.cls):
        raise AssertionError("object class changed")
    if not np.array_equal(original.track_id, result.track_id):
        raise AssertionError("track ID changed")
    if not np.allclose(original.bbox_size, result.bbox_size, rtol=0.0, atol=bbox_size_atol):
        raise AssertionError("bbox size changed")
