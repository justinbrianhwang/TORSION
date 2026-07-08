"""Semantic-preserving cost-map torsion operators.

Cost maps are 2D arrays ``C in R^{H x W}`` with values in ``[0, 1]``:
``0`` means low-cost/free space and ``1`` means high-cost/blocked space.

Operator coordinates are grid-cell coordinates, not meters.  A point
``(x, y)`` means ``x == column`` and ``y == row``.  Scenario runners that work
in meters are responsible for converting between metric coordinates and this
grid coordinate frame.

The B.4 preservation constraints are enforced consistently:

* the cost value range is clipped back to ``[0, 1]``;
* deformations are local, with exact no-op behavior outside a finite support;
* operators are deterministic and never mutate their input array;
* edits preserve the grid shape and avoid random global destruction of drivable
  topology.

Callers that have an explicit road/lane-boundary mask can pass ``fixed_mask``.
Those cells are copied back exactly from the input after the local edit.  The
Phase 2b cost-map runner uses the cleaner decomposition from design B.4: it
keeps the road prior fixed, perturbs only the obstacle/free-space residual, and
then pins road/lane-boundary cells back to the clean map on recombination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.operators.twist import twist_grid_inverse

COST_MIN = 0.0
COST_MAX = 1.0
LOCAL_SUPPORT_SIGMAS = 4.0


def spatial_cost_warp(
    C: NDArray[np.float64] | Any,
    pivot: tuple[float, float] | NDArray[np.float64],
    alpha: float,
    sigma: float,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Warp a cost grid by the canonical local swirl field.

    The returned map is ``C_tilde(p) = C(T^{-1}(p))`` where ``T`` is the spatial
    twist from :mod:`torsion.operators.twist`.  Cells farther than
    ``4 * sigma`` from ``pivot`` are copied exactly from the input, giving the
    compact local support required by the cost-map representation contract.
    """

    cost = _as_cost_map(C)
    center = _as_vector2(pivot, "pivot")
    alpha_value = _finite_float(alpha, "alpha")
    sigma_value = _positive_float(sigma, "sigma")

    if abs(alpha_value) <= 0.0:
        return cost.copy()

    coords = _grid_coordinates(cost.shape)
    radius = np.linalg.norm(coords - center, axis=-1)
    support = radius <= LOCAL_SUPPORT_SIGMAS * sigma_value

    source = twist_grid_inverse(coords, center, alpha_value, sigma_value)
    sampled = _bilinear_sample(cost, source[..., 0], source[..., 1])

    out = cost.copy()
    out[support] = sampled[support]
    return _preserve_fixed_cells(cost, _clip_cost(out), fixed_mask)


def translate_cost_field(
    C: NDArray[np.float64] | Any,
    shift_xy: tuple[float, float] | NDArray[np.float64],
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Translate a cost field by a uniform grid-cell ``(dx, dy)`` shift.

    Positive ``dx`` moves content toward larger column indices; positive ``dy``
    moves content toward larger row indices.  Sampling uses the inverse shift
    and clipped bilinear interpolation, so the output shape and cost range are
    preserved.
    """

    cost = _as_cost_map(C)
    shift = _as_vector2(shift_xy, "shift_xy")
    if np.allclose(shift, 0.0, rtol=0.0, atol=0.0):
        return cost.copy()

    coords = _grid_coordinates(cost.shape)
    source = coords - shift
    out = _bilinear_sample(cost, source[..., 0], source[..., 1])
    return _preserve_fixed_cells(cost, _clip_cost(out), fixed_mask)


def gaussian_cost_noise(
    C: NDArray[np.float64] | Any,
    noise: NDArray[np.float64] | Any,
    scale: float,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Add a deterministic caller-supplied Gaussian noise field to a cost map."""

    cost = _as_cost_map(C)
    noise_grid = np.asarray(noise, dtype=np.float64)
    if noise_grid.shape != cost.shape or not np.all(np.isfinite(noise_grid)):
        raise ValueError("noise must be a finite array matching the cost map shape")
    scale_value = _finite_float(scale, "scale")
    out = cost + scale_value * noise_grid
    return _preserve_fixed_cells(cost, _clip_cost(out), fixed_mask)


def directional_obstacle_inflation(
    C: NDArray[np.float64] | Any,
    center: tuple[float, float] | NDArray[np.float64],
    beta: float,
    cov: NDArray[np.float64] | Any,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Add a local anisotropic Gaussian obstacle-cost bump.

    ``cov`` is a 2x2 covariance matrix in grid-cell units.  The bump is
    evaluated only inside a finite Mahalanobis-radius support, so cells far from
    the obstacle are unchanged exactly.
    """

    cost = _as_cost_map(C)
    beta_value = _nonnegative_float(beta, "beta")
    bump = _gaussian_bump(cost.shape, center=center, cov=cov)
    return _preserve_fixed_cells(cost, _clip_cost(cost + beta_value * bump), fixed_mask)


def obstacle_deflation(
    C: NDArray[np.float64] | Any,
    center: tuple[float, float] | NDArray[np.float64],
    beta: float,
    cov: NDArray[np.float64] | Any,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Subtract a local anisotropic Gaussian obstacle-cost bump."""

    cost = _as_cost_map(C)
    beta_value = _nonnegative_float(beta, "beta")
    bump = _gaussian_bump(cost.shape, center=center, cov=cov)
    return _preserve_fixed_cells(cost, _clip_cost(cost - beta_value * bump), fixed_mask)


def lane_boundary_shear(
    C: NDArray[np.float64] | Any,
    s0: float,
    kappa: float,
    sigma: float,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Apply a local Frenet-style shear to lane-boundary cost.

    Grid column ``x`` is treated as longitudinal coordinate ``s`` and grid row
    ``y`` as lateral coordinate ``n``.  The inverse shear samples

    ``source_y = y - kappa * (x - s0)^2 * exp(-((x - s0)^2 / sigma^2))``.

    Columns farther than ``4 * sigma`` from ``s0`` are copied exactly from the
    input to keep the deformation local.
    """

    cost = _as_cost_map(C)
    s0_value = _finite_float(s0, "s0")
    kappa_value = _finite_float(kappa, "kappa")
    sigma_value = _positive_float(sigma, "sigma")

    yy, xx = np.indices(cost.shape, dtype=np.float64)
    dx = xx - s0_value
    support = np.abs(dx) <= LOCAL_SUPPORT_SIGMAS * sigma_value
    shear = kappa_value * dx * dx * np.exp(-((dx * dx) / (sigma_value * sigma_value)))
    sampled = _bilinear_sample(cost, xx, yy - shear)

    out = cost.copy()
    out[support] = sampled[support]
    return _preserve_fixed_cells(cost, _clip_cost(out), fixed_mask)


def false_free_space(
    C: NDArray[np.float64] | Any,
    region: tuple[float, float, float, float] | NDArray[np.bool_] | Any,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Create a deterministic local cost hole in ``region``."""

    cost = _as_cost_map(C)
    mask = _region_mask(cost.shape, region)
    out = cost.copy()
    out[mask] = COST_MIN
    return _preserve_fixed_cells(cost, out, fixed_mask)


def false_wall(
    C: NDArray[np.float64] | Any,
    region: tuple[float, float, float, float] | NDArray[np.bool_] | Any,
    *,
    fixed_mask: NDArray[np.bool_] | Any | None = None,
) -> NDArray[np.float64]:
    """Create a deterministic local high-cost wall in ``region``."""

    cost = _as_cost_map(C)
    mask = _region_mask(cost.shape, region)
    out = cost.copy()
    out[mask] = COST_MAX
    return _preserve_fixed_cells(cost, out, fixed_mask)


def sample_cost_grid(
    C: NDArray[np.float64] | Any,
    points_xy: NDArray[np.float64] | Any,
) -> NDArray[np.float64]:
    """Bilinearly sample a cost grid at grid-cell ``(x, y)`` points."""

    cost = _as_cost_map(C)
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("points_xy must have shape (..., 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xy must contain only finite values")
    return _bilinear_sample(cost, points[..., 0], points[..., 1])


def _gaussian_bump(
    shape: tuple[int, int],
    *,
    center: tuple[float, float] | NDArray[np.float64],
    cov: NDArray[np.float64] | Any,
) -> NDArray[np.float64]:
    center_xy = _as_vector2(center, "center")
    covariance = _as_covariance(cov)
    inv_cov = np.linalg.inv(covariance)

    coords = _grid_coordinates(shape)
    delta = coords - center_xy
    maha2 = np.einsum("...i,ij,...j->...", delta, inv_cov, delta)
    support = maha2 <= LOCAL_SUPPORT_SIGMAS * LOCAL_SUPPORT_SIGMAS

    bump = np.zeros(shape, dtype=np.float64)
    bump[support] = np.exp(-0.5 * maha2[support])
    return bump


def _bilinear_sample(
    cost: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    h, w = cost.shape
    x_clipped = np.clip(x, 0.0, float(w - 1))
    y_clipped = np.clip(y, 0.0, float(h - 1))

    x0 = np.floor(x_clipped).astype(np.int64)
    y0 = np.floor(y_clipped).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = x_clipped - x0
    wy = y_clipped - y0

    top = (1.0 - wx) * cost[y0, x0] + wx * cost[y0, x1]
    bottom = (1.0 - wx) * cost[y1, x0] + wx * cost[y1, x1]
    return (1.0 - wy) * top + wy * bottom


def _grid_coordinates(shape: tuple[int, int]) -> NDArray[np.float64]:
    yy, xx = np.indices(shape, dtype=np.float64)
    return np.stack((xx, yy), axis=-1)


def _region_mask(
    shape: tuple[int, int],
    region: tuple[float, float, float, float] | NDArray[np.bool_] | Any,
) -> NDArray[np.bool_]:
    mask = np.asarray(region)
    if mask.dtype == bool:
        if mask.shape != shape:
            raise ValueError("boolean region mask must match cost grid shape")
        return mask.astype(bool, copy=True)

    values = np.asarray(region, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("region must be a boolean mask or (x_min, y_min, x_max, y_max)")
    x_min, y_min, x_max, y_max = values
    if x_max < x_min or y_max < y_min:
        raise ValueError("region bounds must be ordered")

    yy, xx = np.indices(shape, dtype=np.float64)
    return (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)


def _preserve_fixed_cells(
    original: NDArray[np.float64],
    modified: NDArray[np.float64],
    fixed_mask: NDArray[np.bool_] | Any | None,
) -> NDArray[np.float64]:
    if fixed_mask is None:
        return modified
    mask = np.asarray(fixed_mask, dtype=bool)
    if mask.shape != original.shape:
        raise ValueError("fixed_mask must match cost grid shape")
    out = modified.copy()
    out[mask] = original[mask]
    return out


def _as_cost_map(value: NDArray[np.float64] | Any) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("cost map must be a 2D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cost map must contain only finite values")
    return arr.copy()


def _as_vector2(value: tuple[float, float] | NDArray[np.float64], name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr.copy()


def _as_covariance(value: NDArray[np.float64] | Any) -> NDArray[np.float64]:
    cov = np.asarray(value, dtype=np.float64)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError("cov must be a finite 2x2 matrix")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12):
        raise ValueError("cov must be symmetric")
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals <= 0.0):
        raise ValueError("cov must be positive definite")
    return cov.copy()


def _clip_cost(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(value, COST_MIN, COST_MAX).astype(np.float64, copy=False)


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


def _nonnegative_float(value: float, name: str) -> float:
    out = _finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out
