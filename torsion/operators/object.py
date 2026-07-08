"""Semantic-preserving object-set torsion operators.

The object-set contract preserves object count, class labels, track IDs, and
bounding-box dimensions. Operators only perturb semantics inside that contract.

Velocity is represented as a 2D vector ``v = [vx, vy]`` in the same frame as
``x`` and ``y``. For target selection, the default ego frame assumes +x is
forward and +y is left.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

MagnitudeName = Literal["low", "medium", "high"]
TargetSelector = str | Iterable[Any] | NDArray[np.bool_]

DEFAULT_MAX_POSITION_SHIFT_M = 2.0
DEFAULT_MAX_YAW_DELTA_RAD = float(np.deg2rad(30.0))
DEFAULT_MAX_VELOCITY_ROTATION_RAD = float(np.deg2rad(30.0))
DEFAULT_MAX_CONFIDENCE_DELTA = 0.5


@dataclass(frozen=True)
class ObjectMagnitude:
    """Magnitude values for object-set torsion."""

    position_shift_m: float
    yaw_shift_rad: float
    velocity_rotation_rad: float
    confidence_delta: float

    @property
    def yaw_shift_deg(self) -> float:
        return float(np.rad2deg(self.yaw_shift_rad))

    @property
    def velocity_rotation_deg(self) -> float:
        return float(np.rad2deg(self.velocity_rotation_rad))


MAGNITUDE_LEVELS: dict[str, ObjectMagnitude] = {
    "low": ObjectMagnitude(
        position_shift_m=0.2,
        yaw_shift_rad=float(np.deg2rad(2.0)),
        velocity_rotation_rad=float(np.deg2rad(2.0)),
        confidence_delta=0.05,
    ),
    "medium": ObjectMagnitude(
        position_shift_m=0.5,
        yaw_shift_rad=float(np.deg2rad(5.0)),
        velocity_rotation_rad=float(np.deg2rad(5.0)),
        confidence_delta=0.10,
    ),
    "high": ObjectMagnitude(
        position_shift_m=1.0,
        yaw_shift_rad=float(np.deg2rad(10.0)),
        velocity_rotation_rad=float(np.deg2rad(10.0)),
        confidence_delta=0.20,
    ),
}


@dataclass(frozen=True)
class ObjectSet:
    """Object-set representation used by Phase 1 torsion operators."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    w: NDArray[np.float64]
    h: NDArray[np.float64]
    l: NDArray[np.float64]
    yaw: NDArray[np.float64]
    v: NDArray[np.float64]
    cls: NDArray[np.object_]
    conf: NDArray[np.float64]
    track_id: NDArray[np.object_]

    def __post_init__(self) -> None:
        numeric_1d = ("x", "y", "z", "w", "h", "l", "yaw", "conf")
        n: int | None = None

        for name in numeric_1d:
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array")
            if n is None:
                n = int(arr.shape[0])
            elif arr.shape[0] != n:
                raise ValueError(f"{name} length {arr.shape[0]} does not match {n}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")
            arr = arr.copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

        if n is None:
            n = 0

        v = np.asarray(self.v, dtype=np.float64)
        if v.ndim == 1 and v.shape == (2,) and n == 1:
            v = v.reshape(1, 2)
        if v.ndim != 2 or v.shape != (n, 2):
            raise ValueError(f"v must have shape ({n}, 2)")
        if not np.all(np.isfinite(v)):
            raise ValueError("v must contain only finite values")
        v = v.copy()
        v.setflags(write=False)
        object.__setattr__(self, "v", v)

        cls = np.asarray(self.cls, dtype=object)
        track_id = np.asarray(self.track_id, dtype=object)
        for name, arr in (("cls", cls), ("track_id", track_id)):
            if arr.ndim == 0 and n == 1:
                arr = arr.reshape(1)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(f"{name} must be a 1D array with length {n}")
            arr = arr.copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

        if np.any(self.w <= 0.0) or np.any(self.h <= 0.0) or np.any(self.l <= 0.0):
            raise ValueError("bbox dimensions w, h, and l must be positive")
        if np.any((self.conf < 0.0) | (self.conf > 1.0)):
            raise ValueError("conf must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def xy(self) -> NDArray[np.float64]:
        return np.column_stack((self.x, self.y))

    @property
    def bbox_size(self) -> NDArray[np.float64]:
        return np.column_stack((self.w, self.h, self.l))

    def copy(self) -> ObjectSet:
        """Return a deep copy with read-only arrays."""

        return self.replace()

    def replace(self, **updates: Any) -> ObjectSet:
        """Return a new ObjectSet with selected fields replaced."""

        data = {
            "x": self.x.copy(),
            "y": self.y.copy(),
            "z": self.z.copy(),
            "w": self.w.copy(),
            "h": self.h.copy(),
            "l": self.l.copy(),
            "yaw": self.yaw.copy(),
            "v": self.v.copy(),
            "cls": self.cls.copy(),
            "conf": self.conf.copy(),
            "track_id": self.track_id.copy(),
        }
        data.update(updates)
        return ObjectSet(**data)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> ObjectSet:
        """Build an ObjectSet from dictionaries with A.2 field names."""

        rows = list(records)
        if not rows:
            empty = np.array([], dtype=np.float64)
            return cls(
                x=empty,
                y=empty,
                z=empty,
                w=empty,
                h=empty,
                l=empty,
                yaw=empty,
                v=np.empty((0, 2), dtype=np.float64),
                cls=np.array([], dtype=object),
                conf=empty,
                track_id=np.array([], dtype=object),
            )

        velocities = []
        for row in rows:
            if "v" in row:
                velocities.append(row["v"])
            else:
                velocities.append((row["vx"], row["vy"]))

        return cls(
            x=[row["x"] for row in rows],
            y=[row["y"] for row in rows],
            z=[row["z"] for row in rows],
            w=[row["w"] for row in rows],
            h=[row["h"] for row in rows],
            l=[row["l"] for row in rows],
            yaw=[row["yaw"] for row in rows],
            v=velocities,
            cls=[row["cls"] for row in rows],
            conf=[row["conf"] for row in rows],
            track_id=[row["track_id"] for row in rows],
        )


def resolve_magnitude(magnitude: MagnitudeName | str | ObjectMagnitude) -> ObjectMagnitude:
    """Resolve a named object-set magnitude to numeric values."""

    if isinstance(magnitude, ObjectMagnitude):
        return magnitude
    key = str(magnitude).lower().strip()
    try:
        return MAGNITUDE_LEVELS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MAGNITUDE_LEVELS))
        raise ValueError(f"unknown magnitude {magnitude!r}; expected one of {valid}") from exc


def select_targets(
    objects: ObjectSet,
    target: TargetSelector = "all",
    *,
    track_ids: Iterable[Any] | Any | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
) -> NDArray[np.bool_]:
    """Return a boolean mask for all, explicit track IDs, or nearest-front target."""

    if track_ids is not None:
        return _mask_for_track_ids(objects, track_ids)

    target_array = np.asarray(target) if not isinstance(target, str) else None
    if target_array is not None and target_array.dtype == bool:
        if target_array.ndim != 1 or target_array.shape[0] != len(objects):
            raise ValueError("boolean target mask must be 1D and match object count")
        return target_array.astype(bool, copy=True)

    if not isinstance(target, str):
        return _mask_for_track_ids(objects, target)

    selector = target.lower().strip().replace("-", "_")
    if selector == "all":
        return np.ones(len(objects), dtype=bool)
    if selector == "none":
        return np.zeros(len(objects), dtype=bool)
    if selector in {"track_id", "track_ids"}:
        raise ValueError("track_ids must be provided when target='track_ids'")
    if selector in {"nearest_front", "nearest_front_vehicle"}:
        return _nearest_front_mask(objects, ego_xy=ego_xy, front_axis=front_axis)

    raise ValueError(f"unknown target selector {target!r}")


def position_torsion(
    objects: ObjectSet,
    *,
    magnitude: MagnitudeName | str | ObjectMagnitude = "medium",
    dx: float | None = None,
    dy: float | None = None,
    target: TargetSelector = "all",
    track_ids: Iterable[Any] | Any | None = None,
    seed: int | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
    max_shift_m: float = DEFAULT_MAX_POSITION_SHIFT_M,
    bbox_size_atol: float = 0.0,
) -> ObjectSet:
    """Shift selected object centers in x/y without changing identity or size."""

    level = resolve_magnitude(magnitude)
    mask = select_targets(
        objects, target=target, track_ids=track_ids, ego_xy=ego_xy, front_axis=front_axis
    )
    target_count = int(mask.sum())
    new_x = objects.x.copy()
    new_y = objects.y.copy()

    if target_count:
        if dx is None and dy is None:
            rng = np.random.default_rng(seed)
            angles = rng.uniform(-np.pi, np.pi, size=target_count)
            deltas = np.column_stack(
                (
                    level.position_shift_m * np.cos(angles),
                    level.position_shift_m * np.sin(angles),
                )
            )
        else:
            dx_value = 0.0 if dx is None else _finite_float(dx, "dx")
            dy_value = 0.0 if dy is None else _finite_float(dy, "dy")
            deltas = np.tile(np.array([dx_value, dy_value], dtype=np.float64), (target_count, 1))

        _validate_shift_deltas(deltas, max_shift_m=max_shift_m)
        new_x[mask] = new_x[mask] + deltas[:, 0]
        new_y[mask] = new_y[mask] + deltas[:, 1]

    result = objects.replace(x=new_x, y=new_y)
    _assert_contract_preserved(objects, result, bbox_size_atol=bbox_size_atol)
    return result


def yaw_torsion(
    objects: ObjectSet,
    *,
    magnitude: MagnitudeName | str | ObjectMagnitude = "medium",
    d_yaw: float | None = None,
    d_yaw_deg: float | None = None,
    target: TargetSelector = "all",
    track_ids: Iterable[Any] | Any | None = None,
    seed: int | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
    max_yaw_delta_rad: float = DEFAULT_MAX_YAW_DELTA_RAD,
    bbox_size_atol: float = 0.0,
) -> ObjectSet:
    """Add a bounded yaw delta to selected objects."""

    level = resolve_magnitude(magnitude)
    delta_value, explicit = _resolve_angle(
        level.yaw_shift_rad, radians=d_yaw, degrees=d_yaw_deg, name="d_yaw"
    )
    mask = select_targets(
        objects, target=target, track_ids=track_ids, ego_xy=ego_xy, front_axis=front_axis
    )
    target_count = int(mask.sum())
    new_yaw = objects.yaw.copy()

    if target_count:
        if explicit:
            deltas = np.full(target_count, delta_value, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed)
            signs = rng.choice(np.array([-1.0, 1.0]), size=target_count)
            deltas = signs * delta_value

        _validate_angle_deltas(deltas, max_abs_rad=max_yaw_delta_rad, name="yaw delta")
        new_yaw[mask] = _wrap_to_pi(new_yaw[mask] + deltas)

    result = objects.replace(yaw=new_yaw)
    _assert_contract_preserved(objects, result, bbox_size_atol=bbox_size_atol)
    return result


def velocity_direction_torsion(
    objects: ObjectSet,
    *,
    magnitude: MagnitudeName | str | ObjectMagnitude = "medium",
    theta: float | None = None,
    theta_deg: float | None = None,
    target: TargetSelector = "all",
    track_ids: Iterable[Any] | Any | None = None,
    seed: int | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
    max_rotation_rad: float = DEFAULT_MAX_VELOCITY_ROTATION_RAD,
    bbox_size_atol: float = 0.0,
) -> ObjectSet:
    """Rotate selected 2D velocity vectors while preserving speed magnitude."""

    level = resolve_magnitude(magnitude)
    theta_value, explicit = _resolve_angle(
        level.velocity_rotation_rad, radians=theta, degrees=theta_deg, name="theta"
    )
    mask = select_targets(
        objects, target=target, track_ids=track_ids, ego_xy=ego_xy, front_axis=front_axis
    )
    target_count = int(mask.sum())
    new_v = objects.v.copy()

    if target_count:
        if explicit:
            deltas = np.full(target_count, theta_value, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed)
            signs = rng.choice(np.array([-1.0, 1.0]), size=target_count)
            deltas = signs * theta_value

        _validate_angle_deltas(deltas, max_abs_rad=max_rotation_rad, name="velocity rotation")
        c = np.cos(deltas)
        s = np.sin(deltas)
        vx = new_v[mask, 0].copy()
        vy = new_v[mask, 1].copy()
        new_v[mask, 0] = c * vx - s * vy
        new_v[mask, 1] = s * vx + c * vy

    result = objects.replace(v=new_v)
    _assert_contract_preserved(objects, result, bbox_size_atol=bbox_size_atol)
    return result


def confidence_redistribution(
    objects: ObjectSet,
    *,
    magnitude: MagnitudeName | str | ObjectMagnitude = "medium",
    delta: float | None = None,
    target: TargetSelector = "all",
    track_ids: Iterable[Any] | Any | None = None,
    seed: int | None = None,
    ego_xy: tuple[float, float] = (0.0, 0.0),
    front_axis: tuple[float, float] = (1.0, 0.0),
    max_abs_delta: float = DEFAULT_MAX_CONFIDENCE_DELTA,
    bbox_size_atol: float = 0.0,
) -> ObjectSet:
    """Perturb selected confidences, clamp to [0, 1], and preserve identity."""

    level = resolve_magnitude(magnitude)
    mask = select_targets(
        objects, target=target, track_ids=track_ids, ego_xy=ego_xy, front_axis=front_axis
    )
    target_count = int(mask.sum())
    new_conf = objects.conf.copy()

    if target_count:
        if delta is None:
            rng = np.random.default_rng(seed)
            deltas = rng.uniform(-level.confidence_delta, level.confidence_delta, size=target_count)
        else:
            delta_value = _finite_float(delta, "delta")
            deltas = np.full(target_count, delta_value, dtype=np.float64)

        if not np.isfinite(max_abs_delta) or max_abs_delta <= 0.0:
            raise ValueError("max_abs_delta must be positive and finite")
        if np.any(np.abs(deltas) > max_abs_delta):
            raise ValueError("confidence delta exceeds max_abs_delta")
        new_conf[mask] = np.clip(new_conf[mask] + deltas, 0.0, 1.0)

    result = objects.replace(conf=new_conf)
    _assert_contract_preserved(objects, result, bbox_size_atol=bbox_size_atol)
    return result


velocity_torsion = velocity_direction_torsion


def _mask_for_track_ids(objects: ObjectSet, track_ids: Iterable[Any] | Any) -> NDArray[np.bool_]:
    wanted = _as_track_id_list(track_ids)
    if not wanted:
        raise ValueError("at least one track_id must be provided")

    present = set(objects.track_id.tolist())
    missing = [track_id for track_id in wanted if track_id not in present]
    if missing:
        raise ValueError(f"track_id(s) not found: {missing!r}")

    return np.isin(objects.track_id, np.array(wanted, dtype=object))


def _as_track_id_list(track_ids: Iterable[Any] | Any) -> list[Any]:
    if isinstance(track_ids, (str, bytes)):
        return [track_ids]
    try:
        return list(track_ids)
    except TypeError:
        return [track_ids]


def _nearest_front_mask(
    objects: ObjectSet,
    *,
    ego_xy: tuple[float, float],
    front_axis: tuple[float, float],
) -> NDArray[np.bool_]:
    mask = np.zeros(len(objects), dtype=bool)
    if len(objects) == 0:
        return mask

    ego = _xy_vector(ego_xy, "ego_xy")
    axis = _xy_vector(front_axis, "front_axis")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("front_axis must be non-zero")
    axis = axis / norm

    rel = objects.xy - ego
    projection = rel @ axis
    ahead = projection > 0.0
    if not np.any(ahead):
        return mask

    ahead_indices = np.flatnonzero(ahead)
    distances = np.linalg.norm(rel[ahead], axis=1)
    chosen = ahead_indices[int(np.argmin(distances))]
    mask[chosen] = True
    return mask


def _xy_vector(value: tuple[float, float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr


def _resolve_angle(
    level_value: float,
    *,
    radians: float | None,
    degrees: float | None,
    name: str,
) -> tuple[float, bool]:
    if radians is not None and degrees is not None:
        raise ValueError(f"provide either {name} or {name}_deg, not both")
    if degrees is not None:
        return float(np.deg2rad(_finite_float(degrees, f"{name}_deg"))), True
    if radians is not None:
        return _finite_float(radians, name), True
    return level_value, False


def _finite_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _validate_shift_deltas(deltas: NDArray[np.float64], *, max_shift_m: float) -> None:
    if not np.isfinite(max_shift_m) or max_shift_m <= 0.0:
        raise ValueError("max_shift_m must be positive and finite")
    norms = np.linalg.norm(deltas, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("position deltas must be finite")
    if np.any(norms > max_shift_m):
        raise ValueError("position shift exceeds max_shift_m")


def _validate_angle_deltas(
    deltas: NDArray[np.float64],
    *,
    max_abs_rad: float,
    name: str,
) -> None:
    if not np.isfinite(max_abs_rad) or max_abs_rad <= 0.0:
        raise ValueError("angle limit must be positive and finite")
    if not np.all(np.isfinite(deltas)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.abs(deltas) > max_abs_rad):
        raise ValueError(f"{name} exceeds configured plausibility limit")


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
