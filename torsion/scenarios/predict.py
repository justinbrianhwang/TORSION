"""Constant-velocity prediction for synthetic object-set experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.operators.object import ObjectSet


@dataclass(frozen=True)
class PredictedTrajectory:
    """One actor trajectory predicted from a single perceived object state."""

    track_id: Any
    cls: str
    xy: NDArray[np.float64]
    yaw: NDArray[np.float64]
    velocity: NDArray[np.float64]
    width: float
    height: float
    length: float
    confidence: float

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        yaw = np.asarray(self.yaw, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (steps, 2)")
        if yaw.ndim != 1 or yaw.shape[0] != xy.shape[0]:
            raise ValueError("yaw must be 1D and match xy length")
        if velocity.shape != (2,):
            raise ValueError("velocity must have shape (2,)")
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(yaw)):
            raise ValueError("predicted trajectory must be finite")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must be finite")

        xy = xy.copy()
        yaw = yaw.copy()
        velocity = velocity.copy()
        xy.setflags(write=False)
        yaw.setflags(write=False)
        velocity.setflags(write=False)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "yaw", yaw)
        object.__setattr__(self, "velocity", velocity)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable prediction record."""

        return {
            "track_id": self.track_id,
            "cls": self.cls,
            "x": self.xy[:, 0].tolist(),
            "y": self.xy[:, 1].tolist(),
            "yaw": self.yaw.tolist(),
            "velocity": self.velocity.tolist(),
            "width": self.width,
            "height": self.height,
            "length": self.length,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PredictionSet:
    """A collection of actor predictions sharing one time base."""

    trajectories: tuple[PredictedTrajectory, ...]
    times_s: NDArray[np.float64]
    dt: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times_s must be a non-empty 1D array")
        if not np.all(np.isfinite(times)):
            raise ValueError("times_s must be finite")
        times = times.copy()
        times.setflags(write=False)
        object.__setattr__(self, "times_s", times)

    @property
    def horizon_s(self) -> float:
        return float(self.times_s[-1])

    def by_track_id(self, track_id: Any) -> PredictedTrajectory | None:
        for trajectory in self.trajectories:
            if trajectory.track_id == track_id:
                return trajectory
        return None

    def to_records(self) -> list[dict[str, Any]]:
        return [trajectory.to_record() for trajectory in self.trajectories]


def constant_velocity_predict(
    objects: ObjectSet,
    *,
    horizon_s: float = 3.0,
    dt: float = 0.1,
) -> PredictionSet:
    """Predict actor trajectories by holding perceived velocity constant."""

    if not np.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive and finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")

    steps = int(round(horizon_s / dt))
    if steps <= 0:
        raise ValueError("horizon_s must contain at least one dt")
    times = np.arange(steps + 1, dtype=np.float64) * dt

    trajectories: list[PredictedTrajectory] = []
    for idx in range(len(objects)):
        start_xy = np.array([objects.x[idx], objects.y[idx]], dtype=np.float64)
        velocity = objects.v[idx].astype(np.float64, copy=True)
        xy = start_xy[None, :] + times[:, None] * velocity[None, :]
        yaw = np.full(times.shape, float(objects.yaw[idx]), dtype=np.float64)
        trajectories.append(
            PredictedTrajectory(
                track_id=objects.track_id[idx],
                cls=str(objects.cls[idx]),
                xy=xy,
                yaw=yaw,
                velocity=velocity,
                width=float(objects.w[idx]),
                height=float(objects.h[idx]),
                length=float(objects.l[idx]),
                confidence=float(objects.conf[idx]),
            )
        )

    return PredictionSet(trajectories=tuple(trajectories), times_s=times, dt=float(dt))
