"""Pure planning metrics for synthetic closed-loop traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def trajectory_l2_deviation(
    trace: Sequence[Mapping[str, Any]],
    clean_trace: Sequence[Mapping[str, Any]],
) -> float:
    """Return RMS ego xy deviation from a clean reference trace."""

    count = min(len(trace), len(clean_trace))
    if count == 0:
        return float("nan")

    deltas = []
    for frame in range(count):
        ego = _mapping(trace[frame]["ego"], "ego")
        clean_ego = _mapping(clean_trace[frame]["ego"], "clean ego")
        deltas.append(
            (
                _float(ego["x"], "ego.x") - _float(clean_ego["x"], "clean ego.x"),
                _float(ego["y"], "ego.y") - _float(clean_ego["y"], "clean ego.y"),
            )
        )
    arr = np.asarray(deltas, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))))


def braking_reaction_delay(
    trace: Sequence[Mapping[str, Any]],
    *,
    event_time_s: float,
    brake_threshold: float = 0.05,
) -> float:
    """Return seconds from an event to first brake command, or ``inf``."""

    if not np.isfinite(event_time_s):
        raise ValueError("event_time_s must be finite")
    if brake_threshold < 0.0 or not np.isfinite(brake_threshold):
        raise ValueError("brake_threshold must be non-negative and finite")

    for row in trace:
        time_s = _float(row["time_s"], "time_s")
        if time_s < event_time_s:
            continue
        control = _mapping(row["control"], "control")
        brake = _float(control["brake"], "control.brake")
        accel = _float(control["accel_mps2"], "control.accel_mps2")
        if brake >= brake_threshold or accel < -1e-9:
            return float(time_s - event_time_s)
    return float("inf")


def recovery_time(
    trace: Sequence[Mapping[str, Any]],
    *,
    event_end_time_s: float,
    target_speed_mps: float | None = None,
    speed_tol_mps: float = 0.5,
    lateral_tol_m: float = 0.25,
    brake_threshold: float = 0.05,
    stable_frames: int = 5,
) -> float:
    """Return time after an event until ego is stable again, or ``inf``.

    Stability means a consecutive window with no collision, little braking,
    lane-centered ego pose, and speed close to the configured target speed.
    """

    if stable_frames <= 0:
        raise ValueError("stable_frames must be positive")
    if not np.isfinite(event_end_time_s):
        raise ValueError("event_end_time_s must be finite")

    for start_idx, row in enumerate(trace):
        start_time = _float(row["time_s"], "time_s")
        if start_time < event_end_time_s:
            continue
        window = trace[start_idx : start_idx + stable_frames]
        if len(window) < stable_frames:
            break
        if all(
            _is_recovered_frame(
                candidate,
                target_speed_mps=target_speed_mps,
                speed_tol_mps=speed_tol_mps,
                lateral_tol_m=lateral_tol_m,
                brake_threshold=brake_threshold,
            )
            for candidate in window
        ):
            return float(start_time - event_end_time_s)
    return float("inf")


def _is_recovered_frame(
    row: Mapping[str, Any],
    *,
    target_speed_mps: float | None,
    speed_tol_mps: float,
    lateral_tol_m: float,
    brake_threshold: float,
) -> bool:
    ego = _mapping(row["ego"], "ego")
    control = _mapping(row["control"], "control")
    if bool(row.get("collision", False)):
        return False
    if abs(_float(ego["y"], "ego.y")) > lateral_tol_m:
        return False
    if target_speed_mps is not None and (
        abs(_float(ego["speed"], "ego.speed") - target_speed_mps) > speed_tol_mps
    ):
        return False
    if _float(control["brake"], "control.brake") >= brake_threshold:
        return False
    return True


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


# TODO(phase-2/3): Simulator-derived metrics such as route completion from a
# CARLA route graph, lane/off-road violations, and planner failure rate should
# be implemented against simulator/map APIs when those runners are active.
