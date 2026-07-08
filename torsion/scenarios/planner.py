"""Reactive longitudinal planner for the synthetic closed-loop harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torsion.scenarios.predict import PredictionSet


@dataclass(frozen=True)
class EgoState:
    """Minimal ego state for a straight-road synthetic episode."""

    x: float
    y: float
    yaw: float
    speed: float

    @property
    def velocity_xy(self) -> tuple[float, float]:
        return (
            float(self.speed * np.cos(self.yaw)),
            float(self.speed * np.sin(self.yaw)),
        )


@dataclass(frozen=True)
class PlannerConfig:
    """Parameters for the TTC-threshold reactive planner."""

    target_speed_mps: float = 12.0
    max_accel_mps2: float = 1.2
    brake_accel_mps2: float = -5.5
    hard_brake_accel_mps2: float = -8.0
    ttc_brake_threshold_s: float = 3.05
    ttc_hard_brake_threshold_s: float = 1.0
    desired_clearance_m: float = 5.0
    ego_width_m: float = 2.0
    ego_length_m: float = 4.5
    ego_lane_y_m: float = 0.0
    lane_keep_gain: float = 1.0
    lane_conflict_half_width_m: float = 1.2


@dataclass(frozen=True)
class ControlCommand:
    """Longitudinal control selected from predicted actor trajectories."""

    accel_mps2: float
    brake: float
    predicted_ttc_s: float
    min_predicted_clearance_m: float
    target_track_id: Any | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "accel_mps2": self.accel_mps2,
            "brake": self.brake,
            "predicted_ttc_s": self.predicted_ttc_s,
            "min_predicted_clearance_m": self.min_predicted_clearance_m,
            "target_track_id": self.target_track_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PredictionRisk:
    predicted_ttc_s: float
    min_clearance_m: float
    target_track_id: Any | None


class ReactivePlanner:
    """A deterministic TTC policy over constant-velocity predictions.

    The planner projects ego forward at current speed and checks whether any
    predicted actor path enters a narrow ego-lane conflict corridor. If the
    projected longitudinal clearance falls below zero inside the horizon, the
    corresponding time is treated as predicted TTC. This deliberately simple
    policy makes object-set prediction errors visible without depending on a
    simulator or optimizer.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, ego: EgoState, predictions: PredictionSet) -> ControlCommand:
        risk = evaluate_prediction_risk(ego, predictions, self.config)

        if risk.predicted_ttc_s <= self.config.ttc_hard_brake_threshold_s:
            accel = self.config.hard_brake_accel_mps2
            reason = "hard_brake_ttc"
        elif (
            risk.predicted_ttc_s <= self.config.ttc_brake_threshold_s
            or risk.min_clearance_m <= self.config.desired_clearance_m
        ):
            accel = self.config.brake_accel_mps2
            reason = "brake_ttc"
        else:
            speed_error = self.config.target_speed_mps - ego.speed
            accel = float(np.clip(0.8 * speed_error, 0.0, self.config.max_accel_mps2))
            reason = "cruise"

        brake = float(np.clip(-accel / abs(self.config.hard_brake_accel_mps2), 0.0, 1.0))
        return ControlCommand(
            accel_mps2=float(accel),
            brake=brake,
            predicted_ttc_s=float(risk.predicted_ttc_s),
            min_predicted_clearance_m=float(risk.min_clearance_m),
            target_track_id=risk.target_track_id,
            reason=reason,
        )


def evaluate_prediction_risk(
    ego: EgoState,
    predictions: PredictionSet,
    config: PlannerConfig | None = None,
) -> PredictionRisk:
    """Return the nearest predicted ego-lane conflict in the planning horizon."""

    cfg = config or PlannerConfig()
    best_ttc = float("inf")
    best_clearance = float("inf")
    best_track_id: Any | None = None
    ego_x_future = ego.x + ego.speed * predictions.times_s
    half_longitudinal_ego = 0.5 * cfg.ego_length_m

    for trajectory in predictions.trajectories:
        lateral_distance = np.abs(trajectory.xy[:, 1] - cfg.ego_lane_y_m)
        in_conflict_corridor = lateral_distance <= cfg.lane_conflict_half_width_m
        if not np.any(in_conflict_corridor):
            continue

        half_length = half_longitudinal_ego + 0.5 * trajectory.length
        center_dx = trajectory.xy[:, 0] - ego_x_future
        actor_ahead_or_near = center_dx >= -half_length
        clearance = center_dx - half_length
        relevant = in_conflict_corridor & actor_ahead_or_near
        if not np.any(relevant):
            continue

        min_idx = int(np.argmin(np.where(relevant, clearance, np.inf)))
        min_clearance = float(clearance[min_idx])
        if min_clearance < best_clearance:
            best_clearance = min_clearance
            best_track_id = trajectory.track_id

        collision_indices = np.flatnonzero(relevant & (clearance <= 0.0))
        if collision_indices.size:
            ttc = float(predictions.times_s[int(collision_indices[0])])
            if ttc < best_ttc:
                best_ttc = ttc
                best_track_id = trajectory.track_id

    return PredictionRisk(
        predicted_ttc_s=best_ttc,
        min_clearance_m=best_clearance,
        target_track_id=best_track_id,
    )


def step_ego_kinematics(
    ego: EgoState,
    command: ControlCommand,
    *,
    dt: float,
    config: PlannerConfig | None = None,
) -> EgoState:
    """Advance ego one step with longitudinal acceleration and lane keeping."""

    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be positive and finite")

    cfg = config or PlannerConfig()
    accel = float(command.accel_mps2)
    next_speed = max(0.0, ego.speed + accel * dt)
    avg_speed = 0.5 * (ego.speed + next_speed)
    next_x = ego.x + avg_speed * dt * np.cos(ego.yaw)
    lateral_error = cfg.ego_lane_y_m - ego.y
    next_y = ego.y + cfg.lane_keep_gain * lateral_error * dt
    return EgoState(x=float(next_x), y=float(next_y), yaw=0.0, speed=float(next_speed))
