"""Headless synthetic closed-loop harness for object-set TORSION.

Phase 2a budget matching uses the common metric already logged by the harness:
the active-window mean L2 displacement between the perturbed and clean target
actor predictions.  Section 8.1 position levels are interpreted as this realized
prediction budget in meters: low 0.2, medium 0.5, high 1.0.  Each first-class
fault method is calibrated against that metric before safety outcomes are read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

from torsion.metrics.planning import braking_reaction_delay, recovery_time
from torsion.metrics.safety import colliding_track_ids, min_actor_distance, min_ttc
from torsion.operators.object import (
    ObjectSet,
    position_torsion,
    resolve_magnitude,
    velocity_direction_torsion,
    yaw_torsion,
)
from torsion.operators.temporal import is_active
from torsion.operators.twist import (
    scene_swirl_torsion,
    temporal_curl_torsion,
)
from torsion.scenarios.planner import (
    EgoState,
    PlannerConfig,
    ReactivePlanner,
    step_ego_kinematics,
)
from torsion.scenarios.predict import constant_velocity_predict
from torsion.scenarios.synthetic_scenarios import SyntheticScenario, get_scenario

MethodName = Literal[
    "clean",
    "gaussian_matched",
    "random_warp",
    "torsion_displace",
    "torsion_translate",
    "torsion_swirl",
    "torsion_curl",
    "torsion_combined",
    "gaussian_legacy",
    "torsion",
]

_RAYLEIGH_MEAN_FACTOR = float(np.sqrt(np.pi / 2.0))
_FOLDED_NORMAL_MEAN_FACTOR = float(np.sqrt(2.0 / np.pi))
_TWIST_PEAK_FACTOR = float(np.exp(-0.5))
_CALIBRATION_TOLERANCE_M = 1e-4
_MAX_CALIBRATION_SCALE = 64.0


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for one deterministic synthetic closed-loop run."""

    scenario: str = "cut_in"
    method: MethodName = "torsion_swirl"
    magnitude: str = "medium"
    seed: int = 0
    temporal_pattern: str = "burst"
    start_frame: int = 0
    duration_frames: int | None = 30
    prediction_horizon_s: float = 3.0
    dt: float = 0.1
    steps: int | None = None
    target_actor: Any | None = None
    gaussian_sigma_scale: float = 0.25
    swirl_sigma_m: float = 3.0

    @property
    def method_key(self) -> str:
        """Return the configured method name, preserving legacy aliases."""

        return "torsion_displace" if self.method == "torsion" else str(self.method)

    @property
    def operator_name(self) -> str:
        method = self.method_key
        if method == "clean":
            return "none"
        if method == "gaussian_matched":
            return "gaussian_matched_prediction_l2_calibrated"
        if method == "random_warp":
            return "random_semantic_warp_prediction_l2_calibrated"
        if method == "gaussian_legacy":
            return "gaussian_xy_noise"
        if method in ("torsion_displace", "torsion_translate"):
            return "directed_semantic_displacement_translation_uniform_yaw_velocity_torsion"
        if method == "torsion_swirl":
            return "diffeomorphic_local_swirl_torsion"
        if method == "torsion_curl":
            return "temporal_curl_torsion"
        if method == "torsion_combined":
            return "swirl_plus_temporal_curl_torsion"
        raise ValueError(f"unknown method {self.method!r}")

    @property
    def run_id(self) -> str:
        return (
            f"synthetic_{self.scenario}_{self.method_key}_{self.magnitude}_"
            f"seed{self.seed}_{self.temporal_pattern}"
        )


@dataclass(frozen=True)
class FaultApplication:
    """Per-frame fault result plus budget-calibration metadata."""

    objects: ObjectSet
    metadata: dict[str, float]


@dataclass(frozen=True)
class RunResult:
    """Trace and summary from one synthetic closed-loop episode."""

    config: RunnerConfig
    trace: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = {
            "run_id": self.config.run_id,
            "git_commit": "",
            "simulator": "synthetic_headless",
            "simulator_version": "n/a",
            "model": "constant_velocity_predictor_reactive_ttc_planner",
            "scenario_id": self.config.scenario,
            "seed": self.config.seed,
            "injection_point": "object_set" if self.config.method_key != "clean" else "none",
            "operator": self.config.operator_name,
            "method": self.config.method_key,
            "magnitude": self.config.magnitude,
            "temporal_pattern": self.config.temporal_pattern,
            "start_frame": self.config.start_frame,
            "duration_frames": self.config.duration_frames,
            "target_actor": self.summary.get("target_actor"),
        }
        record.update(self.summary)
        return record


def run_synthetic_closed_loop(config: RunnerConfig | Mapping[str, Any]) -> RunResult:
    """Run one CARLA-free closed-loop synthetic episode."""

    cfg = config if isinstance(config, RunnerConfig) else RunnerConfig(**dict(config))
    scenario_kwargs: dict[str, Any] = {"dt": cfg.dt, "seed": cfg.seed}
    if cfg.steps is not None:
        scenario_kwargs["steps"] = cfg.steps
    scenario = get_scenario(cfg.scenario, **scenario_kwargs)

    target_actor = scenario.primary_actor_id if cfg.target_actor is None else cfg.target_actor
    planner_config = PlannerConfig(target_speed_mps=scenario.ego_initial.speed)
    planner = ReactivePlanner(planner_config)
    ego = scenario.ego_initial
    run_calibration = _prepare_run_calibration(
        cfg,
        scenario=scenario,
        target_actor=target_actor,
    )
    trace: list[dict[str, Any]] = []

    for frame_idx in range(scenario.steps):
        time_s = frame_idx * scenario.dt
        gt_objects = scenario.ground_truth_object_set(frame_idx)
        observed_objects = scenario.object_set(frame_idx)
        fault_active = _fault_active(cfg, frame_idx)
        clean_predictions = constant_velocity_predict(
            observed_objects,
            horizon_s=cfg.prediction_horizon_s,
            dt=scenario.dt,
        )
        fault = _apply_object_fault(
            observed_objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
            frame_idx=frame_idx,
            active=fault_active,
            run_calibration=run_calibration,
        )
        perceived_objects = fault.objects
        predictions = constant_velocity_predict(
            perceived_objects,
            horizon_s=cfg.prediction_horizon_s,
            dt=scenario.dt,
        )
        realized_budget = _realized_prediction_perturbation(
            clean_predictions,
            predictions,
            target_actor=target_actor,
        )
        realized_fields = _realized_field_perturbations(
            observed_objects,
            perceived_objects,
            target_actor=target_actor,
        )
        command = planner.plan(ego, predictions)
        ego_xy = (ego.x, ego.y)
        actual_ttc = min_ttc(
            ego_xy,
            ego.velocity_xy,
            gt_objects,
            ego_width=planner_config.ego_width_m,
            ego_length=planner_config.ego_length_m,
            horizon_s=5.0,
        )
        actor_distance = min_actor_distance(
            ego_xy,
            gt_objects,
            clearance=True,
            ego_width=planner_config.ego_width_m,
            ego_length=planner_config.ego_length_m,
        )
        collision_ids = colliding_track_ids(
            ego.x,
            ego.y,
            ego.yaw,
            planner_config.ego_width_m,
            planner_config.ego_length_m,
            gt_objects,
        )

        trace.append(
            {
                "frame": frame_idx,
                "time_s": float(time_s),
                "fault_active": fault_active,
                "ego": _ego_to_record(ego),
                "gt_actors": _object_set_to_records(gt_objects),
                "clean_observed_actors": _object_set_to_records(observed_objects),
                "perceived_actors": _object_set_to_records(perceived_objects),
                "predictions": predictions.to_records(),
                "clean_predictions": clean_predictions.to_records(),
                "realized_prediction_perturbation_m": float(realized_budget),
                "realized_position_shift_m": float(realized_fields["position_shift_m"]),
                "realized_yaw_shift_rad": float(realized_fields["yaw_shift_rad"]),
                "realized_velocity_rotation_rad": float(
                    realized_fields["velocity_rotation_rad"]
                ),
                "target_realized_prediction_budget_m": float(
                    fault.metadata.get("target_realized_prediction_budget_m", 0.0)
                ),
                "calibrated_scale": float(fault.metadata.get("calibrated_scale", 0.0)),
                "calibrated_swirl_alpha_rad": float(
                    fault.metadata.get("calibrated_swirl_alpha_rad", 0.0)
                ),
                "calibrated_curl_alpha_rad": float(
                    fault.metadata.get("calibrated_curl_alpha_rad", 0.0)
                ),
                "twist_sigma_m": float(fault.metadata.get("twist_sigma_m", 0.0)),
                "control": command.to_record(),
                "actual_ttc_s": float(actual_ttc),
                "min_actor_distance_m": float(actor_distance),
                "collision": bool(collision_ids),
                "collision_track_ids": list(collision_ids),
            }
        )
        ego = step_ego_kinematics(ego, command, dt=scenario.dt, config=planner_config)

    summary = _summarize_trace(
        cfg,
        scenario=scenario,
        trace=trace,
        final_ego=ego,
        target_actor=target_actor,
        planner_config=planner_config,
    )
    return RunResult(config=cfg, trace=tuple(trace), summary=summary)


def _fault_active(cfg: RunnerConfig, frame_idx: int) -> bool:
    if cfg.method_key == "clean":
        return False
    return is_active(
        cfg.temporal_pattern,
        frame_idx,
        start_frame=cfg.start_frame,
        duration=cfg.duration_frames,
    )


def _apply_object_fault(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
    frame_idx: int,
    active: bool,
    run_calibration: Mapping[str, float],
) -> FaultApplication:
    method = cfg.method_key
    if method == "clean" or not active:
        return FaultApplication(objects=objects, metadata={})
    if method == "gaussian_matched":
        return _gaussian_matched_object_noise(
            objects,
            cfg=cfg,
            scenario=scenario,
            frame_idx=frame_idx,
            target_actor=target_actor,
        )
    if method == "random_warp":
        return _random_semantic_warp(
            objects, cfg=cfg, scenario=scenario, target_actor=target_actor
        )
    if method in ("torsion_displace", "torsion_translate"):
        return _targeted_translate_torsion(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
        )
    if method == "torsion_swirl":
        return _targeted_swirl_torsion(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
        )
    if method == "torsion_curl":
        return _targeted_temporal_curl_torsion(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
            frame_idx=frame_idx,
            run_calibration=run_calibration,
        )
    if method == "torsion_combined":
        return _targeted_combined_torsion(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
            frame_idx=frame_idx,
            run_calibration=run_calibration,
        )
    if method == "gaussian_legacy":
        return FaultApplication(
            objects=_gaussian_legacy_object_noise(objects, cfg=cfg, frame_idx=frame_idx),
            metadata={},
        )
    raise ValueError(f"unknown method {cfg.method!r} in scenario {scenario.scenario_id!r}")


def _gaussian_matched_object_noise(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    frame_idx: int,
    target_actor: Any,
) -> FaultApplication:
    """Random baseline calibrated on realized prediction L2 for this frame."""

    magnitude = resolve_magnitude(cfg.magnitude)
    rng = np.random.default_rng(_stable_seed(cfg, frame_idx, "gaussian_matched"))
    target_idx = _target_index(objects, target_actor)

    position_sigma = _position_axis_sigma_for_expected_l2(magnitude.position_shift_m)
    yaw_sigma = _scalar_sigma_for_expected_abs(magnitude.yaw_shift_rad)
    velocity_sigma = _scalar_sigma_for_expected_abs(magnitude.velocity_rotation_rad)

    dx, dy = rng.normal(0.0, position_sigma, size=2)
    d_yaw = float(rng.normal(0.0, yaw_sigma))
    theta = float(rng.normal(0.0, velocity_sigma))

    def apply(scale: float) -> tuple[ObjectSet, dict[str, float]]:
        new_x = objects.x.copy()
        new_y = objects.y.copy()
        new_yaw = objects.yaw.copy()
        new_v = objects.v.copy()
        new_x[target_idx] += float(scale * dx)
        new_y[target_idx] += float(scale * dy)
        new_yaw[target_idx] = _wrap_angle_scalar(
            float(new_yaw[target_idx]) + scale * d_yaw
        )
        new_v[target_idx] = _rotate2d(new_v[target_idx], scale * theta)
        return objects.replace(x=new_x, y=new_y, yaw=new_yaw, v=new_v), {}

    return _calibrate_frame_prediction_budget(
        objects,
        cfg=cfg,
        scenario=scenario,
        target_actor=target_actor,
        apply_scaled=apply,
    )


def _random_semantic_warp(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    target_actor: Any,
) -> FaultApplication:
    """Contract-preserving random warp calibrated on realized prediction L2."""

    magnitude = resolve_magnitude(cfg.magnitude)
    rng = np.random.default_rng(_stable_seed(cfg, None, "random_warp"))

    angle = float(rng.uniform(-np.pi, np.pi))
    dx = magnitude.position_shift_m * float(np.cos(angle))
    dy = magnitude.position_shift_m * float(np.sin(angle))
    d_yaw = magnitude.yaw_shift_rad * float(rng.choice(np.array([-1.0, 1.0])))
    theta = magnitude.velocity_rotation_rad * float(rng.choice(np.array([-1.0, 1.0])))
    limits = _operator_calibration_limits(cfg)

    def apply(scale: float) -> tuple[ObjectSet, dict[str, float]]:
        out = position_torsion(
            objects,
            dx=scale * dx,
            dy=scale * dy,
            track_ids=[target_actor],
            max_shift_m=limits["position_shift_m"],
        )
        out = yaw_torsion(
            out,
            d_yaw=scale * d_yaw,
            track_ids=[target_actor],
            max_yaw_delta_rad=limits["yaw_shift_rad"],
        )
        out = velocity_direction_torsion(
            out,
            theta=scale * theta,
            track_ids=[target_actor],
            max_rotation_rad=limits["velocity_rotation_rad"],
        )
        return out, {}

    return _calibrate_frame_prediction_budget(
        objects,
        cfg=cfg,
        scenario=scenario,
        target_actor=target_actor,
        apply_scaled=apply,
    )


def _gaussian_legacy_object_noise(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    frame_idx: int,
) -> ObjectSet:
    """Legacy contract-breaking xy-only noise baseline.

    The object count, class, and ID happen to remain unchanged here, but the
    perturbation is random, temporally incoherent, and not chosen to preserve
    semantic kinematics. It is included as the random-fault contrast to TORSION.
    """

    magnitude = resolve_magnitude(cfg.magnitude)
    rng = np.random.default_rng(int(cfg.seed) * 1_000_003 + frame_idx * 9_176 + 17)
    sigma = cfg.gaussian_sigma_scale * magnitude.position_shift_m
    noisy_x = objects.x + rng.normal(0.0, sigma, size=len(objects))
    noisy_y = objects.y + rng.normal(0.0, sigma, size=len(objects))
    return objects.replace(x=noisy_x, y=noisy_y)


def _targeted_translate_torsion(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
) -> FaultApplication:
    magnitude = resolve_magnitude(cfg.magnitude)
    target_idx = _target_index(objects, target_actor)
    dx, dy = _away_from_ego_position_delta(
        objects,
        target_idx=target_idx,
        ego=ego,
        shift_m=magnitude.position_shift_m,
    )
    theta = _velocity_rotation_away_from_ego(
        objects,
        target_idx=target_idx,
        ego_y=ego.y,
        theta_abs=magnitude.velocity_rotation_rad,
        position_dy=dy,
        horizon_s=cfg.prediction_horizon_s,
    )
    if abs(theta) > 0.0:
        d_yaw = float(np.sign(theta) * magnitude.yaw_shift_rad)
    else:
        d_yaw = float(_away_from_ego_sign(float(objects.y[target_idx]), ego.y))
        d_yaw *= magnitude.yaw_shift_rad
    limits = _operator_calibration_limits(cfg)

    def apply(scale: float) -> tuple[ObjectSet, dict[str, float]]:
        out = position_torsion(
            objects,
            dx=scale * dx,
            dy=scale * dy,
            track_ids=[target_actor],
            max_shift_m=limits["position_shift_m"],
        )
        out = yaw_torsion(
            out,
            d_yaw=scale * d_yaw,
            track_ids=[target_actor],
            max_yaw_delta_rad=limits["yaw_shift_rad"],
        )
        out = velocity_direction_torsion(
            out,
            theta=scale * theta,
            track_ids=[target_actor],
            max_rotation_rad=limits["velocity_rotation_rad"],
        )
        return out, {}

    return _calibrate_frame_prediction_budget(
        objects,
        cfg=cfg,
        scenario=scenario,
        target_actor=target_actor,
        apply_scaled=apply,
    )


def _targeted_swirl_torsion(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
) -> FaultApplication:
    base_alpha_abs = _base_swirl_alpha_abs(cfg)
    sigma = _swirl_sigma(cfg)

    def apply(scale: float) -> tuple[ObjectSet, dict[str, float]]:
        out, metadata = _apply_directed_swirl(
            objects,
            ego=ego,
            target_actor=target_actor,
            alpha_abs=scale * base_alpha_abs,
            sigma=sigma,
        )
        metadata["twist_sigma_m"] = sigma
        return out, metadata

    return _calibrate_frame_prediction_budget(
        objects,
        cfg=cfg,
        scenario=scenario,
        target_actor=target_actor,
        apply_scaled=apply,
    )


def _targeted_temporal_curl_torsion(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
    frame_idx: int,
    run_calibration: Mapping[str, float],
) -> FaultApplication:
    target_idx = _target_index(objects, target_actor)
    t_window = _fault_window_frame_count(cfg, scenario)
    t_offset = frame_idx - cfg.start_frame
    alpha_abs = float(run_calibration.get("curl_alpha_abs_rad", _base_curl_alpha_abs(cfg)))
    sign = _velocity_rotation_toward_ego_sign(
        objects,
        target_idx=target_idx,
        ego_y=ego.y,
        theta_abs=max(alpha_abs * max(t_offset, 0.0) / max(t_window, 1.0), 0.0),
        horizon_s=cfg.prediction_horizon_s,
    )
    alpha = sign * alpha_abs
    out = temporal_curl_torsion(
        objects,
        alpha=alpha,
        T_window=t_window,
        t_offset=t_offset,
        track_ids=[target_actor],
    )
    return FaultApplication(
        objects=out,
        metadata={
            "calibrated_scale": float(run_calibration.get("curl_scale", 1.0)),
            "calibrated_curl_alpha_rad": float(alpha),
            "target_realized_prediction_budget_m": _target_prediction_budget(cfg),
        },
    )


def _targeted_combined_torsion(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
    frame_idx: int,
    run_calibration: Mapping[str, float],
) -> FaultApplication:
    target_idx = _target_index(objects, target_actor)
    t_window = _fault_window_frame_count(cfg, scenario)
    t_offset = frame_idx - cfg.start_frame
    scale = float(run_calibration.get("combined_scale", 1.0))
    sigma = _swirl_sigma(cfg)
    swirl_alpha_abs = scale * _base_swirl_alpha_abs(cfg)
    curl_alpha_abs = scale * _base_curl_alpha_abs(cfg)

    out, swirl_metadata = _apply_directed_swirl(
        objects,
        ego=ego,
        target_actor=target_actor,
        alpha_abs=swirl_alpha_abs,
        sigma=sigma,
    )
    curl_theta_abs = max(curl_alpha_abs * max(t_offset, 0.0) / max(t_window, 1.0), 0.0)
    curl_sign = _velocity_rotation_toward_ego_sign(
        objects,
        target_idx=target_idx,
        ego_y=ego.y,
        theta_abs=curl_theta_abs,
        horizon_s=cfg.prediction_horizon_s,
    )
    curl_alpha = curl_sign * curl_alpha_abs
    out = temporal_curl_torsion(
        out,
        alpha=curl_alpha,
        T_window=t_window,
        t_offset=t_offset,
        track_ids=[target_actor],
    )
    return FaultApplication(
        objects=out,
        metadata={
            "calibrated_scale": scale,
            "calibrated_swirl_alpha_rad": float(
                swirl_metadata["calibrated_swirl_alpha_rad"]
            ),
            "calibrated_curl_alpha_rad": float(curl_alpha),
            "twist_sigma_m": sigma,
            "target_realized_prediction_budget_m": _target_prediction_budget(cfg),
        },
    )


def _apply_directed_swirl(
    objects: ObjectSet,
    *,
    ego: EgoState,
    target_actor: Any,
    alpha_abs: float,
    sigma: float,
) -> tuple[ObjectSet, dict[str, float]]:
    target_idx = _target_index(objects, target_actor)
    pivot, sign = _directed_swirl_pivot_and_sign(
        objects,
        target_idx=target_idx,
        ego=ego,
        sigma=sigma,
    )
    alpha = sign * abs(float(alpha_abs))
    out = scene_swirl_torsion(
        objects,
        pivot=pivot,
        alpha=alpha,
        sigma=sigma,
        track_ids=[target_actor],
    )
    return out, {"calibrated_swirl_alpha_rad": float(alpha)}


def _directed_swirl_pivot_and_sign(
    objects: ObjectSet,
    *,
    target_idx: int,
    ego: EgoState,
    sigma: float,
) -> tuple[tuple[float, float], float]:
    target_xy = objects.xy[target_idx]
    lateral_delta = float(target_xy[1] - ego.y)
    if abs(lateral_delta) > 1e-6:
        radial = np.array([sigma, 0.0], dtype=np.float64)
        desired_dy_sign = float(np.sign(ego.y - target_xy[1]))
        sign = desired_dy_sign if desired_dy_sign != 0.0 else 1.0
    else:
        radial = np.array([0.0, sigma], dtype=np.float64)
        desired_dx_sign = float(np.sign(ego.x - target_xy[0]))
        if desired_dx_sign == 0.0:
            desired_dx_sign = -1.0
        sign = -desired_dx_sign
    pivot = target_xy - radial
    return (float(pivot[0]), float(pivot[1])), float(sign)


def _velocity_rotation_toward_ego_sign(
    objects: ObjectSet,
    *,
    target_idx: int,
    ego_y: float,
    theta_abs: float,
    horizon_s: float,
) -> float:
    if theta_abs <= 0.0:
        return 1.0
    velocity = objects.v[target_idx]
    y0 = float(objects.y[target_idx])
    candidates = (abs(theta_abs), -abs(theta_abs))
    scores = []
    for theta in candidates:
        rotated = _rotate2d(velocity, theta)
        future_y = y0 + float(rotated[1]) * horizon_s
        scores.append(abs(future_y - ego_y))
    return float(np.sign(candidates[int(np.argmin(scores))]))


def _target_index(objects: ObjectSet, target_actor: Any) -> int:
    matches = np.flatnonzero(objects.track_id == target_actor)
    if matches.size != 1:
        raise ValueError(f"target actor {target_actor!r} not found exactly once")
    return int(matches[0])


def _away_from_ego_sign(actor_y: float, ego_y: float) -> float:
    delta = actor_y - ego_y
    if abs(delta) < 1e-9:
        return 1.0
    return float(np.sign(delta))


def _away_from_ego_position_delta(
    objects: ObjectSet,
    *,
    target_idx: int,
    ego: EgoState,
    shift_m: float,
) -> tuple[float, float]:
    if shift_m == 0.0:
        return 0.0, 0.0
    direction = np.array(
        [
            _away_from_ego_sign(float(objects.x[target_idx]), ego.x),
            _away_from_ego_sign(float(objects.y[target_idx]), ego.y),
        ],
        dtype=np.float64,
    )
    direction = direction / float(np.linalg.norm(direction))
    delta = shift_m * direction
    return float(delta[0]), float(delta[1])


def _velocity_rotation_away_from_ego(
    objects: ObjectSet,
    *,
    target_idx: int,
    ego_y: float,
    theta_abs: float,
    position_dy: float,
    horizon_s: float,
) -> float:
    if theta_abs == 0.0:
        return 0.0
    velocity = objects.v[target_idx]
    y0 = float(objects.y[target_idx]) + position_dy
    candidates = (abs(theta_abs), -abs(theta_abs))
    scores = []
    for theta in candidates:
        rotated = _rotate2d(velocity, theta)
        future_y = y0 + float(rotated[1]) * horizon_s
        scores.append(abs(future_y - ego_y))
    return float(candidates[int(np.argmax(scores))])


def _prepare_run_calibration(
    cfg: RunnerConfig,
    *,
    scenario: SyntheticScenario,
    target_actor: Any,
) -> dict[str, float]:
    method = cfg.method_key
    if method == "torsion_curl":
        scale = _calibrate_temporal_scale(
            cfg,
            scenario=scenario,
            target_actor=target_actor,
            apply_scaled=lambda objects, frame_idx, scale_value: _apply_temporal_curl_for_calibration(
                objects,
                cfg=cfg,
                scenario=scenario,
                target_actor=target_actor,
                frame_idx=frame_idx,
                scale=scale_value,
            ),
        )
        return {
            "curl_scale": scale,
            "curl_alpha_abs_rad": scale * _base_curl_alpha_abs(cfg),
        }
    if method == "torsion_combined":
        scale = _calibrate_temporal_scale(
            cfg,
            scenario=scenario,
            target_actor=target_actor,
            apply_scaled=lambda objects, frame_idx, scale_value: _apply_combined_for_calibration(
                objects,
                cfg=cfg,
                scenario=scenario,
                target_actor=target_actor,
                frame_idx=frame_idx,
                scale=scale_value,
            ),
        )
        return {"combined_scale": scale}
    return {}


def _calibrate_temporal_scale(
    cfg: RunnerConfig,
    *,
    scenario: SyntheticScenario,
    target_actor: Any,
    apply_scaled: Any,
) -> float:
    target_budget = _target_prediction_budget(cfg)
    active_frames = _fault_active_frames(cfg, scenario)
    if target_budget <= 0.0 or not active_frames:
        return 0.0

    def budget_at(scale: float) -> float:
        values = []
        for frame_idx in active_frames:
            objects = scenario.object_set(frame_idx)
            perturbed = apply_scaled(objects, frame_idx, scale)
            clean_predictions = constant_velocity_predict(
                objects,
                horizon_s=cfg.prediction_horizon_s,
                dt=scenario.dt,
            )
            perturbed_predictions = constant_velocity_predict(
                perturbed,
                horizon_s=cfg.prediction_horizon_s,
                dt=scenario.dt,
            )
            values.append(
                _realized_prediction_perturbation(
                    clean_predictions,
                    perturbed_predictions,
                    target_actor=target_actor,
                )
            )
        return float(np.mean(values))

    return _bisect_scale_for_budget(budget_at, target_budget)


def _apply_temporal_curl_for_calibration(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    target_actor: Any,
    frame_idx: int,
    scale: float,
) -> ObjectSet:
    target_idx = _target_index(objects, target_actor)
    t_window = _fault_window_frame_count(cfg, scenario)
    t_offset = frame_idx - cfg.start_frame
    alpha_abs = scale * _base_curl_alpha_abs(cfg)
    theta_abs = max(alpha_abs * max(t_offset, 0.0) / max(t_window, 1.0), 0.0)
    sign = _velocity_rotation_toward_ego_sign(
        objects,
        target_idx=target_idx,
        ego_y=scenario.ego_initial.y,
        theta_abs=theta_abs,
        horizon_s=cfg.prediction_horizon_s,
    )
    return temporal_curl_torsion(
        objects,
        alpha=sign * alpha_abs,
        T_window=t_window,
        t_offset=t_offset,
        track_ids=[target_actor],
    )


def _apply_combined_for_calibration(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    target_actor: Any,
    frame_idx: int,
    scale: float,
) -> ObjectSet:
    sigma = _swirl_sigma(cfg)
    swirl, _ = _apply_directed_swirl(
        objects,
        ego=scenario.ego_initial,
        target_actor=target_actor,
        alpha_abs=scale * _base_swirl_alpha_abs(cfg),
        sigma=sigma,
    )
    target_idx = _target_index(objects, target_actor)
    t_window = _fault_window_frame_count(cfg, scenario)
    t_offset = frame_idx - cfg.start_frame
    curl_alpha_abs = scale * _base_curl_alpha_abs(cfg)
    theta_abs = max(curl_alpha_abs * max(t_offset, 0.0) / max(t_window, 1.0), 0.0)
    sign = _velocity_rotation_toward_ego_sign(
        objects,
        target_idx=target_idx,
        ego_y=scenario.ego_initial.y,
        theta_abs=theta_abs,
        horizon_s=cfg.prediction_horizon_s,
    )
    return temporal_curl_torsion(
        swirl,
        alpha=sign * curl_alpha_abs,
        T_window=t_window,
        t_offset=t_offset,
        track_ids=[target_actor],
    )


def _calibrate_frame_prediction_budget(
    objects: ObjectSet,
    *,
    cfg: RunnerConfig,
    scenario: SyntheticScenario,
    target_actor: Any,
    apply_scaled: Any,
) -> FaultApplication:
    target_budget = _target_prediction_budget(cfg)
    if target_budget <= 0.0:
        out, metadata = apply_scaled(0.0)
        metadata = dict(metadata)
        metadata.update(
            {
                "calibrated_scale": 0.0,
                "target_realized_prediction_budget_m": 0.0,
            }
        )
        return FaultApplication(objects=out, metadata=metadata)

    clean_predictions = constant_velocity_predict(
        objects,
        horizon_s=cfg.prediction_horizon_s,
        dt=scenario.dt,
    )

    def budget_at(scale: float) -> float:
        perturbed, _ = apply_scaled(scale)
        perturbed_predictions = constant_velocity_predict(
            perturbed,
            horizon_s=cfg.prediction_horizon_s,
            dt=scenario.dt,
        )
        return _realized_prediction_perturbation(
            clean_predictions,
            perturbed_predictions,
            target_actor=target_actor,
        )

    scale = _bisect_scale_for_budget(budget_at, target_budget)
    out, metadata = apply_scaled(scale)
    metadata = dict(metadata)
    metadata.update(
        {
            "calibrated_scale": float(scale),
            "target_realized_prediction_budget_m": float(target_budget),
        }
    )
    return FaultApplication(objects=out, metadata=metadata)


def _bisect_scale_for_budget(budget_at: Any, target_budget: float) -> float:
    lower = 0.0
    upper = 1.0
    upper_budget = float(budget_at(upper))
    while upper_budget < target_budget and upper < _MAX_CALIBRATION_SCALE:
        upper *= 2.0
        upper_budget = float(budget_at(upper))

    if upper_budget < target_budget:
        return float(upper)

    for _ in range(40):
        mid = 0.5 * (lower + upper)
        mid_budget = float(budget_at(mid))
        if abs(mid_budget - target_budget) <= _CALIBRATION_TOLERANCE_M:
            return float(mid)
        if mid_budget < target_budget:
            lower = mid
        else:
            upper = mid
    return float(0.5 * (lower + upper))


def _target_prediction_budget(cfg: RunnerConfig) -> float:
    if cfg.method_key == "clean":
        return 0.0
    return float(resolve_magnitude(cfg.magnitude).position_shift_m)


def _operator_calibration_limits(cfg: RunnerConfig) -> dict[str, float]:
    magnitude = resolve_magnitude(cfg.magnitude)
    return {
        "position_shift_m": float(
            _MAX_CALIBRATION_SCALE * max(magnitude.position_shift_m, 1e-9)
        ),
        "yaw_shift_rad": float(
            _MAX_CALIBRATION_SCALE * max(magnitude.yaw_shift_rad, 1e-9)
        ),
        "velocity_rotation_rad": float(
            _MAX_CALIBRATION_SCALE * max(magnitude.velocity_rotation_rad, 1e-9)
        ),
    }


def _base_swirl_alpha_abs(cfg: RunnerConfig) -> float:
    magnitude = resolve_magnitude(cfg.magnitude)
    return float(magnitude.velocity_rotation_rad / _TWIST_PEAK_FACTOR)


def _base_curl_alpha_abs(cfg: RunnerConfig) -> float:
    return float(resolve_magnitude(cfg.magnitude).velocity_rotation_rad)


def _swirl_sigma(cfg: RunnerConfig) -> float:
    sigma = float(cfg.swirl_sigma_m)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("swirl_sigma_m must be positive and finite")
    return sigma


def _fault_active_frames(cfg: RunnerConfig, scenario: SyntheticScenario) -> list[int]:
    return [
        frame_idx
        for frame_idx in range(scenario.steps)
        if _fault_active(cfg, frame_idx)
    ]


def _fault_window_frame_count(cfg: RunnerConfig, scenario: SyntheticScenario) -> int:
    frames = _fault_active_frames(cfg, scenario)
    return max(len(frames), 1)


def _rotate2d(vector: np.ndarray, theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _realized_prediction_perturbation(
    clean_predictions: Any,
    perturbed_predictions: Any,
    *,
    target_actor: Any,
) -> float:
    clean = clean_predictions.by_track_id(target_actor)
    perturbed = perturbed_predictions.by_track_id(target_actor)
    if clean is None or perturbed is None:
        return 0.0
    if clean.xy.shape != perturbed.xy.shape:
        raise ValueError("clean and perturbed predictions must use the same horizon")
    displacement = np.linalg.norm(perturbed.xy - clean.xy, axis=1)
    return float(np.mean(displacement))


def _realized_field_perturbations(
    clean_objects: ObjectSet,
    perturbed_objects: ObjectSet,
    *,
    target_actor: Any,
) -> dict[str, float]:
    clean_idx = _target_index(clean_objects, target_actor)
    perturbed_idx = _target_index(perturbed_objects, target_actor)
    position_shift = float(
        np.linalg.norm(
            [
                perturbed_objects.x[perturbed_idx] - clean_objects.x[clean_idx],
                perturbed_objects.y[perturbed_idx] - clean_objects.y[clean_idx],
            ]
        )
    )
    yaw_shift = abs(
        _angle_delta(
            float(perturbed_objects.yaw[perturbed_idx]),
            float(clean_objects.yaw[clean_idx]),
        )
    )
    velocity_rotation = abs(
        _velocity_angle_delta(
            clean_objects.v[clean_idx],
            perturbed_objects.v[perturbed_idx],
        )
    )
    return {
        "position_shift_m": float(position_shift),
        "yaw_shift_rad": float(yaw_shift),
        "velocity_rotation_rad": float(velocity_rotation),
    }


def _position_axis_sigma_for_expected_l2(expected_l2_m: float) -> float:
    return float(expected_l2_m / _RAYLEIGH_MEAN_FACTOR)


def _scalar_sigma_for_expected_abs(expected_abs: float) -> float:
    return float(expected_abs / _FOLDED_NORMAL_MEAN_FACTOR)


def _stable_seed(cfg: RunnerConfig, frame_idx: int | None, salt: str) -> int:
    text = "|".join(
        (
            "torsion-synthetic",
            cfg.scenario,
            cfg.method_key,
            cfg.magnitude,
            str(cfg.seed),
            "" if frame_idx is None else str(frame_idx),
            salt,
        )
    )
    value = 0x345678
    for byte in text.encode("utf-8"):
        value = ((value * 1_000_003) ^ byte) & 0xFFFFFFFF
    return int(value)


def _angle_delta(a: float, b: float) -> float:
    return _wrap_angle_scalar(a - b)


def _velocity_angle_delta(clean: np.ndarray, perturbed: np.ndarray) -> float:
    clean_norm = float(np.linalg.norm(clean))
    perturbed_norm = float(np.linalg.norm(perturbed))
    if clean_norm <= 1e-12 or perturbed_norm <= 1e-12:
        return 0.0
    clean_angle = float(np.arctan2(clean[1], clean[0]))
    perturbed_angle = float(np.arctan2(perturbed[1], perturbed[0]))
    return _angle_delta(perturbed_angle, clean_angle)


def _wrap_angle_scalar(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _summarize_trace(
    cfg: RunnerConfig,
    *,
    scenario: SyntheticScenario,
    trace: list[dict[str, Any]],
    final_ego: EgoState,
    target_actor: Any,
    planner_config: PlannerConfig,
) -> dict[str, Any]:
    collision_frames = [row["frame"] for row in trace if row["collision"]]
    min_ttc_value = min(float(row["actual_ttc_s"]) for row in trace)
    min_distance_value = min(float(row["min_actor_distance_m"]) for row in trace)
    magnitude = resolve_magnitude(cfg.magnitude)
    active_rows = [row for row in trace if row["fault_active"]]
    target_position_budget = 0.0 if cfg.method_key == "clean" else magnitude.position_shift_m
    target_yaw_budget = 0.0 if cfg.method_key == "clean" else magnitude.yaw_shift_rad
    target_velocity_budget = (
        0.0 if cfg.method_key == "clean" else magnitude.velocity_rotation_rad
    )
    target_prediction_budget = _target_prediction_budget(cfg)
    mean_realized_budget = _mean_trace_value(
        active_rows, "realized_prediction_perturbation_m"
    )
    mean_realized_position = _mean_trace_value(active_rows, "realized_position_shift_m")
    mean_realized_yaw = _mean_trace_value(active_rows, "realized_yaw_shift_rad")
    mean_realized_velocity = _mean_trace_value(
        active_rows, "realized_velocity_rotation_rad"
    )
    fault_start_time = cfg.start_frame * scenario.dt
    fault_end_time = _fault_end_time(cfg, scenario)
    delay = braking_reaction_delay(trace, event_time_s=fault_start_time)
    recovery = recovery_time(
        trace,
        event_end_time_s=fault_end_time,
        target_speed_mps=None,
    )
    route_completion = float(np.clip(final_ego.x / scenario.route_length_m, 0.0, 1.0))

    return {
        "collision": bool(collision_frames),
        "collision_frame": collision_frames[0] if collision_frames else None,
        "collision_time_s": (
            float(collision_frames[0] * scenario.dt) if collision_frames else None
        ),
        "min_ttc": float(min_ttc_value),
        "min_actor_distance": float(min_distance_value),
        "braking_reaction_delay_s": float(delay),
        "recovery_time_s": float(recovery),
        "route_completion": route_completion,
        "lane_departure": bool(any(abs(row["ego"]["y"]) > 1.75 for row in trace)),
        "fault_active_frames": int(sum(bool(row["fault_active"]) for row in trace)),
        "fault_start_time_s": float(fault_start_time),
        "fault_end_time_s": float(fault_end_time),
        "target_actor": target_actor,
        "target_position_shift_m": float(target_position_budget),
        "target_yaw_shift_rad": float(target_yaw_budget),
        "target_velocity_rotation_rad": float(target_velocity_budget),
        "target_realized_prediction_budget_m": float(target_prediction_budget),
        "mean_realized_budget": float(mean_realized_budget),
        "mean_realized_prediction_perturbation_m": float(mean_realized_budget),
        "max_realized_prediction_perturbation_m": float(
            max(
                (
                    float(row["realized_prediction_perturbation_m"])
                    for row in active_rows
                ),
                default=0.0,
            )
        ),
        "mean_realized_position_shift_m": float(mean_realized_position),
        "mean_realized_yaw_shift_rad": float(mean_realized_yaw),
        "mean_realized_velocity_rotation_rad": float(mean_realized_velocity),
        "mean_calibrated_scale": float(_mean_trace_value(active_rows, "calibrated_scale")),
        "mean_calibrated_swirl_alpha_rad": float(
            _mean_trace_value(active_rows, "calibrated_swirl_alpha_rad")
        ),
        "mean_abs_calibrated_swirl_alpha_rad": float(
            _mean_abs_trace_value(active_rows, "calibrated_swirl_alpha_rad")
        ),
        "mean_calibrated_curl_alpha_rad": float(
            _mean_trace_value(active_rows, "calibrated_curl_alpha_rad")
        ),
        "mean_abs_calibrated_curl_alpha_rad": float(
            _mean_abs_trace_value(active_rows, "calibrated_curl_alpha_rad")
        ),
        "twist_sigma_m": float(_mean_trace_value(active_rows, "twist_sigma_m")),
        "final_ego_x": float(final_ego.x),
        "final_ego_speed": float(final_ego.speed),
    }


def _mean_trace_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_abs_trace_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([abs(float(row[key])) for row in rows]))


def _fault_end_time(cfg: RunnerConfig, scenario: SyntheticScenario) -> float:
    pattern = cfg.temporal_pattern.lower().strip().replace("_", "-")
    if cfg.method_key == "clean":
        return float(cfg.start_frame * scenario.dt)
    if pattern == "single-frame":
        return float((cfg.start_frame + 1) * scenario.dt)
    if pattern == "burst":
        duration = cfg.duration_frames if cfg.duration_frames is not None else 1
        return float((cfg.start_frame + duration) * scenario.dt)
    if pattern == "persistent":
        return float((scenario.steps - 1) * scenario.dt)
    return float(cfg.start_frame * scenario.dt)


def _ego_to_record(ego: EgoState) -> dict[str, float]:
    return asdict(ego)


def _object_set_to_records(objects: ObjectSet) -> list[dict[str, Any]]:
    rows = []
    for idx in range(len(objects)):
        rows.append(
            {
                "track_id": objects.track_id[idx],
                "cls": str(objects.cls[idx]),
                "x": float(objects.x[idx]),
                "y": float(objects.y[idx]),
                "z": float(objects.z[idx]),
                "w": float(objects.w[idx]),
                "h": float(objects.h[idx]),
                "l": float(objects.l[idx]),
                "yaw": float(objects.yaw[idx]),
                "vx": float(objects.v[idx, 0]),
                "vy": float(objects.v[idx, 1]),
                "conf": float(objects.conf[idx]),
            }
        )
    return rows
