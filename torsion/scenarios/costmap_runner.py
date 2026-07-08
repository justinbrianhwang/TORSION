"""Dense cost-map closed-loop runner for Phase 2b.

The runner uses the existing scripted scenarios but changes the injection point
from sparse object sets to a dense ego-centric cost map.  Each frame:

1. rasterizes the road/lane prior and actor obstacle/free-space component
   separately;
2. optionally perturbs only the obstacle/free-space component;
3. plans with a deterministic receding-horizon lateral-offset sampler;
4. executes the first step of the selected path and logs safety metrics.

Grid convention
---------------
The cost map is ego-centric in meters with default extent:

* x: ``[-6, 54]`` meters, where positive x is forward;
* y: ``[-8, 8]`` meters, where positive y is left;
* resolution: ``0.4`` meters per cell.

Cost values are in ``[0, 1]``.  Low values are drivable/free space, and high
values are lane edges, off-road regions, or inflated obstacle occupancy.

Topology preservation
---------------------
Design B.4 requires the drivable topology not be destroyed.  This runner keeps
the road/lane-boundary prior fixed and applies every non-clean method only to
the obstacle/free-space component.  Recombination uses ``max(C_road,
C_obstacle_tilde)`` and then copies road/lane-boundary cells exactly from the
clean map, so the visual road edge in Figure 6 cannot be bent by a fault.

Budget matching
---------------
The low/medium/high cost-map budgets are the same realized metric logged in
the trace: mean L2 deviation of the chosen ego path versus the clean chosen
path during active frames.  Targets are 0.2 m, 0.5 m, and 1.0 m respectively.
Per-frame calibration searches only over operator strength and minimizes
absolute error to that path-budget target; it never reads collision/TTC
outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from torsion.metrics.safety import colliding_track_ids, min_actor_distance, min_ttc
from torsion.operators.costmap import (
    COST_MAX,
    COST_MIN,
    gaussian_cost_noise,
    sample_cost_grid,
    spatial_cost_warp,
    translate_cost_field,
)
from torsion.operators.temporal import is_active
from torsion.operators.twist import twist_points
from torsion.scenarios.predict import constant_velocity_predict
from torsion.scenarios.planner import EgoState
from torsion.scenarios.synthetic_scenarios import SyntheticScenario, get_scenario

CostMapMethodName = Literal[
    "clean",
    "cost_translate",
    "torsion_displace",
    "gaussian_cost",
    "random_warp_cost",
    "torsion_swirl",
    "swirl_illegal",
]

COSTMAP_METHODS: tuple[str, ...] = (
    "clean",
    "cost_translate",
    "gaussian_cost",
    "random_warp_cost",
    "torsion_swirl",
)
COSTMAP_ALIAS_METHODS: tuple[str, ...] = ("torsion_displace",)
COSTMAP_CONTRACT_VIOLATING_METHODS: tuple[str, ...] = ("swirl_illegal",)
COSTMAP_FAULT_METHODS: tuple[str, ...] = tuple(
    method for method in COSTMAP_METHODS if method != "clean"
) + COSTMAP_ALIAS_METHODS
COSTMAP_ALLOWED_FAULT_METHODS: tuple[str, ...] = (
    COSTMAP_FAULT_METHODS + COSTMAP_CONTRACT_VIOLATING_METHODS
)
MAGNITUDE_PATH_BUDGETS_M: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 1.0,
}
_CALIBRATION_SCALES = np.array(
    [
        0.0,
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
        1.2,
        1.6,
        2.4,
        3.2,
        4.8,
        6.4,
        9.6,
        12.8,
        16.0,
        20.0,
        24.0,
        32.0,
        48.0,
        64.0,
        96.0,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CostMapSpec:
    """Metric extent and resolution for the ego-centric cost map."""

    resolution_m: float = 0.4
    x_min_m: float = -6.0
    x_max_m: float = 54.0
    y_min_m: float = -8.0
    y_max_m: float = 8.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive and finite")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("grid max bounds must exceed min bounds")

    @property
    def width(self) -> int:
        return int(round((self.x_max_m - self.x_min_m) / self.resolution_m)) + 1

    @property
    def height(self) -> int:
        return int(round((self.y_max_m - self.y_min_m) / self.resolution_m)) + 1

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def metric_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return local ``(x, y)`` mesh arrays in meters."""

        x = self.x_min_m + np.arange(self.width, dtype=np.float64) * self.resolution_m
        y = self.y_max_m - np.arange(self.height, dtype=np.float64) * self.resolution_m
        return np.meshgrid(x, y)

    def metric_to_grid(self, points_xy: NDArray[np.float64] | Any) -> NDArray[np.float64]:
        """Convert local metric ``(x, y)`` points to grid-cell ``(col, row)``."""

        points = np.asarray(points_xy, dtype=np.float64)
        if points.shape[-1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("points_xy must have finite shape (..., 2)")
        out = np.empty_like(points, dtype=np.float64)
        out[..., 0] = (points[..., 0] - self.x_min_m) / self.resolution_m
        out[..., 1] = (self.y_max_m - points[..., 1]) / self.resolution_m
        return out

    def grid_to_metric(self, points_xy: NDArray[np.float64] | Any) -> NDArray[np.float64]:
        """Convert grid-cell ``(col, row)`` points to local metric ``(x, y)``."""

        points = np.asarray(points_xy, dtype=np.float64)
        if points.shape[-1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("points_xy must have finite shape (..., 2)")
        out = np.empty_like(points, dtype=np.float64)
        out[..., 0] = self.x_min_m + points[..., 0] * self.resolution_m
        out[..., 1] = self.y_max_m - points[..., 1] * self.resolution_m
        return out

    def to_record(self) -> dict[str, float | int]:
        return {
            "resolution_m": float(self.resolution_m),
            "x_min_m": float(self.x_min_m),
            "x_max_m": float(self.x_max_m),
            "y_min_m": float(self.y_min_m),
            "y_max_m": float(self.y_max_m),
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class CostMapPlannerConfig:
    """Deterministic receding-horizon sampling planner parameters."""

    target_speed_mps: float | None = None
    horizon_s: float = 3.0
    path_samples: int = 41
    lateral_targets_m: tuple[float, ...] = (
        0.0,
        -0.4,
        0.4,
        -0.8,
        0.8,
        -1.0,
        1.0,
        -1.2,
        1.2,
        -1.6,
        1.6,
        -2.0,
        2.0,
        -2.4,
        2.4,
        -3.2,
        3.2,
    )
    min_horizon_distance_m: float = 24.0
    max_accel_mps2: float = 1.2
    brake_accel_mps2: float = -5.5
    hard_brake_accel_mps2: float = -8.0
    collision_cost_threshold: float = 0.78
    slow_cost_threshold: float = 0.52
    curvature_cost_weight: float = 1.8
    lateral_cost_weight: float = 0.018
    ego_width_m: float = 2.0
    ego_length_m: float = 4.5
    lane_half_width_m: float = 1.75
    road_half_width_m: float = 5.4
    dynamic_inflation_horizon_s: float = 1.5
    max_dynamic_inflation_m: float = 4.0
    selection_mode: str = "argmin"
    selection_temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.target_speed_mps is not None and self.target_speed_mps <= 0.0:
            raise ValueError("target_speed_mps must be positive when provided")
        if self.horizon_s <= 0.0 or self.path_samples < 2:
            raise ValueError("horizon_s must be positive and path_samples >= 2")
        if self.road_half_width_m <= self.lane_half_width_m:
            raise ValueError("road_half_width_m must exceed lane_half_width_m")
        if self.dynamic_inflation_horizon_s < 0.0 or self.max_dynamic_inflation_m < 0.0:
            raise ValueError("dynamic inflation parameters must be non-negative")
        if self.selection_mode not in {"argmin", "softmax"}:
            raise ValueError("selection_mode must be 'argmin' or 'softmax'")
        if self.selection_mode == "softmax" and (
            self.selection_temperature <= 0.0
            or not np.isfinite(float(self.selection_temperature))
        ):
            raise ValueError("selection_temperature must be positive and finite for softmax")


@dataclass(frozen=True)
class CostMapPlan:
    """One selected candidate path and the planner's bookkeeping."""

    path_xy: NDArray[np.float64]
    target_lateral_m: float
    score: float
    mean_cost: float
    max_cost: float
    collision_free: bool
    mean_curvature: float
    max_curvature: float
    target_speed_mps: float
    accel_mps2: float
    reason: str
    alternatives: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "path_xy": _path_to_list(self.path_xy),
            "target_lateral_m": float(self.target_lateral_m),
            "score": float(self.score),
            "mean_cost": float(self.mean_cost),
            "max_cost": float(self.max_cost),
            "collision_free": bool(self.collision_free),
            "mean_curvature": float(self.mean_curvature),
            "max_curvature": float(self.max_curvature),
            "target_speed_mps": float(self.target_speed_mps),
            "accel_mps2": float(self.accel_mps2),
            "reason": self.reason,
            "alternatives": self.alternatives,
        }


@dataclass(frozen=True)
class CostMapRunnerConfig:
    """Configuration for one deterministic dense cost-map run."""

    scenario: str = "cut_in"
    method: CostMapMethodName = "torsion_swirl"
    magnitude: str = "medium"
    seed: int = 0
    temporal_pattern: str = "burst"
    start_frame: int = 0
    duration_frames: int | None = 30
    dt: float = 0.1
    steps: int | None = None
    target_actor: Any | None = None
    grid: CostMapSpec = field(default_factory=CostMapSpec)
    planner: CostMapPlannerConfig = field(default_factory=CostMapPlannerConfig)
    swirl_sigma_m: float = 7.0
    swirl_alpha_rad: float | None = None
    translate_shift_m: float | None = None
    gaussian_cost_scale: float | None = None
    calibrate_budget: bool = True

    @property
    def method_key(self) -> str:
        return str(self.method)

    @property
    def operator_name(self) -> str:
        method = self.method_key
        if method == "clean":
            return "none"
        if method in ("cost_translate", "torsion_displace"):
            return "directed_false_free_space_obstacle_relocation_path_l2_calibrated"
        if method == "gaussian_cost":
            return "gaussian_cost_noise_path_l2_calibrated"
        if method == "random_warp_cost":
            return "random_sign_spatial_cost_warp_path_l2_calibrated"
        if method == "torsion_swirl":
            return "directed_spatial_cost_warp_path_l2_calibrated"
        if method == "swirl_illegal":
            return "contract_violating_boundary_warping_spatial_cost_warp_path_l2_calibrated"
        raise ValueError(f"unknown cost-map method {self.method!r}")

    @property
    def run_id(self) -> str:
        return (
            f"costmap_{self.scenario}_{self.method_key}_{self.magnitude}_"
            f"seed{self.seed}_{self.temporal_pattern}"
        )


@dataclass(frozen=True)
class CostMapRunResult:
    """Trace and summary from one dense cost-map closed-loop episode."""

    config: CostMapRunnerConfig
    trace: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = {
            "run_id": self.config.run_id,
            "git_commit": "",
            "simulator": "synthetic_headless",
            "simulator_version": "n/a",
            "model": "ego_centric_costmap_sampling_planner",
            "scenario_id": self.config.scenario,
            "seed": self.config.seed,
            "injection_point": "cost_map" if self.config.method_key != "clean" else "none",
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


RunResult = CostMapRunResult


@dataclass(frozen=True)
class CostMapFaultApplication:
    cost_grid: NDArray[np.float64]
    metadata: dict[str, float]


@dataclass(frozen=True)
class CostMapComponents:
    """Separated road prior and obstacle/free-space component for B.4."""

    road_grid: NDArray[np.float64]
    obstacle_grid: NDArray[np.float64]
    boundary_mask: NDArray[np.bool_]

    @property
    def combined(self) -> NDArray[np.float64]:
        combined = np.maximum(self.road_grid, self.obstacle_grid)
        return np.clip(combined, COST_MIN, COST_MAX)


class CostMapPlanner:
    """Sample lateral-offset paths and select the lowest-cost feasible path.

    The planner samples a small deterministic library of smooth lateral-offset
    paths over the receding horizon.  Each path is integrated over the cost map,
    rejected when it crosses high-cost occupancy/off-road cells, and scored
    with small curvature and lateral-deviation penalties.  Longitudinal speed is
    then selected by a simple rule based on the chosen path's maximum cost.
    """

    def __init__(self, config: CostMapPlannerConfig | None = None) -> None:
        self.config = config or CostMapPlannerConfig()

    def plan(
        self,
        ego: EgoState,
        cost_grid: NDArray[np.float64] | Any,
        spec: CostMapSpec,
    ) -> CostMapPlan:
        cost = np.asarray(cost_grid, dtype=np.float64)
        if cost.shape != spec.shape:
            raise ValueError(f"cost grid shape {cost.shape} does not match {spec.shape}")

        candidates = self._candidate_paths(ego, spec)
        scored = [
            self._score_candidate(path, target_lateral, ego, cost, spec, order)
            for order, (path, target_lateral) in enumerate(candidates)
        ]
        feasible = [row for row in scored if bool(row["collision_free"])]
        pool = feasible if feasible else scored
        if self.config.selection_mode == "argmin":
            chosen = self._argmin_selection(pool)
        else:
            chosen = self._softmax_selection(pool, ego, cost, spec)

        target_speed, accel, reason = self._speed_rule(ego, chosen)
        alternatives = tuple(_alternative_record(row) for row in scored)
        return CostMapPlan(
            path_xy=np.asarray(chosen["path_xy"], dtype=np.float64),
            target_lateral_m=float(chosen["target_lateral_m"]),
            score=float(chosen["score"]),
            mean_cost=float(chosen["mean_cost"]),
            max_cost=float(chosen["max_cost"]),
            collision_free=bool(chosen["collision_free"]),
            mean_curvature=float(chosen["mean_curvature"]),
            max_curvature=float(chosen["max_curvature"]),
            target_speed_mps=float(target_speed),
            accel_mps2=float(accel),
            reason=reason,
            alternatives=alternatives,
        )

    @staticmethod
    def _argmin_selection(scored: list[dict[str, Any]]) -> dict[str, Any]:
        return min(
            scored,
            key=lambda row: (
                float(row["score"]),
                abs(float(row["target_lateral_m"])),
                int(row["order"]),
            ),
        )

    def _softmax_selection(
        self,
        scored: list[dict[str, Any]],
        ego: EgoState,
        cost_grid: NDArray[np.float64],
        spec: CostMapSpec,
    ) -> dict[str, Any]:
        tau = float(self.config.selection_temperature)
        scores = np.asarray([float(row["score"]) for row in scored], dtype=np.float64)
        targets = np.asarray(
            [float(row["target_lateral_m"]) for row in scored],
            dtype=np.float64,
        )
        if scores.size == 0 or not np.all(np.isfinite(scores)):
            raise ValueError("softmax selection requires finite candidate scores")

        logits = -(scores - float(np.min(scores))) / tau
        weights = np.exp(logits)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            raise ValueError("softmax selection produced invalid weights")
        weights = weights / weight_sum

        target_lateral = float(np.sum(weights * targets))
        path = self._path_for_target(ego, spec, target_lateral)
        return self._score_candidate(
            path,
            target_lateral,
            ego,
            cost_grid,
            spec,
            order=-1,
        )

    def _candidate_paths(
        self,
        ego: EgoState,
        spec: CostMapSpec,
    ) -> tuple[tuple[NDArray[np.float64], float], ...]:
        candidates: list[tuple[NDArray[np.float64], float]] = []
        for target_lateral in self.config.lateral_targets_m:
            candidates.append(
                (
                    self._path_for_target(ego, spec, float(target_lateral)),
                    float(target_lateral),
                )
            )
        return tuple(candidates)

    def _path_for_target(
        self,
        ego: EgoState,
        spec: CostMapSpec,
        target_lateral: float,
    ) -> NDArray[np.float64]:
        cfg = self.config
        distance = max(cfg.min_horizon_distance_m, ego.speed * cfg.horizon_s)
        distance = min(distance, spec.x_max_m - 1.0)
        x = np.linspace(0.0, distance, cfg.path_samples, dtype=np.float64)
        u = x / max(distance, 1e-9)
        smooth = 3.0 * u * u - 2.0 * u * u * u
        y = float(target_lateral) * smooth
        return np.column_stack((x, y))

    def _score_candidate(
        self,
        path_xy: NDArray[np.float64],
        target_lateral: float,
        ego: EgoState,
        cost_grid: NDArray[np.float64],
        spec: CostMapSpec,
        order: int,
    ) -> dict[str, Any]:
        cfg = self.config
        grid_points = spec.metric_to_grid(path_xy)
        samples = sample_cost_grid(cost_grid, grid_points)
        world_y = ego.y + path_xy[:, 1]
        in_bounds = (
            (path_xy[:, 0] >= spec.x_min_m)
            & (path_xy[:, 0] <= spec.x_max_m)
            & (path_xy[:, 1] >= spec.y_min_m)
            & (path_xy[:, 1] <= spec.y_max_m)
        )
        off_road = np.abs(world_y) > cfg.road_half_width_m

        mean_cost = float(np.mean(samples))
        max_cost = float(np.max(samples))
        mean_curvature, max_curvature = _path_curvature(path_xy)
        collision_free = bool(
            np.all(samples < cfg.collision_cost_threshold)
            and not np.any(off_road)
            and np.all(in_bounds)
        )

        score = (
            mean_cost
            + cfg.curvature_cost_weight * mean_curvature
            + cfg.lateral_cost_weight * abs(target_lateral)
        )
        if not collision_free:
            score += 10.0 + max_cost

        return {
            "order": int(order),
            "path_xy": path_xy,
            "target_lateral_m": float(target_lateral),
            "score": float(score),
            "mean_cost": mean_cost,
            "max_cost": max_cost,
            "collision_free": collision_free,
            "mean_curvature": float(mean_curvature),
            "max_curvature": float(max_curvature),
        }

    def _speed_rule(self, ego: EgoState, chosen: Mapping[str, Any]) -> tuple[float, float, str]:
        cfg = self.config
        nominal = float(cfg.target_speed_mps if cfg.target_speed_mps is not None else ego.speed)
        max_cost = float(chosen["max_cost"])
        lateral = abs(float(chosen["target_lateral_m"]))

        if not bool(chosen["collision_free"]):
            target_speed = min(ego.speed, 0.25 * nominal)
            accel = cfg.hard_brake_accel_mps2
            reason = "hard_brake_no_collision_free_path"
        elif max_cost >= cfg.slow_cost_threshold:
            cost_ratio = np.clip(max_cost / max(cfg.collision_cost_threshold, 1e-9), 0.0, 1.0)
            target_speed = nominal * max(0.0, 1.0 - cost_ratio)
            accel = np.clip(0.9 * (target_speed - ego.speed), cfg.brake_accel_mps2, cfg.max_accel_mps2)
            reason = "slow_high_path_cost"
        else:
            target_speed = nominal * max(0.72, 1.0 - 0.04 * lateral)
            accel = np.clip(0.8 * (target_speed - ego.speed), cfg.brake_accel_mps2, cfg.max_accel_mps2)
            reason = "track_low_cost_path"

        return float(target_speed), float(accel), reason


class PotentialFieldPlanner(CostMapPlanner):
    """Continuous potential-field planner with no candidate enumeration.

    The planner takes one field-driven lateral step from the local cost
    gradient near the look-ahead distance and a lane-center attraction term.
    Its decision margin is therefore undefined, so returned plans intentionally
    carry no discrete alternatives.
    """

    def __init__(
        self,
        config: CostMapPlannerConfig | None = None,
        *,
        obstacle_gain: float = 8.0,
        lane_gain: float = 0.35,
        step_gain: float = 1.0,
        lateral_profile_samples: int = 81,
        lookahead_band_samples: int = 5,
    ) -> None:
        super().__init__(config)
        self.obstacle_gain = _positive_finite_float(obstacle_gain, "obstacle_gain")
        self.lane_gain = _nonnegative_finite_float(lane_gain, "lane_gain")
        self.step_gain = _positive_finite_float(step_gain, "step_gain")
        if lateral_profile_samples < 5:
            raise ValueError("lateral_profile_samples must be at least 5")
        if lookahead_band_samples < 1:
            raise ValueError("lookahead_band_samples must be positive")
        self.lateral_profile_samples = int(lateral_profile_samples)
        self.lookahead_band_samples = int(lookahead_band_samples)

    def plan(
        self,
        ego: EgoState,
        cost_grid: NDArray[np.float64] | Any,
        spec: CostMapSpec,
    ) -> CostMapPlan:
        cost = np.asarray(cost_grid, dtype=np.float64)
        if cost.shape != spec.shape:
            raise ValueError(f"cost grid shape {cost.shape} does not match {spec.shape}")

        distance = self._lookahead_distance(ego, spec)
        cost_gradient = self._lookahead_lateral_gradient(ego, cost, spec, distance)

        # Modeling choice: obstacle force descends the local cost field, while
        # the lane term attracts the ego vehicle's world y position to zero.
        force = -self.obstacle_gain * cost_gradient - self.lane_gain * float(ego.y)
        lower, upper = self._target_lateral_bounds(ego, spec)
        target_lateral = float(np.clip(self.step_gain * force, lower, upper))

        path = self._path_for_target(ego, spec, target_lateral)
        evaluated = self._evaluate_field_path(path, target_lateral, ego, cost, spec)
        target_speed, accel, reason = self._speed_rule(ego, evaluated)

        return CostMapPlan(
            path_xy=np.asarray(evaluated["path_xy"], dtype=np.float64),
            target_lateral_m=float(evaluated["target_lateral_m"]),
            score=float(evaluated["score"]),
            mean_cost=float(evaluated["mean_cost"]),
            max_cost=float(evaluated["max_cost"]),
            collision_free=bool(evaluated["collision_free"]),
            mean_curvature=float(evaluated["mean_curvature"]),
            max_curvature=float(evaluated["max_curvature"]),
            target_speed_mps=float(target_speed),
            accel_mps2=float(accel),
            reason=f"potential_field_{reason}",
            alternatives=(),
        )

    def _lookahead_distance(self, ego: EgoState, spec: CostMapSpec) -> float:
        cfg = self.config
        distance = max(cfg.min_horizon_distance_m, ego.speed * cfg.horizon_s)
        return float(min(distance, spec.x_max_m - 1.0))

    def _lookahead_lateral_gradient(
        self,
        ego: EgoState,
        cost_grid: NDArray[np.float64],
        spec: CostMapSpec,
        distance: float,
    ) -> float:
        lower, upper = self._target_lateral_bounds(ego, spec)
        if upper <= lower:
            return 0.0

        x = np.linspace(
            0.0,
            float(distance),
            self.config.path_samples,
            dtype=np.float64,
        )
        band_count = min(self.lookahead_band_samples, x.size)
        x_band = x[-band_count:]
        lateral = np.linspace(
            lower,
            upper,
            self.lateral_profile_samples,
            dtype=np.float64,
        )
        xx, yy = np.meshgrid(x_band, lateral, indexing="ij")
        metric_points = np.stack((xx, yy), axis=-1)
        grid_points = spec.metric_to_grid(metric_points)
        samples = sample_cost_grid(cost_grid, grid_points)
        profile = np.mean(samples, axis=0)
        gradient = np.gradient(profile, lateral)
        return float(np.interp(np.clip(0.0, lower, upper), lateral, gradient))

    def _target_lateral_bounds(
        self,
        ego: EgoState,
        spec: CostMapSpec,
    ) -> tuple[float, float]:
        cfg = self.config
        configured_limit = max(abs(float(value)) for value in cfg.lateral_targets_m)
        limit = min(configured_limit, cfg.road_half_width_m)
        lower = max(-limit, spec.y_min_m, -cfg.road_half_width_m - float(ego.y))
        upper = min(limit, spec.y_max_m, cfg.road_half_width_m - float(ego.y))
        if upper < lower:
            midpoint = 0.5 * (lower + upper)
            return float(midpoint), float(midpoint)
        return float(lower), float(upper)

    def _evaluate_field_path(
        self,
        path_xy: NDArray[np.float64],
        target_lateral: float,
        ego: EgoState,
        cost_grid: NDArray[np.float64],
        spec: CostMapSpec,
    ) -> dict[str, Any]:
        cfg = self.config
        grid_points = spec.metric_to_grid(path_xy)
        samples = sample_cost_grid(cost_grid, grid_points)
        world_y = ego.y + path_xy[:, 1]
        in_bounds = (
            (path_xy[:, 0] >= spec.x_min_m)
            & (path_xy[:, 0] <= spec.x_max_m)
            & (path_xy[:, 1] >= spec.y_min_m)
            & (path_xy[:, 1] <= spec.y_max_m)
        )
        off_road = np.abs(world_y) > cfg.road_half_width_m

        mean_cost = float(np.mean(samples))
        max_cost = float(np.max(samples))
        mean_curvature, max_curvature = _path_curvature(path_xy)
        collision_free = bool(
            np.all(samples < cfg.collision_cost_threshold)
            and not np.any(off_road)
            and np.all(in_bounds)
        )

        score = (
            mean_cost
            + cfg.curvature_cost_weight * mean_curvature
            + cfg.lateral_cost_weight * abs(float(target_lateral))
        )
        if not collision_free:
            score += 10.0 + max_cost

        return {
            "path_xy": path_xy,
            "target_lateral_m": float(target_lateral),
            "score": float(score),
            "mean_cost": mean_cost,
            "max_cost": max_cost,
            "collision_free": collision_free,
            "mean_curvature": float(mean_curvature),
            "max_curvature": float(max_curvature),
        }


def build_planner(
    planner_type: str,
    config: CostMapPlannerConfig | None = None,
) -> CostMapPlanner:
    """Build the requested downstream planner with sampling as the default."""

    key = str(planner_type).lower().strip().replace("-", "_")
    if key in {"", "sampling", "sample", "costmap", "argmin"}:
        return CostMapPlanner(config)
    if key in {"potential_field", "field", "gradient"}:
        return PotentialFieldPlanner(config)
    raise ValueError(
        "unknown planner_type "
        f"{planner_type!r}; expected 'sampling' or 'potential_field'"
    )


def run_costmap_closed_loop(
    config: CostMapRunnerConfig | Mapping[str, Any],
) -> CostMapRunResult:
    """Run one deterministic dense cost-map closed-loop episode."""

    cfg = config if isinstance(config, CostMapRunnerConfig) else CostMapRunnerConfig(**dict(config))
    scenario_kwargs: dict[str, Any] = {"dt": cfg.dt, "seed": cfg.seed}
    if cfg.steps is not None:
        scenario_kwargs["steps"] = cfg.steps
    scenario = get_scenario(cfg.scenario, **scenario_kwargs)

    target_actor = scenario.primary_actor_id if cfg.target_actor is None else cfg.target_actor
    planner_config = _planner_config_for_scenario(cfg.planner, scenario)
    planner = CostMapPlanner(planner_config)
    ego = scenario.ego_initial
    trace: list[dict[str, Any]] = []

    for frame_idx in range(scenario.steps):
        time_s = frame_idx * scenario.dt
        gt_objects = scenario.ground_truth_object_set(frame_idx)
        observed_objects = scenario.object_set(frame_idx)
        components = build_cost_grid_components(observed_objects, ego, cfg.grid, planner_config)
        clean_grid = components.combined
        clean_reference_plan = planner.plan(ego, clean_grid, cfg.grid)
        fault_active = _fault_active(cfg, frame_idx)
        fault = _apply_costmap_fault(
            components,
            cfg=cfg,
            spec=cfg.grid,
            ego=ego,
            scenario=scenario,
            objects=observed_objects,
            target_actor=target_actor,
            active=fault_active,
            planner=planner,
            clean_reference_plan=clean_reference_plan,
            frame_idx=frame_idx,
        )
        warped_grid = fault.cost_grid

        chosen_plan = planner.plan(ego, warped_grid, cfg.grid)
        realized_path_budget = (
            _path_l2_deviation(clean_reference_plan.path_xy, chosen_plan.path_xy)
            if fault_active
            else 0.0
        )

        actual_ttc = min_ttc(
            (ego.x, ego.y),
            ego.velocity_xy,
            gt_objects,
            ego_width=planner_config.ego_width_m,
            ego_length=planner_config.ego_length_m,
            horizon_s=5.0,
        )
        obstacle_distance = min_actor_distance(
            (ego.x, ego.y),
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
        lane_departure = abs(ego.y) > planner_config.lane_half_width_m
        off_road = abs(ego.y) > planner_config.road_half_width_m

        trace.append(
            {
                "frame": frame_idx,
                "time_s": float(time_s),
                "fault_active": bool(fault_active),
                "ego": _ego_to_record(ego),
                "gt_actors": _object_set_to_records(gt_objects),
                "clean_observed_actors": _object_set_to_records(observed_objects),
                "clean_cost_grid": _grid_to_list(clean_grid),
                "warped_cost_grid": _grid_to_list(warped_grid),
                "cost_grid_spec": cfg.grid.to_record(),
                "chosen_path": chosen_plan.to_record(),
                "clean_reference_path": clean_reference_plan.to_record(),
                "realized_path_deviation_m": float(realized_path_budget),
                "target_realized_path_budget_m": float(
                    fault.metadata.get("target_realized_path_budget_m", 0.0)
                ),
                "calibrated_scale": float(fault.metadata.get("calibrated_scale", 0.0)),
                "operator_strength": float(fault.metadata.get("operator_strength", 0.0)),
                "swirl_alpha_rad": float(fault.metadata.get("swirl_alpha_rad", 0.0)),
                "swirl_sigma_m": float(fault.metadata.get("swirl_sigma_m", 0.0)),
                "swirl_pivot_x_m": float(fault.metadata.get("swirl_pivot_x_m", 0.0)),
                "swirl_pivot_y_m": float(fault.metadata.get("swirl_pivot_y_m", 0.0)),
                "translate_shift_y_m": float(fault.metadata.get("translate_shift_y_m", 0.0)),
                "gaussian_cost_scale": float(fault.metadata.get("gaussian_cost_scale", 0.0)),
                "actual_ttc_s": float(actual_ttc),
                "min_obstacle_distance_m": float(obstacle_distance),
                "min_actor_distance_m": float(obstacle_distance),
                "collision": bool(collision_ids),
                "collision_track_ids": list(collision_ids),
                "lane_departure": bool(lane_departure),
                "off_road": bool(off_road),
                "path_curvature": float(chosen_plan.mean_curvature),
                "path_max_curvature": float(chosen_plan.max_curvature),
            }
        )
        ego = step_ego_on_costmap_path(ego, chosen_plan, dt=scenario.dt)

    summary = _summarize_trace(
        cfg,
        scenario=scenario,
        trace=trace,
        final_ego=ego,
        target_actor=target_actor,
    )
    return CostMapRunResult(config=cfg, trace=tuple(trace), summary=summary)


def build_cost_grid(
    objects: Any,
    ego: EgoState,
    spec: CostMapSpec | None = None,
    planner_config: CostMapPlannerConfig | None = None,
) -> NDArray[np.float64]:
    """Rasterize lane prior and inflated actor obstacles into a cost grid."""

    return build_cost_grid_components(objects, ego, spec, planner_config).combined


def build_cost_grid_components(
    objects: Any,
    ego: EgoState,
    spec: CostMapSpec | None = None,
    planner_config: CostMapPlannerConfig | None = None,
) -> CostMapComponents:
    """Return separated road and obstacle/free-space cost components.

    The road component contains the lane center prior, lane-boundary ridge, and
    off-road hard edge.  Fault operators are applied only to the obstacle
    component and recombined with this road component to preserve topology.
    """

    grid_spec = spec or CostMapSpec()
    cfg = planner_config or CostMapPlannerConfig()
    local_x, local_y = grid_spec.metric_mesh()

    road = build_road_cost_grid(ego, grid_spec, cfg)
    obstacle = np.zeros(grid_spec.shape, dtype=np.float64)

    for idx in range(len(objects)):
        actor_x = float(objects.x[idx] - ego.x)
        actor_y = float(objects.y[idx] - ego.y)
        if (
            actor_x < grid_spec.x_min_m - 8.0
            or actor_x > grid_spec.x_max_m + 8.0
            or actor_y < grid_spec.y_min_m - 8.0
            or actor_y > grid_spec.y_max_m + 8.0
        ):
            continue
        bump = _actor_cost_blob(
            local_x,
            local_y,
            center=(actor_x, actor_y),
            yaw=float(objects.yaw[idx] - ego.yaw),
            width=float(objects.w[idx]),
            length=float(objects.l[idx]),
            cls=str(objects.cls[idx]),
            velocity=(float(objects.v[idx, 0]), float(objects.v[idx, 1])),
            dynamic_horizon_s=cfg.dynamic_inflation_horizon_s,
            max_dynamic_inflation_m=cfg.max_dynamic_inflation_m,
        )
        obstacle = np.maximum(obstacle, bump)

    return CostMapComponents(
        road_grid=np.clip(road, COST_MIN, COST_MAX),
        obstacle_grid=np.clip(obstacle, COST_MIN, COST_MAX),
        boundary_mask=road_boundary_mask(ego, grid_spec, cfg),
    )


def build_predicted_cost_grid_components(
    objects: Any,
    ego: EgoState,
    spec: CostMapSpec | None = None,
    planner_config: CostMapPlannerConfig | None = None,
    *,
    horizon_s: float,
    dt: float,
    samples: int,
) -> CostMapComponents:
    """Return a swept predicted-occupancy cost map from CV actor trajectories.

    Prediction sampling is a modeling choice: each actor's constant-velocity
    trajectory is sampled uniformly over ``[0, horizon_s]`` and the obstacle
    blobs are combined with ``maximum`` to form a future occupancy sweep.
    Dynamic inflation is disabled for each sampled blob because prediction now
    supplies the forward sweep explicitly.
    """

    sample_count = int(samples)
    if sample_count <= 0:
        raise ValueError("samples must be positive")

    grid_spec = spec or CostMapSpec()
    cfg = planner_config or CostMapPlannerConfig()
    prediction = constant_velocity_predict(objects, horizon_s=horizon_s, dt=dt)
    local_x, local_y = grid_spec.metric_mesh()

    road = build_road_cost_grid(ego, grid_spec, cfg)
    obstacle = np.zeros(grid_spec.shape, dtype=np.float64)
    sample_times = np.linspace(0.0, float(horizon_s), sample_count, dtype=np.float64)

    for trajectory in prediction.trajectories:
        xs = np.interp(sample_times, prediction.times_s, trajectory.xy[:, 0])
        ys = np.interp(sample_times, prediction.times_s, trajectory.xy[:, 1])
        yaws = np.interp(sample_times, prediction.times_s, trajectory.yaw)
        for actor_x_world, actor_y_world, actor_yaw in zip(xs, ys, yaws, strict=True):
            actor_x = float(actor_x_world - ego.x)
            actor_y = float(actor_y_world - ego.y)
            if (
                actor_x < grid_spec.x_min_m - 8.0
                or actor_x > grid_spec.x_max_m + 8.0
                or actor_y < grid_spec.y_min_m - 8.0
                or actor_y > grid_spec.y_max_m + 8.0
            ):
                continue
            bump = _actor_cost_blob(
                local_x,
                local_y,
                center=(actor_x, actor_y),
                yaw=float(actor_yaw - ego.yaw),
                width=float(trajectory.width),
                length=float(trajectory.length),
                cls=str(trajectory.cls),
                velocity=(float(trajectory.velocity[0]), float(trajectory.velocity[1])),
                dynamic_horizon_s=0.0,
                max_dynamic_inflation_m=0.0,
            )
            obstacle = np.maximum(obstacle, bump)

    return CostMapComponents(
        road_grid=np.clip(road, COST_MIN, COST_MAX),
        obstacle_grid=np.clip(obstacle, COST_MIN, COST_MAX),
        boundary_mask=road_boundary_mask(ego, grid_spec, cfg),
    )


def build_road_cost_grid(
    ego: EgoState,
    spec: CostMapSpec | None = None,
    planner_config: CostMapPlannerConfig | None = None,
) -> NDArray[np.float64]:
    """Rasterize only the drivable-area and road/lane-boundary prior."""

    grid_spec = spec or CostMapSpec()
    cfg = planner_config or CostMapPlannerConfig()
    _, local_y = grid_spec.metric_mesh()
    world_y = ego.y + local_y

    cost = np.full(grid_spec.shape, 0.03, dtype=np.float64)
    lane_norm = np.abs(world_y) / max(cfg.lane_half_width_m, 1e-9)
    cost += 0.055 * lane_norm * lane_norm

    outside_lane = np.maximum(np.abs(world_y) - cfg.lane_half_width_m, 0.0)
    shoulder_width = cfg.road_half_width_m - cfg.lane_half_width_m
    cost += 0.20 * np.minimum(outside_lane / shoulder_width, 1.0) ** 2

    for boundary_y in (-cfg.lane_half_width_m, cfg.lane_half_width_m):
        cost += 0.08 * np.exp(-0.5 * ((world_y - boundary_y) / 0.22) ** 2)

    road_edge = np.abs(world_y) > cfg.road_half_width_m
    cost[road_edge] = COST_MAX

    return np.clip(cost, COST_MIN, COST_MAX)


def road_boundary_mask(
    ego: EgoState,
    spec: CostMapSpec | None = None,
    planner_config: CostMapPlannerConfig | None = None,
) -> NDArray[np.bool_]:
    """Return cells that encode lane boundaries or the road edge."""

    grid_spec = spec or CostMapSpec()
    cfg = planner_config or CostMapPlannerConfig()
    _, local_y = grid_spec.metric_mesh()
    world_y = ego.y + local_y
    lane_tolerance = max(0.5 * grid_spec.resolution_m, 0.24)
    lane_boundary = (
        np.abs(np.abs(world_y) - cfg.lane_half_width_m) <= lane_tolerance
    )
    road_edge = np.abs(world_y) >= cfg.road_half_width_m - 0.5 * grid_spec.resolution_m
    return (lane_boundary | road_edge).astype(bool, copy=False)


def step_ego_on_costmap_path(
    ego: EgoState,
    plan: CostMapPlan,
    *,
    dt: float,
) -> EgoState:
    """Execute the first step of a planned local path."""

    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    next_speed = max(0.0, ego.speed + plan.accel_mps2 * dt)
    avg_speed = 0.5 * (ego.speed + next_speed)
    step_x = max(0.0, avg_speed * dt)
    step_y = _path_y_at_x(plan.path_xy, step_x)
    return EgoState(
        x=float(ego.x + step_x),
        y=float(ego.y + step_y),
        yaw=0.0,
        speed=float(next_speed),
    )


def _apply_costmap_fault(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    spec: CostMapSpec,
    ego: EgoState,
    scenario: SyntheticScenario,
    objects: Any,
    target_actor: Any,
    active: bool,
    planner: CostMapPlanner,
    clean_reference_plan: CostMapPlan,
    frame_idx: int,
) -> CostMapFaultApplication:
    clean_grid = components.combined
    if cfg.method_key == "clean" or not active:
        return CostMapFaultApplication(cost_grid=clean_grid.copy(), metadata={})

    method = cfg.method_key
    if method not in COSTMAP_ALLOWED_FAULT_METHODS:
        raise ValueError(f"unknown cost-map method {cfg.method!r} in {scenario.scenario_id!r}")

    def apply_scaled(scale: float) -> CostMapFaultApplication:
        return _apply_scaled_costmap_fault(
            components,
            cfg=cfg,
            spec=spec,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            frame_idx=frame_idx,
            clean_reference_plan=clean_reference_plan,
            scale=scale,
        )

    target_budget = _target_path_budget(cfg)
    if not cfg.calibrate_budget:
        out = apply_scaled(1.0)
        metadata = dict(out.metadata)
        metadata.update(
            {
                "calibrated_scale": 1.0,
                "target_realized_path_budget_m": float(target_budget),
            }
        )
        return CostMapFaultApplication(cost_grid=out.cost_grid, metadata=metadata)

    return _calibrate_frame_path_budget(
        target_budget=target_budget,
        ego=ego,
        planner=planner,
        spec=spec,
        clean_reference_plan=clean_reference_plan,
        apply_scaled=apply_scaled,
    )


def _apply_scaled_costmap_fault(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    spec: CostMapSpec,
    ego: EgoState,
    objects: Any,
    target_actor: Any,
    frame_idx: int,
    clean_reference_plan: CostMapPlan,
    scale: float,
) -> CostMapFaultApplication:
    method = cfg.method_key
    if method in ("cost_translate", "torsion_displace"):
        return _apply_cost_translate(
            components,
            cfg=cfg,
            spec=spec,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            scale=scale,
        )
    if method == "gaussian_cost":
        return _apply_gaussian_cost(
            components,
            cfg=cfg,
            frame_idx=frame_idx,
            scale=scale,
        )
    if method == "random_warp_cost":
        return _apply_random_warp_cost(
            components,
            cfg=cfg,
            spec=spec,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            frame_idx=frame_idx,
            scale=scale,
        )
    if method == "torsion_swirl":
        return _apply_torsion_swirl(
            components,
            cfg=cfg,
            spec=spec,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            clean_reference_plan=clean_reference_plan,
            scale=scale,
        )
    if method == "swirl_illegal":
        return _apply_swirl_illegal(
            components,
            cfg=cfg,
            spec=spec,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            clean_reference_plan=clean_reference_plan,
            scale=scale,
        )
    raise ValueError(f"unknown cost-map method {cfg.method!r}")


def _apply_cost_translate(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    spec: CostMapSpec,
    ego: EgoState,
    objects: Any,
    target_actor: Any,
    scale: float,
) -> CostMapFaultApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    shift_m = scale * _base_translate_shift_m(cfg)
    sign = _away_from_route_lateral_sign(ego_y=ego.y, target_local_y=target_xy[1])
    shift_y_m = sign * shift_m
    shift_grid = np.array([0.0, -shift_y_m / spec.resolution_m], dtype=np.float64)
    obstacle = translate_cost_field(components.obstacle_grid, shift_grid)
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(abs(shift_y_m)),
            "translate_shift_y_m": float(shift_y_m),
        },
    )


def _apply_gaussian_cost(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    frame_idx: int,
    scale: float,
) -> CostMapFaultApplication:
    base = _base_gaussian_cost_scale(cfg)
    noise = _smooth_gaussian_noise(components.obstacle_grid.shape, cfg=cfg, frame_idx=frame_idx)
    obstacle = gaussian_cost_noise(components.obstacle_grid, noise, scale * base)
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(abs(scale * base)),
            "gaussian_cost_scale": float(scale * base),
        },
    )


def _apply_random_warp_cost(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    spec: CostMapSpec,
    ego: EgoState,
    objects: Any,
    target_actor: Any,
    frame_idx: int,
    scale: float,
) -> CostMapFaultApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    target_grid = spec.metric_to_grid(np.asarray(target_xy, dtype=np.float64))
    sigma_cells = _swirl_sigma_cells(cfg, spec)
    rng = np.random.default_rng(_stable_seed(cfg, frame_idx, "random_warp_cost"))
    angle = float(rng.uniform(-np.pi, np.pi))
    radial = sigma_cells * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    pivot_grid = target_grid - radial
    sign = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float64)))
    alpha = sign * scale * _base_swirl_alpha_abs(cfg)
    obstacle = spatial_cost_warp(
        components.obstacle_grid,
        tuple(pivot_grid),
        alpha=alpha,
        sigma=sigma_cells,
    )
    pivot_metric = spec.grid_to_metric(pivot_grid)
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(abs(alpha)),
            "swirl_alpha_rad": float(alpha),
            "swirl_sigma_m": float(cfg.swirl_sigma_m),
            "swirl_pivot_x_m": float(pivot_metric[0]),
            "swirl_pivot_y_m": float(pivot_metric[1]),
        },
    )


def _apply_torsion_swirl(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    spec: CostMapSpec,
    ego: EgoState,
    objects: Any,
    target_actor: Any,
    clean_reference_plan: CostMapPlan,
    scale: float,
) -> CostMapFaultApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    sigma_cells = _swirl_sigma_cells(cfg, spec)
    alpha_abs = scale * _base_swirl_alpha_abs(cfg)
    target_grid = spec.metric_to_grid(np.asarray(target_xy, dtype=np.float64))
    pivot_grid = np.array([target_grid[0] - sigma_cells, target_grid[1]], dtype=np.float64)
    alpha = _directed_swirl_alpha(
        target_grid=target_grid,
        pivot_grid=pivot_grid,
        alpha_abs=alpha_abs,
        sigma_cells=sigma_cells,
        spec=spec,
        ego=ego,
        preferred_lateral_m=clean_reference_plan.target_lateral_m,
    )
    obstacle = spatial_cost_warp(
        components.obstacle_grid,
        tuple(pivot_grid),
        alpha=alpha,
        sigma=sigma_cells,
    )
    pivot_metric = spec.grid_to_metric(pivot_grid)
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(abs(alpha)),
            "swirl_alpha_rad": float(alpha),
            "swirl_sigma_m": float(cfg.swirl_sigma_m),
            "swirl_pivot_x_m": float(pivot_metric[0]),
            "swirl_pivot_y_m": float(pivot_metric[1]),
        },
    )


def _apply_swirl_illegal(
    components: CostMapComponents,
    *,
    cfg: CostMapRunnerConfig,
    spec: CostMapSpec,
    ego: EgoState,
    objects: Any,
    target_actor: Any,
    clean_reference_plan: CostMapPlan,
    scale: float,
) -> CostMapFaultApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    sigma_cells = _swirl_sigma_cells(cfg, spec)
    alpha_abs = scale * _base_swirl_alpha_abs(cfg)
    target_grid = spec.metric_to_grid(np.asarray(target_xy, dtype=np.float64))
    pivot_grid = np.array([target_grid[0] - sigma_cells, target_grid[1]], dtype=np.float64)
    alpha = _directed_swirl_alpha(
        target_grid=target_grid,
        pivot_grid=pivot_grid,
        alpha_abs=alpha_abs,
        sigma_cells=sigma_cells,
        spec=spec,
        ego=ego,
        preferred_lateral_m=clean_reference_plan.target_lateral_m,
    )
    warped = spatial_cost_warp(
        components.combined,
        tuple(pivot_grid),
        alpha=alpha,
        sigma=sigma_cells,
    )
    pivot_metric = spec.grid_to_metric(pivot_grid)
    return CostMapFaultApplication(
        cost_grid=np.clip(warped, COST_MIN, COST_MAX),
        metadata={
            "operator_strength": float(abs(alpha)),
            "swirl_alpha_rad": float(alpha),
            "swirl_sigma_m": float(cfg.swirl_sigma_m),
            "swirl_pivot_x_m": float(pivot_metric[0]),
            "swirl_pivot_y_m": float(pivot_metric[1]),
            "contract_violation": 1.0,
        },
    )


def _recombine_fault(
    components: CostMapComponents,
    obstacle_grid: NDArray[np.float64],
    *,
    metadata: dict[str, float],
) -> CostMapFaultApplication:
    obstacle = np.clip(np.asarray(obstacle_grid, dtype=np.float64), COST_MIN, COST_MAX)
    if obstacle.shape != components.obstacle_grid.shape:
        raise ValueError("obstacle_grid shape changed during cost-map fault")
    clean = components.combined
    out = np.maximum(components.road_grid, obstacle)
    out = np.clip(out, COST_MIN, COST_MAX)
    out[components.boundary_mask] = clean[components.boundary_mask]
    return CostMapFaultApplication(cost_grid=out, metadata=dict(metadata))


def _calibrate_frame_path_budget(
    *,
    target_budget: float,
    ego: EgoState,
    planner: CostMapPlanner,
    spec: CostMapSpec,
    clean_reference_plan: CostMapPlan,
    apply_scaled: Any,
) -> CostMapFaultApplication:
    if target_budget <= 0.0:
        out = apply_scaled(0.0)
        metadata = dict(out.metadata)
        metadata.update(
            {
                "calibrated_scale": 0.0,
                "target_realized_path_budget_m": 0.0,
            }
        )
        return CostMapFaultApplication(cost_grid=out.cost_grid, metadata=metadata)

    best: tuple[float, float, float, float, CostMapFaultApplication] | None = None
    for scale in _CALIBRATION_SCALES:
        out = apply_scaled(float(scale))
        plan = planner.plan(ego, out.cost_grid, spec)
        budget = _path_l2_deviation(clean_reference_plan.path_xy, plan.path_xy)
        error = abs(budget - target_budget)
        tie = abs(float(scale))
        candidate = (error, -budget, tie, float(scale), out)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    assert best is not None
    best_error, _, _, best_scale, best_out = best
    metadata = dict(best_out.metadata)
    metadata.update(
        {
            "calibrated_scale": float(best_scale),
            "target_realized_path_budget_m": float(target_budget),
            "calibration_abs_error_m": float(best_error),
        }
    )
    return CostMapFaultApplication(cost_grid=best_out.cost_grid, metadata=metadata)


def _directed_swirl_alpha(
    *,
    target_grid: NDArray[np.float64],
    pivot_grid: NDArray[np.float64],
    alpha_abs: float,
    sigma_cells: float,
    spec: CostMapSpec,
    ego: EgoState,
    preferred_lateral_m: float,
) -> float:
    candidates = (abs(alpha_abs), -abs(alpha_abs))
    best_alpha = candidates[0]
    best_score = -float("inf")
    for alpha in candidates:
        moved_grid = twist_points(target_grid, pivot_grid, alpha=alpha, sigma=sigma_cells)
        moved_metric = spec.grid_to_metric(moved_grid)
        world_y = ego.y + float(moved_metric[1])
        if abs(preferred_lateral_m) > 1e-9:
            score = -abs(float(moved_metric[1]) - float(preferred_lateral_m))
        else:
            score = abs(world_y)
        if score > best_score + 1e-12:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def _target_path_budget(cfg: CostMapRunnerConfig) -> float:
    if cfg.method_key == "clean":
        return 0.0
    key = str(cfg.magnitude).lower().strip()
    try:
        return float(MAGNITUDE_PATH_BUDGETS_M[key])
    except KeyError as exc:
        valid = ", ".join(sorted(MAGNITUDE_PATH_BUDGETS_M))
        raise ValueError(f"unknown cost-map magnitude {cfg.magnitude!r}; expected {valid}") from exc


def _base_translate_shift_m(cfg: CostMapRunnerConfig) -> float:
    if cfg.translate_shift_m is not None:
        value = float(cfg.translate_shift_m)
    else:
        value = 1.0
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("translate_shift_m must be non-negative and finite")
    return value


def _base_gaussian_cost_scale(cfg: CostMapRunnerConfig) -> float:
    if cfg.gaussian_cost_scale is not None:
        value = float(cfg.gaussian_cost_scale)
    else:
        value = 0.10
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("gaussian_cost_scale must be non-negative and finite")
    return value


def _base_swirl_alpha_abs(cfg: CostMapRunnerConfig) -> float:
    if cfg.swirl_alpha_rad is not None:
        value = float(cfg.swirl_alpha_rad)
    else:
        value = 0.35
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("swirl_alpha_rad must be non-negative and finite")
    return value


def _away_from_route_lateral_sign(*, ego_y: float, target_local_y: float) -> float:
    world_y = float(ego_y + target_local_y)
    if abs(world_y) <= 1e-9:
        return 1.0
    return float(np.sign(world_y))


def _smooth_gaussian_noise(
    shape: tuple[int, int],
    *,
    cfg: CostMapRunnerConfig,
    frame_idx: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(_stable_seed(cfg, frame_idx, "gaussian_cost"))
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float64)
    for _ in range(5):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0
    noise -= float(np.mean(noise))
    max_abs = float(np.max(np.abs(noise)))
    if max_abs <= 1e-12:
        return np.zeros(shape, dtype=np.float64)
    return noise / max_abs


def _stable_seed(cfg: CostMapRunnerConfig, frame_idx: int | None, salt: str) -> int:
    text = "|".join(
        (
            "torsion-costmap",
            cfg.scenario,
            cfg.method_key,
            cfg.magnitude,
            str(cfg.seed),
            "" if frame_idx is None else str(frame_idx),
            salt,
        )
    )
    value = 0x9E3779B9
    for byte in text.encode("utf-8"):
        value = ((value * 1_000_003) ^ byte) & 0xFFFFFFFF
    return int(value)


def _actor_cost_blob(
    local_x: NDArray[np.float64],
    local_y: NDArray[np.float64],
    *,
    center: tuple[float, float],
    yaw: float,
    width: float,
    length: float,
    cls: str,
    velocity: tuple[float, float],
    dynamic_horizon_s: float,
    max_dynamic_inflation_m: float,
) -> NDArray[np.float64]:
    dx = local_x - center[0]
    dy = local_y - center[1]
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    longitudinal = c * dx + s * dy
    lateral = -s * dx + c * dy

    if cls == "pedestrian":
        sigma_x = max(1.0, 0.7 * length + 0.7)
        sigma_y = max(1.0, 0.9 * width + 0.7)
    else:
        sigma_x = max(1.8, 0.55 * length + 0.8)
        sigma_y = max(1.1, 0.70 * width + 0.7)

    speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
    dynamic_extension = min(max_dynamic_inflation_m, dynamic_horizon_s * speed)
    sigma_x += dynamic_extension

    maha = (longitudinal / sigma_x) ** 2 + (lateral / sigma_y) ** 2
    bump = 0.98 * np.exp(-0.5 * maha)
    bump[maha > 16.0] = 0.0
    return bump


def _target_local_xy(objects: Any, *, target_actor: Any, ego: EgoState) -> tuple[float, float]:
    matches = np.flatnonzero(objects.track_id == target_actor)
    if matches.size != 1:
        raise ValueError(f"target actor {target_actor!r} not found exactly once")
    idx = int(matches[0])
    return float(objects.x[idx] - ego.x), float(objects.y[idx] - ego.y)


def _planner_config_for_scenario(
    config: CostMapPlannerConfig,
    scenario: SyntheticScenario,
) -> CostMapPlannerConfig:
    if config.target_speed_mps is not None:
        return config
    return replace(config, target_speed_mps=scenario.ego_initial.speed)


def _fault_active(cfg: CostMapRunnerConfig, frame_idx: int) -> bool:
    if cfg.method_key == "clean":
        return False
    return is_active(
        cfg.temporal_pattern,
        frame_idx,
        start_frame=cfg.start_frame,
        duration=cfg.duration_frames,
    )


def _swirl_sigma_cells(cfg: CostMapRunnerConfig, spec: CostMapSpec) -> float:
    sigma_m = float(cfg.swirl_sigma_m)
    if not np.isfinite(sigma_m) or sigma_m <= 0.0:
        raise ValueError("swirl_sigma_m must be positive and finite")
    return sigma_m / spec.resolution_m


def _positive_finite_float(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be positive and finite")
    return out


def _nonnegative_finite_float(value: float, name: str) -> float:
    out = float(value)
    if out < 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be non-negative and finite")
    return out


def _path_l2_deviation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.shape != b.shape:
        raise ValueError("paths must have the same shape")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _path_curvature(path_xy: NDArray[np.float64]) -> tuple[float, float]:
    x = path_xy[:, 0]
    y = path_xy[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return float(np.mean(curvature)), float(np.max(curvature))


def _path_y_at_x(path_xy: NDArray[np.float64], x: float) -> float:
    x_values = path_xy[:, 0]
    y_values = path_xy[:, 1]
    return float(np.interp(float(x), x_values, y_values))


def _alternative_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_lateral_m": float(row["target_lateral_m"]),
        "score": float(row["score"]),
        "mean_cost": float(row["mean_cost"]),
        "max_cost": float(row["max_cost"]),
        "collision_free": bool(row["collision_free"]),
        "mean_curvature": float(row["mean_curvature"]),
        "max_curvature": float(row["max_curvature"]),
        "path_xy": _path_to_list(np.asarray(row["path_xy"], dtype=np.float64)),
    }


def _summarize_trace(
    cfg: CostMapRunnerConfig,
    *,
    scenario: SyntheticScenario,
    trace: list[dict[str, Any]],
    final_ego: EgoState,
    target_actor: Any,
) -> dict[str, Any]:
    collision_frames = [row["frame"] for row in trace if row["collision"]]
    active_rows = [row for row in trace if row["fault_active"]]
    min_ttc_value = min(float(row["actual_ttc_s"]) for row in trace)
    min_distance_value = min(float(row["min_obstacle_distance_m"]) for row in trace)
    route_completion = float(np.clip(final_ego.x / scenario.route_length_m, 0.0, 1.0))

    return {
        "collision": bool(collision_frames),
        "collision_frame": collision_frames[0] if collision_frames else None,
        "collision_time_s": (
            float(collision_frames[0] * scenario.dt) if collision_frames else None
        ),
        "min_ttc": float(min_ttc_value),
        "min_obstacle_distance": float(min_distance_value),
        "min_actor_distance": float(min_distance_value),
        "off_road": bool(any(row["off_road"] for row in trace)),
        "off_road_rate": float(np.mean([float(row["off_road"]) for row in trace])),
        "lane_departure": bool(any(row["lane_departure"] for row in trace)),
        "lane_departure_rate": float(np.mean([float(row["lane_departure"]) for row in trace])),
        "mean_path_curvature": float(np.mean([float(row["path_curvature"]) for row in trace])),
        "max_path_curvature": float(max(float(row["path_max_curvature"]) for row in trace)),
        "route_completion": route_completion,
        "fault_active_frames": int(sum(bool(row["fault_active"]) for row in trace)),
        "fault_start_time_s": float(cfg.start_frame * scenario.dt),
        "fault_end_time_s": float(_fault_end_time(cfg, scenario)),
        "target_actor": target_actor,
        "target_realized_path_budget_m": float(
            _mean_trace_value(active_rows, "target_realized_path_budget_m")
        ),
        "mean_realized_budget": float(_mean_trace_value(active_rows, "realized_path_deviation_m")),
        "mean_realized_path_deviation_m": float(
            _mean_trace_value(active_rows, "realized_path_deviation_m")
        ),
        "max_realized_path_deviation_m": float(
            max((float(row["realized_path_deviation_m"]) for row in active_rows), default=0.0)
        ),
        "mean_calibrated_scale": float(_mean_trace_value(active_rows, "calibrated_scale")),
        "mean_operator_strength": float(_mean_trace_value(active_rows, "operator_strength")),
        "mean_swirl_alpha_rad": float(_mean_trace_value(active_rows, "swirl_alpha_rad")),
        "mean_abs_swirl_alpha_rad": float(_mean_abs_trace_value(active_rows, "swirl_alpha_rad")),
        "swirl_sigma_m": float(_mean_trace_value(active_rows, "swirl_sigma_m")),
        "mean_translate_shift_y_m": float(_mean_trace_value(active_rows, "translate_shift_y_m")),
        "mean_abs_translate_shift_y_m": float(
            _mean_abs_trace_value(active_rows, "translate_shift_y_m")
        ),
        "mean_gaussian_cost_scale": float(_mean_trace_value(active_rows, "gaussian_cost_scale")),
        "final_ego_x": float(final_ego.x),
        "final_ego_y": float(final_ego.y),
        "final_ego_speed": float(final_ego.speed),
    }


def _fault_end_time(cfg: CostMapRunnerConfig, scenario: SyntheticScenario) -> float:
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


def _mean_trace_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_abs_trace_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([abs(float(row[key])) for row in rows]))


def _ego_to_record(ego: EgoState) -> dict[str, float]:
    return asdict(ego)


def _object_set_to_records(objects: Any) -> list[dict[str, Any]]:
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


def _grid_to_list(grid: NDArray[np.float64]) -> list[list[float]]:
    return [[float(value) for value in row] for row in grid]


def _path_to_list(path_xy: NDArray[np.float64]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in path_xy]
