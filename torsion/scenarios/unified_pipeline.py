"""Unified cross-representation TORSION pipeline.

This module implements the Phase 5 comparison protocol:

    scenario -> object-set -> cost-map -> one cost-map sampling planner -> ego step

The downstream planner, ego update, and safety metrics are shared for object
and cost-map injection.  Faults are budget-matched on the realized downstream
planned-path L2 deviation versus the clean plan at the same frame.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from torsion.metrics.safety import colliding_track_ids, min_actor_distance, min_ttc
from torsion.metrics.statistics import summarize_paired_delta, summarize_safety_group
from torsion.operators.costmap import (
    COST_MAX,
    COST_MIN,
    directional_obstacle_inflation,
    gaussian_cost_noise,
    spatial_cost_warp,
    translate_cost_field,
)
from torsion.operators.object import (
    ObjectSet,
    position_torsion,
    resolve_magnitude,
    velocity_direction_torsion,
    yaw_torsion,
)
from torsion.operators.temporal import is_active
from torsion.operators.twist import scene_swirl_torsion
from torsion.scenarios.costmap_runner import (
    CostMapComponents,
    CostMapPlan,
    CostMapPlanner,
    CostMapPlannerConfig,
    CostMapSpec,
    build_planner,
    build_cost_grid_components,
    build_predicted_cost_grid_components,
    step_ego_on_costmap_path,
)
from torsion.scenarios.planner import EgoState
from torsion.scenarios.predict import PredictionSet, constant_velocity_predict
from torsion.scenarios.synthetic_scenarios import SyntheticScenario, get_scenario

InjectionPoint = Literal["none", "object", "prediction", "costmap"]
UnifiedMethodName = Literal[
    "clean",
    "torsion_displace",
    "torsion_translate",
    "displacement",
    "gaussian",
    "gaussian_matched",
    "random_warp",
    "rotation",
    "torsion_swirl",
    "inflation",
]

SCENARIOS: tuple[str, ...] = ("cut_in", "leading_vehicle", "pedestrian_crossing")
INJECTION_POINTS: tuple[str, ...] = ("object", "costmap")
MAGNITUDES: tuple[str, ...] = ("low", "medium", "high")
MAGNITUDE_PATH_BUDGETS_M: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 1.0,
}
PRIMARY_METHOD = "torsion_displace"
DEFAULT_SEEDS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
DEFAULT_BOOTSTRAP_SEED = 0
TTC_CENSOR_S = 5.0
DEGRADATION_COLLISION_SECONDS = 5.0

_CALIBRATION_SCALES = np.array(
    [
        0.0,
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
        1.0,
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
_RAYLEIGH_MEAN_FACTOR = float(np.sqrt(np.pi / 2.0))
_FOLDED_NORMAL_MEAN_FACTOR = float(np.sqrt(2.0 / np.pi))
_TWIST_PEAK_FACTOR = float(np.exp(-0.5))


def default_unified_planner_config() -> CostMapPlannerConfig:
    """Return the shared downstream planner config for injection-point studies."""

    return CostMapPlannerConfig(
        lateral_targets_m=(
            0.0,
            -0.2,
            0.2,
            -0.4,
            0.4,
            -0.6,
            0.6,
            -0.8,
            0.8,
            -1.0,
            1.0,
            -1.2,
            1.2,
            -1.4,
            1.4,
            -1.6,
            1.6,
            -2.0,
            2.0,
            -2.4,
            2.4,
            -3.2,
            3.2,
        )
    )


@dataclass(frozen=True)
class UnifiedPipelineConfig:
    """Configuration for one deterministic unified closed-loop run."""

    scenario: str = "cut_in"
    injection_point: InjectionPoint = "object"
    method: UnifiedMethodName = "torsion_displace"
    magnitude: str = "medium"
    seed: int = 0
    temporal_pattern: str = "burst"
    start_frame: int = 0
    duration_frames: int | None = 30
    dt: float = 0.1
    steps: int | None = None
    target_actor: Any | None = None
    grid: CostMapSpec = field(default_factory=CostMapSpec)
    planner: CostMapPlannerConfig = field(default_factory=default_unified_planner_config)
    planner_type: str = "sampling"
    calibrate_budget: bool = True
    match_cross_representation_budget: bool = True
    swirl_sigma_m: float = 7.0
    trace_grids: bool = True
    use_prediction: bool = False
    prediction_horizon_s: float = 2.0
    prediction_samples: int = 5
    target_path_budget_m: float | None = None

    @property
    def magnitude_key(self) -> str:
        return _normalize_magnitude(self.magnitude)

    @property
    def method_key(self) -> str:
        method = str(self.method).lower().strip().replace("-", "_")
        if method in {"torsion_translate", "displacement"}:
            return "torsion_displace"
        if method == "gaussian_matched":
            return "gaussian"
        if method == "rotation":
            return "torsion_swirl"
        return method

    @property
    def injection_key(self) -> str:
        point = str(self.injection_point).lower().strip().replace("-", "_")
        if point in {"object", "object_set", "objects", "stage_a"}:
            return "object"
        if point in {"prediction", "predict", "stage_p"}:
            return "prediction"
        if point in {"costmap", "cost_map", "cost", "stage_b"}:
            return "costmap"
        if point in {"none", "clean"}:
            return "none"
        raise ValueError(f"unknown injection point {self.injection_point!r}")

    @property
    def operator_name(self) -> str:
        method = self.method_key
        if method == "clean":
            return "none"
        if method == "torsion_displace":
            return "directed_semantic_displacement_path_l2_calibrated"
        if method == "gaussian":
            return "gaussian_baseline_path_l2_calibrated"
        if method == "random_warp":
            return "random_warp_baseline_path_l2_calibrated"
        if method == "torsion_swirl":
            return "rotation_swirl_ablation_path_l2_calibrated"
        if method == "inflation":
            return "cost_inflation_ablation_path_l2_calibrated"
        raise ValueError(f"unknown unified method {self.method!r}")

    @property
    def run_id(self) -> str:
        return (
            f"unified_{self.scenario}_{self.injection_key}_{self.method_key}_"
            f"{self.magnitude_key}_seed{self.seed}_{self.temporal_pattern}"
        )


@dataclass(frozen=True)
class UnifiedRunResult:
    """Full per-frame trace and run summary for the unified pipeline."""

    config: UnifiedPipelineConfig
    trace: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = {
            "run_id": self.config.run_id,
            "git_commit": "",
            "simulator": "synthetic_headless",
            "simulator_version": "n/a",
            "model": "unified_object_to_costmap_sampling_planner",
            "scenario_id": self.config.scenario,
            "seed": int(self.config.seed),
            "injection_point": self.config.injection_key,
            "operator": self.config.operator_name,
            "method": self.config.method_key,
            "magnitude": self.config.magnitude_key,
            "temporal_pattern": self.config.temporal_pattern,
            "start_frame": int(self.config.start_frame),
            "duration_frames": self.config.duration_frames,
            "target_actor": self.summary.get("target_actor"),
        }
        record.update(self.summary)
        return record


@dataclass(frozen=True)
class UnifiedFaultApplication:
    """Per-frame Stage A/Stage B fault result."""

    objects: ObjectSet
    components: CostMapComponents
    cost_grid: NDArray[np.float64]
    metadata: dict[str, float]
    prediction: PredictionSet | None = None


@dataclass(frozen=True)
class _BudgetCandidate:
    """One calibrated candidate and its downstream path-L2 budget."""

    scale: float
    budget: float
    application: UnifiedFaultApplication


def run_unified_pipeline(
    config: UnifiedPipelineConfig | Mapping[str, Any],
) -> UnifiedRunResult:
    """Run the unified object-to-cost-map closed loop with optional staged injection."""

    cfg = (
        config
        if isinstance(config, UnifiedPipelineConfig)
        else UnifiedPipelineConfig(**dict(config))
    )
    scenario_kwargs: dict[str, Any] = {"dt": cfg.dt, "seed": cfg.seed}
    if cfg.steps is not None:
        scenario_kwargs["steps"] = cfg.steps
    scenario = get_scenario(cfg.scenario, **scenario_kwargs)

    target_actor = scenario.primary_actor_id if cfg.target_actor is None else cfg.target_actor
    planner_config = _planner_config_for_scenario(cfg.planner, scenario)
    planner = build_planner(cfg.planner_type, planner_config)
    ego = scenario.ego_initial
    trace: list[dict[str, Any]] = []

    for frame_idx in range(scenario.steps):
        time_s = frame_idx * scenario.dt
        gt_objects = scenario.ground_truth_object_set(frame_idx)
        clean_objects = scenario.object_set(frame_idx)
        clean_prediction = (
            _predict_objects_for_config(clean_objects, cfg) if cfg.use_prediction else None
        )
        clean_components = _build_cost_grid_components_for_config(
            clean_objects,
            ego,
            cfg,
            planner_config,
        )
        clean_grid = clean_components.combined
        clean_reference_plan = planner.plan(ego, clean_grid, cfg.grid)
        fault_active = _fault_active(cfg, frame_idx)

        if cfg.injection_key == "object":
            fault = _apply_object_stage_fault(
                clean_objects,
                cfg=cfg,
                scenario=scenario,
                ego=ego,
                planner=planner,
                planner_config=planner_config,
                clean_reference_plan=clean_reference_plan,
                target_actor=target_actor,
                frame_idx=frame_idx,
                active=fault_active,
            )
            stage_a_objects = fault.objects
            stage_b_grid = fault.components.combined
            final_grid = fault.cost_grid
        elif cfg.injection_key == "prediction":
            if not cfg.use_prediction:
                raise ValueError("prediction injection requires use_prediction=True")
            stage_a_objects = clean_objects
            fault = _apply_prediction_stage_fault(
                clean_objects,
                clean_components=clean_components,
                cfg=cfg,
                scenario=scenario,
                ego=ego,
                planner=planner,
                planner_config=planner_config,
                clean_reference_plan=clean_reference_plan,
                target_actor=target_actor,
                frame_idx=frame_idx,
                active=fault_active,
            )
            stage_b_grid = fault.components.combined
            final_grid = fault.cost_grid
        elif cfg.injection_key == "costmap":
            stage_a_objects = clean_objects
            stage_b_grid = clean_grid
            fault = _apply_costmap_stage_fault(
                clean_components,
                cfg=cfg,
                scenario=scenario,
                ego=ego,
                planner=planner,
                clean_reference_plan=clean_reference_plan,
                objects=clean_objects,
                target_actor=target_actor,
                frame_idx=frame_idx,
                active=fault_active,
            )
            final_grid = fault.cost_grid
        elif cfg.injection_key == "none" or cfg.method_key == "clean":
            stage_a_objects = clean_objects
            stage_b_grid = clean_grid
            fault = UnifiedFaultApplication(
                objects=clean_objects,
                components=clean_components,
                cost_grid=clean_grid.copy(),
                metadata={},
            )
            final_grid = fault.cost_grid
        else:
            raise ValueError(f"unknown injection point {cfg.injection_point!r}")

        chosen_plan = planner.plan(ego, final_grid, cfg.grid)
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
            horizon_s=TTC_CENSOR_S,
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
        object_delta = _object_delta_metrics(clean_objects, stage_a_objects, target_actor=target_actor)
        cost_delta = _cost_grid_delta_metrics(clean_grid, final_grid)
        if cfg.use_prediction:
            fault_prediction = (
                fault.prediction
                if fault.prediction is not None
                else _predict_objects_for_config(stage_a_objects, cfg)
            )
            prediction_delta = _prediction_delta_metrics(
                clean_prediction,
                fault_prediction,
                target_actor=target_actor,
            )
        else:
            prediction_delta = {"prediction_traj_l2_delta": 0.0}

        row = {
            "frame": int(frame_idx),
            "time_s": float(time_s),
            "fault_active": bool(fault_active),
            "injection_point": cfg.injection_key,
            "ego": _ego_to_record(ego),
            "clean_object_set": _object_set_to_records(clean_objects),
            "stage_a_object_set": _object_set_to_records(stage_a_objects),
            "gt_actors": _object_set_to_records(gt_objects),
            "cost_grid_spec": cfg.grid.to_record(),
            "clean_reference_path": clean_reference_plan.to_record(),
            "chosen_path": chosen_plan.to_record(),
            "control": _plan_to_control_record(chosen_plan, planner_config),
            "realized_path_deviation_m": float(realized_path_budget),
            "target_realized_path_budget_m": float(
                fault.metadata.get("target_realized_path_budget_m", 0.0)
            ),
            "calibrated_scale": float(fault.metadata.get("calibrated_scale", 0.0)),
            "calibration_abs_error_m": float(
                fault.metadata.get("calibration_abs_error_m", 0.0)
            ),
            "operator_strength": float(fault.metadata.get("operator_strength", 0.0)),
            "object_position_shift_m": float(object_delta["object_position_shift_m"]),
            "object_yaw_shift_rad": float(object_delta["object_yaw_shift_rad"]),
            "object_velocity_rotation_rad": float(
                object_delta["object_velocity_rotation_rad"]
            ),
            "object_shift_x_m": float(fault.metadata.get("object_shift_x_m", 0.0)),
            "object_shift_y_m": float(fault.metadata.get("object_shift_y_m", 0.0)),
            "cost_grid_l2_delta": float(cost_delta["cost_grid_l2_delta"]),
            "cost_grid_mean_abs_delta": float(cost_delta["cost_grid_mean_abs_delta"]),
            "cost_grid_max_abs_delta": float(cost_delta["cost_grid_max_abs_delta"]),
            "translate_shift_y_m": float(fault.metadata.get("translate_shift_y_m", 0.0)),
            "swirl_alpha_rad": float(fault.metadata.get("swirl_alpha_rad", 0.0)),
            "swirl_sigma_m": float(fault.metadata.get("swirl_sigma_m", 0.0)),
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
        row["prediction_traj_l2_delta"] = float(
            prediction_delta["prediction_traj_l2_delta"]
        )
        if cfg.trace_grids:
            row.update(
                {
                    "clean_cost_grid": _grid_to_list(clean_grid),
                    "stage_b_cost_grid": _grid_to_list(stage_b_grid),
                    "final_cost_grid": _grid_to_list(final_grid),
                    "warped_cost_grid": _grid_to_list(final_grid),
                }
            )
        trace.append(row)
        ego = step_ego_on_costmap_path(ego, chosen_plan, dt=scenario.dt)

    summary = _summarize_trace(
        cfg,
        scenario=scenario,
        trace=trace,
        final_ego=ego,
        target_actor=target_actor,
    )
    return UnifiedRunResult(config=cfg, trace=tuple(trace), summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--method", default=PRIMARY_METHOD)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--runs-output",
        type=Path,
        default=Path("results/metrics/unified_runs.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/unified_injection_sensitivity.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    jobs = _sweep_jobs(method=str(args.method), seeds=int(args.seeds))
    records = _run_jobs(jobs, workers=int(args.workers))
    records = _sort_records(records)
    _write_csv(args.runs_output, [_csv_record(row) for row in records])

    summary = summarize_injection_sensitivity(
        records,
        n_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    _write_csv(args.summary_output, summary, fieldnames=SUMMARY_COLUMNS)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure12 = args.figures_dir / "figure12_injection_sensitivity.png"
    figure13 = args.figures_dir / "figure13_error_propagation.png"
    figure14 = args.figures_dir / "figure14_scenario_heatmap.png"
    make_figure12(summary, figure12)
    make_figure14(summary, figure14)
    representative = _select_representative_pair(records)
    make_figure13(representative, figure13)

    _print_sensitivity_table(summary)
    print(f"\nWrote {len(records)} runs to {args.runs_output}")
    print(f"Wrote {len(summary)} sensitivity rows to {args.summary_output}")
    print(f"Wrote figures to {args.figures_dir}")


SUMMARY_COLUMNS = (
    "scenario_id",
    "injection_point",
    "method",
    "magnitude",
    "n_runs",
    "collision_rate",
    "collision_rate_ci_low",
    "collision_rate_ci_high",
    "mean_min_ttc",
    "mean_min_ttc_ci_low",
    "mean_min_ttc_ci_high",
    "std_min_ttc",
    "iqr_min_ttc",
    "worst5pct_min_ttc",
    "worst_case_min_ttc",
    "mean_realized_budget",
    "paired_n",
    "paired_mean_min_ttc_delta_s",
    "paired_mean_min_ttc_delta_ci_low",
    "paired_mean_min_ttc_delta_ci_high",
    "paired_std_min_ttc_delta_s",
    "mean_target_budget",
    "mean_calibration_abs_error_m",
    "clean_collision_rate",
    "clean_mean_min_ttc",
    "collision_rate_degradation",
    "mean_min_ttc_drop_s",
    "degradation_score",
)


def summarize_injection_sensitivity(
    records: list[dict[str, Any]],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return per-scenario and aggregate object-vs-costmap sensitivity rows."""

    rows: list[dict[str, Any]] = []
    scopes = (*SCENARIOS, "ALL")
    for scope in scopes:
        scoped = (
            records
            if scope == "ALL"
            else [row for row in records if row["scenario_id"] == scope]
        )
        for injection_point in INJECTION_POINTS:
            for magnitude in MAGNITUDES:
                group = [
                    row
                    for row in scoped
                    if row["injection_point"] == injection_point
                    and row["magnitude"] == magnitude
                ]
                if not group:
                    continue
                clean = [
                    row
                    for row in scoped
                    if row["injection_point"] == "none" and row["magnitude"] == magnitude
                ]
                if not clean:
                    raise ValueError(f"missing clean baseline for {scope}/{magnitude}")
                record = {
                    "scenario_id": scope,
                    "injection_point": injection_point,
                    "method": group[0]["method"],
                    "magnitude": magnitude,
                }
                record.update(
                    summarize_safety_group(
                        collision=[float(row["collision"]) for row in group],
                        min_ttc=[float(row["min_ttc_censored"]) for row in group],
                        realized_budget=[float(row["mean_realized_budget"]) for row in group],
                        n_resamples=n_resamples,
                        seed=bootstrap_seed,
                    )
                )
                paired = _paired_clean_fault_ttc(
                    clean,
                    group,
                    n_resamples=n_resamples,
                    bootstrap_seed=bootstrap_seed,
                )
                record.update(paired)
                clean_collision = float(np.mean([float(row["collision"]) for row in clean]))
                clean_ttc = float(np.mean([float(row["min_ttc_censored"]) for row in clean]))
                collision_delta = float(record["collision_rate"] - clean_collision)
                ttc_drop = float(clean_ttc - float(record["mean_min_ttc"]))
                record["mean_target_budget"] = float(
                    np.mean([float(row["target_realized_path_budget_m"]) for row in group])
                )
                record["mean_calibration_abs_error_m"] = float(
                    np.mean([float(row["mean_calibration_abs_error_m"]) for row in group])
                )
                record["clean_collision_rate"] = clean_collision
                record["clean_mean_min_ttc"] = clean_ttc
                record["collision_rate_degradation"] = collision_delta
                record["mean_min_ttc_drop_s"] = ttc_drop
                record["degradation_score"] = float(
                    max(0.0, ttc_drop)
                    + DEGRADATION_COLLISION_SECONDS * max(0.0, collision_delta)
                )
                rows.append(record)
    return _sort_summary(rows)


def _paired_clean_fault_ttc(
    clean: list[dict[str, Any]],
    group: list[dict[str, Any]],
    *,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    clean_by_key = {
        (row["scenario_id"], row["magnitude"], int(row["seed"])): row
        for row in clean
    }
    baseline: list[float] = []
    treatment: list[float] = []
    for row in group:
        key = (row["scenario_id"], row["magnitude"], int(row["seed"]))
        clean_row = clean_by_key.get(key)
        if clean_row is None:
            raise ValueError(f"missing paired clean row for {key!r}")
        baseline.append(float(clean_row["min_ttc_censored"]))
        treatment.append(float(row["min_ttc_censored"]))

    paired = summarize_paired_delta(
        baseline,
        treatment,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
    )
    return {
        "paired_n": paired["paired_n"],
        "paired_mean_min_ttc_delta_s": paired["paired_mean_delta"],
        "paired_mean_min_ttc_delta_ci_low": paired["paired_delta_ci_low"],
        "paired_mean_min_ttc_delta_ci_high": paired["paired_delta_ci_high"],
        "paired_std_min_ttc_delta_s": paired["paired_std_delta"],
    }


def make_figure12(summary: list[dict[str, Any]], path: Path) -> None:
    """Write collision/TTC magnitude-response small multiples."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"object": "#1f77b4", "costmap": "#d62728"}
    x = np.arange(len(MAGNITUDES), dtype=np.float64)
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(13.5, 6.8), sharex=True)
    for col, scenario in enumerate(SCENARIOS):
        scenario_rows = [row for row in summary if row["scenario_id"] == scenario]
        for injection_point in INJECTION_POINTS:
            rows = [
                _find_summary_row(scenario_rows, injection_point, magnitude)
                for magnitude in MAGNITUDES
            ]
            collision = np.array([float(row["collision_rate"]) for row in rows])
            collision_low = np.array([float(row["collision_rate_ci_low"]) for row in rows])
            collision_high = np.array([float(row["collision_rate_ci_high"]) for row in rows])
            ttc = np.array([float(row["mean_min_ttc"]) for row in rows])
            ttc_low = np.array([float(row["mean_min_ttc_ci_low"]) for row in rows])
            ttc_high = np.array([float(row["mean_min_ttc_ci_high"]) for row in rows])
            label = "inject@object" if injection_point == "object" else "inject@costmap"

            axes[0, col].errorbar(
                x,
                collision,
                yerr=np.vstack((collision - collision_low, collision_high - collision)),
                color=colors[injection_point],
                marker="o",
                capsize=3,
                label=label,
            )
            axes[1, col].errorbar(
                x,
                ttc,
                yerr=np.vstack((ttc - ttc_low, ttc_high - ttc)),
                color=colors[injection_point],
                marker="o",
                capsize=3,
                label=label,
            )
        axes[0, col].set_title(_scenario_label(scenario))
        axes[0, col].set_ylim(-0.05, 1.05)
        axes[0, col].grid(True, alpha=0.25)
        axes[1, col].set_ylim(0.0, TTC_CENSOR_S + 0.2)
        axes[1, col].grid(True, alpha=0.25)
        axes[1, col].set_xticks(x, ["Low", "Med", "High"])
    axes[0, 0].set_ylabel("Collision rate")
    axes[1, 0].set_ylabel("Mean min-TTC (s)")
    axes[0, -1].legend(loc="upper right", frameon=False)
    fig.suptitle("Figure 12. Injection-point sensitivity at matched path budget", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figure14(summary: list[dict[str, Any]], path: Path) -> None:
    """Write scenario x injection-point paired TTC-degradation heatmap."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.zeros((len(SCENARIOS), len(INJECTION_POINTS)), dtype=np.float64)
    ci_low = np.zeros_like(matrix)
    ci_high = np.zeros_like(matrix)
    magnitude = "high"
    for row_idx, scenario in enumerate(SCENARIOS):
        scenario_rows = [row for row in summary if row["scenario_id"] == scenario]
        for col_idx, injection_point in enumerate(INJECTION_POINTS):
            row = _find_summary_row(scenario_rows, injection_point, magnitude)
            matrix[row_idx, col_idx] = float(row["paired_mean_min_ttc_delta_s"])
            ci_low[row_idx, col_idx] = float(row["paired_mean_min_ttc_delta_ci_low"])
            ci_high[row_idx, col_idx] = float(row["paired_mean_min_ttc_delta_ci_high"])

    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    vmax = max(0.05, float(np.max(matrix)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(
        np.arange(len(INJECTION_POINTS)),
        ["inject@object", "inject@costmap"],
    )
    ax.set_yticks(np.arange(len(SCENARIOS)), [_scenario_label(name) for name in SCENARIOS])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}\n[{ci_low[row_idx, col_idx]:.2f}, {ci_high[row_idx, col_idx]:.2f}]",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_title("Figure 14. High-magnitude paired min-TTC degradation")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Clean - fault min-TTC (s); cell text is mean [95% CI]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figure13(
    representative: tuple[str, str, int],
    path: Path,
) -> None:
    """Write one representative error-propagation trace for both stages."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario, magnitude, seed = representative
    clean = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario=scenario,
            injection_point="none",
            method="clean",
            magnitude=magnitude,
            seed=seed,
        )
    )
    object_result = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario=scenario,
            injection_point="object",
            method=PRIMARY_METHOD,
            magnitude=magnitude,
            seed=seed,
        )
    )
    costmap_result = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario=scenario,
            injection_point="costmap",
            method=PRIMARY_METHOD,
            magnitude=magnitude,
            seed=seed,
        )
    )

    results = (("inject@object", object_result), ("inject@costmap", costmap_result))
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2))
    for row_idx, (label, result) in enumerate(results):
        frame = _representative_frame(result)
        _plot_object_stage(axes[row_idx, 0], frame, label)
        _plot_costmap_delta(axes[row_idx, 1], frame)
        _plot_path_shift(axes[row_idx, 2], frame)
        _plot_ttc_control(axes[row_idx, 3], clean, result)
    fig.suptitle(
        (
            "Figure 13. Error propagation through object-set, cost-map, "
            "plan, control, and safety"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_cost_grid_components_for_config(
    objects: ObjectSet,
    ego: EgoState,
    cfg: UnifiedPipelineConfig,
    planner_config: CostMapPlannerConfig,
) -> CostMapComponents:
    if not cfg.use_prediction:
        return build_cost_grid_components(objects, ego, cfg.grid, planner_config)
    return build_predicted_cost_grid_components(
        objects,
        ego,
        cfg.grid,
        planner_config,
        horizon_s=cfg.prediction_horizon_s,
        dt=cfg.dt,
        samples=cfg.prediction_samples,
    )


def _predict_objects_for_config(
    objects: ObjectSet,
    cfg: UnifiedPipelineConfig,
) -> PredictionSet:
    return constant_velocity_predict(
        objects,
        horizon_s=cfg.prediction_horizon_s,
        dt=cfg.dt,
    )


def _apply_object_stage_fault(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    planner: CostMapPlanner,
    planner_config: CostMapPlannerConfig,
    clean_reference_plan: CostMapPlan,
    target_actor: Any,
    frame_idx: int,
    active: bool,
) -> UnifiedFaultApplication:
    clean_components = _build_cost_grid_components_for_config(
        objects,
        ego,
        cfg,
        planner_config,
    )
    if cfg.method_key == "clean" or not active:
        return UnifiedFaultApplication(
            objects=objects,
            components=clean_components,
            cost_grid=clean_components.combined,
            metadata={},
        )
    if cfg.calibrate_budget and cfg.match_cross_representation_budget:
        return _calibrate_cross_representation_frame(
            choose="object",
            objects=objects,
            clean_components=clean_components,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            planner=planner,
            planner_config=planner_config,
            clean_reference_plan=clean_reference_plan,
            target_actor=target_actor,
            frame_idx=frame_idx,
        )

    def apply_scaled(scale: float) -> UnifiedFaultApplication:
        perturbed, metadata = _apply_scaled_object_fault(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
            frame_idx=frame_idx,
            scale=scale,
        )
        components = _build_cost_grid_components_for_config(
            perturbed,
            ego,
            cfg,
            planner_config,
        )
        prediction = _predict_objects_for_config(perturbed, cfg) if cfg.use_prediction else None
        return UnifiedFaultApplication(
            objects=perturbed,
            components=components,
            cost_grid=components.combined,
            metadata=metadata,
            prediction=prediction,
        )

    if not cfg.calibrate_budget:
        out = apply_scaled(1.0)
        metadata = dict(out.metadata)
        metadata.update(
            {
                "calibrated_scale": 1.0,
                "target_realized_path_budget_m": _target_path_budget(cfg),
            }
        )
        return UnifiedFaultApplication(
            objects=out.objects,
            components=out.components,
            cost_grid=out.cost_grid,
            metadata=metadata,
            prediction=out.prediction,
        )

    return _calibrate_frame_path_budget(
        target_budget=_target_path_budget(cfg),
        ego=ego,
        planner=planner,
        spec=cfg.grid,
        clean_reference_plan=clean_reference_plan,
        apply_scaled=apply_scaled,
    )


def _apply_prediction_stage_fault(
    objects: ObjectSet,
    *,
    clean_components: CostMapComponents,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    planner: CostMapPlanner,
    planner_config: CostMapPlannerConfig,
    clean_reference_plan: CostMapPlan,
    target_actor: Any,
    frame_idx: int,
    active: bool,
) -> UnifiedFaultApplication:
    if not cfg.use_prediction:
        raise ValueError("prediction-stage faults require use_prediction=True")
    if cfg.method_key == "clean" or not active:
        return UnifiedFaultApplication(
            objects=objects,
            components=clean_components,
            cost_grid=clean_components.combined.copy(),
            metadata={},
            prediction=_predict_objects_for_config(objects, cfg),
        )
    if cfg.method_key == "gaussian":
        raise ValueError("gaussian prediction-stage faults are not contract-preserving")

    def apply_scaled(scale: float) -> UnifiedFaultApplication:
        motion_objects, metadata = _apply_scaled_prediction_fault(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
            frame_idx=frame_idx,
            scale=scale,
        )
        components = build_predicted_cost_grid_components(
            motion_objects,
            ego,
            cfg.grid,
            planner_config,
            horizon_s=cfg.prediction_horizon_s,
            dt=cfg.dt,
            samples=cfg.prediction_samples,
        )
        return UnifiedFaultApplication(
            objects=objects,
            components=components,
            cost_grid=components.combined,
            metadata=metadata,
            prediction=_predict_objects_for_config(motion_objects, cfg),
        )

    if not cfg.calibrate_budget:
        out = apply_scaled(1.0)
        metadata = dict(out.metadata)
        metadata.update(
            {
                "calibrated_scale": 1.0,
                "target_realized_path_budget_m": _target_path_budget(cfg),
            }
        )
        return UnifiedFaultApplication(
            objects=out.objects,
            components=out.components,
            cost_grid=out.cost_grid,
            metadata=metadata,
            prediction=out.prediction,
        )

    return _calibrate_frame_path_budget(
        target_budget=_target_path_budget(cfg),
        ego=ego,
        planner=planner,
        spec=cfg.grid,
        clean_reference_plan=clean_reference_plan,
        apply_scaled=apply_scaled,
    )


def _apply_costmap_stage_fault(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    planner: CostMapPlanner,
    clean_reference_plan: CostMapPlan,
    objects: ObjectSet,
    target_actor: Any,
    frame_idx: int,
    active: bool,
) -> UnifiedFaultApplication:
    if cfg.method_key == "clean" or not active:
        return UnifiedFaultApplication(
            objects=objects,
            components=components,
            cost_grid=components.combined.copy(),
            metadata={},
        )
    if cfg.calibrate_budget and cfg.match_cross_representation_budget:
        return _calibrate_cross_representation_frame(
            choose="costmap",
            objects=objects,
            clean_components=components,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            planner=planner,
            planner_config=_planner_config_for_scenario(cfg.planner, scenario),
            clean_reference_plan=clean_reference_plan,
            target_actor=target_actor,
            frame_idx=frame_idx,
        )

    def apply_scaled(scale: float) -> UnifiedFaultApplication:
        out = _apply_scaled_costmap_fault(
            components,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            frame_idx=frame_idx,
            clean_reference_plan=clean_reference_plan,
            scale=scale,
        )
        return UnifiedFaultApplication(
            objects=objects,
            components=components,
            cost_grid=out.cost_grid,
            metadata=out.metadata,
        )

    if not cfg.calibrate_budget:
        out = apply_scaled(1.0)
        metadata = dict(out.metadata)
        metadata.update(
            {
                "calibrated_scale": 1.0,
                "target_realized_path_budget_m": _target_path_budget(cfg),
            }
        )
        return UnifiedFaultApplication(
            objects=out.objects,
            components=out.components,
            cost_grid=out.cost_grid,
            metadata=metadata,
        )

    return _calibrate_frame_path_budget(
        target_budget=_target_path_budget(cfg),
        ego=ego,
        planner=planner,
        spec=cfg.grid,
        clean_reference_plan=clean_reference_plan,
        apply_scaled=apply_scaled,
    )


def _calibrate_cross_representation_frame(
    *,
    choose: Literal["object", "costmap"],
    objects: ObjectSet,
    clean_components: CostMapComponents,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    planner: CostMapPlanner,
    planner_config: CostMapPlannerConfig,
    clean_reference_plan: CostMapPlan,
    target_actor: Any,
    frame_idx: int,
) -> UnifiedFaultApplication:
    target_budget = _target_path_budget(cfg)

    def object_application(scale: float) -> UnifiedFaultApplication:
        perturbed, metadata = _apply_scaled_object_fault(
            objects,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            target_actor=target_actor,
            frame_idx=frame_idx,
            scale=scale,
        )
        components = _build_cost_grid_components_for_config(
            perturbed,
            ego,
            cfg,
            planner_config,
        )
        prediction = _predict_objects_for_config(perturbed, cfg) if cfg.use_prediction else None
        return UnifiedFaultApplication(
            objects=perturbed,
            components=components,
            cost_grid=components.combined,
            metadata=metadata,
            prediction=prediction,
        )

    def costmap_application(scale: float) -> UnifiedFaultApplication:
        out = _apply_scaled_costmap_fault(
            clean_components,
            cfg=cfg,
            scenario=scenario,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            frame_idx=frame_idx,
            clean_reference_plan=clean_reference_plan,
            scale=scale,
        )
        return UnifiedFaultApplication(
            objects=objects,
            components=clean_components,
            cost_grid=out.cost_grid,
            metadata=out.metadata,
        )

    object_candidates = _budget_candidates(
        object_application,
        planner=planner,
        ego=ego,
        spec=cfg.grid,
        clean_reference_plan=clean_reference_plan,
    )
    costmap_candidates = _budget_candidates(
        costmap_application,
        planner=planner,
        ego=ego,
        spec=cfg.grid,
        clean_reference_plan=clean_reference_plan,
    )
    object_idx, costmap_idx = _select_matched_budget_pair(
        object_candidates,
        costmap_candidates,
        target_budget=target_budget,
    )
    chosen = object_candidates[object_idx] if choose == "object" else costmap_candidates[costmap_idx]
    peer = costmap_candidates[costmap_idx] if choose == "object" else object_candidates[object_idx]
    metadata = dict(chosen.application.metadata)
    metadata.update(
        {
            "calibrated_scale": float(chosen.scale),
            "target_realized_path_budget_m": float(target_budget),
            "calibration_abs_error_m": float(abs(chosen.budget - target_budget)),
            "matched_peer_realized_path_budget_m": float(peer.budget),
            "matched_pair_budget_gap_m": float(abs(chosen.budget - peer.budget)),
            "matched_pair_mean_budget_m": float(0.5 * (chosen.budget + peer.budget)),
        }
    )
    return UnifiedFaultApplication(
        chosen.application.objects,
        chosen.application.components,
        chosen.application.cost_grid,
        metadata,
        prediction=chosen.application.prediction,
    )


def _budget_candidates(
    apply_scaled: Any,
    *,
    planner: CostMapPlanner,
    ego: EgoState,
    spec: CostMapSpec,
    clean_reference_plan: CostMapPlan,
) -> list[_BudgetCandidate]:
    candidates: list[_BudgetCandidate] = []
    for scale in _CALIBRATION_SCALES:
        application = apply_scaled(float(scale))
        plan = planner.plan(ego, application.cost_grid, spec)
        budget = _path_l2_deviation(clean_reference_plan.path_xy, plan.path_xy)
        candidates.append(
            _BudgetCandidate(
                scale=float(scale),
                budget=float(budget),
                application=application,
            )
        )
    return candidates


def _select_matched_budget_pair(
    object_candidates: list[_BudgetCandidate],
    costmap_candidates: list[_BudgetCandidate],
    *,
    target_budget: float,
) -> tuple[int, int]:
    best: tuple[float, float, float, float, int, int] | None = None
    for object_idx, object_candidate in enumerate(object_candidates):
        for costmap_idx, costmap_candidate in enumerate(costmap_candidates):
            mean_budget = 0.5 * (object_candidate.budget + costmap_candidate.budget)
            budget_gap = abs(object_candidate.budget - costmap_candidate.budget)
            target_error = abs(mean_budget - target_budget)
            scale_tie = abs(object_candidate.scale) + abs(costmap_candidate.scale)
            candidate = (
                budget_gap,
                target_error,
                -mean_budget,
                scale_tie,
                object_idx,
                costmap_idx,
            )
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return int(best[4]), int(best[5])


def _apply_scaled_object_fault(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
    frame_idx: int,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    method = cfg.method_key
    if method == "torsion_displace":
        return _apply_object_displacement(
            objects, cfg=cfg, ego=ego, target_actor=target_actor, scale=scale
        )
    if method == "gaussian":
        return _apply_object_gaussian(
            objects,
            cfg=cfg,
            frame_idx=frame_idx,
            target_actor=target_actor,
            scale=scale,
        )
    if method == "random_warp":
        return _apply_object_random_warp(
            objects, cfg=cfg, target_actor=target_actor, scale=scale
        )
    if method == "torsion_swirl":
        return _apply_object_swirl(
            objects, cfg=cfg, ego=ego, target_actor=target_actor, scale=scale
        )
    if method == "inflation":
        return _apply_object_displacement(
            objects, cfg=cfg, ego=ego, target_actor=target_actor, scale=scale
        )
    raise ValueError(f"unknown object-stage method {cfg.method!r} in {scenario.scenario_id!r}")


def _apply_scaled_prediction_fault(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    target_actor: Any,
    frame_idx: int,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    method = cfg.method_key
    if method == "torsion_displace":
        return _apply_prediction_displacement(
            objects,
            cfg=cfg,
            ego=ego,
            target_actor=target_actor,
            scale=scale,
        )
    if method == "random_warp":
        return _apply_prediction_random_warp(
            objects,
            cfg=cfg,
            frame_idx=frame_idx,
            target_actor=target_actor,
            scale=scale,
        )
    if method == "gaussian":
        raise ValueError("gaussian prediction-stage faults are not contract-preserving")
    raise ValueError(
        f"unknown prediction-stage method {cfg.method!r} in {scenario.scenario_id!r}"
    )


@dataclass(frozen=True)
class _CostmapStageApplication:
    cost_grid: NDArray[np.float64]
    metadata: dict[str, float]


def _apply_scaled_costmap_fault(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    scenario: SyntheticScenario,
    ego: EgoState,
    objects: ObjectSet,
    target_actor: Any,
    frame_idx: int,
    clean_reference_plan: CostMapPlan,
    scale: float,
) -> _CostmapStageApplication:
    method = cfg.method_key
    if method == "torsion_displace":
        return _apply_costmap_translate(
            components, cfg=cfg, ego=ego, objects=objects, target_actor=target_actor, scale=scale
        )
    if method == "gaussian":
        return _apply_costmap_gaussian(components, cfg=cfg, frame_idx=frame_idx, scale=scale)
    if method == "random_warp":
        return _apply_costmap_random_warp(
            components,
            cfg=cfg,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            frame_idx=frame_idx,
            scale=scale,
        )
    if method == "torsion_swirl":
        return _apply_costmap_swirl(
            components,
            cfg=cfg,
            ego=ego,
            objects=objects,
            target_actor=target_actor,
            clean_reference_plan=clean_reference_plan,
            scale=scale,
        )
    if method == "inflation":
        return _apply_costmap_inflation(
            components, cfg=cfg, ego=ego, objects=objects, target_actor=target_actor, scale=scale
        )
    raise ValueError(f"unknown cost-map method {cfg.method!r} in {scenario.scenario_id!r}")


def _apply_object_displacement(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    target_actor: Any,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    magnitude = resolve_magnitude(cfg.magnitude_key)
    target_idx = _target_index(objects, target_actor)
    dx, dy = _away_from_ego_position_delta(
        objects,
        target_idx=target_idx,
        ego=ego,
        shift_m=scale * magnitude.position_shift_m,
    )
    theta = _velocity_rotation_away_from_ego(
        objects,
        target_idx=target_idx,
        ego_y=ego.y,
        theta_abs=scale * magnitude.velocity_rotation_rad,
        position_dy=dy,
        horizon_s=3.0,
    )
    d_yaw = (
        float(np.sign(theta) * scale * magnitude.yaw_shift_rad)
        if abs(theta) > 0.0
        else _away_from_ego_sign(float(objects.y[target_idx]), ego.y)
        * scale
        * magnitude.yaw_shift_rad
    )
    limits = _object_limits(cfg)
    out = position_torsion(
        objects,
        dx=dx,
        dy=dy,
        track_ids=[target_actor],
        max_shift_m=limits["position_shift_m"],
    )
    out = yaw_torsion(
        out,
        d_yaw=d_yaw,
        track_ids=[target_actor],
        max_yaw_delta_rad=limits["yaw_shift_rad"],
    )
    out = velocity_direction_torsion(
        out,
        theta=theta,
        track_ids=[target_actor],
        max_rotation_rad=limits["velocity_rotation_rad"],
    )
    return out, {
        "operator_strength": float(np.hypot(dx, dy)),
        "object_shift_x_m": float(dx),
        "object_shift_y_m": float(dy),
        "object_yaw_delta_rad": float(d_yaw),
        "object_velocity_rotation_delta_rad": float(theta),
    }


def _apply_object_gaussian(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    frame_idx: int,
    target_actor: Any,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    magnitude = resolve_magnitude(cfg.magnitude_key)
    rng = np.random.default_rng(_stable_seed(cfg, frame_idx, "object_gaussian"))
    target_idx = _target_index(objects, target_actor)
    dx, dy = rng.normal(
        0.0,
        magnitude.position_shift_m / _RAYLEIGH_MEAN_FACTOR,
        size=2,
    )
    d_yaw = float(
        rng.normal(0.0, magnitude.yaw_shift_rad / _FOLDED_NORMAL_MEAN_FACTOR)
    )
    theta = float(
        rng.normal(0.0, magnitude.velocity_rotation_rad / _FOLDED_NORMAL_MEAN_FACTOR)
    )
    limits = _object_limits(cfg)
    out = position_torsion(
        objects,
        dx=scale * float(dx),
        dy=scale * float(dy),
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
    return out, {
        "operator_strength": float(scale * np.hypot(dx, dy)),
        "object_shift_x_m": float(scale * dx),
        "object_shift_y_m": float(scale * dy),
        "object_yaw_delta_rad": float(scale * d_yaw),
        "object_velocity_rotation_delta_rad": float(scale * theta),
    }


def _apply_object_random_warp(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    target_actor: Any,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    magnitude = resolve_magnitude(cfg.magnitude_key)
    rng = np.random.default_rng(_stable_seed(cfg, None, "object_random_warp"))
    angle = float(rng.uniform(-np.pi, np.pi))
    dx = magnitude.position_shift_m * float(np.cos(angle))
    dy = magnitude.position_shift_m * float(np.sin(angle))
    d_yaw = magnitude.yaw_shift_rad * float(rng.choice(np.array([-1.0, 1.0])))
    theta = magnitude.velocity_rotation_rad * float(rng.choice(np.array([-1.0, 1.0])))
    limits = _object_limits(cfg)
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
    return out, {
        "operator_strength": float(scale * np.hypot(dx, dy)),
        "object_shift_x_m": float(scale * dx),
        "object_shift_y_m": float(scale * dy),
        "object_yaw_delta_rad": float(scale * d_yaw),
        "object_velocity_rotation_delta_rad": float(scale * theta),
    }


def _apply_prediction_displacement(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    target_actor: Any,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    magnitude = resolve_magnitude(cfg.magnitude_key)
    target_idx = _target_index(objects, target_actor)
    theta = _velocity_rotation_away_from_ego(
        objects,
        target_idx=target_idx,
        ego_y=ego.y,
        theta_abs=scale * magnitude.velocity_rotation_rad,
        position_dy=0.0,
        horizon_s=cfg.prediction_horizon_s,
    )
    d_yaw = (
        float(np.sign(theta) * scale * magnitude.yaw_shift_rad)
        if abs(theta) > 0.0
        else _away_from_ego_sign(float(objects.y[target_idx]), ego.y)
        * scale
        * magnitude.yaw_shift_rad
    )
    limits = _object_limits(cfg)
    out = yaw_torsion(
        objects,
        d_yaw=d_yaw,
        track_ids=[target_actor],
        max_yaw_delta_rad=limits["yaw_shift_rad"],
    )
    out = velocity_direction_torsion(
        out,
        theta=theta,
        track_ids=[target_actor],
        max_rotation_rad=limits["velocity_rotation_rad"],
    )
    return out, {
        "operator_strength": float(abs(theta)),
        "object_shift_x_m": 0.0,
        "object_shift_y_m": 0.0,
        "prediction_yaw_delta_rad": float(d_yaw),
        "prediction_velocity_rotation_delta_rad": float(theta),
    }


def _apply_prediction_random_warp(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    frame_idx: int,
    target_actor: Any,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    magnitude = resolve_magnitude(cfg.magnitude_key)
    rng = np.random.default_rng(_stable_seed(cfg, frame_idx, "prediction_random_warp"))
    d_yaw = magnitude.yaw_shift_rad * float(rng.choice(np.array([-1.0, 1.0])))
    theta = magnitude.velocity_rotation_rad * float(rng.choice(np.array([-1.0, 1.0])))
    limits = _object_limits(cfg)
    out = yaw_torsion(
        objects,
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
    return out, {
        "operator_strength": float(abs(scale * theta)),
        "object_shift_x_m": 0.0,
        "object_shift_y_m": 0.0,
        "prediction_yaw_delta_rad": float(scale * d_yaw),
        "prediction_velocity_rotation_delta_rad": float(scale * theta),
    }


def _apply_object_swirl(
    objects: ObjectSet,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    target_actor: Any,
    scale: float,
) -> tuple[ObjectSet, dict[str, float]]:
    sigma = _swirl_sigma(cfg)
    magnitude = resolve_magnitude(cfg.magnitude_key)
    alpha_abs = scale * magnitude.velocity_rotation_rad / _TWIST_PEAK_FACTOR
    target_idx = _target_index(objects, target_actor)
    pivot, sign = _directed_swirl_pivot_and_sign(
        objects,
        target_idx=target_idx,
        ego=ego,
        sigma=sigma,
    )
    alpha = sign * abs(alpha_abs)
    out = scene_swirl_torsion(
        objects,
        pivot=pivot,
        alpha=alpha,
        sigma=sigma,
        track_ids=[target_actor],
    )
    return out, {
        "operator_strength": float(abs(alpha)),
        "swirl_alpha_rad": float(alpha),
        "swirl_sigma_m": float(sigma),
    }


def _apply_costmap_translate(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    objects: ObjectSet,
    target_actor: Any,
    scale: float,
) -> _CostmapStageApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    shift_m = scale * 1.0
    sign = _away_from_route_lateral_sign(ego_y=ego.y, target_local_y=target_xy[1])
    shift_y_m = sign * shift_m
    shift_grid = np.array([0.0, -shift_y_m / cfg.grid.resolution_m], dtype=np.float64)
    obstacle = translate_cost_field(components.obstacle_grid, shift_grid)
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(abs(shift_y_m)),
            "translate_shift_y_m": float(shift_y_m),
        },
    )


def _apply_costmap_gaussian(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    frame_idx: int,
    scale: float,
) -> _CostmapStageApplication:
    noise = _smooth_gaussian_noise(
        components.obstacle_grid.shape,
        seed=_stable_seed(cfg, frame_idx, "costmap_gaussian"),
    )
    cost_scale = scale * 0.10
    obstacle = gaussian_cost_noise(components.obstacle_grid, noise, cost_scale)
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(abs(cost_scale)),
            "gaussian_cost_scale": float(cost_scale),
        },
    )


def _apply_costmap_random_warp(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    objects: ObjectSet,
    target_actor: Any,
    frame_idx: int,
    scale: float,
) -> _CostmapStageApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    target_grid = cfg.grid.metric_to_grid(np.asarray(target_xy, dtype=np.float64))
    sigma_cells = _swirl_sigma_cells(cfg)
    rng = np.random.default_rng(_stable_seed(cfg, frame_idx, "costmap_random_warp"))
    angle = float(rng.uniform(-np.pi, np.pi))
    radial = sigma_cells * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    pivot_grid = target_grid - radial
    sign = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float64)))
    alpha = sign * scale * 0.35
    obstacle = spatial_cost_warp(
        components.obstacle_grid,
        tuple(pivot_grid),
        alpha=alpha,
        sigma=sigma_cells,
    )
    pivot_metric = cfg.grid.grid_to_metric(pivot_grid)
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


def _apply_costmap_swirl(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    objects: ObjectSet,
    target_actor: Any,
    clean_reference_plan: CostMapPlan,
    scale: float,
) -> _CostmapStageApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    sigma_cells = _swirl_sigma_cells(cfg)
    alpha_abs = scale * 0.35
    target_grid = cfg.grid.metric_to_grid(np.asarray(target_xy, dtype=np.float64))
    pivot_grid = np.array([target_grid[0] - sigma_cells, target_grid[1]], dtype=np.float64)
    alpha = _directed_swirl_alpha(
        target_grid=target_grid,
        pivot_grid=pivot_grid,
        alpha_abs=alpha_abs,
        sigma_cells=sigma_cells,
        cfg=cfg,
        ego=ego,
        preferred_lateral_m=clean_reference_plan.target_lateral_m,
    )
    obstacle = spatial_cost_warp(
        components.obstacle_grid,
        tuple(pivot_grid),
        alpha=alpha,
        sigma=sigma_cells,
    )
    pivot_metric = cfg.grid.grid_to_metric(pivot_grid)
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


def _apply_costmap_inflation(
    components: CostMapComponents,
    *,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    objects: ObjectSet,
    target_actor: Any,
    scale: float,
) -> _CostmapStageApplication:
    target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
    sign = _away_from_route_lateral_sign(ego_y=ego.y, target_local_y=target_xy[1])
    false_center = np.asarray([target_xy[0], target_xy[1] - sign * 1.6], dtype=np.float64)
    center_grid = cfg.grid.metric_to_grid(false_center)
    cov = np.diag([(3.0 / cfg.grid.resolution_m) ** 2, (1.0 / cfg.grid.resolution_m) ** 2])
    obstacle = directional_obstacle_inflation(
        components.obstacle_grid,
        center=(float(center_grid[0]), float(center_grid[1])),
        beta=min(0.8, 0.08 * scale),
        cov=cov,
    )
    return _recombine_fault(
        components,
        obstacle,
        metadata={
            "operator_strength": float(min(0.8, 0.08 * scale)),
            "inflation_beta": float(min(0.8, 0.08 * scale)),
        },
    )


def _calibrate_frame_path_budget(
    *,
    target_budget: float,
    ego: EgoState,
    planner: CostMapPlanner,
    spec: CostMapSpec,
    clean_reference_plan: CostMapPlan,
    apply_scaled: Any,
) -> UnifiedFaultApplication:
    if target_budget <= 0.0:
        out = apply_scaled(0.0)
        metadata = dict(out.metadata)
        metadata.update(
            {
                "calibrated_scale": 0.0,
                "target_realized_path_budget_m": 0.0,
                "calibration_abs_error_m": 0.0,
            }
        )
        return UnifiedFaultApplication(
            out.objects,
            out.components,
            out.cost_grid,
            metadata,
            prediction=out.prediction,
        )

    best: tuple[float, float, float, float, UnifiedFaultApplication] | None = None
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
    return UnifiedFaultApplication(
        best_out.objects,
        best_out.components,
        best_out.cost_grid,
        metadata,
        prediction=best_out.prediction,
    )


def _recombine_fault(
    components: CostMapComponents,
    obstacle_grid: NDArray[np.float64],
    *,
    metadata: dict[str, float],
) -> _CostmapStageApplication:
    obstacle = np.clip(np.asarray(obstacle_grid, dtype=np.float64), COST_MIN, COST_MAX)
    if obstacle.shape != components.obstacle_grid.shape:
        raise ValueError("obstacle_grid shape changed during cost-map fault")
    clean = components.combined
    out = np.maximum(components.road_grid, obstacle)
    out = np.clip(out, COST_MIN, COST_MAX)
    out[components.boundary_mask] = clean[components.boundary_mask]
    return _CostmapStageApplication(cost_grid=out, metadata=dict(metadata))


def _target_path_budget(cfg: UnifiedPipelineConfig) -> float:
    if cfg.method_key == "clean" or cfg.injection_key == "none":
        return 0.0
    if cfg.target_path_budget_m is not None:
        return float(cfg.target_path_budget_m)
    return float(MAGNITUDE_PATH_BUDGETS_M[cfg.magnitude_key])


def _summarize_trace(
    cfg: UnifiedPipelineConfig,
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
        "lane_departure_rate": float(
            np.mean([float(row["lane_departure"]) for row in trace])
        ),
        "mean_path_curvature": float(np.mean([float(row["path_curvature"]) for row in trace])),
        "max_path_curvature": float(max(float(row["path_max_curvature"]) for row in trace)),
        "route_completion": route_completion,
        "fault_active_frames": int(sum(bool(row["fault_active"]) for row in trace)),
        "fault_start_time_s": float(cfg.start_frame * scenario.dt),
        "fault_end_time_s": float(_fault_end_time(cfg, scenario)),
        "target_actor": target_actor,
        "target_realized_path_budget_m": float(_mean_trace_value(active_rows, "target_realized_path_budget_m")),
        "mean_realized_budget": float(_mean_trace_value(active_rows, "realized_path_deviation_m")),
        "mean_realized_path_deviation_m": float(
            _mean_trace_value(active_rows, "realized_path_deviation_m")
        ),
        "max_realized_path_deviation_m": float(
            max((float(row["realized_path_deviation_m"]) for row in active_rows), default=0.0)
        ),
        "mean_calibration_abs_error_m": float(
            _mean_trace_value(active_rows, "calibration_abs_error_m")
        ),
        "mean_calibrated_scale": float(_mean_trace_value(active_rows, "calibrated_scale")),
        "mean_operator_strength": float(_mean_trace_value(active_rows, "operator_strength")),
        "mean_object_position_shift_m": float(
            _mean_trace_value(active_rows, "object_position_shift_m")
        ),
        "mean_object_yaw_shift_rad": float(
            _mean_trace_value(active_rows, "object_yaw_shift_rad")
        ),
        "mean_object_velocity_rotation_rad": float(
            _mean_trace_value(active_rows, "object_velocity_rotation_rad")
        ),
        "mean_cost_grid_l2_delta": float(_mean_trace_value(active_rows, "cost_grid_l2_delta")),
        "mean_cost_grid_mean_abs_delta": float(
            _mean_trace_value(active_rows, "cost_grid_mean_abs_delta")
        ),
        "mean_translate_shift_y_m": float(_mean_trace_value(active_rows, "translate_shift_y_m")),
        "mean_abs_translate_shift_y_m": float(
            _mean_abs_trace_value(active_rows, "translate_shift_y_m")
        ),
        "mean_swirl_alpha_rad": float(_mean_trace_value(active_rows, "swirl_alpha_rad")),
        "mean_abs_swirl_alpha_rad": float(
            _mean_abs_trace_value(active_rows, "swirl_alpha_rad")
        ),
        "mean_gaussian_cost_scale": float(_mean_trace_value(active_rows, "gaussian_cost_scale")),
        "final_ego_x": float(final_ego.x),
        "final_ego_y": float(final_ego.y),
        "final_ego_speed": float(final_ego.speed),
    }


def _sweep_jobs(method: str, seeds: int) -> list[tuple[str, str, str, str, int]]:
    jobs: list[tuple[str, str, str, str, int]] = []
    for scenario in SCENARIOS:
        for magnitude in MAGNITUDES:
            for seed in range(seeds):
                jobs.append((scenario, "none", "clean", magnitude, seed))
                for injection_point in INJECTION_POINTS:
                    jobs.append((scenario, injection_point, method, magnitude, seed))
    return jobs


def _run_jobs(
    jobs: list[tuple[str, str, str, str, int]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        return [_run_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs))


def _run_one(job: tuple[str, str, str, str, int]) -> dict[str, Any]:
    scenario, injection_point, method, magnitude, seed = job
    result = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario=scenario,
            injection_point=injection_point,  # type: ignore[arg-type]
            method=method,  # type: ignore[arg-type]
            magnitude=magnitude,
            seed=seed,
            trace_grids=False,
        )
    )
    record = result.to_record()
    record["min_ttc_censored"] = _finite_metric(record["min_ttc"], cap=TTC_CENSOR_S)
    return record


def _select_representative_pair(records: list[dict[str, Any]]) -> tuple[str, str, int]:
    best: tuple[float, str, str, int] | None = None
    clean_by_key = {
        (row["scenario_id"], row["magnitude"], int(row["seed"])): row
        for row in records
        if row["injection_point"] == "none"
    }
    for scenario in SCENARIOS:
        for magnitude in MAGNITUDES:
            for seed in sorted({int(row["seed"]) for row in records}):
                key = (scenario, magnitude, seed)
                clean = clean_by_key.get(key)
                object_row = _find_record(records, scenario, "object", magnitude, seed)
                costmap_row = _find_record(records, scenario, "costmap", magnitude, seed)
                if clean is None or object_row is None or costmap_row is None:
                    continue
                clean_ttc = float(clean["min_ttc_censored"])
                score = (
                    max(0.0, clean_ttc - float(object_row["min_ttc_censored"]))
                    + max(0.0, clean_ttc - float(costmap_row["min_ttc_censored"]))
                    + float(object_row["mean_realized_budget"])
                    + float(costmap_row["mean_realized_budget"])
                )
                candidate = (score, scenario, magnitude, seed)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return ("cut_in", "high", 0)
    return best[1], best[2], best[3]


def _find_record(
    records: list[dict[str, Any]],
    scenario: str,
    injection_point: str,
    magnitude: str,
    seed: int,
) -> dict[str, Any] | None:
    for row in records:
        if (
            row["scenario_id"] == scenario
            and row["injection_point"] == injection_point
            and row["magnitude"] == magnitude
            and int(row["seed"]) == seed
        ):
            return row
    return None


def _representative_frame(result: UnifiedRunResult) -> dict[str, Any]:
    active = [row for row in result.trace if row["fault_active"]]
    rows = active if active else list(result.trace)
    return max(rows, key=lambda row: float(row["realized_path_deviation_m"]))


def _plot_object_stage(ax: Any, frame: dict[str, Any], label: str) -> None:
    ego = frame["ego"]
    clean = frame["clean_object_set"]
    stage = frame["stage_a_object_set"]
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axhline(1.75, color="#dddddd", linewidth=0.8)
    ax.axhline(-1.75, color="#dddddd", linewidth=0.8)
    for clean_obj, stage_obj in zip(clean, stage, strict=True):
        x0 = float(clean_obj["x"]) - float(ego["x"])
        y0 = float(clean_obj["y"]) - float(ego["y"])
        x1 = float(stage_obj["x"]) - float(ego["x"])
        y1 = float(stage_obj["y"]) - float(ego["y"])
        ax.scatter([x0], [y0], color="#333333", s=26, label="clean object")
        ax.scatter([x1], [y1], color="#1f77b4", s=26, label="Stage A")
        ax.arrow(
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            color="#1f77b4",
            width=0.015,
            head_width=0.25,
            length_includes_head=True,
        )
    ax.scatter([0.0], [0.0], color="#000000", marker="s", s=28)
    ax.set_xlim(-4, 42)
    ax.set_ylim(-6, 6)
    ax.set_title(f"{label}\nStage A object-set")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.grid(True, alpha=0.2)


def _plot_costmap_delta(ax: Any, frame: dict[str, Any]) -> None:
    clean = np.asarray(frame["clean_cost_grid"], dtype=np.float64)
    final = np.asarray(frame["final_cost_grid"], dtype=np.float64)
    spec = frame["cost_grid_spec"]
    delta = final - clean
    vmax = max(0.15, float(np.max(np.abs(delta))))
    extent = [
        float(spec["x_min_m"]),
        float(spec["x_max_m"]),
        float(spec["y_min_m"]),
        float(spec["y_max_m"]),
    ]
    ax.imshow(
        delta,
        origin="upper",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_title("Stage B cost-map delta")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")


def _plot_path_shift(ax: Any, frame: dict[str, Any]) -> None:
    clean_path = np.asarray(frame["clean_reference_path"]["path_xy"], dtype=np.float64)
    chosen_path = np.asarray(frame["chosen_path"]["path_xy"], dtype=np.float64)
    ax.plot(clean_path[:, 0], clean_path[:, 1], color="#333333", label="clean plan")
    ax.plot(chosen_path[:, 0], chosen_path[:, 1], color="#d62728", label="faulted plan")
    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.set_xlim(0, 36)
    ax.set_ylim(-4.2, 4.2)
    ax.set_title(
        f"Plan shift\nL2={float(frame['realized_path_deviation_m']):.2f} m"
    )
    ax.set_xlabel("path x (m)")
    ax.set_ylabel("path y (m)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def _plot_ttc_control(ax: Any, clean: UnifiedRunResult, result: UnifiedRunResult) -> None:
    time = np.array([float(row["time_s"]) for row in result.trace], dtype=np.float64)
    clean_ttc = np.array(
        [_finite_metric(row["actual_ttc_s"], cap=TTC_CENSOR_S) for row in clean.trace],
        dtype=np.float64,
    )
    fault_ttc = np.array(
        [_finite_metric(row["actual_ttc_s"], cap=TTC_CENSOR_S) for row in result.trace],
        dtype=np.float64,
    )
    accel = np.array([float(row["control"]["accel_mps2"]) for row in result.trace])
    ax.plot(time[: clean_ttc.size], clean_ttc[: time.size], color="#333333", label="clean TTC")
    ax.plot(time, fault_ttc, color="#d62728", label="fault TTC")
    ax.set_ylim(0.0, TTC_CENSOR_S + 0.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("min-TTC (s)")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(time, accel, color="#1f77b4", alpha=0.75, linestyle="--", label="accel")
    ax2.set_ylabel("accel (m/s^2)")
    ax2.set_ylim(-8.5, 1.5)
    ax.set_title("Control and safety")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right", frameon=False, fontsize=8)


def _find_summary_row(rows: list[dict[str, Any]], injection_point: str, magnitude: str) -> dict[str, Any]:
    for row in rows:
        if row["injection_point"] == injection_point and row["magnitude"] == magnitude:
            return row
    raise ValueError(f"missing summary row for {injection_point}/{magnitude}")


def _print_sensitivity_table(summary: list[dict[str, Any]]) -> None:
    for scenario in (*SCENARIOS, "ALL"):
        rows = [
            row
            for row in summary
            if row["scenario_id"] == scenario and row["magnitude"] == "high"
        ]
        print(f"\nScenario: {scenario}, magnitude: high")
        print(
            "injection      collision_rate [95% CI]    mean_min_ttc [95% CI]     "
            "paired_drop [95% CI]       std_ttc  budget  degradation"
        )
        for row in rows:
            print(
                f"{row['injection_point']:<14}"
                f"{_format_ci(row, 'collision_rate'):<28}"
                f"{_format_ci(row, 'mean_min_ttc'):<28}"
                f"{_format_paired_ttc_delta(row):<27}"
                f"{float(row['std_min_ttc']):<9.3f}"
                f"{float(row['mean_realized_budget']):<8.3f}"
                f"{float(row['degradation_score']):.3f}"
            )


def _format_ci(row: dict[str, Any], metric: str) -> str:
    return (
        f"{float(row[metric]):.3f} "
        f"[{float(row[f'{metric}_ci_low']):.3f}, "
        f"{float(row[f'{metric}_ci_high']):.3f}]"
    )


def _format_paired_ttc_delta(row: dict[str, Any]) -> str:
    return (
        f"{float(row['paired_mean_min_ttc_delta_s']):.3f} "
        f"[{float(row['paired_mean_min_ttc_delta_ci_low']):.3f}, "
        f"{float(row['paired_mean_min_ttc_delta_ci_high']):.3f}]"
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    names = list(fieldnames if fieldnames is not None else rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_record(row) for row in rows)


def _csv_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = ""
        else:
            out[key] = value
    return out


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate(SCENARIOS)}
    injection_order = {"none": 0, "object": 1, "costmap": 2}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        records,
        key=lambda row: (
            scenario_order[row["scenario_id"]],
            magnitude_order[row["magnitude"]],
            injection_order[row["injection_point"]],
            int(row["seed"]),
        ),
    )


def _sort_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate((*SCENARIOS, "ALL"))}
    injection_order = {name: idx for idx, name in enumerate(INJECTION_POINTS)}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order[row["scenario_id"]],
            injection_order[row["injection_point"]],
            magnitude_order[row["magnitude"]],
        ),
    )


def _normalize_magnitude(value: str) -> str:
    key = str(value).lower().strip().replace("-", "_")
    if key == "med":
        key = "medium"
    if key not in MAGNITUDE_PATH_BUDGETS_M:
        valid = ", ".join(("low", "med", "medium", "high"))
        raise ValueError(f"unknown magnitude {value!r}; expected one of {valid}")
    return key


def _planner_config_for_scenario(
    config: CostMapPlannerConfig,
    scenario: SyntheticScenario,
) -> CostMapPlannerConfig:
    if config.target_speed_mps is not None:
        return config
    return replace(config, target_speed_mps=scenario.ego_initial.speed)


def _fault_active(cfg: UnifiedPipelineConfig, frame_idx: int) -> bool:
    if cfg.method_key == "clean" or cfg.injection_key == "none":
        return False
    return is_active(
        cfg.temporal_pattern,
        frame_idx,
        start_frame=cfg.start_frame,
        duration=cfg.duration_frames,
    )


def _fault_end_time(cfg: UnifiedPipelineConfig, scenario: SyntheticScenario) -> float:
    pattern = cfg.temporal_pattern.lower().strip().replace("_", "-")
    if cfg.method_key == "clean" or cfg.injection_key == "none":
        return float(cfg.start_frame * scenario.dt)
    if pattern == "single-frame":
        return float((cfg.start_frame + 1) * scenario.dt)
    if pattern == "burst":
        duration = cfg.duration_frames if cfg.duration_frames is not None else 1
        return float((cfg.start_frame + duration) * scenario.dt)
    if pattern == "persistent":
        return float((scenario.steps - 1) * scenario.dt)
    return float(cfg.start_frame * scenario.dt)


def _path_l2_deviation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.shape != b.shape:
        raise ValueError("paths must have the same shape")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _target_index(objects: ObjectSet, target_actor: Any) -> int:
    matches = np.flatnonzero(objects.track_id == target_actor)
    if matches.size != 1:
        raise ValueError(f"target actor {target_actor!r} not found exactly once")
    return int(matches[0])


def _target_local_xy(objects: ObjectSet, *, target_actor: Any, ego: EgoState) -> tuple[float, float]:
    idx = _target_index(objects, target_actor)
    return float(objects.x[idx] - ego.x), float(objects.y[idx] - ego.y)


def _away_from_ego_sign(actor_y: float, ego_y: float) -> float:
    delta = actor_y - ego_y
    if abs(delta) < 1e-9:
        return 1.0
    return float(np.sign(delta))


def _away_from_route_lateral_sign(*, ego_y: float, target_local_y: float) -> float:
    world_y = float(ego_y + target_local_y)
    if abs(world_y) <= 1e-9:
        return 1.0
    return float(np.sign(world_y))


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
    delta = float(shift_m) * direction
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


def _directed_swirl_alpha(
    *,
    target_grid: NDArray[np.float64],
    pivot_grid: NDArray[np.float64],
    alpha_abs: float,
    sigma_cells: float,
    cfg: UnifiedPipelineConfig,
    ego: EgoState,
    preferred_lateral_m: float,
) -> float:
    candidates = (abs(alpha_abs), -abs(alpha_abs))
    best_alpha = candidates[0]
    best_score = -float("inf")
    for alpha in candidates:
        from torsion.operators.twist import twist_points

        moved_grid = twist_points(target_grid, pivot_grid, alpha=alpha, sigma=sigma_cells)
        moved_metric = cfg.grid.grid_to_metric(moved_grid)
        world_y = ego.y + float(moved_metric[1])
        if abs(preferred_lateral_m) > 1e-9:
            score = -abs(float(moved_metric[1]) - float(preferred_lateral_m))
        else:
            score = abs(world_y)
        if score > best_score + 1e-12:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def _object_limits(cfg: UnifiedPipelineConfig) -> dict[str, float]:
    magnitude = resolve_magnitude(cfg.magnitude_key)
    max_scale = float(np.max(_CALIBRATION_SCALES))
    return {
        "position_shift_m": float(max_scale * max(magnitude.position_shift_m, 1e-9) + 1e-9),
        "yaw_shift_rad": float(max_scale * max(magnitude.yaw_shift_rad, 1e-9) + 1e-9),
        "velocity_rotation_rad": float(
            max_scale * max(magnitude.velocity_rotation_rad, 1e-9) + 1e-9
        ),
    }


def _swirl_sigma(cfg: UnifiedPipelineConfig) -> float:
    sigma = float(cfg.swirl_sigma_m)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("swirl_sigma_m must be positive and finite")
    return sigma


def _swirl_sigma_cells(cfg: UnifiedPipelineConfig) -> float:
    return _swirl_sigma(cfg) / cfg.grid.resolution_m


def _rotate2d(vector: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _smooth_gaussian_noise(shape: tuple[int, int], *, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
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


def _stable_seed(cfg: UnifiedPipelineConfig, frame_idx: int | None, salt: str) -> int:
    text = "|".join(
        (
            "torsion-unified",
            cfg.scenario,
            cfg.injection_key,
            cfg.method_key,
            cfg.magnitude_key,
            str(cfg.seed),
            "" if frame_idx is None else str(frame_idx),
            salt,
        )
    )
    value = 0x85EBCA6B
    for byte in text.encode("utf-8"):
        value = ((value * 1_000_003) ^ byte) & 0xFFFFFFFF
    return int(value)


def _object_delta_metrics(
    clean: ObjectSet,
    perturbed: ObjectSet,
    *,
    target_actor: Any,
) -> dict[str, float]:
    clean_idx = _target_index(clean, target_actor)
    perturbed_idx = _target_index(perturbed, target_actor)
    shift = float(
        np.linalg.norm(
            [
                perturbed.x[perturbed_idx] - clean.x[clean_idx],
                perturbed.y[perturbed_idx] - clean.y[clean_idx],
            ]
        )
    )
    yaw_delta = abs(_wrap_angle_scalar(float(perturbed.yaw[perturbed_idx] - clean.yaw[clean_idx])))
    velocity_delta = abs(_velocity_angle_delta(clean.v[clean_idx], perturbed.v[perturbed_idx]))
    return {
        "object_position_shift_m": shift,
        "object_yaw_shift_rad": float(yaw_delta),
        "object_velocity_rotation_rad": float(velocity_delta),
    }


def _prediction_delta_metrics(
    clean: PredictionSet | None,
    perturbed: PredictionSet | None,
    *,
    target_actor: Any,
) -> dict[str, float]:
    if clean is None or perturbed is None:
        return {"prediction_traj_l2_delta": 0.0}
    clean_trajectory = clean.by_track_id(target_actor)
    perturbed_trajectory = perturbed.by_track_id(target_actor)
    if clean_trajectory is None or perturbed_trajectory is None:
        raise ValueError(f"target actor {target_actor!r} not found in predictions")
    if clean_trajectory.xy.shape != perturbed_trajectory.xy.shape:
        raise ValueError("prediction trajectories must have the same shape")
    delta = perturbed_trajectory.xy - clean_trajectory.xy
    return {
        "prediction_traj_l2_delta": float(np.mean(np.linalg.norm(delta, axis=1))),
    }


def _cost_grid_delta_metrics(
    clean: NDArray[np.float64],
    perturbed: NDArray[np.float64],
) -> dict[str, float]:
    delta = np.asarray(perturbed, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    return {
        "cost_grid_l2_delta": float(np.sqrt(np.mean(delta * delta))),
        "cost_grid_mean_abs_delta": float(np.mean(np.abs(delta))),
        "cost_grid_max_abs_delta": float(np.max(np.abs(delta))),
    }


def _velocity_angle_delta(clean: NDArray[np.float64], perturbed: NDArray[np.float64]) -> float:
    clean_norm = float(np.linalg.norm(clean))
    perturbed_norm = float(np.linalg.norm(perturbed))
    if clean_norm <= 1e-12 or perturbed_norm <= 1e-12:
        return 0.0
    clean_angle = float(np.arctan2(clean[1], clean[0]))
    perturbed_angle = float(np.arctan2(perturbed[1], perturbed[0]))
    return _wrap_angle_scalar(perturbed_angle - clean_angle)


def _wrap_angle_scalar(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _plan_to_control_record(
    plan: CostMapPlan,
    planner_config: CostMapPlannerConfig,
) -> dict[str, Any]:
    brake = float(
        np.clip(
            -float(plan.accel_mps2) / max(abs(planner_config.hard_brake_accel_mps2), 1e-9),
            0.0,
            1.0,
        )
    )
    return {
        "accel_mps2": float(plan.accel_mps2),
        "brake": brake,
        "target_speed_mps": float(plan.target_speed_mps),
        "target_lateral_m": float(plan.target_lateral_m),
        "mean_cost": float(plan.mean_cost),
        "max_cost": float(plan.max_cost),
        "reason": plan.reason,
    }


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


def _grid_to_list(grid: NDArray[np.float64]) -> list[list[float]]:
    return [[float(value) for value in row] for row in grid]


def _finite_metric(value: Any, *, cap: float) -> float:
    out = float(value)
    if math.isnan(out):
        return cap
    if math.isinf(out):
        return cap if out > 0.0 else 0.0
    return float(np.clip(out, 0.0, cap))


def _scenario_label(name: str) -> str:
    return name.replace("_", " ").title()


if __name__ == "__main__":
    main()
