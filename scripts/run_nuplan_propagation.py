"""Run TORSION open-loop propagation on real nuPlan frames.

The runner keeps the synthetic propagation machinery fixed and swaps in real
nuPlan inputs: ego-local boxes from ``nuplan_adapter`` and a map-derived road
prior from ``nuplan_map``.  Real cost maps are combined with the same rule as
``CostMapComponents.combined``: ``clip(max(road_grid, obstacle_grid), 0, 1)``.
Cost-map faults only perturb the obstacle component and pin map road-boundary
cells back to the clean map during recombination.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torsion.data.nuplan_adapter import (  # noqa: E402
    NuPlanFrame,
    NuPlanLog,
    find_frames_with_tag,
    list_logs,
    load_log,
    select_target_actor,
)
from torsion.data.nuplan_map import (  # noqa: E402
    build_map_road_cost_grid,
    epsg_for_location,
    load_map,
    map_path_for_log,
    road_boundary_mask_from_map,
)
from torsion.analysis.mechanism import decision_margin  # noqa: E402
from torsion.metrics.statistics import bootstrap_ci  # noqa: E402
from torsion.operators.costmap import COST_MAX, COST_MIN, spatial_cost_warp, translate_cost_field  # noqa: E402
from torsion.operators.object import (  # noqa: E402
    ObjectSet,
    position_torsion,
    velocity_direction_torsion,
    yaw_torsion,
)
from torsion.scenarios.costmap_runner import (  # noqa: E402
    CostMapComponents,
    CostMapPlan,
    CostMapPlannerConfig,
    CostMapSpec,
    build_cost_grid_components,
    build_planner,
    build_predicted_cost_grid_components,
)
from torsion.scenarios.planner import EgoState  # noqa: E402

DEFAULT_LOGS_ROOT = Path("Dataset/nuplan-v1.0_val/data/cache/public_set_val")
DEFAULT_MAPS_ROOT = Path("Dataset/nuplan-maps-v1.0")
DEFAULT_OUT_DIR = Path("results/metrics")

CATEGORIES: tuple[str, ...] = ("FOLLOWING", "INTERSECTION", "LANE_CHANGE")
CATEGORY_TAGS: dict[str, tuple[str, ...]] = {
    "FOLLOWING": (
        "high_magnitude_speed",
        "medium_magnitude_speed",
        "high_speed",
        "medium_speed",
    ),
    "INTERSECTION": ("traversing_intersection", "on_intersection"),
    "LANE_CHANGE": (
        "changing_lane",
        "changing_lane_left",
        "changing_lane_right",
        "changing_lane_to_left",
        "changing_lane_to_right",
    ),
}
DEFAULT_INJECTIONS: tuple[str, ...] = ("object", "costmap", "prediction")
DEFAULT_METHODS: tuple[str, ...] = ("torsion_displace", "random_warp")
DEFAULT_PLANNERS: tuple[str, ...] = ("sampling", "potential_field")
DEFAULT_MAGNITUDES_M: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_MAX_LOGS = 12
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_PREDICTION_HORIZON_S = 2.0
DEFAULT_PREDICTION_DT_S = 0.1
DEFAULT_PREDICTION_SAMPLES = 5
DEFAULT_SWIRL_SIGMA_M = 7.0
TWIST_PEAK_FACTOR = float(np.exp(-0.5))
EPS_RATIO = 1.0e-12

RUN_COLUMNS: tuple[str, ...] = (
    "status",
    "error",
    "log_path",
    "log_name",
    "frame_index",
    "frame_token",
    "time_s",
    "category",
    "scenario_tags",
    "target_actor",
    "injection",
    "method",
    "planner",
    "magnitude_m",
    "ego_speed_mps",
    "object_shift_m",
    "cost_l2",
    "plan_dev",
    "object__cost_gain",
    "cost__plan_gain",
    "argmin_flip",
    "planner_gateway",
    "decision_margin_score",
    "n_candidates",
    "n_feasible",
    "decision_margin_pool",
    "safety1_plan_dev",
    "clean_min_dist_m",
    "fault_min_dist_m",
    "safety2_mindist_drop",
    "clean_cv_ttc_s",
    "fault_cv_ttc_s",
    "safety2_ttc_drop",
    "operator_strength",
    "prediction_l2",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "row_type",
    "category",
    "injection",
    "method",
    "planner",
    "n_runs",
    "n_valid",
    "n_frames",
    "object__cost_gain_n",
    "object__cost_gain_mean",
    "object__cost_gain_ci_low",
    "object__cost_gain_ci_high",
    "cost__plan_gain_n",
    "cost__plan_gain_mean",
    "cost__plan_gain_ci_low",
    "cost__plan_gain_ci_high",
    "gateway_rate",
    "argmin_flip_rate",
    "decision_margin_score_n",
    "decision_margin_score_mean",
    "decision_margin_score_ci_low",
    "decision_margin_score_ci_high",
    "n_feasible_n",
    "n_feasible_mean",
    "n_feasible_ci_low",
    "n_feasible_ci_high",
    "safety1_plan_dev_n",
    "safety1_plan_dev_mean",
    "safety1_plan_dev_ci_low",
    "safety1_plan_dev_ci_high",
    "safety2_mindist_drop_n",
    "safety2_mindist_drop_mean",
    "safety2_mindist_drop_ci_low",
    "safety2_mindist_drop_ci_high",
    "safety2_ttc_drop_n",
    "safety2_ttc_drop_mean",
    "safety2_ttc_drop_ci_low",
    "safety2_ttc_drop_ci_high",
    "costmap_minus_object_plan_dev_n",
    "costmap_minus_object_plan_dev_mean",
    "costmap_minus_object_plan_dev_ci_low",
    "costmap_minus_object_plan_dev_ci_high",
    "costmap_minus_object_mindist_drop_n",
    "costmap_minus_object_mindist_drop_mean",
    "costmap_minus_object_mindist_drop_ci_low",
    "costmap_minus_object_mindist_drop_ci_high",
    "costmap_minus_object_ttc_drop_n",
    "costmap_minus_object_ttc_drop_mean",
    "costmap_minus_object_ttc_drop_ci_low",
    "costmap_minus_object_ttc_drop_ci_high",
)


@dataclass(frozen=True)
class NuPlanPropagationConfig:
    logs_root: Path = DEFAULT_LOGS_ROOT
    maps_root: Path = DEFAULT_MAPS_ROOT
    n_frames: int = 150
    categories: tuple[str, ...] = CATEGORIES
    injections: tuple[str, ...] = DEFAULT_INJECTIONS
    methods: tuple[str, ...] = DEFAULT_METHODS
    planners: tuple[str, ...] = DEFAULT_PLANNERS
    magnitudes_m: tuple[float, ...] = DEFAULT_MAGNITUDES_M
    out_dir: Path = DEFAULT_OUT_DIR
    max_logs: int | None = DEFAULT_MAX_LOGS
    seed: int = 0
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    prediction_horizon_s: float = DEFAULT_PREDICTION_HORIZON_S
    prediction_dt_s: float = DEFAULT_PREDICTION_DT_S
    prediction_samples: int = DEFAULT_PREDICTION_SAMPLES
    swirl_sigma_m: float = DEFAULT_SWIRL_SIGMA_M


@dataclass(frozen=True)
class SelectedFrame:
    category: str
    log_index: int
    frame_index: int
    target_actor: str


@dataclass(frozen=True)
class LoadedLog:
    db_path: Path
    log: NuPlanLog
    map_path: Path
    epsg: str


@dataclass(frozen=True)
class ExperimentResult:
    runs: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    stdout: str
    runs_path: Path
    summary_path: Path


@dataclass(frozen=True)
class PathSafety:
    min_distance_m: float | None
    cv_ttc_s: float | None


@dataclass(frozen=True)
class FaultApplication:
    objects: ObjectSet
    cost_grid: NDArray[np.float64]
    metadata: dict[str, float]
    prediction_l2: float | None = None


def run_experiment(config: NuPlanPropagationConfig) -> ExperimentResult:
    cfg = _validate_config(config)
    runs_path = cfg.out_dir / "nuplan_propagation_runs.csv"
    summary_path = cfg.out_dir / "nuplan_propagation_summary.csv"

    loaded_logs = _load_candidate_logs(cfg)
    selected = _select_frames(loaded_logs, cfg)
    runs = _run_selected_frames(loaded_logs, selected, cfg)
    runs = _sort_run_rows(runs)
    summary = summarize_runs(runs, cfg)

    _write_csv(runs_path, runs, RUN_COLUMNS)
    _write_csv(summary_path, summary, SUMMARY_COLUMNS)
    stdout = format_stdout_summary(runs, summary, cfg, len(loaded_logs), len(selected))
    return ExperimentResult(
        runs=runs,
        summary=summary,
        stdout=stdout,
        runs_path=runs_path,
        summary_path=summary_path,
    )


def summarize_runs(
    runs: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = [row for row in runs if row.get("status") == "completed"]
    for key in _group_keys(valid, ("category", "injection", "method", "planner")):
        group = [
            row
            for row in valid
            if all(
                row[column] == value
                for column, value in zip(("category", "injection", "method", "planner"), key, strict=True)
            )
        ]
        record: dict[str, Any] = {
            "row_type": "group",
            "category": key[0],
            "injection": key[1],
            "method": key[2],
            "planner": key[3],
            "n_runs": len(group),
            "n_valid": len(group),
            "n_frames": _count_frames(group),
        }
        _add_mean_ci(record, group, "object__cost_gain", cfg=cfg)
        _add_mean_ci(record, group, "cost__plan_gain", cfg=cfg)
        record["gateway_rate"] = _mean_bool(group, "planner_gateway")
        record["argmin_flip_rate"] = _mean_bool(group, "argmin_flip")
        _add_mean_ci(record, group, "decision_margin_score", cfg=cfg)
        _add_mean_ci(record, group, "n_feasible", cfg=cfg)
        _add_mean_ci(record, group, "safety1_plan_dev", cfg=cfg)
        _add_mean_ci(record, group, "safety2_mindist_drop", cfg=cfg)
        _add_mean_ci(record, group, "safety2_ttc_drop", cfg=cfg)
        _blank_comparison_fields(record)
        rows.append(record)

    rows.extend(_object_vs_costmap_rows(valid, cfg))
    return _sort_summary_rows(rows)


def format_stdout_summary(
    runs: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
    n_loaded_logs: int,
    n_selected_frames: int,
) -> str:
    del summary
    valid = [row for row in runs if row.get("status") == "completed"]
    lines = [
        "nuPlan open-loop propagation summary",
        f"loaded_logs={n_loaded_logs} selected_frames={n_selected_frames} completed_runs={len(valid)}",
        (
            "real_cost_map=clip(max(map_road_grid, real_agent_obstacle_grid), "
            f"{COST_MIN:.0f}, {COST_MAX:.0f}); costmap_faults_pin_map_boundary_cells=True"
        ),
        (
            "prediction_injection="
            + (
                "included_cv_predicted_occupancy_composed_with_real_map"
                if "prediction" in cfg.injections
                else "skipped_by_cli"
            )
        ),
    ]
    if not valid:
        lines.extend(
            [
                "",
                "No completed propagation rows. Check nuPlan logs/maps and scenario-tag coverage.",
            ]
        )
        return "\n".join(lines)

    claim_rows = [
        _claim_object_cost_attenuation(valid, cfg),
        _claim_cost_plan_amplification(valid, cfg),
        _claim_costmap_vs_object_safety(valid, cfg),
        _claim_gateway_argmin(valid),
        _claim_directed_vs_random(valid, cfg),
    ]
    lines.extend(
        [
            "",
            "Claim                                           metric                                               reproduces",
            "----------------------------------------------  ---------------------------------------------------  ----------",
        ]
    )
    for label, metric, verdict in claim_rows:
        lines.append(f"{label:<46}  {metric:<51}  {verdict}")
    lines.extend(
        [
            "",
            f"wrote_runs={cfg.out_dir / 'nuplan_propagation_runs.csv'}",
            f"wrote_summary={cfg.out_dir / 'nuplan_propagation_summary.csv'}",
        ]
    )
    return "\n".join(lines)


def _run_selected_frames(
    loaded_logs: list[LoadedLog],
    selected: list[SelectedFrame],
    cfg: NuPlanPropagationConfig,
) -> list[dict[str, Any]]:
    spec = CostMapSpec()
    planner_config = CostMapPlannerConfig()
    rows: list[dict[str, Any]] = []
    map_cache: dict[tuple[Path, str], Any] = {}

    selected_by_log: dict[int, list[SelectedFrame]] = {}
    for item in selected:
        selected_by_log.setdefault(item.log_index, []).append(item)

    for log_index in sorted(selected_by_log):
        loaded = loaded_logs[log_index]
        map_key = (loaded.map_path, loaded.epsg)
        if map_key not in map_cache:
            map_cache[map_key] = load_map(loaded.map_path, loaded.epsg)
        nmap = map_cache[map_key]

        for selected_frame in sorted(
            selected_by_log[log_index],
            key=lambda item: (_category_rank(item.category), item.frame_index),
        ):
            frame = loaded.log.frames[selected_frame.frame_index]
            try:
                base_context = _frame_context(
                    frame,
                    nmap=nmap,
                    spec=spec,
                    planner_config=planner_config,
                )
            except Exception as exc:
                rows.extend(
                    _failed_condition_rows(
                        loaded=loaded,
                        frame=frame,
                        selected_frame=selected_frame,
                        cfg=cfg,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            for planner_name in cfg.planners:
                planner = build_planner(planner_name, planner_config)
                clean_current_plan = planner.plan(
                    base_context["ego"],
                    base_context["current_components"].combined,
                    spec,
                )
                clean_prediction_components = None
                clean_prediction_plan = None
                if "prediction" in cfg.injections:
                    clean_prediction_components = _prediction_components(
                        frame.object_set,
                        base_context=base_context,
                        cfg=cfg,
                        spec=spec,
                        planner_config=planner_config,
                    )
                    clean_prediction_plan = planner.plan(
                        base_context["ego"],
                        clean_prediction_components.combined,
                        spec,
                    )

                for injection in cfg.injections:
                    for method in cfg.methods:
                        for magnitude_m in cfg.magnitudes_m:
                            row = _run_condition(
                                loaded=loaded,
                                frame=frame,
                                selected_frame=selected_frame,
                                cfg=cfg,
                                spec=spec,
                                planner_config=planner_config,
                                planner_name=planner_name,
                                planner=planner,
                                injection=injection,
                                method=method,
                                magnitude_m=float(magnitude_m),
                                base_context=base_context,
                                clean_current_plan=clean_current_plan,
                                clean_prediction_components=clean_prediction_components,
                                clean_prediction_plan=clean_prediction_plan,
                            )
                            rows.append(row)
    return rows


def _run_condition(
    *,
    loaded: LoadedLog,
    frame: NuPlanFrame,
    selected_frame: SelectedFrame,
    cfg: NuPlanPropagationConfig,
    spec: CostMapSpec,
    planner_config: CostMapPlannerConfig,
    planner_name: str,
    planner: Any,
    injection: str,
    method: str,
    magnitude_m: float,
    base_context: Mapping[str, Any],
    clean_current_plan: CostMapPlan,
    clean_prediction_components: CostMapComponents | None,
    clean_prediction_plan: CostMapPlan | None,
) -> dict[str, Any]:
    base_record = _base_run_record(
        loaded=loaded,
        frame=frame,
        selected_frame=selected_frame,
        injection=injection,
        method=method,
        planner_name=planner_name,
        magnitude_m=magnitude_m,
        ego=base_context["ego"],
    )
    try:
        if injection == "object":
            clean_grid = base_context["current_components"].combined
            clean_plan = clean_current_plan
            fault = _apply_object_injection(
                frame.object_set,
                selected_frame.target_actor,
                method=method,
                magnitude_m=magnitude_m,
                seed=_stable_seed(cfg, loaded.db_path, frame, injection, method, magnitude_m),
                base_context=base_context,
                spec=spec,
                planner_config=planner_config,
            )
            object_shift = _object_position_shift(
                frame.object_set,
                fault.objects,
                selected_frame.target_actor,
            )
        elif injection == "costmap":
            clean_grid = base_context["current_components"].combined
            clean_plan = clean_current_plan
            fault = _apply_costmap_injection(
                base_context["current_components"],
                frame.object_set,
                selected_frame.target_actor,
                clean_plan=clean_current_plan,
                method=method,
                magnitude_m=magnitude_m,
                seed=_stable_seed(cfg, loaded.db_path, frame, injection, method, magnitude_m),
                cfg=cfg,
                spec=spec,
                ego=base_context["ego"],
            )
            object_shift = 0.0
        elif injection == "prediction":
            if clean_prediction_components is None or clean_prediction_plan is None:
                raise ValueError("prediction injection was not initialized")
            clean_grid = clean_prediction_components.combined
            clean_plan = clean_prediction_plan
            fault = _apply_prediction_injection(
                frame.object_set,
                selected_frame.target_actor,
                method=method,
                magnitude_m=magnitude_m,
                seed=_stable_seed(cfg, loaded.db_path, frame, injection, method, magnitude_m),
                base_context=base_context,
                cfg=cfg,
                spec=spec,
                planner_config=planner_config,
            )
            object_shift = 0.0
        else:
            raise ValueError(f"unknown injection {injection!r}")

        fault_plan = planner.plan(base_context["ego"], fault.cost_grid, spec)
        metrics = _condition_metrics(
            clean_grid=clean_grid,
            fault_grid=fault.cost_grid,
            clean_plan=clean_plan,
            fault_plan=fault_plan,
            real_objects=frame.object_set,
            planner_name=planner_name,
            object_shift_m=object_shift,
            planner_config=planner_config,
        )
        out = dict(base_record)
        out.update(metrics)
        out["status"] = "completed"
        out["error"] = ""
        out["operator_strength"] = fault.metadata.get("operator_strength")
        out["prediction_l2"] = fault.prediction_l2
        return out
    except Exception as exc:
        out = dict(base_record)
        out.update(_empty_metric_record())
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _frame_context(
    frame: NuPlanFrame,
    *,
    nmap: Any,
    spec: CostMapSpec,
    planner_config: CostMapPlannerConfig,
) -> dict[str, Any]:
    ego_speed = float(np.linalg.norm(np.asarray(frame.ego_v, dtype=np.float64)))
    ego = EgoState(x=0.0, y=0.0, yaw=0.0, speed=ego_speed)
    road_grid = build_map_road_cost_grid(
        nmap,
        ego_global_xy=frame.ego_xy,
        ego_yaw=frame.ego_yaw,
        spec=spec,
        planner_config=planner_config,
    )
    boundary_mask = road_boundary_mask_from_map(
        nmap,
        ego_global_xy=frame.ego_xy,
        ego_yaw=frame.ego_yaw,
        spec=spec,
        planner_config=planner_config,
    )
    obstacle_components = build_cost_grid_components(
        frame.object_set,
        ego,
        spec,
        planner_config,
    )
    current_components = CostMapComponents(
        road_grid=np.asarray(road_grid, dtype=np.float64),
        obstacle_grid=np.asarray(obstacle_components.obstacle_grid, dtype=np.float64),
        boundary_mask=np.asarray(boundary_mask, dtype=bool),
    )
    return {
        "ego": ego,
        "road_grid": current_components.road_grid,
        "boundary_mask": current_components.boundary_mask,
        "current_components": current_components,
    }


def _prediction_components(
    objects: ObjectSet,
    *,
    base_context: Mapping[str, Any],
    cfg: NuPlanPropagationConfig,
    spec: CostMapSpec,
    planner_config: CostMapPlannerConfig,
) -> CostMapComponents:
    predicted = build_predicted_cost_grid_components(
        objects,
        base_context["ego"],
        spec,
        planner_config,
        horizon_s=cfg.prediction_horizon_s,
        dt=cfg.prediction_dt_s,
        samples=cfg.prediction_samples,
    )
    return CostMapComponents(
        road_grid=np.asarray(base_context["road_grid"], dtype=np.float64),
        obstacle_grid=np.asarray(predicted.obstacle_grid, dtype=np.float64),
        boundary_mask=np.asarray(base_context["boundary_mask"], dtype=bool),
    )


def _apply_object_injection(
    objects: ObjectSet,
    target_actor: str,
    *,
    method: str,
    magnitude_m: float,
    seed: int,
    base_context: Mapping[str, Any],
    spec: CostMapSpec,
    planner_config: CostMapPlannerConfig,
) -> FaultApplication:
    perturbed, metadata = _perturb_objects(
        objects,
        target_actor,
        method=method,
        magnitude_m=magnitude_m,
        seed=seed,
        include_position=True,
        ego=base_context["ego"],
    )
    components = _components_for_objects(
        perturbed,
        base_context=base_context,
        spec=spec,
        planner_config=planner_config,
    )
    return FaultApplication(objects=perturbed, cost_grid=components.combined, metadata=metadata)


def _apply_prediction_injection(
    objects: ObjectSet,
    target_actor: str,
    *,
    method: str,
    magnitude_m: float,
    seed: int,
    base_context: Mapping[str, Any],
    cfg: NuPlanPropagationConfig,
    spec: CostMapSpec,
    planner_config: CostMapPlannerConfig,
) -> FaultApplication:
    motion_objects, metadata = _perturb_objects(
        objects,
        target_actor,
        method=method,
        magnitude_m=magnitude_m,
        seed=seed,
        include_position=False,
        ego=base_context["ego"],
    )
    clean_components = _prediction_components(
        objects,
        base_context=base_context,
        cfg=cfg,
        spec=spec,
        planner_config=planner_config,
    )
    fault_components = _prediction_components(
        motion_objects,
        base_context=base_context,
        cfg=cfg,
        spec=spec,
        planner_config=planner_config,
    )
    prediction_l2 = _rms_delta(clean_components.obstacle_grid, fault_components.obstacle_grid)
    return FaultApplication(
        objects=objects,
        cost_grid=fault_components.combined,
        metadata=metadata,
        prediction_l2=prediction_l2,
    )


def _apply_costmap_injection(
    components: CostMapComponents,
    objects: ObjectSet,
    target_actor: str,
    *,
    clean_plan: CostMapPlan,
    method: str,
    magnitude_m: float,
    seed: int,
    cfg: NuPlanPropagationConfig,
    spec: CostMapSpec,
    ego: EgoState,
) -> FaultApplication:
    del clean_plan
    if method == "torsion_displace":
        target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
        sign = _away_from_route_lateral_sign(ego_y=ego.y, target_local_y=target_xy[1])
        shift_y_m = sign * magnitude_m
        shift_grid = np.array([0.0, -shift_y_m / spec.resolution_m], dtype=np.float64)
        obstacle = translate_cost_field(components.obstacle_grid, shift_grid)
        metadata = {
            "operator_strength": float(abs(shift_y_m)),
            "translate_shift_y_m": float(shift_y_m),
        }
    elif method == "random_warp":
        target_xy = _target_local_xy(objects, target_actor=target_actor, ego=ego)
        target_grid = spec.metric_to_grid(np.asarray(target_xy, dtype=np.float64))
        sigma_cells = _swirl_sigma_cells(cfg, spec)
        rng = np.random.default_rng(seed)
        angle = float(rng.uniform(-np.pi, np.pi))
        radial = sigma_cells * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        pivot_grid = target_grid - radial
        sign = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float64)))
        alpha = sign * _alpha_for_physical_shift(magnitude_m, cfg.swirl_sigma_m)
        obstacle = spatial_cost_warp(
            components.obstacle_grid,
            tuple(pivot_grid),
            alpha=alpha,
            sigma=sigma_cells,
        )
        pivot_metric = spec.grid_to_metric(pivot_grid)
        metadata = {
            "operator_strength": float(abs(alpha)),
            "swirl_alpha_rad": float(alpha),
            "swirl_sigma_m": float(cfg.swirl_sigma_m),
            "swirl_pivot_x_m": float(pivot_metric[0]),
            "swirl_pivot_y_m": float(pivot_metric[1]),
        }
    else:
        raise ValueError(f"unknown method {method!r}")
    return FaultApplication(
        objects=objects,
        cost_grid=_recombine_costmap_fault(components, obstacle),
        metadata=metadata,
    )


def _perturb_objects(
    objects: ObjectSet,
    target_actor: str,
    *,
    method: str,
    magnitude_m: float,
    seed: int,
    include_position: bool,
    ego: EgoState,
) -> tuple[ObjectSet, dict[str, float]]:
    target_idx = _target_index(objects, target_actor)
    rng = np.random.default_rng(seed)
    if method == "torsion_displace":
        if include_position:
            dx, dy = _away_from_ego_position_delta(
                objects,
                target_idx=target_idx,
                ego=ego,
                shift_m=magnitude_m,
            )
        else:
            dx, dy = 0.0, 0.0
        theta = _velocity_rotation_away_from_ego(
            objects,
            target_idx=target_idx,
            ego_y=ego.y,
            theta_abs=_angle_for_magnitude(magnitude_m),
            position_dy=dy,
            horizon_s=3.0,
        )
        d_yaw = _yaw_delta_for_theta(objects, target_idx, ego, theta, magnitude_m)
    elif method == "random_warp":
        if include_position:
            angle = float(rng.uniform(-np.pi, np.pi))
            dx = magnitude_m * float(np.cos(angle))
            dy = magnitude_m * float(np.sin(angle))
        else:
            dx, dy = 0.0, 0.0
        d_yaw = _angle_for_magnitude(magnitude_m) * float(rng.choice(np.array([-1.0, 1.0])))
        theta = _angle_for_magnitude(magnitude_m) * float(rng.choice(np.array([-1.0, 1.0])))
    else:
        raise ValueError(f"unknown method {method!r}")

    max_shift = max(2.0, float(magnitude_m) + 1.0e-9)
    max_angle = float(np.deg2rad(30.0))
    out = objects
    if include_position:
        out = position_torsion(
            out,
            dx=dx,
            dy=dy,
            track_ids=[target_actor],
            max_shift_m=max_shift,
        )
    out = yaw_torsion(
        out,
        d_yaw=d_yaw,
        track_ids=[target_actor],
        max_yaw_delta_rad=max_angle,
    )
    out = velocity_direction_torsion(
        out,
        theta=theta,
        track_ids=[target_actor],
        max_rotation_rad=max_angle,
    )
    return out, {
        "operator_strength": float(np.hypot(dx, dy) if include_position else abs(theta)),
        "object_shift_x_m": float(dx),
        "object_shift_y_m": float(dy),
        "object_yaw_delta_rad": float(d_yaw),
        "object_velocity_rotation_delta_rad": float(theta),
    }


def _components_for_objects(
    objects: ObjectSet,
    *,
    base_context: Mapping[str, Any],
    spec: CostMapSpec,
    planner_config: CostMapPlannerConfig,
) -> CostMapComponents:
    obstacle_components = build_cost_grid_components(
        objects,
        base_context["ego"],
        spec,
        planner_config,
    )
    return CostMapComponents(
        road_grid=np.asarray(base_context["road_grid"], dtype=np.float64),
        obstacle_grid=np.asarray(obstacle_components.obstacle_grid, dtype=np.float64),
        boundary_mask=np.asarray(base_context["boundary_mask"], dtype=bool),
    )


def _condition_metrics(
    *,
    clean_grid: NDArray[np.float64],
    fault_grid: NDArray[np.float64],
    clean_plan: CostMapPlan,
    fault_plan: CostMapPlan,
    real_objects: ObjectSet,
    planner_name: str,
    object_shift_m: float,
    planner_config: CostMapPlannerConfig,
) -> dict[str, Any]:
    cost_l2 = _rms_delta(clean_grid, fault_grid)
    plan_dev = _path_l2_deviation(clean_plan.path_xy, fault_plan.path_xy)
    object_cost_gain = _safe_ratio(cost_l2, object_shift_m) if object_shift_m > EPS_RATIO else None
    cost_plan_gain = _safe_ratio(plan_dev, cost_l2)
    argmin_flip = _argmin_flip(clean_plan, fault_plan, planner_name)
    gains = [value for value in (object_cost_gain, cost_plan_gain) if value is not None]
    planner_gateway = (
        bool(cost_plan_gain is not None and cost_plan_gain > 1.0 and cost_plan_gain >= max(gains))
        if gains
        else False
    )

    horizon_s = float(planner_config.horizon_s)
    clean_safety = planned_path_safety(clean_plan.path_xy, real_objects, horizon_s=horizon_s)
    fault_safety = planned_path_safety(fault_plan.path_xy, real_objects, horizon_s=horizon_s)
    mindist_drop = _finite_difference(clean_safety.min_distance_m, fault_safety.min_distance_m)
    ttc_drop = _finite_difference(clean_safety.cv_ttc_s, fault_safety.cv_ttc_s)

    # Log the clean plan's decision margin so the gateway-collapse hypothesis --
    # "dense scenes have well-separated cost minima, so the argmin cannot flip" --
    # can be tested on real data instead of only asserted.
    margin = decision_margin(clean_plan.to_record())

    return {
        "object_shift_m": float(object_shift_m),
        "cost_l2": cost_l2,
        "plan_dev": plan_dev,
        "object__cost_gain": object_cost_gain,
        "cost__plan_gain": cost_plan_gain,
        "argmin_flip": argmin_flip,
        "planner_gateway": planner_gateway,
        "decision_margin_score": margin["decision_margin_score"],
        "n_candidates": margin["n_candidates"],
        "n_feasible": margin["n_feasible"],
        "decision_margin_pool": margin["decision_margin_pool"],
        "safety1_plan_dev": plan_dev,
        "clean_min_dist_m": clean_safety.min_distance_m,
        "fault_min_dist_m": fault_safety.min_distance_m,
        "safety2_mindist_drop": mindist_drop,
        "clean_cv_ttc_s": clean_safety.cv_ttc_s,
        "fault_cv_ttc_s": fault_safety.cv_ttc_s,
        "safety2_ttc_drop": ttc_drop,
    }


def planned_path_safety(
    path_xy: NDArray[np.float64],
    objects: ObjectSet,
    *,
    horizon_s: float,
) -> PathSafety:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or not np.all(np.isfinite(path)):
        raise ValueError("path_xy must have finite shape (N, 2)")
    if horizon_s <= 0.0 or not math.isfinite(horizon_s):
        raise ValueError("horizon_s must be positive and finite")
    if len(objects) == 0:
        return PathSafety(min_distance_m=None, cv_ttc_s=None)

    actor_xy = np.asarray(objects.xy, dtype=np.float64)
    deltas_current = path[:, None, :] - actor_xy[None, :, :]
    distances = np.linalg.norm(deltas_current, axis=2)
    min_distance = float(np.min(distances)) if distances.size else None

    times = np.linspace(0.0, float(horizon_s), path.shape[0], dtype=np.float64)
    if path.shape[0] == 1:
        ego_velocity = np.zeros((1, 2), dtype=np.float64)
    else:
        ego_velocity = np.gradient(path, times, axis=0)
    actor_future = actor_xy[None, :, :] + times[:, None, None] * objects.v[None, :, :]
    rel_pos = actor_future - path[:, None, :]
    rel_vel = objects.v[None, :, :] - ego_velocity[:, None, :]
    ranges = np.linalg.norm(rel_pos, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = -np.sum(rel_pos * rel_vel, axis=2) / np.maximum(ranges, EPS_RATIO)
        ttc = np.where(ranges <= EPS_RATIO, 0.0, ranges / closing)
    valid = np.isfinite(ttc) & (closing > EPS_RATIO) & (ttc >= 0.0)
    cv_ttc = float(np.min(ttc[valid])) if np.any(valid) else None
    return PathSafety(min_distance_m=min_distance, cv_ttc_s=cv_ttc)


def _load_candidate_logs(cfg: NuPlanPropagationConfig) -> list[LoadedLog]:
    db_paths = list_logs(cfg.logs_root)
    if cfg.max_logs is not None:
        db_paths = db_paths[: cfg.max_logs]
    loaded: list[LoadedLog] = []
    for db_path in db_paths:
        try:
            map_path = map_path_for_log(db_path, cfg.maps_root)
            metadata = _log_metadata(db_path)
            epsg = epsg_for_location(str(metadata.get("map_version") or metadata.get("location")))
            log = load_log(db_path)
        except Exception:
            continue
        loaded.append(LoadedLog(db_path=Path(db_path), log=log, map_path=map_path, epsg=epsg))
    return loaded


def _select_frames(
    loaded_logs: list[LoadedLog],
    cfg: NuPlanPropagationConfig,
) -> list[SelectedFrame]:
    by_category: dict[str, list[SelectedFrame]] = {category: [] for category in cfg.categories}
    for log_index, loaded in enumerate(loaded_logs):
        for category in cfg.categories:
            candidate_indices = _category_frame_indices(loaded.log, category)
            for frame_idx in candidate_indices:
                frame = loaded.log.frames[frame_idx]
                target = select_target_actor(frame, (0.0, 0.0))
                if target is None:
                    continue
                by_category[category].append(
                    SelectedFrame(
                        category=category,
                        log_index=log_index,
                        frame_index=int(frame_idx),
                        target_actor=str(target),
                    )
                )

    selected: list[SelectedFrame] = []
    for category in cfg.categories:
        selected.extend(_sample_category_frames(by_category.get(category, []), cfg.n_frames))
    return sorted(
        selected,
        key=lambda item: (_category_rank(item.category), item.log_index, item.frame_index),
    )


def _category_frame_indices(log: NuPlanLog, category: str) -> list[int]:
    key = str(category).upper()
    if key not in CATEGORY_TAGS:
        raise ValueError(f"unknown category {category!r}")
    exact_tags = CATEGORY_TAGS[key]
    indices: set[int] = set()
    for tag in exact_tags:
        if tag.endswith("*"):
            prefix = tag[:-1]
            indices.update(
                idx
                for idx, frame in enumerate(log.frames)
                if any(str(value).startswith(prefix) for value in frame.scenario_tags)
            )
        elif tag == "changing_lane":
            indices.update(
                idx
                for idx, frame in enumerate(log.frames)
                if any(str(value).startswith("changing_lane") for value in frame.scenario_tags)
            )
        else:
            indices.update(find_frames_with_tag(log, tag))
    return sorted(indices)


def _sample_category_frames(candidates: list[SelectedFrame], limit: int) -> list[SelectedFrame]:
    if limit <= 0 or not candidates:
        return []
    by_log: dict[int, list[SelectedFrame]] = {}
    for item in candidates:
        by_log.setdefault(item.log_index, []).append(item)
    per_log = {
        log_index: _evenly_spaced(items, limit)
        for log_index, items in sorted(by_log.items())
    }
    out: list[SelectedFrame] = []
    offsets = {log_index: 0 for log_index in per_log}
    while len(out) < limit:
        progressed = False
        for log_index in sorted(per_log):
            values = per_log[log_index]
            offset = offsets[log_index]
            if offset >= len(values):
                continue
            out.append(values[offset])
            offsets[log_index] = offset + 1
            progressed = True
            if len(out) >= limit:
                break
        if not progressed:
            break
    return out


def _evenly_spaced(items: list[SelectedFrame], limit: int) -> list[SelectedFrame]:
    values = sorted(items, key=lambda item: item.frame_index)
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
    seen: set[int] = set()
    out: list[SelectedFrame] = []
    for idx in indices.tolist():
        if int(idx) in seen:
            continue
        seen.add(int(idx))
        out.append(values[int(idx)])
    return out


def _object_vs_costmap_rows(
    valid: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
) -> list[dict[str, Any]]:
    del cfg
    out: list[dict[str, Any]] = []
    index: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in valid:
        if row.get("injection") not in {"object", "costmap"}:
            continue
        key = (
            row["category"],
            row["method"],
            row["planner"],
            row["log_path"],
            row["frame_index"],
            row["magnitude_m"],
        )
        index.setdefault(key, {})[str(row["injection"])] = row

    deltas_by_group: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for key, pair in index.items():
        if "object" not in pair or "costmap" not in pair:
            continue
        category, method, planner = str(key[0]), str(key[1]), str(key[2])
        group = deltas_by_group.setdefault(
            (category, method, planner),
            {"plan": [], "mindist": [], "ttc": []},
        )
        object_row = pair["object"]
        costmap_row = pair["costmap"]
        _append_delta(group["plan"], costmap_row.get("safety1_plan_dev"), object_row.get("safety1_plan_dev"))
        _append_delta(
            group["mindist"],
            costmap_row.get("safety2_mindist_drop"),
            object_row.get("safety2_mindist_drop"),
        )
        _append_delta(group["ttc"], costmap_row.get("safety2_ttc_drop"), object_row.get("safety2_ttc_drop"))

    for (category, method, planner), values in sorted(deltas_by_group.items()):
        record: dict[str, Any] = {
            "row_type": "object_vs_costmap",
            "category": category,
            "injection": "costmap_minus_object",
            "method": method,
            "planner": planner,
            "n_runs": None,
            "n_valid": None,
            "n_frames": None,
        }
        _blank_group_metric_fields(record)
        _add_comparison_values(record, "plan_dev", values["plan"])
        _add_comparison_values(record, "mindist_drop", values["mindist"])
        _add_comparison_values(record, "ttc_drop", values["ttc"])
        out.append(record)
    return out


def _claim_object_cost_attenuation(
    rows: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
) -> tuple[str, str, str]:
    values = _finite_values(
        row.get("object__cost_gain")
        for row in rows
        if row.get("injection") == "object"
    )
    return _mean_claim(
        "(a) object->cost attenuation",
        "mean object__cost_gain",
        values,
        lambda mean: mean < 1.0,
        cfg,
    )


def _claim_cost_plan_amplification(
    rows: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
) -> tuple[str, str, str]:
    values = _finite_values(
        row.get("cost__plan_gain")
        for row in rows
        if row.get("planner") == "sampling"
    )
    return _mean_claim(
        "(b) sampling cost->plan amp",
        "mean sampling cost__plan_gain",
        values,
        lambda mean: mean > 1.0,
        cfg,
    )


def _claim_costmap_vs_object_safety(
    rows: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
) -> tuple[str, str, str]:
    del cfg
    object_plan = _finite_values(
        row.get("safety1_plan_dev") for row in rows if row.get("injection") == "object"
    )
    costmap_plan = _finite_values(
        row.get("safety1_plan_dev") for row in rows if row.get("injection") == "costmap"
    )
    object_mindist = _finite_values(
        row.get("safety2_mindist_drop") for row in rows if row.get("injection") == "object"
    )
    costmap_mindist = _finite_values(
        row.get("safety2_mindist_drop") for row in rows if row.get("injection") == "costmap"
    )
    object_ttc = _finite_values(
        row.get("safety2_ttc_drop") for row in rows if row.get("injection") == "object"
    )
    costmap_ttc = _finite_values(
        row.get("safety2_ttc_drop") for row in rows if row.get("injection") == "costmap"
    )
    if not object_plan or not costmap_plan or not object_mindist or not costmap_mindist:
        return ("(c) costmap safety > object", "insufficient paired safety data", "N/A")
    object_plan_mean = float(np.mean(object_plan))
    costmap_plan_mean = float(np.mean(costmap_plan))
    object_mindist_mean = float(np.mean(object_mindist))
    costmap_mindist_mean = float(np.mean(costmap_mindist))
    metric = (
        f"plan {costmap_plan_mean:.4g}>{object_plan_mean:.4g}; "
        f"mindist_drop {costmap_mindist_mean:.4g}>{object_mindist_mean:.4g}"
    )
    if object_ttc and costmap_ttc:
        metric += f"; ttc_drop {float(np.mean(costmap_ttc)):.4g}>{float(np.mean(object_ttc)):.4g}"
    verdict = "Y" if costmap_plan_mean > object_plan_mean and costmap_mindist_mean > object_mindist_mean else "N"
    return ("(c) costmap safety > object", metric, verdict)


def _claim_gateway_argmin(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    sampling = [row for row in rows if row.get("planner") == "sampling"]
    field = [row for row in rows if row.get("planner") == "potential_field"]
    sampling_gateway = _mean_bool(sampling, "planner_gateway")
    field_gateway = _mean_bool(field, "planner_gateway")
    sampling_argmin = _mean_bool(sampling, "argmin_flip")
    field_argmin = _mean_bool(field, "argmin_flip")
    if sampling_gateway is None or field_gateway is None:
        return ("(d) gateway/argmin planner", "insufficient planner comparison data", "N/A")
    metric = (
        f"gateway sampling={sampling_gateway:.3g}, field={field_gateway:.3g}; "
        f"argmin sampling={_format_optional_float(sampling_argmin)}, "
        f"field={_format_optional_float(field_argmin)}"
    )
    field_argmin_ok = field_argmin is None or abs(field_argmin) <= EPS_RATIO
    verdict = "Y" if sampling_gateway > field_gateway and field_argmin_ok else "N"
    return ("(d) gateway/argmin planner", metric, verdict)


def _claim_directed_vs_random(
    rows: list[dict[str, Any]],
    cfg: NuPlanPropagationConfig,
) -> tuple[str, str, str]:
    del cfg
    directed_plan = _finite_values(
        row.get("safety1_plan_dev") for row in rows if row.get("method") == "torsion_displace"
    )
    random_plan = _finite_values(
        row.get("safety1_plan_dev") for row in rows if row.get("method") == "random_warp"
    )
    directed_mindist = _finite_values(
        row.get("safety2_mindist_drop") for row in rows if row.get("method") == "torsion_displace"
    )
    random_mindist = _finite_values(
        row.get("safety2_mindist_drop") for row in rows if row.get("method") == "random_warp"
    )
    if not directed_plan or not random_plan or not directed_mindist or not random_mindist:
        return ("(e) directed vs random", "insufficient method comparison data", "N/A")
    directed_plan_mean = float(np.mean(directed_plan))
    random_plan_mean = float(np.mean(random_plan))
    directed_mindist_mean = float(np.mean(directed_mindist))
    random_mindist_mean = float(np.mean(random_mindist))
    metric = (
        f"plan {directed_plan_mean:.4g} vs {random_plan_mean:.4g}; "
        f"mindist_drop {directed_mindist_mean:.4g} vs {random_mindist_mean:.4g}"
    )
    verdict = "Y" if directed_plan_mean >= random_plan_mean and directed_mindist_mean >= random_mindist_mean else "N"
    return ("(e) directed vs random", metric, verdict)


def _mean_claim(
    label: str,
    metric_name: str,
    values: list[float],
    predicate: Any,
    cfg: NuPlanPropagationConfig,
) -> tuple[str, str, str]:
    if not values:
        return (label, f"{metric_name}: insufficient finite values", "N/A")
    mean = float(np.mean(values))
    low, high = _ci(values, cfg)
    metric = f"{metric_name}={mean:.4g} [{low:.4g},{high:.4g}] n={len(values)}"
    return (label, metric, "Y" if predicate(mean) else "N")


def _base_run_record(
    *,
    loaded: LoadedLog,
    frame: NuPlanFrame,
    selected_frame: SelectedFrame,
    injection: str,
    method: str,
    planner_name: str,
    magnitude_m: float,
    ego: EgoState,
) -> dict[str, Any]:
    return {
        "status": "",
        "error": "",
        "log_path": str(loaded.db_path),
        "log_name": loaded.db_path.stem,
        "frame_index": int(selected_frame.frame_index),
        "frame_token": frame.frame_token,
        "time_s": float(frame.time_s),
        "category": selected_frame.category,
        "scenario_tags": ";".join(frame.scenario_tags),
        "target_actor": selected_frame.target_actor,
        "injection": injection,
        "method": method,
        "planner": planner_name,
        "magnitude_m": float(magnitude_m),
        "ego_speed_mps": float(ego.speed),
    }


def _failed_condition_rows(
    *,
    loaded: LoadedLog,
    frame: NuPlanFrame,
    selected_frame: SelectedFrame,
    cfg: NuPlanPropagationConfig,
    error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ego_speed = float(np.linalg.norm(np.asarray(frame.ego_v, dtype=np.float64)))
    ego = EgoState(x=0.0, y=0.0, yaw=0.0, speed=ego_speed)
    for planner_name in cfg.planners:
        for injection in cfg.injections:
            for method in cfg.methods:
                for magnitude_m in cfg.magnitudes_m:
                    row = _base_run_record(
                        loaded=loaded,
                        frame=frame,
                        selected_frame=selected_frame,
                        injection=injection,
                        method=method,
                        planner_name=planner_name,
                        magnitude_m=float(magnitude_m),
                        ego=ego,
                    )
                    row.update(_empty_metric_record())
                    row["status"] = "failed"
                    row["error"] = error
                    rows.append(row)
    return rows


def _empty_metric_record() -> dict[str, Any]:
    return {
        "object_shift_m": None,
        "cost_l2": None,
        "plan_dev": None,
        "object__cost_gain": None,
        "cost__plan_gain": None,
        "argmin_flip": None,
        "planner_gateway": None,
        "safety1_plan_dev": None,
        "clean_min_dist_m": None,
        "fault_min_dist_m": None,
        "safety2_mindist_drop": None,
        "clean_cv_ttc_s": None,
        "fault_cv_ttc_s": None,
        "safety2_ttc_drop": None,
        "operator_strength": None,
        "prediction_l2": None,
    }


def _recombine_costmap_fault(
    components: CostMapComponents,
    obstacle_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    obstacle = np.clip(np.asarray(obstacle_grid, dtype=np.float64), COST_MIN, COST_MAX)
    if obstacle.shape != components.obstacle_grid.shape:
        raise ValueError("obstacle_grid shape changed during cost-map fault")
    clean = components.combined
    out = np.maximum(components.road_grid, obstacle)
    out = np.clip(out, COST_MIN, COST_MAX)
    out[components.boundary_mask] = clean[components.boundary_mask]
    return out


def _object_position_shift(
    clean: ObjectSet,
    faulted: ObjectSet,
    target_actor: str,
) -> float:
    clean_idx = _target_index(clean, target_actor)
    fault_idx = _target_index(faulted, target_actor)
    delta = np.array(
        [
            float(faulted.x[fault_idx] - clean.x[clean_idx]),
            float(faulted.y[fault_idx] - clean.y[clean_idx]),
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(delta))


def _target_index(objects: ObjectSet, target_actor: str) -> int:
    matches = np.flatnonzero(objects.track_id == target_actor)
    if matches.size != 1:
        raise ValueError(f"target actor {target_actor!r} not found exactly once")
    return int(matches[0])


def _target_local_xy(objects: ObjectSet, *, target_actor: str, ego: EgoState) -> tuple[float, float]:
    idx = _target_index(objects, target_actor)
    return float(objects.x[idx] - ego.x), float(objects.y[idx] - ego.y)


def _away_from_ego_position_delta(
    objects: ObjectSet,
    *,
    target_idx: int,
    ego: EgoState,
    shift_m: float,
) -> tuple[float, float]:
    if abs(float(shift_m)) <= 0.0:
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
    if abs(float(theta_abs)) <= 0.0:
        return 0.0
    velocity = np.asarray(objects.v[target_idx], dtype=np.float64)
    y0 = float(objects.y[target_idx]) + float(position_dy)
    candidates = (abs(float(theta_abs)), -abs(float(theta_abs)))
    scores = []
    for theta in candidates:
        rotated = _rotate2d(velocity, theta)
        future_y = y0 + float(rotated[1]) * float(horizon_s)
        scores.append(abs(future_y - float(ego_y)))
    return float(candidates[int(np.argmax(scores))])


def _yaw_delta_for_theta(
    objects: ObjectSet,
    target_idx: int,
    ego: EgoState,
    theta: float,
    magnitude_m: float,
) -> float:
    angle = _angle_for_magnitude(magnitude_m)
    if abs(float(theta)) > 0.0:
        return float(np.sign(theta) * angle)
    return float(_away_from_ego_sign(float(objects.y[target_idx]), ego.y) * angle)


def _angle_for_magnitude(magnitude_m: float) -> float:
    return float(np.clip(np.deg2rad(10.0) * float(magnitude_m), 0.0, np.deg2rad(30.0)))


def _alpha_for_physical_shift(magnitude_m: float, sigma_m: float) -> float:
    sigma = float(sigma_m)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("swirl_sigma_m must be positive and finite")
    return float(abs(float(magnitude_m)) / max(sigma * TWIST_PEAK_FACTOR, EPS_RATIO))


def _swirl_sigma_cells(cfg: NuPlanPropagationConfig, spec: CostMapSpec) -> float:
    sigma_m = float(cfg.swirl_sigma_m)
    if sigma_m <= 0.0 or not math.isfinite(sigma_m):
        raise ValueError("swirl_sigma_m must be positive and finite")
    return sigma_m / float(spec.resolution_m)


def _away_from_ego_sign(actor_value: float, ego_value: float) -> float:
    delta = float(actor_value) - float(ego_value)
    if abs(delta) < 1.0e-9:
        return 1.0
    return float(np.sign(delta))


def _away_from_route_lateral_sign(*, ego_y: float, target_local_y: float) -> float:
    world_y = float(ego_y) + float(target_local_y)
    if abs(world_y) <= 1.0e-9:
        return 1.0
    return float(np.sign(world_y))


def _rotate2d(vector: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _argmin_flip(
    clean_plan: CostMapPlan,
    fault_plan: CostMapPlan,
    planner_name: str,
) -> bool | None:
    if _planner_key(planner_name) != "sampling":
        return None
    return bool(
        not math.isclose(
            float(clean_plan.target_lateral_m),
            float(fault_plan.target_lateral_m),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    )


def _path_l2_deviation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("paths must have the same shape")
    return float(np.mean(np.linalg.norm(first - second, axis=1)))


def _rms_delta(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("arrays must have the same shape")
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not _is_finite_number(numerator) or not _is_finite_number(denominator):
        return None
    denom = float(denominator)
    if abs(denom) <= EPS_RATIO:
        return None
    return float(float(numerator) / denom)


def _finite_difference(left: Any, right: Any) -> float | None:
    if not _is_finite_number(left) or not _is_finite_number(right):
        return None
    return float(float(left) - float(right))


def _log_metadata(db_path: Path) -> dict[str, str | None]:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError(f"nuPlan DB has no log row: {db_path}")
    return {"location": None if row[0] is None else str(row[0]), "map_version": None if row[1] is None else str(row[1])}


def _stable_seed(
    cfg: NuPlanPropagationConfig,
    db_path: Path,
    frame: NuPlanFrame,
    injection: str,
    method: str,
    magnitude_m: float,
) -> int:
    text = "|".join(
        (
            "torsion-nuplan-openloop",
            str(cfg.seed),
            str(db_path),
            frame.frame_token,
            injection,
            method,
            f"{float(magnitude_m):.12g}",
        )
    )
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "little")


def _validate_config(config: NuPlanPropagationConfig) -> NuPlanPropagationConfig:
    cfg = replace(
        config,
        logs_root=Path(config.logs_root),
        maps_root=Path(config.maps_root),
        out_dir=Path(config.out_dir),
        categories=tuple(str(value).upper() for value in config.categories),
        injections=tuple(_injection_key(value) for value in config.injections),
        methods=tuple(_method_key(value) for value in config.methods),
        planners=tuple(_planner_key(value) for value in config.planners),
        magnitudes_m=tuple(float(value) for value in config.magnitudes_m),
    )
    if cfg.n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if cfg.max_logs is not None and cfg.max_logs <= 0:
        raise ValueError("max_logs must be positive or omitted")
    if cfg.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if cfg.prediction_horizon_s <= 0.0 or not math.isfinite(cfg.prediction_horizon_s):
        raise ValueError("prediction_horizon_s must be positive and finite")
    if cfg.prediction_dt_s <= 0.0 or not math.isfinite(cfg.prediction_dt_s):
        raise ValueError("prediction_dt_s must be positive and finite")
    if cfg.prediction_samples <= 0:
        raise ValueError("prediction_samples must be positive")
    if not cfg.categories:
        raise ValueError("at least one category is required")
    unknown_categories = [value for value in cfg.categories if value not in CATEGORIES]
    if unknown_categories:
        raise ValueError(f"unknown categories {unknown_categories!r}; expected {', '.join(CATEGORIES)}")
    if not cfg.injections or not cfg.methods or not cfg.planners or not cfg.magnitudes_m:
        raise ValueError("injections, methods, planners, and magnitudes must be non-empty")
    for magnitude in cfg.magnitudes_m:
        if magnitude <= 0.0 or not math.isfinite(magnitude):
            raise ValueError("magnitudes must be positive finite meters")
    return cfg


def _injection_key(value: str) -> str:
    key = str(value).lower().strip().replace("-", "_")
    if key in {"object", "objects", "object_set"}:
        return "object"
    if key in {"costmap", "cost_map", "cost"}:
        return "costmap"
    if key in {"prediction", "predict"}:
        return "prediction"
    raise ValueError(f"unknown injection {value!r}; expected object,costmap,prediction")


def _method_key(value: str) -> str:
    key = str(value).lower().strip().replace("-", "_")
    if key in {"torsion_displace", "torsion_translate", "displacement"}:
        return "torsion_displace"
    if key in {"random_warp", "random"}:
        return "random_warp"
    raise ValueError(f"unknown method {value!r}; expected torsion_displace,random_warp")


def _planner_key(value: str) -> str:
    key = str(value).lower().strip().replace("-", "_")
    if key in {"sampling", "sample", "argmin", "costmap"}:
        return "sampling"
    if key in {"potential_field", "field", "gradient"}:
        return "potential_field"
    raise ValueError(f"unknown planner {value!r}; expected sampling,potential_field")


def _group_keys(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    keys: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return sorted(keys)


def _add_mean_ci(
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    column: str,
    *,
    cfg: NuPlanPropagationConfig,
) -> None:
    values = _finite_values(row.get(column) for row in rows)
    record[f"{column}_n"] = len(values)
    if not values:
        record[f"{column}_mean"] = None
        record[f"{column}_ci_low"] = None
        record[f"{column}_ci_high"] = None
        return
    low, high = _ci(values, cfg)
    record[f"{column}_mean"] = float(np.mean(np.asarray(values, dtype=np.float64)))
    record[f"{column}_ci_low"] = low
    record[f"{column}_ci_high"] = high


def _add_comparison_values(record: dict[str, Any], suffix: str, values: list[float]) -> None:
    prefix = f"costmap_minus_object_{suffix}"
    record[f"{prefix}_n"] = len(values)
    if not values:
        record[f"{prefix}_mean"] = None
        record[f"{prefix}_ci_low"] = None
        record[f"{prefix}_ci_high"] = None
        return
    low, high = bootstrap_ci(values, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES, seed=DEFAULT_BOOTSTRAP_SEED)
    record[f"{prefix}_mean"] = float(np.mean(np.asarray(values, dtype=np.float64)))
    record[f"{prefix}_ci_low"] = low
    record[f"{prefix}_ci_high"] = high


def _ci(values: list[float], cfg: NuPlanPropagationConfig) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    return bootstrap_ci(
        values,
        n_resamples=cfg.bootstrap_resamples,
        seed=cfg.bootstrap_seed,
    )


def _blank_group_metric_fields(record: dict[str, Any]) -> None:
    for column in (
        "object__cost_gain",
        "cost__plan_gain",
        "safety1_plan_dev",
        "safety2_mindist_drop",
        "safety2_ttc_drop",
    ):
        record[f"{column}_n"] = None
        record[f"{column}_mean"] = None
        record[f"{column}_ci_low"] = None
        record[f"{column}_ci_high"] = None
    record["gateway_rate"] = None
    record["argmin_flip_rate"] = None


def _blank_comparison_fields(record: dict[str, Any]) -> None:
    for suffix in ("plan_dev", "mindist_drop", "ttc_drop"):
        prefix = f"costmap_minus_object_{suffix}"
        record[f"{prefix}_n"] = None
        record[f"{prefix}_mean"] = None
        record[f"{prefix}_ci_low"] = None
        record[f"{prefix}_ci_high"] = None


def _append_delta(out: list[float], left: Any, right: Any) -> None:
    if _is_finite_number(left) and _is_finite_number(right):
        out.append(float(left) - float(right))


def _mean_bool(rows: list[dict[str, Any]], column: str) -> float | None:
    values = [row.get(column) for row in rows if row.get(column) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray([1.0 if bool(value) else 0.0 for value in values], dtype=np.float64)))


def _finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if _is_finite_number(value):
            out.append(float(value))
    return out


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _count_frames(rows: list[dict[str, Any]]) -> int:
    return len({(row["log_path"], int(row["frame_index"])) for row in rows})


def _sort_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _category_rank(str(row["category"])),
            str(row["log_path"]),
            int(row["frame_index"]),
            _injection_rank(str(row["injection"])),
            _method_rank(str(row["method"])),
            _planner_rank(str(row["planner"])),
            float(row["magnitude_m"]),
        ),
    )


def _sort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_rank = {"group": 0, "object_vs_costmap": 1}
    return sorted(
        rows,
        key=lambda row: (
            row_rank.get(str(row["row_type"]), 99),
            _category_rank(str(row["category"])),
            _injection_rank(str(row["injection"])),
            _method_rank(str(row["method"])),
            _planner_rank(str(row["planner"])),
        ),
    )


def _category_rank(category: str) -> int:
    try:
        return CATEGORIES.index(str(category).upper())
    except ValueError:
        return len(CATEGORIES)


def _injection_rank(injection: str) -> int:
    order = {name: idx for idx, name in enumerate(("object", "costmap", "prediction", "costmap_minus_object"))}
    return order.get(str(injection), len(order))


def _method_rank(method: str) -> int:
    order = {name: idx for idx, name in enumerate(DEFAULT_METHODS)}
    return order.get(str(method), len(order))


def _planner_rank(planner: str) -> int:
    order = {name: idx for idx, name in enumerate(DEFAULT_PLANNERS)}
    return order.get(str(planner), len(order))


def _format_optional_float(value: Any) -> str:
    if not _is_finite_number(value):
        return "None"
    return f"{float(value):.3g}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_record(row) for row in rows)


def _csv_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            out[key] = ""
        elif isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, float) and not math.isfinite(value):
            out[key] = ""
        else:
            out[key] = value
    return out


def _split_csv(value: str) -> list[str]:
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError("at least one comma-separated value is required")
    return parsed


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in _split_csv(value))


def _parse_optional_max_logs(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def parse_args(argv: Sequence[str] | None = None) -> NuPlanPropagationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--n-frames", type=int, default=150)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--injections", default=",".join(DEFAULT_INJECTIONS))
    parser.add_argument("--planners", default=",".join(DEFAULT_PLANNERS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--magnitudes", default=",".join(str(value) for value in DEFAULT_MAGNITUDES_M))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-logs",
        type=int,
        default=DEFAULT_MAX_LOGS,
        help="maximum number of validation logs to scan; use 0 for all logs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--prediction-horizon", type=float, default=DEFAULT_PREDICTION_HORIZON_S)
    parser.add_argument("--prediction-dt", type=float, default=DEFAULT_PREDICTION_DT_S)
    parser.add_argument("--prediction-samples", type=int, default=DEFAULT_PREDICTION_SAMPLES)
    args = parser.parse_args(argv)
    return NuPlanPropagationConfig(
        logs_root=args.logs_root,
        maps_root=args.maps_root,
        n_frames=int(args.n_frames),
        categories=tuple(_split_csv(args.categories)),
        injections=tuple(_split_csv(args.injections)),
        methods=tuple(_split_csv(args.methods)),
        planners=tuple(_split_csv(args.planners)),
        magnitudes_m=_parse_float_csv(args.magnitudes),
        out_dir=args.out,
        max_logs=_parse_optional_max_logs(args.max_logs),
        seed=int(args.seed),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        prediction_horizon_s=float(args.prediction_horizon),
        prediction_dt_s=float(args.prediction_dt),
        prediction_samples=int(args.prediction_samples),
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = run_experiment(parse_args(argv))
    print(result.stdout)


if __name__ == "__main__":
    main()
