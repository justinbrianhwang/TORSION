"""Analysis-only mechanism metrics for Phase A+.

These functions explain the measured interface gains by reading existing
TORSION representations and traces. They do not modify operators, the
pipeline, or the planner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.operators.object import ObjectSet
from torsion.scenarios.costmap_runner import (
    CostMapComponents,
    CostMapPlanner,
    CostMapPlannerConfig,
    CostMapSpec,
    build_cost_grid_components,
    build_predicted_cost_grid_components,
    step_ego_on_costmap_path,
)
from torsion.scenarios.planner import EgoState
from torsion.scenarios.predict import constant_velocity_predict
from torsion.scenarios.synthetic_scenarios import SyntheticScenario, get_scenario
from torsion.scenarios.unified_pipeline import (
    UnifiedPipelineConfig,
    UnifiedRunResult,
    default_unified_planner_config,
    run_unified_pipeline,
)

DEFAULT_EPS_SWEEP: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
DEFAULT_PREDICTION_HORIZON_S = 2.0
DEFAULT_PREDICTION_SAMPLES = 5
DEFAULT_REPRESENTATIVE_FRAME = 10
EPS_MARGIN = 1e-9
TARGET_LATERAL_ATOL_M = 1e-9


def rasterization_jacobian(
    scenario: str | SyntheticScenario,
    *,
    magnitude: str | float,
    seed: int,
    use_prediction: bool = False,
    eps_sweep: Sequence[float] = DEFAULT_EPS_SWEEP,
    frame_idx: int | None = None,
    prediction_horizon_s: float = DEFAULT_PREDICTION_HORIZON_S,
    prediction_samples: int = DEFAULT_PREDICTION_SAMPLES,
    grid: CostMapSpec | None = None,
    planner_config: CostMapPlannerConfig | None = None,
) -> dict[str, Any]:
    """Finite-difference cost-grid sensitivity to target position.

    The target actor is shifted away from the ego position by each epsilon in
    ``eps_sweep``. The reported Jacobian is the RMS cost-grid delta divided by
    meters of position shift, using the same RMS-over-grid metric as the
    unified pipeline.
    """

    _validate_eps_sweep(eps_sweep)
    if prediction_samples <= 0:
        raise ValueError("prediction_samples must be positive")
    if prediction_horizon_s <= 0.0 or not math.isfinite(float(prediction_horizon_s)):
        raise ValueError("prediction_horizon_s must be positive and finite")

    scenario_obj = _scenario_instance(scenario, seed=seed)
    grid_spec = grid or CostMapSpec()
    planner_cfg = _scenario_planner_config(
        planner_config or default_unified_planner_config(),
        scenario_obj,
    )
    frame = _representative_frame(scenario_obj, frame_idx)
    ego = _clean_ego_at_frame(
        scenario_obj,
        frame,
        grid=grid_spec,
        planner_config=planner_cfg,
        use_prediction=use_prediction,
        prediction_horizon_s=prediction_horizon_s,
        prediction_samples=prediction_samples,
    )
    objects = scenario_obj.object_set(frame)
    target_actor = scenario_obj.primary_actor_id
    target_idx = _target_index(objects, target_actor)
    direction = _away_from_ego_direction(objects, target_idx, ego)

    clean_components = _build_components(
        objects,
        ego,
        grid_spec,
        planner_cfg,
        use_prediction=use_prediction,
        prediction_horizon_s=prediction_horizon_s,
        prediction_samples=prediction_samples,
        dt=scenario_obj.dt,
    )
    clean_grid = clean_components.combined

    sweep_rows: list[dict[str, float]] = []
    for eps in eps_sweep:
        eps_value = _positive_finite_float(eps, "eps")
        shifted = _shift_target_position(objects, target_idx, direction, eps_value)
        shifted_components = _build_components(
            shifted,
            ego,
            grid_spec,
            planner_cfg,
            use_prediction=use_prediction,
            prediction_horizon_s=prediction_horizon_s,
            prediction_samples=prediction_samples,
            dt=scenario_obj.dt,
        )
        cost_delta_l2 = _rms_delta(clean_grid, shifted_components.combined)
        sweep_rows.append(
            {
                "eps_m": float(eps_value),
                "cost_delta_l2": float(cost_delta_l2),
                "j_raster": float(cost_delta_l2 / eps_value),
            }
        )

    eps_array = np.asarray([row["eps_m"] for row in sweep_rows], dtype=np.float64)
    delta_array = np.asarray([row["cost_delta_l2"] for row in sweep_rows], dtype=np.float64)
    jacobians = np.asarray([row["j_raster"] for row in sweep_rows], dtype=np.float64)
    j_mean = float(np.mean(jacobians))
    j_std = float(np.std(jacobians, ddof=0))
    cv = float(j_std / abs(j_mean)) if abs(j_mean) > 1e-15 else 0.0
    spread = (
        float((np.max(jacobians) - np.min(jacobians)) / abs(j_mean))
        if abs(j_mean) > 1e-15
        else 0.0
    )
    slope = _slope_through_origin(eps_array, delta_array)

    return {
        "scenario_id": scenario_obj.scenario_id,
        "interface": "prediction__cost" if use_prediction else "object__cost",
        "use_prediction": bool(use_prediction),
        "magnitude": str(magnitude),
        "seed": int(seed),
        "frame": int(frame),
        "target_actor": target_actor,
        "target_direction_x": float(direction[0]),
        "target_direction_y": float(direction[1]),
        "n_eps": int(len(sweep_rows)),
        "eps_min_m": float(np.min(eps_array)),
        "eps_max_m": float(np.max(eps_array)),
        "j_raster": j_mean,
        "j_raster_std": j_std,
        "j_raster_min": float(np.min(jacobians)),
        "j_raster_max": float(np.max(jacobians)),
        "slope_through_origin": slope,
        "linearity_cv": cv,
        "linearity_max_rel_spread": spread,
        "linearity_r2": _linearity_r2(eps_array, delta_array),
        "sweep": tuple(sweep_rows),
    }


def decision_margin_analysis(
    records_or_configs: Any,
    *,
    eps: float = EPS_MARGIN,
    fault_active_only: bool = True,
) -> dict[str, Any]:
    """Quantify argmin switching near low clean planner decision margin."""

    eps_value = _positive_finite_float(eps, "eps")
    results, direct_rows = _coerce_results_and_rows(records_or_configs)

    per_frame: list[dict[str, Any]] = []
    for result in results:
        cfg = getattr(result, "config", None)
        identity = _result_identity(cfg)
        for row in getattr(result, "trace"):
            if fault_active_only and not bool(row.get("fault_active", False)):
                continue
            per_frame.append(
                _decision_margin_row(
                    row,
                    identity=identity,
                    eps=eps_value,
                )
            )

    for row in direct_rows:
        if fault_active_only and not bool(row.get("fault_active", False)):
            continue
        per_frame.append(
            _decision_margin_row(
                row,
                identity={},
                eps=eps_value,
            )
        )

    quartiles = _quartile_summary(per_frame)
    return {
        "per_frame": per_frame,
        "quartile_summary": quartiles,
        "flip_rates": _flip_rates_from_quartiles(quartiles),
        "correlations": _decision_correlations(per_frame, eps=eps_value),
    }


def prediction_jacobian(
    scenario: str | SyntheticScenario,
    *,
    horizon_s: float,
    eps: float,
    seed: int = 0,
    frame_idx: int | None = 0,
    dt: float | None = None,
    velocity_direction: tuple[float, float] = (1.0, 0.0),
) -> dict[str, Any]:
    """Finite-difference CV prediction sensitivity to target velocity."""

    horizon = _positive_finite_float(horizon_s, "horizon_s")
    eps_value = _positive_finite_float(eps, "eps")
    scenario_obj = _scenario_instance(scenario, seed=seed)
    dt_value = float(scenario_obj.dt if dt is None else dt)
    if dt_value <= 0.0 or not math.isfinite(dt_value):
        raise ValueError("dt must be positive and finite")

    frame = _representative_frame(scenario_obj, frame_idx)
    objects = scenario_obj.object_set(frame)
    target_actor = scenario_obj.primary_actor_id
    target_idx = _target_index(objects, target_actor)
    direction = _unit_vector(np.asarray(velocity_direction, dtype=np.float64), name="velocity_direction")

    clean = constant_velocity_predict(objects, horizon_s=horizon, dt=dt_value)
    perturbed_objects = _shift_target_velocity(objects, target_idx, direction, eps_value)
    perturbed = constant_velocity_predict(perturbed_objects, horizon_s=horizon, dt=dt_value)
    clean_traj = clean.by_track_id(target_actor)
    perturbed_traj = perturbed.by_track_id(target_actor)
    if clean_traj is None or perturbed_traj is None:
        raise ValueError(f"target actor {target_actor!r} not found in prediction")

    delta_xy = perturbed_traj.xy - clean_traj.xy
    trajectory_l2 = float(np.mean(np.linalg.norm(delta_xy, axis=1)))
    empirical = float(trajectory_l2 / eps_value)
    analytic = float(np.mean(clean.times_s))
    abs_error = abs(empirical - analytic)
    rel_error = abs_error / max(abs(analytic), 1e-15)
    return {
        "scenario_id": scenario_obj.scenario_id,
        "seed": int(seed),
        "frame": int(frame),
        "target_actor": target_actor,
        "horizon_s": horizon,
        "dt": dt_value,
        "eps_mps": eps_value,
        "n_times": int(clean.times_s.size),
        "sample_time_mean_s": analytic,
        "j_pred_analytic": analytic,
        "j_pred_empirical": empirical,
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "trajectory_l2_delta_m": trajectory_l2,
        "velocity_direction_x": float(direction[0]),
        "velocity_direction_y": float(direction[1]),
    }


def _scenario_instance(scenario: str | SyntheticScenario, *, seed: int) -> SyntheticScenario:
    if isinstance(scenario, SyntheticScenario):
        return scenario
    return get_scenario(str(scenario), seed=int(seed))


def _scenario_planner_config(
    config: CostMapPlannerConfig,
    scenario: SyntheticScenario,
) -> CostMapPlannerConfig:
    if config.target_speed_mps is not None:
        return config
    return replace(config, target_speed_mps=scenario.ego_initial.speed)


def _representative_frame(
    scenario: SyntheticScenario,
    frame_idx: int | None,
) -> int:
    requested = DEFAULT_REPRESENTATIVE_FRAME if frame_idx is None else int(frame_idx)
    return int(np.clip(requested, 0, scenario.steps - 1))


def _clean_ego_at_frame(
    scenario: SyntheticScenario,
    frame_idx: int,
    *,
    grid: CostMapSpec,
    planner_config: CostMapPlannerConfig,
    use_prediction: bool,
    prediction_horizon_s: float,
    prediction_samples: int,
) -> EgoState:
    ego = scenario.ego_initial
    planner = CostMapPlanner(planner_config)
    for frame in range(int(frame_idx)):
        objects = scenario.object_set(frame)
        components = _build_components(
            objects,
            ego,
            grid,
            planner_config,
            use_prediction=use_prediction,
            prediction_horizon_s=prediction_horizon_s,
            prediction_samples=prediction_samples,
            dt=scenario.dt,
        )
        plan = planner.plan(ego, components.combined, grid)
        ego = step_ego_on_costmap_path(ego, plan, dt=scenario.dt)
    return ego


def _build_components(
    objects: ObjectSet,
    ego: EgoState,
    grid: CostMapSpec,
    planner_config: CostMapPlannerConfig,
    *,
    use_prediction: bool,
    prediction_horizon_s: float,
    prediction_samples: int,
    dt: float,
) -> CostMapComponents:
    if use_prediction:
        return build_predicted_cost_grid_components(
            objects,
            ego,
            grid,
            planner_config,
            horizon_s=prediction_horizon_s,
            dt=dt,
            samples=prediction_samples,
        )
    return build_cost_grid_components(objects, ego, grid, planner_config)


def _target_index(objects: ObjectSet, target_actor: Any) -> int:
    matches = np.flatnonzero(objects.track_id == target_actor)
    if matches.size != 1:
        raise ValueError(f"target actor {target_actor!r} not found exactly once")
    return int(matches[0])


def _away_from_ego_direction(
    objects: ObjectSet,
    target_idx: int,
    ego: EgoState,
) -> NDArray[np.float64]:
    vector = np.asarray(
        [float(objects.x[target_idx] - ego.x), float(objects.y[target_idx] - ego.y)],
        dtype=np.float64,
    )
    return _unit_vector(vector, name="away_from_ego", fallback=np.array([1.0, 0.0]))


def _unit_vector(
    vector: NDArray[np.float64],
    *,
    name: str,
    fallback: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-15:
        if fallback is None:
            raise ValueError(f"{name} must be non-zero")
        return _unit_vector(fallback, name=f"{name}_fallback", fallback=None)
    return arr / norm


def _shift_target_position(
    objects: ObjectSet,
    target_idx: int,
    direction: NDArray[np.float64],
    eps: float,
) -> ObjectSet:
    x = objects.x.copy()
    y = objects.y.copy()
    x[target_idx] = x[target_idx] + eps * float(direction[0])
    y[target_idx] = y[target_idx] + eps * float(direction[1])
    return objects.replace(x=x, y=y)


def _shift_target_velocity(
    objects: ObjectSet,
    target_idx: int,
    direction: NDArray[np.float64],
    eps: float,
) -> ObjectSet:
    velocity = objects.v.copy()
    velocity[target_idx, :] = velocity[target_idx, :] + eps * direction
    return objects.replace(v=velocity)


def _rms_delta(clean: NDArray[np.float64], perturbed: NDArray[np.float64]) -> float:
    delta = np.asarray(perturbed, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _validate_eps_sweep(eps_sweep: Sequence[float]) -> None:
    if not eps_sweep:
        raise ValueError("eps_sweep must contain at least one epsilon")
    for eps in eps_sweep:
        _positive_finite_float(eps, "eps")


def _positive_finite_float(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or not math.isfinite(out):
        raise ValueError(f"{name} must be positive and finite")
    return out


def _slope_through_origin(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _linearity_r2(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    slope = _slope_through_origin(x, y)
    fitted = slope * x
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1e-30:
        return 1.0 if residual <= 1e-30 else 0.0
    return float(1.0 - residual / total)


def _coerce_results_and_rows(records_or_configs: Any) -> tuple[list[Any], list[Mapping[str, Any]]]:
    if records_or_configs is None:
        return [], []
    items = _as_items(records_or_configs)
    results: list[Any] = []
    rows: list[Mapping[str, Any]] = []
    for item in items:
        if hasattr(item, "trace"):
            results.append(item)
            continue
        if isinstance(item, UnifiedPipelineConfig):
            results.append(run_unified_pipeline(item))
            continue
        if isinstance(item, Mapping):
            if "chosen_path" in item and "clean_reference_path" in item:
                rows.append(item)
            else:
                results.append(run_unified_pipeline(UnifiedPipelineConfig(**dict(item))))
            continue
        raise TypeError(
            "records_or_configs must contain UnifiedRunResult, UnifiedPipelineConfig, "
            "config mappings, or trace-row mappings"
        )
    return results, rows


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, (UnifiedPipelineConfig, UnifiedRunResult, Mapping)) or hasattr(
        value,
        "trace",
    ):
        return [value]
    if isinstance(value, (str, bytes)):
        raise TypeError("records_or_configs cannot be a string")
    return list(value)


def _result_identity(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    return {
        "scenario_id": str(getattr(cfg, "scenario", "")),
        "injection_point": str(getattr(cfg, "injection_key", getattr(cfg, "injection_point", ""))),
        "method": str(getattr(cfg, "method_key", getattr(cfg, "method", ""))),
        "magnitude": str(getattr(cfg, "magnitude_key", getattr(cfg, "magnitude", ""))),
        "seed": int(getattr(cfg, "seed", 0)),
    }


def _decision_margin_row(
    row: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    eps: float,
) -> dict[str, Any]:
    clean = _mapping(row["clean_reference_path"], name="clean_reference_path")
    chosen = _mapping(row["chosen_path"], name="chosen_path")
    margin = _decision_margin(clean)
    clean_target = float(clean["target_lateral_m"])
    fault_target = float(chosen["target_lateral_m"])
    argmin_defined = bool(margin["argmin_defined"])
    argmin_flip = (
        argmin_defined
        and not math.isclose(
            clean_target,
            fault_target,
            rel_tol=0.0,
            abs_tol=TARGET_LATERAL_ATOL_M,
        )
    )

    out = {
        "scenario_id": identity.get("scenario_id", row.get("scenario_id", "")),
        "injection_point": identity.get("injection_point", row.get("injection_point", "")),
        "method": identity.get("method", row.get("method", "")),
        "magnitude": identity.get("magnitude", row.get("magnitude", "")),
        "seed": identity.get("seed", row.get("seed", "")),
        "frame": int(row.get("frame", -1)),
        "time_s": float(row.get("time_s", float("nan"))),
        "fault_active": bool(row.get("fault_active", False)),
        "decision_margin_score": margin["decision_margin_score"],
        "inverse_margin": (
            float(1.0 / (margin["decision_margin_score"] + eps))
            if argmin_defined
            else float("nan")
        ),
        "argmin_defined": bool(argmin_defined),
        "decision_margin_pool": margin["decision_margin_pool"],
        "fallback_all_candidates": margin["fallback_all_candidates"],
        "n_candidates": margin["n_candidates"],
        "n_feasible": margin["n_feasible"],
        "clean_target_lateral_m": clean_target,
        "fault_target_lateral_m": fault_target,
        "argmin_flip": bool(argmin_flip),
        "realized_path_deviation_m": float(row.get("realized_path_deviation_m", 0.0)),
    }
    return out


def decision_margin(clean_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Cost margin between the planner's best and second-best candidate.

    Public entry point so callers outside the synthetic harness -- notably the
    nuPlan runner -- can log the same margin the M2 analysis uses, which is what
    lets the gateway-collapse hypothesis be tested on real data.
    """

    return _decision_margin(clean_plan)


def _decision_margin(clean_plan: Mapping[str, Any]) -> dict[str, Any]:
    alternatives = list(clean_plan.get("alternatives", ()))
    if len(alternatives) < 2:
        return {
            "decision_margin_score": float("nan"),
            "argmin_defined": False,
            "decision_margin_pool": "undefined",
            "fallback_all_candidates": False,
            "n_candidates": int(len(alternatives)),
            "n_feasible": int(sum(bool(alt.get("collision_free", False)) for alt in alternatives)),
        }
    feasible = [alt for alt in alternatives if bool(alt.get("collision_free", False))]
    pool = feasible if len(feasible) >= 2 else alternatives
    fallback = len(feasible) < 2
    scores = sorted(float(alt["score"]) for alt in pool)
    if len(scores) < 2:
        return {
            "decision_margin_score": float("nan"),
            "argmin_defined": False,
            "decision_margin_pool": "undefined",
            "fallback_all_candidates": bool(fallback),
            "n_candidates": int(len(alternatives)),
            "n_feasible": int(len(feasible)),
        }
    margin = float(scores[1] - scores[0])
    if margin < -1e-12:
        raise ValueError("decision margin must be non-negative after sorting")
    return {
        "decision_margin_score": float(max(0.0, margin)),
        "argmin_defined": True,
        "decision_margin_pool": "all" if fallback else "feasible",
        "fallback_all_candidates": bool(fallback),
        "n_candidates": int(len(alternatives)),
        "n_feasible": int(len(feasible)),
    }


def _quartile_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        row
        for row in rows
        if _is_finite_number(row.get("decision_margin_score"))
        and _is_finite_number(row.get("realized_path_deviation_m"))
    ]
    if not valid:
        return []
    ordered = sorted(
        valid,
        key=lambda row: (
            float(row["decision_margin_score"]),
            str(row.get("scenario_id", "")),
            int(row.get("seed", 0) or 0),
            int(row.get("frame", 0) or 0),
        ),
    )
    groups = [group.tolist() for group in np.array_split(np.asarray(ordered, dtype=object), 4)]
    out: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        if not group:
            continue
        margins = np.asarray([float(row["decision_margin_score"]) for row in group], dtype=np.float64)
        deviations = np.asarray([float(row["realized_path_deviation_m"]) for row in group], dtype=np.float64)
        flips = np.asarray([float(bool(row["argmin_flip"])) for row in group], dtype=np.float64)
        out.append(
            {
                "quartile": f"Q{index}",
                "margin_rank": "lowest" if index == 1 else "highest" if index == 4 else "middle",
                "n_frames": int(len(group)),
                "margin_min": float(np.min(margins)),
                "margin_max": float(np.max(margins)),
                "mean_margin": float(np.mean(margins)),
                "mean_inverse_margin": float(np.mean(1.0 / (margins + EPS_MARGIN))),
                "mean_realized_path_deviation_m": float(np.mean(deviations)),
                "argmin_flip_rate": float(np.mean(flips)),
            }
        )
    return out


def _flip_rates_from_quartiles(quartiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not quartiles:
        return {
            "small_margin_n": 0,
            "small_margin_flip_rate": None,
            "large_margin_n": 0,
            "large_margin_flip_rate": None,
            "small_margin_mean_deviation_m": None,
            "large_margin_mean_deviation_m": None,
        }
    small = quartiles[0]
    large = quartiles[-1]
    return {
        "small_margin_n": int(small["n_frames"]),
        "small_margin_flip_rate": float(small["argmin_flip_rate"]),
        "large_margin_n": int(large["n_frames"]),
        "large_margin_flip_rate": float(large["argmin_flip_rate"]),
        "small_margin_mean_deviation_m": float(small["mean_realized_path_deviation_m"]),
        "large_margin_mean_deviation_m": float(large["mean_realized_path_deviation_m"]),
    }


def _decision_correlations(
    rows: Sequence[Mapping[str, Any]],
    *,
    eps: float,
) -> dict[str, Any]:
    margins: list[float] = []
    deviations: list[float] = []
    for row in rows:
        if _is_finite_number(row.get("decision_margin_score")) and _is_finite_number(
            row.get("realized_path_deviation_m")
        ):
            margins.append(float(row["decision_margin_score"]))
            deviations.append(float(row["realized_path_deviation_m"]))
    if not margins:
        return {"n": 0, "pearson": None, "spearman": None}
    margin_array = np.asarray(margins, dtype=np.float64)
    deviation_array = np.asarray(deviations, dtype=np.float64)
    inverse_margin = 1.0 / (margin_array + eps)
    return {
        "n": int(margin_array.size),
        "pearson": _pearson(inverse_margin, deviation_array),
        "spearman": _spearman(inverse_margin, deviation_array),
    }


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    if x.size != y.size:
        raise ValueError("correlation arrays must have equal length")
    if x.size < 2:
        return None
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= 1e-15:
        return None
    return float(np.dot(x_centered, y_centered) / denom)


def _spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    if x.size != y.size:
        raise ValueError("correlation arrays must have equal length")
    if x.size < 2:
        return None
    return _pearson(_rank_average(x), _rank_average(y))


def _rank_average(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


__all__ = [
    "DEFAULT_EPS_SWEEP",
    "decision_margin",
    "decision_margin_analysis",
    "prediction_jacobian",
    "rasterization_jacobian",
]
