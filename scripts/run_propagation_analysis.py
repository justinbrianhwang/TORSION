"""Run Phase A cross-representation propagation analysis on unified traces."""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torsion.analysis.propagation import (  # noqa: E402
    EPS_RATIO,
    censor_ttc,
    compute_propagation_metrics,
    critical_interface_score,
)
from torsion.metrics.statistics import bootstrap_ci  # noqa: E402
from torsion.scenarios.unified_pipeline import (  # noqa: E402
    INJECTION_POINTS,
    MAGNITUDES,
    SCENARIOS,
    UnifiedPipelineConfig,
    run_unified_pipeline,
)

DEFAULT_METHODS = ("torsion_displace", "random_warp", "gaussian")
PREDICTION_METHODS = ("torsion_displace", "random_warp")
DEFAULT_SEEDS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
PREDICTION_INJECTION_POINTS = ("object", "prediction", "costmap")
INTERFACE_GAIN_COLUMNS = (
    "object__cost_gain",
    "cost__plan_gain",
    "plan__control_gain",
    "control__safety_gain",
)
PREDICTION_INTERFACE_GAIN_COLUMNS = (
    "object__prediction_gain",
    "prediction__cost_gain",
    "cost__plan_gain",
    "plan__control_gain",
    "control__safety_gain",
)
STAGE_COLUMNS = (
    "scenario_id",
    "injection_point",
    "method",
    "magnitude",
    "seed",
    "status",
    "error",
    "clean_min_ttc_censored",
    "fault_min_ttc_censored",
    "clean_collision",
    "fault_collision",
    "raw_object_shift_m",
    "raw_prediction_l2",
    "raw_cost_l2",
    "raw_plan_dev_m",
    "raw_control_dev",
    "raw_safety_drop_s",
    "raw_collision_delta",
    "scale_object_shift_m",
    "scale_prediction_l2",
    "scale_cost_l2",
    "scale_plan_dev_m",
    "scale_control_dev",
    "scale_safety_drop_s",
    "scale_collision_delta",
    "n_object",
    "n_prediction",
    "n_cost",
    "n_plan",
    "n_control",
    "n_safety",
    "n_collision_delta",
    "object__cost_gain",
    "object__prediction_gain",
    "prediction__cost_gain",
    "cost__plan_gain",
    "plan__control_gain",
    "control__safety_gain",
    "far",
    "far_source",
    "propagation_depth",
    "deepest_stage",
    "reached_safety",
    "recovery_time_s",
    "recovered",
)

LEGACY_STAGE_COLUMNS = tuple(
    column
    for column in STAGE_COLUMNS
    if column
    not in {
        "raw_prediction_l2",
        "scale_prediction_l2",
        "n_prediction",
        "object__prediction_gain",
        "prediction__cost_gain",
        "reached_safety",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated scenarios",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated unified fault methods",
    )
    parser.add_argument(
        "--magnitudes",
        default=",".join(MAGNITUDES),
        help="comma-separated magnitudes",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--use-prediction", action="store_true")
    parser.add_argument("--prediction-horizon", type=float, default=2.0)
    parser.add_argument("--prediction-samples", type=int, default=5)
    args = parser.parse_args()

    scenarios = _parse_choices(args.scenarios, choices=SCENARIOS, name="scenario")
    default_methods = PREDICTION_METHODS if args.use_prediction else DEFAULT_METHODS
    methods = _parse_methods(
        ",".join(default_methods) if args.methods is None else args.methods,
        allow_gaussian=not args.use_prediction or args.methods is not None,
    )
    magnitudes = _parse_choices(args.magnitudes, choices=MAGNITUDES, name="magnitude")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.prediction_samples <= 0:
        raise ValueError("--prediction-samples must be positive")
    if args.prediction_horizon <= 0.0 or not math.isfinite(float(args.prediction_horizon)):
        raise ValueError("--prediction-horizon must be positive and finite")

    injection_points = (
        PREDICTION_INJECTION_POINTS if args.use_prediction else INJECTION_POINTS
    )
    jobs = [
        (
            scenario,
            magnitude,
            seed,
            tuple(methods),
            tuple(injection_points),
            bool(args.use_prediction),
            float(args.prediction_horizon),
            int(args.prediction_samples),
        )
        for scenario in scenarios
        for magnitude in magnitudes
        for seed in range(int(args.seeds))
    ]
    stage_rows = _run_jobs(jobs, workers=int(args.workers))
    stage_rows = _sort_stage_rows(
        stage_rows,
        injection_points=tuple(injection_points),
        methods=tuple(methods),
    )
    gain_columns = (
        PREDICTION_INTERFACE_GAIN_COLUMNS if args.use_prediction else INTERFACE_GAIN_COLUMNS
    )
    stage_columns = STAGE_COLUMNS if args.use_prediction else LEGACY_STAGE_COLUMNS

    propagation_map = _summarize_propagation_map(
        stage_rows,
        scenarios=scenarios,
        gain_columns=gain_columns,
        include_reach_safety=bool(args.use_prediction),
        n_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    propagation_cis = _summarize_cis(
        stage_rows,
        scenarios=scenarios,
        n_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "propagation_pred" if args.use_prediction else "propagation"
    stage_path = args.out_dir / f"{prefix}_stage_errors.csv"
    map_path = args.out_dir / f"{prefix}_map.csv"
    cis_path = args.out_dir / f"{prefix}_cis.csv"
    _write_csv(stage_path, stage_rows, fieldnames=stage_columns)
    _write_csv(
        map_path,
        propagation_map,
        fieldnames=_map_columns(
            gain_columns=gain_columns,
            include_reach_safety=bool(args.use_prediction),
        ),
    )
    _write_csv(cis_path, propagation_cis, fieldnames=_cis_columns())

    _print_summary_table(
        propagation_map,
        propagation_cis,
        use_prediction=bool(args.use_prediction),
    )
    if args.use_prediction:
        horizon_rows = _run_prediction_horizon_sweep(
            workers=int(args.workers),
            prediction_samples=int(args.prediction_samples),
            n_resamples=int(args.bootstrap_resamples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        horizon_path = args.out_dir / "propagation_pred_horizon.csv"
        _write_csv(horizon_path, horizon_rows, fieldnames=_horizon_columns())
        _print_horizon_table(horizon_rows)
        print(f"Wrote {len(horizon_rows)} horizon rows to {horizon_path}")
    print(f"\nWrote {len(stage_rows)} propagation rows to {stage_path}")
    print(f"Wrote {len(propagation_map)} propagation-map rows to {map_path}")
    print(f"Wrote {len(propagation_cis)} CIS rows to {cis_path}")


def _run_jobs(
    jobs: list[
        tuple[str, str, int, tuple[str, ...], tuple[str, ...], bool, float, int]
    ],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        rows: list[dict[str, Any]] = []
        for job in jobs:
            rows.extend(_run_one_clean_group(job))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        nested = list(pool.map(_run_one_clean_group, jobs))
    return [row for group in nested for row in group]


def _run_one_clean_group(
    job: tuple[str, str, int, tuple[str, ...], tuple[str, ...], bool, float, int],
) -> list[dict[str, Any]]:
    (
        scenario,
        magnitude,
        seed,
        methods,
        injection_points,
        use_prediction,
        prediction_horizon,
        prediction_samples,
    ) = job
    clean = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario=scenario,
            injection_point="none",
            method="clean",
            magnitude=magnitude,
            seed=seed,
            trace_grids=False,
            use_prediction=use_prediction,
            prediction_horizon_s=prediction_horizon,
            prediction_samples=prediction_samples,
        )
    )
    rows: list[dict[str, Any]] = []
    for injection_point in injection_points:
        for method in methods:
            fault_config = UnifiedPipelineConfig(
                scenario=scenario,
                injection_point=injection_point,  # type: ignore[arg-type]
                method=method,  # type: ignore[arg-type]
                magnitude=magnitude,
                seed=seed,
                trace_grids=False,
                use_prediction=use_prediction,
                prediction_horizon_s=prediction_horizon,
                prediction_samples=prediction_samples,
            )
            try:
                fault = run_unified_pipeline(fault_config)
            except Exception as exc:
                record = _failed_stage_record(
                    scenario=scenario,
                    injection_point=injection_point,
                    method=method,
                    magnitude=magnitude,
                    seed=seed,
                    clean=clean,
                    exc=exc,
                )
            else:
                record = compute_propagation_metrics(fault, clean)
                record["status"] = "completed"
                record["error"] = ""
                record["clean_min_ttc_censored"] = _min_ttc_censored(clean)
                record["fault_min_ttc_censored"] = _min_ttc_censored(fault)
                record["clean_collision"] = bool(clean.summary.get("collision", False))
                record["fault_collision"] = bool(fault.summary.get("collision", False))
            rows.append(record)
    return rows


def _failed_stage_record(
    *,
    scenario: str,
    injection_point: str,
    method: str,
    magnitude: str,
    seed: int,
    clean: Any,
    exc: Exception,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "scenario_id": scenario,
        "injection_point": injection_point,
        "method": method,
        "magnitude": magnitude,
        "seed": seed,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "clean_min_ttc_censored": _min_ttc_censored(clean),
        "fault_min_ttc_censored": None,
        "clean_collision": bool(clean.summary.get("collision", False)),
        "fault_collision": None,
    }
    for column in STAGE_COLUMNS:
        record.setdefault(column, None)
    return record


def _summarize_propagation_map(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    gain_columns: tuple[str, ...],
    include_reach_safety: bool,
    n_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in (*scenarios, "ALL"):
        scoped = rows if scope == "ALL" else [row for row in rows if row["scenario_id"] == scope]
        for key in _group_keys(scoped, ("injection_point", "method", "magnitude")):
            group = [
                row
                for row in scoped
                if all(
                    row[column] == value
                    for column, value in zip(("injection_point", "method", "magnitude"), key, strict=True)
                )
            ]
            record: dict[str, Any] = {
                "scenario_id": scope,
                "injection_point": key[0],
                "method": key[1],
                "magnitude": key[2],
                "n_runs": len(group),
                "n_valid": sum(row.get("status") == "completed" for row in group),
            }
            for column in gain_columns:
                _add_optional_mean_ci(
                    record,
                    group,
                    column,
                    out_prefix=column,
                    n_resamples=n_resamples,
                    bootstrap_seed=bootstrap_seed,
                )
            for column in ("far", "propagation_depth", "recovery_time_s"):
                _add_optional_mean_ci(
                    record,
                    group,
                    column,
                    out_prefix=column,
                    n_resamples=n_resamples,
                    bootstrap_seed=bootstrap_seed,
                )
            if include_reach_safety:
                reach_values = [
                    float(row["reached_safety"])
                    for row in group
                    if row.get("status") == "completed"
                    and row.get("reached_safety") is not None
                ]
                record["reach_safety_rate"] = (
                    float(np.mean(reach_values)) if reach_values else None
                )
            recovered_values = [
                float(row["recovered"])
                for row in group
                if row.get("status") == "completed" and row.get("recovered") is not None
            ]
            record["recovered_rate"] = (
                float(np.mean(recovered_values)) if recovered_values else None
            )
            out.append(record)
    return _sort_summary_rows(out, scenarios=scenarios)


def _summarize_cis(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    n_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in (*scenarios, "ALL"):
        scoped = rows if scope == "ALL" else [row for row in rows if row["scenario_id"] == scope]
        for key in _group_keys(scoped, ("injection_point", "method", "magnitude")):
            group = [
                row
                for row in scoped
                if all(
                    row[column] == value
                    for column, value in zip(("injection_point", "method", "magnitude"), key, strict=True)
                )
            ]
            valid_group = [
                row
                for row in group
                if row.get("status") == "completed"
                and _is_finite_number(row.get("raw_safety_drop_s"))
                and _is_finite_number(row.get("raw_plan_dev_m"))
            ]
            if not valid_group:
                out.append(
                    {
                        "scenario_id": scope,
                        "injection_point": key[0],
                        "method": key[1],
                        "magnitude": key[2],
                        "n_runs": len(group),
                        "n_valid": 0,
                        "cis": None,
                        "cis_ci_low": None,
                        "cis_ci_high": None,
                        "mean_matched_plan_budget_m": None,
                        "mean_matched_plan_budget_ci_low": None,
                        "mean_matched_plan_budget_ci_high": None,
                        "mean_paired_ttc_drop_s": None,
                        "mean_paired_ttc_drop_ci_low": None,
                        "mean_paired_ttc_drop_ci_high": None,
                    }
                )
                continue
            drops = np.asarray(
                [max(0.0, float(row["raw_safety_drop_s"])) for row in valid_group],
                dtype=np.float64,
            )
            budgets = np.asarray(
                [float(row["raw_plan_dev_m"]) for row in valid_group],
                dtype=np.float64,
            )
            cis = critical_interface_score(drops, budgets)
            cis_low, cis_high = _bootstrap_cis(
                drops,
                budgets,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            drop_low, drop_high = bootstrap_ci(
                drops,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            budget_low, budget_high = bootstrap_ci(
                budgets,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            out.append(
                {
                    "scenario_id": scope,
                    "injection_point": key[0],
                    "method": key[1],
                    "magnitude": key[2],
                    "n_runs": len(group),
                    "n_valid": len(valid_group),
                    "cis": cis,
                    "cis_ci_low": cis_low,
                    "cis_ci_high": cis_high,
                    "mean_matched_plan_budget_m": float(np.mean(budgets)),
                    "mean_matched_plan_budget_ci_low": budget_low,
                    "mean_matched_plan_budget_ci_high": budget_high,
                    "mean_paired_ttc_drop_s": float(np.mean(drops)),
                    "mean_paired_ttc_drop_ci_low": drop_low,
                    "mean_paired_ttc_drop_ci_high": drop_high,
                }
            )
    return _sort_summary_rows(out, scenarios=scenarios)


def _run_prediction_horizon_sweep(
    *,
    workers: int,
    prediction_samples: int,
    n_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_s in (1.0, 2.0, 3.0):
        jobs = [
            (
                "leading_vehicle",
                "high",
                seed,
                ("torsion_displace",),
                ("prediction",),
                True,
                horizon_s,
                prediction_samples,
            )
            for seed in range(15)
        ]
        stage_rows = _run_jobs(jobs, workers=workers)
        valid_rows = [
            row
            for row in stage_rows
            if row.get("status") == "completed"
            and _is_finite_number(row.get("raw_safety_drop_s"))
            and _is_finite_number(row.get("raw_plan_dev_m"))
        ]
        record: dict[str, Any] = {
            "scenario_id": "leading_vehicle",
            "injection_point": "prediction",
            "method": "torsion_displace",
            "magnitude": "high",
            "prediction_horizon_s": horizon_s,
            "n_runs": len(stage_rows),
            "n_valid": len(valid_rows),
        }
        _add_optional_mean_ci(
            record,
            valid_rows,
            "raw_prediction_l2",
            out_prefix="raw_prediction_l2",
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        _add_optional_mean_ci(
            record,
            valid_rows,
            "prediction__cost_gain",
            out_prefix="prediction__cost_gain",
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        if valid_rows:
            drops = np.asarray(
                [max(0.0, float(row["raw_safety_drop_s"])) for row in valid_rows],
                dtype=np.float64,
            )
            budgets = np.asarray(
                [float(row["raw_plan_dev_m"]) for row in valid_rows],
                dtype=np.float64,
            )
            cis = critical_interface_score(drops, budgets)
            cis_low, cis_high = _bootstrap_cis(
                drops,
                budgets,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            record["cis"] = cis
            record["cis_ci_low"] = cis_low
            record["cis_ci_high"] = cis_high
            record["mean_matched_plan_budget_m"] = float(np.mean(budgets))
            record["mean_paired_ttc_drop_s"] = float(np.mean(drops))
        else:
            record["cis"] = None
            record["cis_ci_low"] = None
            record["cis_ci_high"] = None
            record["mean_matched_plan_budget_m"] = None
            record["mean_paired_ttc_drop_s"] = None
        rows.append(record)
    return rows


def _add_optional_mean_ci(
    record: dict[str, Any],
    group: list[dict[str, Any]],
    column: str,
    *,
    out_prefix: str,
    n_resamples: int,
    bootstrap_seed: int,
) -> None:
    values = [
        float(row[column])
        for row in group
        if _is_finite_number(row.get(column))
    ]
    record[f"{out_prefix}_n"] = len(values)
    if not values:
        record[f"{out_prefix}_mean"] = None
        record[f"{out_prefix}_ci_low"] = None
        record[f"{out_prefix}_ci_high"] = None
        return
    low, high = bootstrap_ci(values, n_resamples=n_resamples, seed=bootstrap_seed)
    record[f"{out_prefix}_mean"] = float(np.mean(np.asarray(values, dtype=np.float64)))
    record[f"{out_prefix}_ci_low"] = low
    record[f"{out_prefix}_ci_high"] = high


def _bootstrap_cis(
    drops: np.ndarray,
    budgets: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if drops.size != budgets.size:
        raise ValueError("drops and budgets must have equal length")
    if drops.size == 0 or critical_interface_score(drops, budgets) is None:
        return None, None
    if drops.size == 1:
        value = critical_interface_score(drops, budgets)
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, drops.size, size=(n_resamples, drops.size))
    sampled_drops = drops[indices]
    sampled_budgets = budgets[indices]
    denominator = np.mean(sampled_budgets, axis=1)
    valid = denominator > EPS_RATIO
    if not np.any(valid):
        return None, None
    statistics = np.mean(sampled_drops, axis=1)[valid] / denominator[valid]
    low, high = np.percentile(statistics, [2.5, 97.5])
    return float(low), float(high)


def _print_summary_table(
    propagation_map: list[dict[str, Any]],
    propagation_cis: list[dict[str, Any]],
    *,
    use_prediction: bool = False,
) -> None:
    map_rows = [
        row
        for row in propagation_map
        if row["scenario_id"] == "ALL"
        and row["magnitude"] == "high"
        and row["method"] == "torsion_displace"
    ]
    cis_rows = [
        row
        for row in propagation_cis
        if row["scenario_id"] == "ALL"
        and row["magnitude"] == "high"
        and row["method"] == "torsion_displace"
    ]
    cis_by_injection = {row["injection_point"]: row for row in cis_rows}
    table: list[dict[str, Any]] = []
    for row in map_rows:
        cis = cis_by_injection.get(row["injection_point"], {})
        if use_prediction:
            table.append(
                {
                    "injection": row["injection_point"],
                    "CIS [95% CI]": _format_ci(cis, "cis"),
                    "object__prediction": _format_float(
                        row.get("object__prediction_gain_mean")
                    ),
                    "prediction__cost": _format_float(
                        row.get("prediction__cost_gain_mean")
                    ),
                    "cost__plan": _format_float(row.get("cost__plan_gain_mean")),
                    "plan__control": _format_float(row.get("plan__control_gain_mean")),
                    "control__safety": _format_float(
                        row.get("control__safety_gain_mean")
                    ),
                    "FAR": _format_float(row.get("far_mean")),
                    "reach_safety": _format_float(row.get("reach_safety_rate")),
                }
            )
        else:
            table.append(
                {
                    "injection": row["injection_point"],
                    "CIS [95% CI]": _format_ci(cis, "cis"),
                    "object__cost": _format_float(row.get("object__cost_gain_mean")),
                    "cost__plan": _format_float(row.get("cost__plan_gain_mean")),
                    "plan__control": _format_float(row.get("plan__control_gain_mean")),
                    "control__safety": _format_float(row.get("control__safety_gain_mean")),
                    "FAR": _format_float(row.get("far_mean")),
                    "depth": _format_float(row.get("propagation_depth_mean")),
                    "recovery_s": _format_float(row.get("recovery_time_s_mean")),
                }
            )
    label = "Phase B prediction" if use_prediction else "Phase A"
    print(
        f"\n{label} propagation summary: scenario=ALL, magnitude=high, "
        "method=torsion_displace"
    )
    _print_table(table)


def _parse_choices(value: str, *, choices: Iterable[str], name: str) -> list[str]:
    allowed = tuple(choices)
    parsed = _split_csv(value)
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"unknown {name} value(s) {invalid!r}; expected {', '.join(allowed)}")
    return parsed


def _parse_methods(value: str, *, allow_gaussian: bool = True) -> list[str]:
    allowed = {"torsion_displace", "random_warp"}
    if allow_gaussian:
        allowed.add("gaussian")
    parsed = _split_csv(value)
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"unknown method value(s) {invalid!r}; expected {', '.join(sorted(allowed))}")
    return parsed


def _split_csv(value: str) -> list[str]:
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError("at least one comma-separated value is required")
    return parsed


def _group_keys(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    keys: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _sort_stage_rows(
    rows: list[dict[str, Any]],
    *,
    injection_points: tuple[str, ...],
    methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate(SCENARIOS)}
    injection_order = {name: idx for idx, name in enumerate(injection_points)}
    method_order = {name: idx for idx, name in enumerate(methods)}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order[str(row["scenario_id"])],
            injection_order[str(row["injection_point"])],
            method_order.get(str(row["method"]), len(method_order)),
            magnitude_order[str(row["magnitude"])],
            int(row["seed"]),
        ),
    )


def _sort_summary_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate((*scenarios, "ALL"))}
    injection_order = {name: idx for idx, name in enumerate(PREDICTION_INJECTION_POINTS)}
    method_order = {name: idx for idx, name in enumerate(DEFAULT_METHODS)}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order[str(row["scenario_id"])],
            injection_order[str(row["injection_point"])],
            method_order.get(str(row["method"]), len(method_order)),
            magnitude_order[str(row["magnitude"])],
        ),
    )


def _map_columns(
    *,
    gain_columns: tuple[str, ...] = INTERFACE_GAIN_COLUMNS,
    include_reach_safety: bool = False,
) -> tuple[str, ...]:
    columns = ["scenario_id", "injection_point", "method", "magnitude", "n_runs", "n_valid"]
    for column in gain_columns:
        columns.extend((f"{column}_n", f"{column}_mean", f"{column}_ci_low", f"{column}_ci_high"))
    for column in ("far", "propagation_depth", "recovery_time_s"):
        columns.extend((f"{column}_n", f"{column}_mean", f"{column}_ci_low", f"{column}_ci_high"))
    if include_reach_safety:
        columns.append("reach_safety_rate")
    columns.append("recovered_rate")
    return tuple(columns)


def _cis_columns() -> tuple[str, ...]:
    return (
        "scenario_id",
        "injection_point",
        "method",
        "magnitude",
        "n_runs",
        "n_valid",
        "cis",
        "cis_ci_low",
        "cis_ci_high",
        "mean_matched_plan_budget_m",
        "mean_matched_plan_budget_ci_low",
        "mean_matched_plan_budget_ci_high",
        "mean_paired_ttc_drop_s",
        "mean_paired_ttc_drop_ci_low",
        "mean_paired_ttc_drop_ci_high",
    )


def _horizon_columns() -> tuple[str, ...]:
    return (
        "scenario_id",
        "injection_point",
        "method",
        "magnitude",
        "prediction_horizon_s",
        "n_runs",
        "n_valid",
        "raw_prediction_l2_n",
        "raw_prediction_l2_mean",
        "raw_prediction_l2_ci_low",
        "raw_prediction_l2_ci_high",
        "prediction__cost_gain_n",
        "prediction__cost_gain_mean",
        "prediction__cost_gain_ci_low",
        "prediction__cost_gain_ci_high",
        "cis",
        "cis_ci_low",
        "cis_ci_high",
        "mean_matched_plan_budget_m",
        "mean_paired_ttc_drop_s",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_record(row) for row in rows)


def _csv_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            out[key] = ""
        elif isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = ""
        else:
            out[key] = value
    return out


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _format_float(value: Any) -> str:
    if not _is_finite_number(value):
        return ""
    return f"{float(value):.3f}"


def _format_ci(row: dict[str, Any], metric: str) -> str:
    if not row or not _is_finite_number(row.get(metric)):
        return ""
    return (
        f"{float(row[metric]):.3f} "
        f"[{float(row[f'{metric}_ci_low']):.3f}, "
        f"{float(row[f'{metric}_ci_high']):.3f}]"
    )


def _print_horizon_table(rows: list[dict[str, Any]]) -> None:
    table = [
        {
            "horizon_s": _format_float(row.get("prediction_horizon_s")),
            "raw_prediction_l2": _format_float(row.get("raw_prediction_l2_mean")),
            "prediction__cost": _format_float(row.get("prediction__cost_gain_mean")),
            "CIS [95% CI]": _format_ci(row, "cis"),
            "budget_m": _format_float(row.get("mean_matched_plan_budget_m")),
            "ttc_drop_s": _format_float(row.get("mean_paired_ttc_drop_s")),
        }
        for row in rows
    ]
    print("\nPrediction horizon sweep: leading_vehicle/high/torsion_displace/inject@prediction")
    _print_table(table)


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(none)")
        return
    columns = tuple(rows[0].keys())
    widths = {
        column: max(len(str(column)), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(str(column).ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def _min_ttc_censored(result: Any) -> float:
    return float(
        np.min(
            np.asarray(
                [censor_ttc(row["actual_ttc_s"]) for row in result.trace],
                dtype=np.float64,
            )
        )
    )


if __name__ == "__main__":
    main()
