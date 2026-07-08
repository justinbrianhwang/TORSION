"""Compare sampling and potential-field planners in the unified pipeline."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torsion.analysis.failure_taxonomy import classify_run  # noqa: E402
from torsion.analysis.propagation import (  # noqa: E402
    compute_stage_errors,
    critical_interface_score,
)
from torsion.analysis.transfer_function import (  # noqa: E402
    characterize_linearity,
    interface_raw_gains,
)
from torsion.scenarios.unified_pipeline import (  # noqa: E402
    MAGNITUDES,
    SCENARIOS,
    UnifiedPipelineConfig,
    run_unified_pipeline,
)

METHOD = "torsion_displace"
PLANNER_TYPES = ("sampling", "potential_field")
INJECTION_POINTS = ("object", "prediction", "costmap")
DEFAULT_SEEDS = 20
DEFAULT_BUDGETS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

OUTPUT_COLUMNS = (
    "planner_type",
    "costmap_cis_high",
    "object_cis_high",
    "prediction_cis_high",
    "cost_plan_gain_mean",
    "cost_plan_cv",
    "cost_plan_verdict",
    "planner_gateway_rate",
    "argmin_flip_rate",
    "collision_rate_object",
    "collision_rate_prediction",
    "collision_rate_costmap",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated scenarios",
    )
    parser.add_argument(
        "--magnitudes",
        default=",".join(MAGNITUDES),
        help="comma-separated magnitudes for taxonomy and CIS",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="comma-separated cost->plan transfer-function budgets",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--use-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the object->prediction->cost chain",
    )
    parser.add_argument("--prediction-horizon", type=float, default=2.0)
    parser.add_argument("--prediction-samples", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/planner_independence.csv"),
    )
    args = parser.parse_args()

    scenarios = _parse_choices(args.scenarios, choices=SCENARIOS, name="scenario")
    magnitudes = _parse_choices(args.magnitudes, choices=MAGNITUDES, name="magnitude")
    budgets = _parse_positive_floats(args.budgets, name="budget")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.prediction_samples <= 0:
        raise ValueError("--prediction-samples must be positive")
    if args.prediction_horizon <= 0.0 or not math.isfinite(float(args.prediction_horizon)):
        raise ValueError("--prediction-horizon must be positive and finite")

    injection_points = INJECTION_POINTS if args.use_prediction else ("object", "costmap")
    rows: list[dict[str, Any]] = []
    for planner_type in PLANNER_TYPES:
        print(f"\nRunning planner_type={planner_type}")
        transfer = _run_transfer_summary(
            planner_type=planner_type,
            scenarios=scenarios,
            budgets=tuple(budgets),
            seeds=int(args.seeds),
            injection_points=tuple(injection_points),
            use_prediction=bool(args.use_prediction),
            prediction_horizon=float(args.prediction_horizon),
            prediction_samples=int(args.prediction_samples),
            workers=int(args.workers),
        )
        taxonomy_records = _run_taxonomy_records(
            planner_type=planner_type,
            scenarios=scenarios,
            magnitudes=magnitudes,
            seeds=int(args.seeds),
            injection_points=tuple(injection_points),
            use_prediction=bool(args.use_prediction),
            prediction_horizon=float(args.prediction_horizon),
            prediction_samples=int(args.prediction_samples),
            workers=int(args.workers),
        )
        row = {
            "planner_type": planner_type,
            "costmap_cis_high": _cis_high(taxonomy_records, "costmap"),
            "object_cis_high": _cis_high(taxonomy_records, "object"),
            "prediction_cis_high": _cis_high(taxonomy_records, "prediction"),
            "cost_plan_gain_mean": transfer.get("mean_gain"),
            "cost_plan_cv": transfer.get("gain_cv"),
            "cost_plan_verdict": transfer.get("verdict"),
            "planner_gateway_rate": _planner_gateway_rate(taxonomy_records),
            "argmin_flip_rate": _rate(
                taxonomy_records,
                lambda record: bool(record.get("argmin_flipped", False)),
            ),
            "collision_rate_object": _collision_rate(taxonomy_records, "object"),
            "collision_rate_prediction": _collision_rate(taxonomy_records, "prediction"),
            "collision_rate_costmap": _collision_rate(taxonomy_records, "costmap"),
        }
        rows.append(row)

    _write_csv(args.output, rows, fieldnames=OUTPUT_COLUMNS)
    _print_table(rows)
    _print_verdict(rows)
    print(f"\nWrote {len(rows)} planner-independence rows to {args.output}")


def _run_transfer_summary(
    *,
    planner_type: str,
    scenarios: list[str],
    budgets: tuple[float, ...],
    seeds: int,
    injection_points: tuple[str, ...],
    use_prediction: bool,
    prediction_horizon: float,
    prediction_samples: int,
    workers: int,
) -> dict[str, Any]:
    jobs = [
        (
            planner_type,
            scenario,
            seed,
            tuple(budgets),
            tuple(injection_points),
            bool(use_prediction),
            float(prediction_horizon),
            int(prediction_samples),
        )
        for scenario in scenarios
        for seed in range(seeds)
    ]
    if workers == 1:
        nested = [_run_transfer_clean_group(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested = list(pool.map(_run_transfer_clean_group, jobs))
    rows = [row for group in nested for row in group]

    gain_by_budget: dict[float, float | None] = {}
    for budget in budgets:
        values = [
            float(row["raw_cost__plan"])
            for row in rows
            if row["injection"] == "costmap"
            and math.isclose(float(row["budget"]), float(budget), rel_tol=0.0, abs_tol=1e-12)
            and _is_finite_number(row.get("raw_cost__plan"))
        ]
        gain_by_budget[float(budget)] = (
            float(np.mean(np.asarray(values, dtype=np.float64))) if values else None
        )
    return characterize_linearity(gain_by_budget)


def _run_transfer_clean_group(
    job: tuple[str, str, int, tuple[float, ...], tuple[str, ...], bool, float, int],
) -> list[dict[str, Any]]:
    (
        planner_type,
        scenario,
        seed,
        budgets,
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
            magnitude="medium",
            seed=seed,
            trace_grids=False,
            use_prediction=use_prediction,
            prediction_horizon_s=prediction_horizon,
            prediction_samples=prediction_samples,
            planner_type=planner_type,
        )
    )
    rows: list[dict[str, Any]] = []
    for injection in injection_points:
        for budget in budgets:
            fault = run_unified_pipeline(
                UnifiedPipelineConfig(
                    scenario=scenario,
                    injection_point=injection,  # type: ignore[arg-type]
                    method=METHOD,
                    magnitude="medium",
                    seed=seed,
                    trace_grids=False,
                    use_prediction=use_prediction,
                    prediction_horizon_s=prediction_horizon,
                    prediction_samples=prediction_samples,
                    target_path_budget_m=float(budget),
                    planner_type=planner_type,
                )
            )
            gains = interface_raw_gains(fault, clean)
            rows.append(
                {
                    "planner_type": planner_type,
                    "scenario": scenario,
                    "injection": injection,
                    "budget": float(budget),
                    "seed": int(seed),
                    "raw_cost__plan": gains["cost__plan"],
                }
            )
    return rows


def _run_taxonomy_records(
    *,
    planner_type: str,
    scenarios: list[str],
    magnitudes: list[str],
    seeds: int,
    injection_points: tuple[str, ...],
    use_prediction: bool,
    prediction_horizon: float,
    prediction_samples: int,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        (
            planner_type,
            scenario,
            magnitude,
            seed,
            tuple(injection_points),
            bool(use_prediction),
            float(prediction_horizon),
            int(prediction_samples),
        )
        for scenario in scenarios
        for magnitude in magnitudes
        for seed in range(seeds)
    ]
    if workers == 1:
        nested = [_run_taxonomy_clean_group(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested = list(pool.map(_run_taxonomy_clean_group, jobs))
    rows = [row for group in nested for row in group]
    return sorted(
        rows,
        key=lambda row: (
            str(row["planner_type"]),
            str(row["scenario"]),
            _injection_rank(str(row["fault_origin"])),
            _magnitude_rank(str(row["magnitude"])),
            int(row["seed"]),
        ),
    )


def _run_taxonomy_clean_group(
    job: tuple[str, str, str, int, tuple[str, ...], bool, float, int],
) -> list[dict[str, Any]]:
    (
        planner_type,
        scenario,
        magnitude,
        seed,
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
            planner_type=planner_type,
        )
    )
    rows: list[dict[str, Any]] = []
    for injection_point in injection_points:
        fault = run_unified_pipeline(
            UnifiedPipelineConfig(
                scenario=scenario,
                injection_point=injection_point,  # type: ignore[arg-type]
                method=METHOD,
                magnitude=magnitude,
                seed=seed,
                trace_grids=False,
                use_prediction=use_prediction,
                prediction_horizon_s=prediction_horizon,
                prediction_samples=prediction_samples,
                planner_type=planner_type,
            )
        )
        record = classify_run(fault, clean)
        stage_error = compute_stage_errors(fault, clean)
        record["planner_type"] = planner_type
        record["_stage_safety_drop_s"] = float(stage_error.safety_drop_s)
        record["_stage_plan_dev_m"] = float(stage_error.plan_dev_m)
        rows.append(record)
    return rows


def _planner_gateway_rate(records: list[dict[str, Any]]) -> float | None:
    return _rate(
        records,
        lambda record: (
            bool(record["planner_gateway"])
            if "planner_gateway" in record
            else (
                record.get("dominant_interface") == "cost__plan"
                and _is_finite_number(record.get("dominant_raw_gain"))
                and float(record["dominant_raw_gain"]) > 1.0
            )
        ),
    )


def _collision_rate(records: list[dict[str, Any]], origin: str) -> float | None:
    return _rate(
        [row for row in records if row.get("fault_origin") == origin],
        lambda row: bool(row.get("collision", False)),
    )


def _cis_high(records: list[dict[str, Any]], origin: str) -> float | None:
    group = [
        row
        for row in records
        if row.get("fault_origin") == origin and row.get("magnitude") == "high"
    ]
    drops: list[float] = []
    budgets: list[float] = []
    for row in group:
        drop = row.get("_stage_safety_drop_s")
        budget = row.get("_stage_plan_dev_m")
        if _is_finite_number(drop) and _is_finite_number(budget):
            drops.append(max(0.0, float(drop)))
            budgets.append(float(budget))
    if not drops:
        return None
    return critical_interface_score(drops, budgets)


def _rate(records: list[dict[str, Any]], predicate: Any) -> float | None:
    if not records:
        return None
    values = np.asarray([1.0 if predicate(row) else 0.0 for row in records], dtype=np.float64)
    return float(np.mean(values))


def _parse_choices(value: str, *, choices: Iterable[str], name: str) -> list[str]:
    allowed = tuple(choices)
    parsed = _split_csv(value)
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"unknown {name} value(s) {invalid!r}; expected {', '.join(allowed)}")
    return parsed


def _parse_positive_floats(value: str, *, name: str) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for item in _split_csv(value):
        parsed = float(item)
        if parsed <= 0.0 or not math.isfinite(parsed):
            raise ValueError(f"{name} values must be positive and finite")
        if parsed not in seen:
            seen.add(parsed)
            out.append(parsed)
    return out


def _split_csv(value: str) -> list[str]:
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError("at least one comma-separated value is required")
    return parsed


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nplanner_independence.csv")
    print(",".join(OUTPUT_COLUMNS))
    for row in rows:
        print(",".join(_format_cell(row.get(column)) for column in OUTPUT_COLUMNS))


def _print_verdict(rows: list[dict[str, Any]]) -> None:
    by_type = {str(row["planner_type"]): row for row in rows}
    sampling = by_type.get("sampling")
    field = by_type.get("potential_field")
    if sampling is None or field is None:
        return
    print("\nPlanner-independence check")
    print(
        "planner_gateway_rate: "
        f"sampling={_format_cell(sampling.get('planner_gateway_rate'))}, "
        f"potential_field={_format_cell(field.get('planner_gateway_rate'))}"
    )
    print(
        "argmin_flip_rate: "
        f"sampling={_format_cell(sampling.get('argmin_flip_rate'))}, "
        f"potential_field={_format_cell(field.get('argmin_flip_rate'))}"
    )
    print(
        "high CIS order for potential_field: "
        f"costmap={_format_cell(field.get('costmap_cis_high'))}, "
        f"object={_format_cell(field.get('object_cis_high'))}, "
        f"prediction={_format_cell(field.get('prediction_cis_high'))}"
    )


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _injection_rank(value: str) -> int:
    order = {name: idx for idx, name in enumerate(INJECTION_POINTS)}
    return order.get(value, len(order))


def _magnitude_rank(value: str) -> int:
    order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return order.get(value, len(order))


if __name__ == "__main__":
    main()
