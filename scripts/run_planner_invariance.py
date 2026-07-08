"""Compare hard argmin and softmax planner selection in the unified pipeline.

The default softmax temperatures, 0.02 and 0.1, are in planner-score units.
They are small relative to the infeasible-candidate penalty and comparable to
typical feasible-path score gaps, so they soften candidate switches without
turning selection into a nearly uniform average.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_transfer_function import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BUDGETS,
    DEFAULT_WORKERS,
    _run_jobs as _run_transfer_jobs,
    _summarize_gain_rows,
    _summarize_linearity_rows,
)
from torsion.analysis.failure_taxonomy import classify_run  # noqa: E402
from torsion.analysis.propagation import (  # noqa: E402
    compute_stage_errors,
    critical_interface_score,
)
from torsion.scenarios.unified_pipeline import (  # noqa: E402
    MAGNITUDES,
    SCENARIOS,
    UnifiedPipelineConfig,
    default_unified_planner_config,
    run_unified_pipeline,
)

METHOD = "torsion_displace"
DEFAULT_SEEDS = 20
DEFAULT_SOFTMAX_TEMPERATURES = (0.02, 0.1)
DEFAULT_INJECTION_POINTS = ("object", "prediction", "costmap")

OUTPUT_COLUMNS = (
    "selection_mode",
    "temperature",
    "costplan_gain_cv",
    "costplan_norm_slope",
    "costplan_verdict",
    "planner_switch_rate_overall",
    "collision_rate_object",
    "collision_rate_prediction",
    "collision_rate_costmap",
    "costmap_cis_high",
    "object_cis_high",
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
        help="comma-separated transfer-function target path budgets in meters",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--use-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the six-stage object->prediction->cost chain",
    )
    parser.add_argument("--prediction-horizon", type=float, default=2.0)
    parser.add_argument("--prediction-samples", type=int, default=5)
    parser.add_argument(
        "--transfer-magnitude",
        default="medium",
        help="base transfer-function operator magnitude before budget override",
    )
    parser.add_argument(
        "--softmax-temperatures",
        default=",".join(str(value) for value in DEFAULT_SOFTMAX_TEMPERATURES),
        help="comma-separated softmax temperatures to compare against argmin",
    )
    args = parser.parse_args()

    scenarios = _parse_choices(args.scenarios, choices=SCENARIOS, name="scenario")
    magnitudes = _parse_choices(args.magnitudes, choices=MAGNITUDES, name="magnitude")
    budgets = _parse_positive_floats(args.budgets, name="budget")
    softmax_temperatures = _parse_positive_floats(
        args.softmax_temperatures,
        name="softmax temperature",
    )
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.prediction_samples <= 0:
        raise ValueError("--prediction-samples must be positive")
    if args.prediction_horizon <= 0.0 or not math.isfinite(float(args.prediction_horizon)):
        raise ValueError("--prediction-horizon must be positive and finite")

    modes = [("argmin", 0.0), *(("softmax", tau) for tau in softmax_temperatures)]
    rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    for selection_mode, temperature in modes:
        print(f"\nRunning planner mode {selection_mode} temperature={temperature:g}")
        transfer = _run_transfer_summary(
            scenarios=scenarios,
            budgets=tuple(budgets),
            seeds=int(args.seeds),
            use_prediction=bool(args.use_prediction),
            prediction_horizon=float(args.prediction_horizon),
            prediction_samples=int(args.prediction_samples),
            magnitude=str(args.transfer_magnitude),
            selection_mode=selection_mode,
            selection_temperature=float(temperature),
            workers=int(args.workers),
            bootstrap_resamples=int(args.bootstrap_resamples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        taxonomy_records = _run_taxonomy_records(
            scenarios=scenarios,
            magnitudes=magnitudes,
            seeds=int(args.seeds),
            use_prediction=bool(args.use_prediction),
            prediction_horizon=float(args.prediction_horizon),
            prediction_samples=int(args.prediction_samples),
            selection_mode=selection_mode,
            selection_temperature=float(temperature),
            workers=int(args.workers),
        )
        row = {
            "selection_mode": selection_mode,
            "temperature": float(temperature),
            "costplan_gain_cv": transfer["costplan"].get("gain_cv"),
            "costplan_norm_slope": transfer["costplan"].get("norm_slope"),
            "costplan_verdict": transfer["costplan"].get("verdict"),
        }
        row.update(_taxonomy_metrics(taxonomy_records))
        rows.append(row)
        topology_rows.append(
            {
                "selection_mode": selection_mode,
                "temperature": float(temperature),
                **transfer["topology"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "planner_invariance.csv"
    _write_csv(output_path, rows, fieldnames=OUTPUT_COLUMNS)
    _print_table(rows)
    _print_topology(topology_rows)
    print(f"\nWrote {len(rows)} planner-invariance rows to {output_path}")


def _run_transfer_summary(
    *,
    scenarios: list[str],
    budgets: tuple[float, ...],
    seeds: int,
    use_prediction: bool,
    prediction_horizon: float,
    prediction_samples: int,
    magnitude: str,
    selection_mode: str,
    selection_temperature: float,
    workers: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    injection_points = DEFAULT_INJECTION_POINTS if use_prediction else ("object", "costmap")
    jobs = [
        (
            scenario,
            seed,
            tuple(budgets),
            tuple(injection_points),
            bool(use_prediction),
            float(prediction_horizon),
            int(prediction_samples),
            str(magnitude),
            str(selection_mode),
            float(selection_temperature),
        )
        for scenario in scenarios
        for seed in range(seeds)
    ]
    run_rows = _run_transfer_jobs(jobs, workers=workers)
    gain_rows = _summarize_gain_rows(
        run_rows,
        scenarios=scenarios,
        injection_points=tuple(injection_points),
        budgets=tuple(budgets),
        n_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    linearity_rows = _summarize_linearity_rows(
        gain_rows,
        scenarios=scenarios,
        injection_points=tuple(injection_points),
    )
    return {
        "costplan": _linearity_row(
            linearity_rows,
            injection="costmap",
            interface="cost__plan",
        ),
        "topology": _topology_from_linearity(
            linearity_rows,
            use_prediction=use_prediction,
        ),
    }


def _run_taxonomy_records(
    *,
    scenarios: list[str],
    magnitudes: list[str],
    seeds: int,
    use_prediction: bool,
    prediction_horizon: float,
    prediction_samples: int,
    selection_mode: str,
    selection_temperature: float,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        (
            scenario,
            magnitude,
            seed,
            bool(use_prediction),
            float(prediction_horizon),
            int(prediction_samples),
            str(selection_mode),
            float(selection_temperature),
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
    rows.sort(
        key=lambda row: (
            str(row["scenario"]),
            _injection_rank(str(row["fault_origin"])),
            _magnitude_rank(str(row["magnitude"])),
            int(row["seed"]),
        )
    )
    return rows


def _run_taxonomy_clean_group(
    job: tuple[str, str, int, bool, float, int, str, float],
) -> list[dict[str, Any]]:
    (
        scenario,
        magnitude,
        seed,
        use_prediction,
        prediction_horizon,
        prediction_samples,
        selection_mode,
        selection_temperature,
    ) = job
    planner_config = _planner_config(
        selection_mode=selection_mode,
        selection_temperature=selection_temperature,
    )
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
            planner=planner_config,
        )
    )
    rows: list[dict[str, Any]] = []
    for injection_point in _taxonomy_injection_points(use_prediction):
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
                planner=planner_config,
            )
        )
        record = classify_run(fault, clean)
        stage_error = compute_stage_errors(fault, clean)
        record["_stage_safety_drop_s"] = float(stage_error.safety_drop_s)
        record["_stage_plan_dev_m"] = float(stage_error.plan_dev_m)
        rows.append(record)
    return rows


def _planner_config(
    *,
    selection_mode: str,
    selection_temperature: float,
):
    return replace(
        default_unified_planner_config(),
        selection_mode=str(selection_mode),
        selection_temperature=float(selection_temperature),
    )


def _linearity_row(
    rows: list[dict[str, Any]],
    *,
    injection: str,
    interface: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["scenario"] == "ALL"
        and row["injection"] == injection
        and row["interface"] == interface
    ]
    if not matches:
        return {"gain_cv": None, "norm_slope": None, "verdict": "nonlinear"}
    return matches[0]


def _topology_from_linearity(
    rows: list[dict[str, Any]],
    *,
    use_prediction: bool,
) -> dict[str, Any]:
    topology = {
        "object__prediction_mean_gain": None,
        "prediction__cost_mean_gain": None,
        "cost__plan_mean_gain": _linearity_row(
            rows,
            injection="costmap",
            interface="cost__plan",
        ).get("mean_gain"),
    }
    if use_prediction:
        topology["object__prediction_mean_gain"] = _linearity_row(
            rows,
            injection="object",
            interface="object__prediction",
        ).get("mean_gain")
        topology["prediction__cost_mean_gain"] = _linearity_row(
            rows,
            injection="prediction",
            interface="prediction__cost",
        ).get("mean_gain")
    return topology


def _taxonomy_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "planner_switch_rate_overall": _rate(
            records,
            lambda row: bool(row.get("argmin_flipped", False)),
        ),
        "collision_rate_object": _collision_rate(records, "object"),
        "collision_rate_prediction": _collision_rate(records, "prediction"),
        "collision_rate_costmap": _collision_rate(records, "costmap"),
        "costmap_cis_high": _cis_high(records, "costmap"),
        "object_cis_high": _cis_high(records, "object"),
    }


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


def _rate(
    records: list[dict[str, Any]],
    predicate: Any,
) -> float | None:
    if not records:
        return None
    values = np.asarray([1.0 if predicate(row) else 0.0 for row in records], dtype=np.float64)
    return float(np.mean(values))


def _taxonomy_injection_points(use_prediction: bool) -> tuple[str, ...]:
    return DEFAULT_INJECTION_POINTS if use_prediction else ("object", "costmap")


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
    print("\nplanner_invariance.csv")
    print(",".join(OUTPUT_COLUMNS))
    for row in rows:
        print(",".join(_format_cell(row.get(column)) for column in OUTPUT_COLUMNS))


def _print_topology(rows: list[dict[str, Any]]) -> None:
    print("\nTopology check mean raw gains")
    for row in rows:
        print(
            f"{row['selection_mode']} tau={float(row['temperature']):g}: "
            f"object__prediction={_format_cell(row.get('object__prediction_mean_gain'))}, "
            f"prediction__cost={_format_cell(row.get('prediction__cost_mean_gain'))}, "
            f"cost__plan={_format_cell(row.get('cost__plan_mean_gain'))}"
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
    order = {name: idx for idx, name in enumerate(DEFAULT_INJECTION_POINTS)}
    return order.get(value, len(order))


def _magnitude_rank(value: str) -> int:
    order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return order.get(value, len(order))


if __name__ == "__main__":
    main()
