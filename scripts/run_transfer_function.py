"""Run transfer-function / linearity characterization on unified traces."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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
    compute_normalized_stage_errors,
    compute_stage_errors,
    interface_gains,
)
from torsion.analysis.transfer_function import (  # noqa: E402
    INTERFACE_ORDER,
    characterize_linearity,
    interface_raw_gains,
)
from torsion.metrics.statistics import bootstrap_ci  # noqa: E402
from torsion.scenarios.unified_pipeline import (  # noqa: E402
    SCENARIOS,
    UnifiedPipelineConfig,
    default_unified_planner_config,
    run_unified_pipeline,
)

DEFAULT_BUDGETS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
DEFAULT_INJECTION_POINTS = ("object", "prediction", "costmap")
DEFAULT_SEEDS = 25
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_WORKERS = max(1, min(16, os.cpu_count() or 1))
METHOD = "torsion_displace"

GAIN_COLUMNS = (
    "scenario",
    "injection",
    "interface",
    "budget",
    "n",
    "raw_gain_mean",
    "ci_low",
    "ci_high",
    "norm_gain_mean",
)

LINEARITY_COLUMNS = (
    "scenario",
    "injection",
    "interface",
    "mean_gain",
    "gain_cv",
    "gain_slope",
    "norm_slope",
    "monotonic",
    "verdict",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated scenarios",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="comma-separated target path budgets in meters",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
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
        "--magnitude",
        default="medium",
        help="base operator magnitude used by calibration before budget override",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("argmin", "softmax"),
        default="argmin",
        help="planner selection rule",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=0.0,
        help="softmax temperature; must be positive when --selection-mode=softmax",
    )
    args = parser.parse_args()

    scenarios = _parse_choices(args.scenarios, choices=SCENARIOS, name="scenario")
    budgets = _parse_budgets(args.budgets)
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
    if args.selection_mode == "softmax" and (
        args.selection_temperature <= 0.0
        or not math.isfinite(float(args.selection_temperature))
    ):
        raise ValueError("--selection-temperature must be positive and finite for softmax")

    injection_points = (
        DEFAULT_INJECTION_POINTS if args.use_prediction else ("object", "costmap")
    )
    jobs = [
        (
            scenario,
            seed,
            tuple(budgets),
            tuple(injection_points),
            bool(args.use_prediction),
            float(args.prediction_horizon),
            int(args.prediction_samples),
            str(args.magnitude),
            str(args.selection_mode),
            float(args.selection_temperature),
        )
        for scenario in scenarios
        for seed in range(int(args.seeds))
    ]

    run_rows = _run_jobs(jobs, workers=int(args.workers))
    gain_rows = _summarize_gain_rows(
        run_rows,
        scenarios=scenarios,
        injection_points=tuple(injection_points),
        budgets=tuple(budgets),
        n_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    linearity_rows = _summarize_linearity_rows(
        gain_rows,
        scenarios=scenarios,
        injection_points=tuple(injection_points),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gains_path = args.out_dir / "transfer_function_gains.csv"
    linearity_path = args.out_dir / "transfer_function_linearity.csv"
    _write_csv(gains_path, gain_rows, fieldnames=GAIN_COLUMNS)
    _write_csv(linearity_path, linearity_rows, fieldnames=LINEARITY_COLUMNS)

    _print_all_object_summary(
        gain_rows,
        linearity_rows,
        budgets=tuple(budgets),
    )
    print(f"\nWrote {len(gain_rows)} transfer-function gain rows to {gains_path}")
    print(f"Wrote {len(linearity_rows)} transfer-function linearity rows to {linearity_path}")


def _run_jobs(
    jobs: list[
        tuple[
            str,
            int,
            tuple[float, ...],
            tuple[str, ...],
            bool,
            float,
            int,
            str,
            str,
            float,
        ]
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
    job: tuple[
        str,
        int,
        tuple[float, ...],
        tuple[str, ...],
        bool,
        float,
        int,
        str,
        str,
        float,
    ],
) -> list[dict[str, Any]]:
    (
        scenario,
        seed,
        budgets,
        injection_points,
        use_prediction,
        prediction_horizon,
        prediction_samples,
        magnitude,
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
    for injection in injection_points:
        for budget in budgets:
            fault_config = UnifiedPipelineConfig(
                scenario=scenario,
                injection_point=injection,  # type: ignore[arg-type]
                method=METHOD,
                magnitude=magnitude,
                seed=seed,
                trace_grids=False,
                use_prediction=use_prediction,
                prediction_horizon_s=prediction_horizon,
                prediction_samples=prediction_samples,
                target_path_budget_m=float(budget),
                planner=planner_config,
            )
            try:
                fault = run_unified_pipeline(fault_config)
            except Exception as exc:
                rows.append(
                    _failed_run_record(
                        scenario=scenario,
                        injection=injection,
                        budget=float(budget),
                        seed=seed,
                        exc=exc,
                    )
                )
                continue

            raw_gains = interface_raw_gains(fault, clean)
            stage_error = compute_stage_errors(fault, clean)
            normalized = compute_normalized_stage_errors(stage_error, clean)
            norm_gains = interface_gains(
                normalized,
                injection,
                use_prediction=use_prediction,
            )
            record: dict[str, Any] = {
                "scenario": scenario,
                "injection": injection,
                "budget": float(budget),
                "seed": int(seed),
                "status": "completed",
                "error": "",
            }
            for interface in INTERFACE_ORDER:
                record[f"raw_{interface}"] = raw_gains[interface]
                record[f"norm_{interface}"] = norm_gains.get(interface)
            rows.append(record)
    return rows


def _failed_run_record(
    *,
    scenario: str,
    injection: str,
    budget: float,
    seed: int,
    exc: Exception,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "scenario": scenario,
        "injection": injection,
        "budget": float(budget),
        "seed": int(seed),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }
    for interface in INTERFACE_ORDER:
        record[f"raw_{interface}"] = None
        record[f"norm_{interface}"] = None
    return record


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


def _summarize_gain_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    injection_points: tuple[str, ...],
    budgets: tuple[float, ...],
    n_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in (*scenarios, "ALL"):
        scoped = rows if scope == "ALL" else [row for row in rows if row["scenario"] == scope]
        for injection in injection_points:
            injected = [row for row in scoped if row["injection"] == injection]
            for interface in INTERFACE_ORDER:
                for budget in budgets:
                    group = [
                        row
                        for row in injected
                        if math.isclose(float(row["budget"]), float(budget), rel_tol=0.0, abs_tol=1e-12)
                    ]
                    raw_values = _finite_values(group, f"raw_{interface}")
                    norm_values = _finite_values(group, f"norm_{interface}")
                    record: dict[str, Any] = {
                        "scenario": scope,
                        "injection": injection,
                        "interface": interface,
                        "budget": float(budget),
                        "n": len(raw_values),
                        "norm_gain_mean": (
                            float(np.mean(np.asarray(norm_values, dtype=np.float64)))
                            if norm_values
                            else None
                        ),
                    }
                    if raw_values:
                        low, high = bootstrap_ci(
                            raw_values,
                            n_resamples=n_resamples,
                            seed=bootstrap_seed,
                        )
                        record["raw_gain_mean"] = float(
                            np.mean(np.asarray(raw_values, dtype=np.float64))
                        )
                        record["ci_low"] = low
                        record["ci_high"] = high
                    else:
                        record["raw_gain_mean"] = None
                        record["ci_low"] = None
                        record["ci_high"] = None
                    out.append(record)
    return _sort_gain_rows(
        out,
        scenarios=scenarios,
        injection_points=injection_points,
        budgets=budgets,
    )


def _summarize_linearity_rows(
    gain_rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    injection_points: tuple[str, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in (*scenarios, "ALL"):
        scoped = [row for row in gain_rows if row["scenario"] == scope]
        for injection in injection_points:
            injected = [row for row in scoped if row["injection"] == injection]
            for interface in INTERFACE_ORDER:
                group = [row for row in injected if row["interface"] == interface]
                gains = {
                    float(row["budget"]): row["raw_gain_mean"]
                    for row in group
                    if _is_finite_number(row.get("raw_gain_mean"))
                }
                summary = characterize_linearity(gains)
                out.append(
                    {
                        "scenario": scope,
                        "injection": injection,
                        "interface": interface,
                        "mean_gain": summary["mean_gain"],
                        "gain_cv": summary["gain_cv"],
                        "gain_slope": summary["gain_slope"],
                        "norm_slope": summary["norm_slope"],
                        "monotonic": summary["monotonic"],
                        "verdict": summary["verdict"],
                    }
                )
    return _sort_linearity_rows(out, scenarios=scenarios, injection_points=injection_points)


def _print_all_object_summary(
    gain_rows: list[dict[str, Any]],
    linearity_rows: list[dict[str, Any]],
    *,
    budgets: tuple[float, ...],
) -> None:
    print("\nTransfer-function summary: scenario=ALL, injection=object, method=torsion_displace")
    linear: list[str] = []
    nonlinear: list[str] = []
    for interface in INTERFACE_ORDER:
        curve_rows = [
            row
            for row in gain_rows
            if row["scenario"] == "ALL"
            and row["injection"] == "object"
            and row["interface"] == interface
        ]
        by_budget = {float(row["budget"]): row for row in curve_rows}
        curve = ", ".join(
            f"{budget:g}:{_format_float(by_budget.get(budget, {}).get('raw_gain_mean'))}"
            for budget in budgets
        )
        summary = next(
            row
            for row in linearity_rows
            if row["scenario"] == "ALL"
            and row["injection"] == "object"
            and row["interface"] == interface
        )
        if summary["verdict"] == "linear":
            linear.append(interface)
        else:
            nonlinear.append(interface)
        print(
            f"{interface}: {curve} | "
            f"gain_cv={_format_float(summary.get('gain_cv'))} "
            f"norm_slope={_format_float(summary.get('norm_slope'))} "
            f"verdict={summary['verdict']}"
        )
    print(
        "Summary: "
        f"linear={', '.join(linear) if linear else 'none'}; "
        f"nonlinear={', '.join(nonlinear) if nonlinear else 'none'}"
    )


def _finite_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if _is_finite_number(value):
            values.append(float(value))
    return values


def _parse_choices(value: str, *, choices: Iterable[str], name: str) -> list[str]:
    allowed = tuple(choices)
    parsed = _split_csv(value)
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"unknown {name} value(s) {invalid!r}; expected {', '.join(allowed)}")
    return parsed


def _parse_budgets(value: str) -> list[float]:
    budgets: list[float] = []
    seen: set[float] = set()
    for item in _split_csv(value):
        budget = float(item)
        if budget <= 0.0 or not math.isfinite(budget):
            raise ValueError("budgets must be positive and finite")
        if budget not in seen:
            seen.add(budget)
            budgets.append(budget)
    return budgets


def _split_csv(value: str) -> list[str]:
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError("at least one comma-separated value is required")
    return parsed


def _sort_gain_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    injection_points: tuple[str, ...],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate((*scenarios, "ALL"))}
    injection_order = {name: idx for idx, name in enumerate(injection_points)}
    interface_order = {name: idx for idx, name in enumerate(INTERFACE_ORDER)}
    budget_order = {float(value): idx for idx, value in enumerate(budgets)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order[str(row["scenario"])],
            injection_order[str(row["injection"])],
            interface_order[str(row["interface"])],
            budget_order[float(row["budget"])],
        ),
    )


def _sort_linearity_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    injection_points: tuple[str, ...],
) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate((*scenarios, "ALL"))}
    injection_order = {name: idx for idx, name in enumerate(injection_points)}
    interface_order = {name: idx for idx, name in enumerate(INTERFACE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order[str(row["scenario"])],
            injection_order[str(row["injection"])],
            interface_order[str(row["interface"])],
        ),
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
        return "NA"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
