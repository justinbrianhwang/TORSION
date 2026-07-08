"""Run the data-grounded failure taxonomy on prediction-enabled unified runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torsion.analysis.failure_taxonomy import build_taxonomy, classify_run  # noqa: E402
from torsion.scenarios.unified_pipeline import (  # noqa: E402
    MAGNITUDES,
    SCENARIOS,
    UnifiedPipelineConfig,
    default_unified_planner_config,
    run_unified_pipeline,
)

INJECTION_POINTS = ("object", "prediction", "costmap")
METHOD = "torsion_displace"
DEFAULT_SEEDS = 30
DEFAULT_WORKERS = max(1, min(16, os.cpu_count() or 1))
RUN_COLUMNS = (
    "scenario",
    "fault_origin",
    "signature",
    "dominant_interface",
    "argmin_flipped",
    "failure_mode",
    "magnitude",
    "seed",
    "min_ttc",
    "collision",
    "reach_safety",
)
PATH_COLUMNS = (
    "fault_origin",
    "signature",
    "failure_mode",
    "count",
    "freq_overall",
    "freq_within_origin",
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
        help="comma-separated magnitudes",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--prediction-horizon", type=float, default=2.0)
    parser.add_argument("--prediction-samples", type=int, default=5)
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
    magnitudes = _parse_choices(args.magnitudes, choices=MAGNITUDES, name="magnitude")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.prediction_samples <= 0:
        raise ValueError("--prediction-samples must be positive")
    if args.prediction_horizon <= 0.0 or not math.isfinite(float(args.prediction_horizon)):
        raise ValueError("--prediction-horizon must be positive and finite")
    if args.selection_mode == "softmax" and (
        args.selection_temperature <= 0.0
        or not math.isfinite(float(args.selection_temperature))
    ):
        raise ValueError("--selection-temperature must be positive and finite for softmax")

    jobs = [
        (
            scenario,
            magnitude,
            seed,
            float(args.prediction_horizon),
            int(args.prediction_samples),
            str(args.selection_mode),
            float(args.selection_temperature),
        )
        for scenario in scenarios
        for magnitude in magnitudes
        for seed in range(int(args.seeds))
    ]
    records = _sort_run_records(
        _run_jobs(jobs, workers=int(args.workers)),
        scenarios=scenarios,
        magnitudes=magnitudes,
    )
    taxonomy = build_taxonomy(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.out_dir / "failure_taxonomy_runs.csv"
    paths_path = args.out_dir / "failure_taxonomy_paths.csv"
    _write_csv(runs_path, records, fieldnames=RUN_COLUMNS)
    _write_csv(paths_path, taxonomy["paths"], fieldnames=PATH_COLUMNS)

    _print_top_paths(taxonomy["paths"], limit=5)
    _print_per_origin(taxonomy["per_origin"])
    _print_scenario_specific(taxonomy["scenario_specific"], limit=5)
    _print_advisor_examples(records)
    print(f"\nWrote {len(records)} classified runs to {runs_path}")
    print(f"Wrote {len(taxonomy['paths'])} taxonomy paths to {paths_path}")


def _run_jobs(
    jobs: list[tuple[str, str, int, float, int, str, float]],
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
    job: tuple[str, str, int, float, int, str, float],
) -> list[dict[str, Any]]:
    (
        scenario,
        magnitude,
        seed,
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
            use_prediction=True,
            prediction_horizon_s=prediction_horizon,
            prediction_samples=prediction_samples,
            planner=planner_config,
        )
    )
    rows: list[dict[str, Any]] = []
    for injection_point in INJECTION_POINTS:
        fault = run_unified_pipeline(
            UnifiedPipelineConfig(
                scenario=scenario,
                injection_point=injection_point,  # type: ignore[arg-type]
                method=METHOD,
                magnitude=magnitude,
                seed=seed,
                trace_grids=False,
                use_prediction=True,
                prediction_horizon_s=prediction_horizon,
                prediction_samples=prediction_samples,
                planner=planner_config,
            )
        )
        rows.append(classify_run(fault, clean))
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


def _print_top_paths(paths: list[dict[str, Any]], *, limit: int) -> None:
    print("\nTop representative propagation paths")
    if not paths:
        print("(none)")
        return
    for row in paths[:limit]:
        print(
            f"{row['fault_origin']} -> {row['signature']} -> {row['failure_mode']}  "
            f"({int(row['count'])} runs, {_format_pct(row['freq_overall'])} overall, "
            f"{_format_pct(row['freq_within_origin'])} of {row['fault_origin']})"
        )


def _print_per_origin(per_origin: dict[str, dict[str, Any]]) -> None:
    print("\nPer-origin dominant path and collision rate")
    if not per_origin:
        print("(none)")
        return
    for fault_origin, row in per_origin.items():
        print(
            f"{fault_origin}: {row['signature']} -> {row['failure_mode']} "
            f"({int(row['count'])}/{int(row['n_runs'])} runs, "
            f"{_format_pct(row['freq_within_origin'])} of {fault_origin}); "
            f"collision_rate={_format_pct(100.0 * float(row['collision_rate']))}"
        )


def _print_scenario_specific(rows: list[dict[str, Any]], *, limit: int) -> None:
    if not rows:
        return
    print("\nNotable scenario-specific paths")
    for row in rows[:limit]:
        print(
            f"{row['scenario']} / {row['fault_origin']}: "
            f"{row['signature']} -> {row['failure_mode']} "
            f"({int(row['count'])}/{int(row['cell_total'])} runs, "
            f"{_format_pct(row['freq_within_scenario_origin'])})"
        )


def _print_advisor_examples(records: list[dict[str, Any]]) -> None:
    print("\nAdvisor example support")

    object_prediction_near = [
        row
        for row in records
        if row["fault_origin"] == "object"
        and row["failure_mode"] == "near_miss"
        and row["signature"] in {"object__prediction", "prediction__cost"}
    ]
    _print_support_line(
        "object -> prediction drift -> wrong yield / near-miss",
        object_prediction_near,
    )

    costmap_switch_lane = [
        row
        for row in records
        if row["fault_origin"] == "costmap"
        and row["signature"] == "planner_switch"
        and row["failure_mode"] == "lane_departure"
    ]
    _print_support_line(
        "costmap -> planner switch -> lane departure",
        costmap_switch_lane,
    )

    prediction_late_brake_collision = [
        row
        for row in records
        if row["fault_origin"] == "prediction"
        and row["failure_mode"] == "collision"
        and bool(row.get("hard_brake_response", False))
    ]
    _print_support_line(
        "prediction -> late brake -> collision",
        prediction_late_brake_collision,
        suffix=_signature_breakdown(prediction_late_brake_collision),
    )


def _print_support_line(
    label: str,
    rows: list[dict[str, Any]],
    *,
    suffix: str = "",
) -> None:
    supported = "supported" if rows else "not supported"
    extra = f"; {suffix}" if rows and suffix else ""
    print(f"{label}: {supported} ({len(rows)} runs{extra})")


def _signature_breakdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    counts = Counter(str(row["signature"]) for row in rows)
    parts = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    return f"taxonomy signatures: {parts}"


def _parse_choices(value: str, *, choices: Iterable[str], name: str) -> list[str]:
    allowed = tuple(choices)
    parsed = _split_csv(value)
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"unknown {name} value(s) {invalid!r}; expected {', '.join(allowed)}")
    return parsed


def _split_csv(value: str) -> list[str]:
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError("at least one comma-separated value is required")
    return parsed


def _sort_run_records(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    magnitudes: list[str],
) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate(scenarios)}
    origin_order = {name: idx for idx, name in enumerate(INJECTION_POINTS)}
    magnitude_order = {name: idx for idx, name in enumerate(magnitudes)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order[str(row["scenario"])],
            origin_order[str(row["fault_origin"])],
            magnitude_order[str(row["magnitude"])],
            int(row["seed"]),
        ),
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
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


def _format_pct(value: Any) -> str:
    return f"{float(value):.1f}%"


if __name__ == "__main__":
    main()
