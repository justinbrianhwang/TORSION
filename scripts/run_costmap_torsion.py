"""Run the Phase 2b budget-matched dense cost-map sweep.

Calibration is documented in :mod:`torsion.scenarios.costmap_runner`: every
non-clean method is calibrated on the realized L2 deviation between the chosen
faulted ego path and the clean ego path.  The sweep writes raw per-run metrics,
per scenario/method/magnitude summaries, and a reproducible leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from torsion.metrics.statistics import bootstrap_ci, population_std, summarize_safety_group
from torsion.scenarios.costmap_runner import (
    COSTMAP_METHODS,
    CostMapRunnerConfig,
    run_costmap_closed_loop,
)

DEFAULT_SEEDS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
DEFAULT_BOOTSTRAP_SEED = 0
TTC_CENSOR_S = 5.0
SCENARIOS = ("cut_in", "leading_vehicle", "pedestrian_crossing")
METHODS = COSTMAP_METHODS
MAGNITUDES = ("low", "medium", "high")
SUMMARY_COLUMNS = (
    "scenario_id",
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
    "worst5pct_min_ttc",
    "off_road_rate",
    "mean_min_distance",
    "mean_realized_budget",
)
LEADERBOARD_COLUMNS = (
    "scope",
    "rank",
    "method",
    "n_runs",
    "collision_rate",
    "mean_min_ttc",
    "std_min_ttc",
    "mean_min_distance",
    "mean_realized_budget",
    "ttc_drop_s",
    "collision_increase",
    "severity_score",
    "consistency_score",
    "efficiency_score",
    "composite_score",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="number of deterministic seeds per scenario/method/magnitude cell",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes for run execution",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/costmap_runs.csv"),
        help="per-run CSV output path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/costmap_summary.csv"),
        help="per scenario/method/magnitude summary CSV output path",
    )
    parser.add_argument(
        "--leaderboard-output",
        type=Path,
        default=Path("results/metrics/costmap_leaderboard.csv"),
        help="method leaderboard CSV output path",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples for 95% confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="deterministic seed for bootstrap confidence intervals",
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    jobs = [
        (scenario, method, magnitude, seed)
        for scenario in SCENARIOS
        for method in METHODS
        for magnitude in MAGNITUDES
        for seed in range(args.seeds)
    ]
    records = _run_jobs(jobs, workers=args.workers)
    records = _sort_records(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output, [_csv_record(row) for row in records])

    summary = _summarize_records(
        records,
        group_columns=("scenario_id", "method", "magnitude"),
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary = _sort_summary(summary)
    _write_csv(args.summary_output, summary, fieldnames=SUMMARY_COLUMNS)

    leaderboard = _build_leaderboard(records)
    _write_csv(args.leaderboard_output, leaderboard, fieldnames=LEADERBOARD_COLUMNS)

    aggregate = _summarize_records(
        records,
        group_columns=("method", "magnitude"),
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    aggregate = _sort_summary(aggregate)
    _print_magnitude_tables(aggregate)
    _print_leaderboard(leaderboard)
    print(f"\nWrote {len(records)} runs to {args.output}")
    print(f"Wrote {len(summary)} group summaries to {args.summary_output}")
    print(f"Wrote {len(leaderboard)} leaderboard rows to {args.leaderboard_output}")


def _run_jobs(
    jobs: list[tuple[str, str, str, int]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        return [_run_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs))


def _run_one(job: tuple[str, str, str, int]) -> dict[str, Any]:
    scenario, method, magnitude, seed = job
    result = run_costmap_closed_loop(
        CostMapRunnerConfig(
            scenario=scenario,
            method=method,  # type: ignore[arg-type]
            magnitude=magnitude,
            seed=seed,
        )
    )
    record = result.to_record()
    record["min_ttc_censored"] = _finite_metric(record["min_ttc"], cap=TTC_CENSOR_S)
    return record


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    group_columns: tuple[str, ...],
    n_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in _group_keys(records, group_columns):
        group = [
            row
            for row in records
            if all(row[column] == value for column, value in zip(group_columns, key, strict=True))
        ]
        record = dict(zip(group_columns, key, strict=True))
        record.update(
            summarize_safety_group(
                collision=[float(row["collision"]) for row in group],
                min_ttc=[float(row["min_ttc_censored"]) for row in group],
                realized_budget=[float(row["mean_realized_budget"]) for row in group],
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
        )
        record["off_road_rate"] = float(np.mean([float(row["off_road"]) for row in group]))
        record["mean_min_distance"] = float(
            np.mean([_finite_metric(row["min_obstacle_distance"], cap=100.0) for row in group])
        )
        rows.append(record)
    return rows


def _build_leaderboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes = ("ALL", *SCENARIOS)
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        scoped = records if scope == "ALL" else [
            row for row in records if row["scenario_id"] == scope
        ]
        clean = [row for row in scoped if row["method"] == "clean"]
        clean_collision = float(np.mean([float(row["collision"]) for row in clean]))
        clean_ttc = float(np.mean([float(row["min_ttc_censored"]) for row in clean]))

        scope_rows: list[dict[str, Any]] = []
        for method in METHODS:
            group = [row for row in scoped if row["method"] == method]
            collision_rate = float(np.mean([float(row["collision"]) for row in group]))
            mean_ttc = float(np.mean([float(row["min_ttc_censored"]) for row in group]))
            std_ttc = population_std([float(row["min_ttc_censored"]) for row in group])
            mean_budget = float(np.mean([float(row["mean_realized_budget"]) for row in group]))
            ttc_drop = max(0.0, clean_ttc - mean_ttc)
            collision_increase = max(0.0, collision_rate - clean_collision)
            normalized_ttc_drop = ttc_drop / max(clean_ttc, 1e-9)
            severity = 0.60 * collision_increase + 0.40 * normalized_ttc_drop
            consistency = severity / (1.0 + std_ttc)
            efficiency = severity / max(mean_budget, 1e-9) if method != "clean" else 0.0
            scope_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "n_runs": len(group),
                    "collision_rate": collision_rate,
                    "mean_min_ttc": mean_ttc,
                    "std_min_ttc": std_ttc,
                    "mean_min_distance": float(
                        np.mean([
                            _finite_metric(row["min_obstacle_distance"], cap=100.0)
                            for row in group
                        ])
                    ),
                    "mean_realized_budget": mean_budget,
                    "ttc_drop_s": ttc_drop,
                    "collision_increase": collision_increase,
                    "severity_score": severity,
                    "consistency_score": consistency,
                    "efficiency_score": efficiency,
                }
            )

        _add_composite(scope_rows)
        scope_rows.sort(key=lambda row: float(row["composite_score"]), reverse=True)
        for rank, row in enumerate(scope_rows, start=1):
            row["rank"] = rank
            rows.append(row)
    return rows


def _add_composite(rows: list[dict[str, Any]]) -> None:
    severity = _minmax([float(row["severity_score"]) for row in rows])
    consistency = _minmax([float(row["consistency_score"]) for row in rows])
    efficiency = _minmax([float(row["efficiency_score"]) for row in rows])
    for row, severity_n, consistency_n, efficiency_n in zip(
        rows, severity, consistency, efficiency, strict=True
    ):
        row["composite_score"] = float(
            0.55 * severity_n + 0.20 * consistency_n + 0.25 * efficiency_n
        )


def _minmax(values: Iterable[float]) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high <= low + 1e-12:
        return [0.0 for _ in arr]
    return [float((value - low) / (high - low)) for value in arr]


def _group_keys(records: list[dict[str, Any]], columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    keys: list[tuple[Any, ...]] = []
    for row in records:
        key = tuple(row[column] for column in columns)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate(SCENARIOS)}
    method_order = {name: idx for idx, name in enumerate(METHODS)}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        records,
        key=lambda row: (
            scenario_order[row["scenario_id"]],
            method_order[row["method"]],
            magnitude_order[row["magnitude"]],
            int(row["seed"]),
        ),
    )


def _sort_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate(SCENARIOS)}
    method_order = {name: idx for idx, name in enumerate(METHODS)}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        rows,
        key=lambda row: (
            scenario_order.get(row.get("scenario_id", ""), -1),
            method_order[row["method"]],
            magnitude_order[row["magnitude"]],
        ),
    )


def _print_magnitude_tables(summary: list[dict[str, Any]]) -> None:
    for magnitude in MAGNITUDES:
        rows = [row for row in summary if row["magnitude"] == magnitude]
        table = [
            {
                "method": row["method"],
                "collision_rate [95% CI]": _format_ci(row, "collision_rate"),
                "mean_min_ttc [95% CI]": _format_ci(row, "mean_min_ttc"),
                "std_min_ttc": f"{float(row['std_min_ttc']):.3f}",
                "worst5pct_min_ttc": f"{float(row['worst5pct_min_ttc']):.3f}",
                "mean_realized_budget": f"{float(row['mean_realized_budget']):.3f}",
            }
            for row in rows
        ]
        print(f"\nMagnitude: {magnitude}")
        _print_table(table)


def _print_leaderboard(rows: list[dict[str, Any]]) -> None:
    print("\nLeaderboard composite: 0.55*severity + 0.20*severity/(1+std_min_ttc) + 0.25*severity/budget, each min-max normalized within scope.")
    for scope in ("ALL", *SCENARIOS):
        scoped = [row for row in rows if row["scope"] == scope]
        table = [
            {
                "rank": row["rank"],
                "method": row["method"],
                "collision_rate": f"{float(row['collision_rate']):.3f}",
                "mean_min_ttc": f"{float(row['mean_min_ttc']):.3f}",
                "std_min_ttc": f"{float(row['std_min_ttc']):.3f}",
                "mean_budget": f"{float(row['mean_realized_budget']):.3f}",
                "composite": f"{float(row['composite_score']):.3f}",
            }
            for row in scoped
        ]
        print(f"\nScope: {scope}")
        _print_table(table)


def _format_ci(row: dict[str, Any], metric: str) -> str:
    return (
        f"{float(row[metric]):.3f} "
        f"[{float(row[f'{metric}_ci_low']):.3f}, "
        f"{float(row[f'{metric}_ci_high']):.3f}]"
    )


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


def _finite_metric(value: Any, *, cap: float) -> float:
    out = float(value)
    if math.isnan(out):
        return cap
    if math.isinf(out):
        return cap if out > 0.0 else 0.0
    return float(np.clip(out, 0.0, cap))


if __name__ == "__main__":
    main()
