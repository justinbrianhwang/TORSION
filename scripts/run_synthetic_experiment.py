"""Run the synthetic Phase 2a FAIR closed-loop experiment sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd

from torsion.metrics.statistics import summarize_safety_group
from torsion.scenarios.synthetic_runner import RunnerConfig, run_synthetic_closed_loop

DEFAULT_SEEDS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
DEFAULT_BOOTSTRAP_SEED = 0
SCENARIOS = ("cut_in", "leading_vehicle", "pedestrian_crossing")
METHODS = (
    "clean",
    "gaussian_matched",
    "random_warp",
    "torsion_translate",
    "torsion_swirl",
    "torsion_curl",
    "torsion_combined",
)
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
    "iqr_min_ttc",
    "worst5pct_min_ttc",
    "worst_case_min_ttc",
    "mean_realized_budget",
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
        "--output",
        type=Path,
        default=Path("results/metrics/synthetic_runs.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/synthetic_summary.csv"),
        help="per scenario/method/magnitude summary CSV output path",
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

    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for magnitude in MAGNITUDES:
                for seed in range(args.seeds):
                    result = run_synthetic_closed_loop(
                        RunnerConfig(
                            scenario=scenario,
                            method=method,  # type: ignore[arg-type]
                            magnitude=magnitude,
                            seed=seed,
                        )
                    )
                    records.append(_csv_record(result.to_record()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    df = pd.DataFrame.from_records(records)
    summary = _summarize_dataframe(
        df,
        group_columns=("scenario_id", "method", "magnitude"),
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary = _sort_summary(summary, scenario_order=SCENARIOS)
    summary = summary[list(SUMMARY_COLUMNS)]
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)

    aggregate = _summarize_dataframe(
        df,
        group_columns=("method", "magnitude"),
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    aggregate = _sort_summary(aggregate, scenario_order=())
    _print_magnitude_tables(aggregate)
    print(f"\nWrote {len(records)} runs to {args.output}")
    print(f"Wrote {len(summary)} group summaries to {args.summary_output}")


def _summarize_dataframe(
    df: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    n_resamples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby(list(group_columns), sort=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(group_columns, keys, strict=True))
        record.update(
            summarize_safety_group(
                collision=group["collision"],
                min_ttc=group["min_ttc"],
                realized_budget=group["mean_realized_budget"],
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _sort_summary(df: pd.DataFrame, *, scenario_order: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    if "scenario_id" in out.columns:
        out["scenario_id"] = pd.Categorical(
            out["scenario_id"], categories=scenario_order, ordered=True
        )
    out["method"] = pd.Categorical(out["method"], categories=METHODS, ordered=True)
    out["magnitude"] = pd.Categorical(
        out["magnitude"], categories=MAGNITUDES, ordered=True
    )
    sort_columns = [
        column
        for column in ("scenario_id", "method", "magnitude")
        if column in out.columns
    ]
    out = out.sort_values(sort_columns).reset_index(drop=True)
    for column in ("scenario_id", "method", "magnitude"):
        if column in out.columns:
            out[column] = out[column].astype(str)
    return out


def _print_magnitude_tables(summary: pd.DataFrame) -> None:
    for magnitude in MAGNITUDES:
        rows = summary[summary["magnitude"] == magnitude]
        row_records = rows.to_dict("records")
        table = pd.DataFrame(
            {
                "method": rows["method"].astype(str).tolist(),
                "collision_rate [95% CI]": [
                    _format_ci(row, "collision_rate") for row in row_records
                ],
                "mean_min_ttc [95% CI]": [
                    _format_ci(row, "mean_min_ttc") for row in row_records
                ],
                "std_min_ttc": rows["std_min_ttc"].map(lambda value: f"{value:.3f}").tolist(),
                "worst5pct_min_ttc": rows["worst5pct_min_ttc"].map(
                    lambda value: f"{value:.3f}"
                ).tolist(),
                "mean_realized_budget": rows["mean_realized_budget"].map(
                    lambda value: f"{value:.3f}"
                ).tolist(),
            }
        )
        print(f"\nMagnitude: {magnitude}")
        print(table.to_string(index=False))


def _format_ci(row: dict[str, Any], metric: str) -> str:
    return (
        f"{float(row[metric]):.3f} "
        f"[{float(row[f'{metric}_ci_low']):.3f}, "
        f"{float(row[f'{metric}_ci_high']):.3f}]"
    )


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


if __name__ == "__main__":
    main()
