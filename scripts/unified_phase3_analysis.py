"""Build the Phase 3 unified leaderboard and ablation figures.

The leaderboard uses existing per-scenario summaries for the legal sweeps and
adds the explicitly contract-violating cost-map ``swirl_illegal`` ablation when
its cached summary is missing.

Composite score:

* ``ttc_drop_fraction = max(0, clean_mean_min_ttc - mean_min_ttc) / clean_mean_min_ttc``
* ``collision_increase = max(0, collision_rate - clean_collision_rate)``
* ``severity_raw = 0.5 * ttc_drop_fraction + 0.5 * collision_increase``
* ``consistency_raw = severity_raw / (1 + std_min_ttc)``
* ``efficiency_raw = severity_raw / mean_realized_budget`` for non-clean methods
* ``composite = 0.55 * minmax(severity_raw)
  + 0.20 * minmax(consistency_raw)
  + 0.25 * minmax(efficiency_raw)``

The min-max normalization is global over all representation/method/magnitude
rows in this unified table.  Clean rows get zero severity and efficiency.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torsion.metrics.statistics import summarize_safety_group
from torsion.scenarios.costmap_runner import CostMapRunnerConfig, run_costmap_closed_loop

METRICS_DIR = Path("results/metrics")
FIGURE_DIR = Path("results/figures")
SYNTHETIC_SUMMARY = METRICS_DIR / "synthetic_summary.csv"
COSTMAP_SUMMARY = METRICS_DIR / "costmap_summary.csv"
SYNTHETIC_RUNS = METRICS_DIR / "synthetic_runs.csv"
COSTMAP_RUNS = METRICS_DIR / "costmap_runs.csv"
SWIRL_ILLEGAL_RUNS = METRICS_DIR / "costmap_swirl_illegal_runs.csv"
SWIRL_ILLEGAL_SUMMARY = METRICS_DIR / "costmap_swirl_illegal_summary.csv"
UNIFIED_LEADERBOARD = METRICS_DIR / "unified_leaderboard.csv"
CONTRACT_ABLATION = METRICS_DIR / "ablation_contract.csv"
DIRECTEDNESS_ABLATION = METRICS_DIR / "ablation_directedness.csv"
FIGURE8 = FIGURE_DIR / "figure8_contract_ablation.png"
FIGURE9 = FIGURE_DIR / "figure9_summary.png"

SCENARIOS = ("cut_in", "leading_vehicle", "pedestrian_crossing")
MAGNITUDES = ("low", "medium", "high")
DEFAULT_SEEDS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
TTC_CENSOR_S = 5.0
EPS = 1e-9

SCORE_WEIGHTS = {
    "severity": 0.55,
    "consistency": 0.20,
    "efficiency": 0.25,
}

OBJECT_METHOD_MAP = {
    "clean": "clean",
    "gaussian_matched": "gaussian",
    "random_warp": "random_warp",
    "torsion": "torsion_displace",
    "torsion_displace": "torsion_displace",
    "torsion_translate": "torsion_displace",
    "torsion_swirl": "torsion_swirl",
    "torsion_curl": "torsion_curl",
    "torsion_combined": "torsion_combined",
}
COSTMAP_METHOD_MAP = {
    "clean": "clean",
    "cost_translate": "torsion_displace",
    "torsion_displace": "torsion_displace",
    "gaussian_cost": "gaussian",
    "random_warp_cost": "random_warp",
    "torsion_swirl": "torsion_swirl",
    "swirl_illegal": "swirl_illegal",
}
CONTRACT_STATUS = {
    "clean": "baseline",
    "gaussian": "contract_violating",
    "swirl_illegal": "contract_violating",
    "random_warp": "contract_preserving",
    "torsion_displace": "contract_preserving",
    "torsion_swirl": "contract_preserving",
    "torsion_curl": "contract_preserving",
    "torsion_combined": "contract_preserving",
}
METHOD_LABELS = {
    "clean": "Clean",
    "gaussian": "Gaussian",
    "random_warp": "Random warp",
    "torsion_displace": "Directed displace",
    "torsion_swirl": "Legal swirl",
    "torsion_curl": "Curl",
    "torsion_combined": "Combined",
    "swirl_illegal": "Boundary-warp swirl",
}

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
    "rank_overall",
    "rank_overall_magnitude",
    "rank_representation_magnitude",
    "representation",
    "method",
    "contract_status",
    "magnitude",
    "n_runs",
    "source_methods",
    "collision_rate",
    "mean_min_ttc",
    "std_min_ttc",
    "worst5pct_min_ttc",
    "mean_realized_budget",
    "clean_collision_rate",
    "clean_mean_min_ttc",
    "ttc_drop_s",
    "ttc_drop_fraction",
    "collision_increase",
    "severity_raw",
    "consistency_raw",
    "efficiency_raw",
    "severity_norm",
    "consistency_norm",
    "efficiency_norm",
    "composite_score",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--refresh-illegal",
        action="store_true",
        help="rerun the cached cost-map swirl_illegal ablation",
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    metrics_dir = args.metrics_dir
    figures_dir = args.figures_dir
    illegal_runs = metrics_dir / SWIRL_ILLEGAL_RUNS.name
    illegal_summary = metrics_dir / SWIRL_ILLEGAL_SUMMARY.name
    _ensure_swirl_illegal_summary(
        runs_path=illegal_runs,
        summary_path=illegal_summary,
        seeds=args.seeds,
        workers=args.workers,
        refresh=args.refresh_illegal,
    )

    summary = _load_unified_summary(
        synthetic_path=metrics_dir / SYNTHETIC_SUMMARY.name,
        costmap_path=metrics_dir / COSTMAP_SUMMARY.name,
        illegal_path=illegal_summary,
    )
    leaderboard = _build_leaderboard(summary)
    _write_csv(metrics_dir / UNIFIED_LEADERBOARD.name, leaderboard, LEADERBOARD_COLUMNS)

    contract = _build_contract_ablation(leaderboard)
    directedness = _build_directedness_ablation(leaderboard)
    _write_csv(metrics_dir / CONTRACT_ABLATION.name, contract)
    _write_csv(metrics_dir / DIRECTEDNESS_ABLATION.name, directedness)

    runs = _load_unified_runs(
        synthetic_path=metrics_dir / SYNTHETIC_RUNS.name,
        costmap_path=metrics_dir / COSTMAP_RUNS.name,
        illegal_path=illegal_runs,
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    _make_figure8(contract, figures_dir / FIGURE8.name)
    _make_figure9(runs, figures_dir / FIGURE9.name)

    _print_leaderboard(leaderboard)
    _print_ablation_results(contract, directedness)
    print(f"\nWrote {metrics_dir / UNIFIED_LEADERBOARD.name}")
    print(f"Wrote {metrics_dir / CONTRACT_ABLATION.name}")
    print(f"Wrote {metrics_dir / DIRECTEDNESS_ABLATION.name}")
    print(f"Wrote {figures_dir / FIGURE8.name}")
    print(f"Wrote {figures_dir / FIGURE9.name}")


def _ensure_swirl_illegal_summary(
    *,
    runs_path: Path,
    summary_path: Path,
    seeds: int,
    workers: int,
    refresh: bool,
) -> None:
    if not refresh and runs_path.exists() and summary_path.exists():
        return

    jobs = [
        (scenario, magnitude, seed)
        for scenario in SCENARIOS
        for magnitude in MAGNITUDES
        for seed in range(seeds)
    ]
    records = _run_swirl_illegal_jobs(jobs, workers=workers)
    records = _sort_cost_records(records)
    _write_csv(runs_path, records)
    summary = _summarize_cost_records(records)
    _write_csv(summary_path, summary, SUMMARY_COLUMNS)


def _run_swirl_illegal_jobs(
    jobs: list[tuple[str, str, int]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        return [_run_swirl_illegal_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_swirl_illegal_one, jobs))


def _run_swirl_illegal_one(job: tuple[str, str, int]) -> dict[str, Any]:
    scenario, magnitude, seed = job
    result = run_costmap_closed_loop(
        CostMapRunnerConfig(
            scenario=scenario,
            method="swirl_illegal",
            magnitude=magnitude,
            seed=seed,
        )
    )
    record = result.to_record()
    record["min_ttc_censored"] = _finite_metric(record["min_ttc"], cap=TTC_CENSOR_S)
    return record


def _summarize_cost_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for magnitude in MAGNITUDES:
            group = [
                row
                for row in records
                if row["scenario_id"] == scenario and row["magnitude"] == magnitude
            ]
            if not group:
                continue
            record: dict[str, Any] = {
                "scenario_id": scenario,
                "method": "swirl_illegal",
                "magnitude": magnitude,
            }
            record.update(
                summarize_safety_group(
                    collision=[float(row["collision"]) for row in group],
                    min_ttc=[float(row["min_ttc_censored"]) for row in group],
                    realized_budget=[float(row["mean_realized_budget"]) for row in group],
                    n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                )
            )
            record["off_road_rate"] = float(np.mean([float(row["off_road"]) for row in group]))
            record["mean_min_distance"] = float(
                np.mean([_finite_metric(row["min_obstacle_distance"], cap=100.0) for row in group])
            )
            rows.append(record)
    return rows


def _load_unified_summary(
    *,
    synthetic_path: Path,
    costmap_path: Path,
    illegal_path: Path,
) -> pd.DataFrame:
    object_summary = _read_summary(
        synthetic_path,
        representation="object_set",
        method_map=OBJECT_METHOD_MAP,
    )
    cost_summary = _read_summary(
        costmap_path,
        representation="cost_map",
        method_map=COSTMAP_METHOD_MAP,
    )
    illegal_summary = _read_summary(
        illegal_path,
        representation="cost_map",
        method_map=COSTMAP_METHOD_MAP,
    )
    return pd.concat([object_summary, cost_summary, illegal_summary], ignore_index=True)


def _read_summary(
    path: Path,
    *,
    representation: str,
    method_map: dict[str, str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    df = pd.read_csv(path)
    required = {
        "scenario_id",
        "method",
        "magnitude",
        "n_runs",
        "collision_rate",
        "mean_min_ttc",
        "std_min_ttc",
        "worst5pct_min_ttc",
        "mean_realized_budget",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["representation"] = representation
    out["source_method"] = out["method"].astype(str)
    out["method"] = out["source_method"].map(method_map).fillna(out["source_method"])
    for column in (
        "n_runs",
        "collision_rate",
        "mean_min_ttc",
        "std_min_ttc",
        "worst5pct_min_ttc",
        "mean_realized_budget",
    ):
        out[column] = pd.to_numeric(out[column], errors="raise")
    return out


def _build_leaderboard(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_columns = ["representation", "method", "magnitude"]
    metric_columns = [
        "collision_rate",
        "mean_min_ttc",
        "std_min_ttc",
        "worst5pct_min_ttc",
        "mean_realized_budget",
    ]
    for key, group in summary.groupby(group_columns, sort=False):
        representation, method, magnitude = (str(value) for value in key)
        weights = group["n_runs"].to_numpy(dtype=np.float64)
        total_runs = int(np.sum(weights))
        record: dict[str, Any] = {
            "representation": representation,
            "method": method,
            "contract_status": CONTRACT_STATUS.get(method, "unknown"),
            "magnitude": magnitude,
            "n_runs": total_runs,
            "source_methods": ";".join(sorted(set(group["source_method"].astype(str)))),
        }
        for column in metric_columns:
            values = group[column].to_numpy(dtype=np.float64)
            record[column] = _weighted_mean(values, weights)
        rows.append(record)

    clean_by_scope = {
        (row["representation"], row["magnitude"]): row
        for row in rows
        if row["method"] == "clean"
    }
    for row in rows:
        clean = clean_by_scope[(row["representation"], row["magnitude"])]
        clean_ttc = float(clean["mean_min_ttc"])
        clean_collision = float(clean["collision_rate"])
        ttc_drop = max(0.0, clean_ttc - float(row["mean_min_ttc"]))
        collision_increase = max(0.0, float(row["collision_rate"]) - clean_collision)
        severity = 0.5 * (ttc_drop / max(clean_ttc, EPS)) + 0.5 * collision_increase
        budget = max(float(row["mean_realized_budget"]), EPS)
        row["clean_collision_rate"] = clean_collision
        row["clean_mean_min_ttc"] = clean_ttc
        row["ttc_drop_s"] = ttc_drop
        row["ttc_drop_fraction"] = ttc_drop / max(clean_ttc, EPS)
        row["collision_increase"] = collision_increase
        row["severity_raw"] = severity
        row["consistency_raw"] = severity / (1.0 + max(float(row["std_min_ttc"]), 0.0))
        row["efficiency_raw"] = 0.0 if row["method"] == "clean" else severity / budget

    _add_normalized_scores(rows)
    rows.sort(
        key=lambda row: (
            -float(row["composite_score"]),
            row["representation"],
            MAGNITUDES.index(row["magnitude"]),
            row["method"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank_overall"] = rank
    _add_group_rank(rows, "rank_overall_magnitude", ("magnitude",))
    _add_group_rank(rows, "rank_representation_magnitude", ("representation", "magnitude"))
    return rows


def _add_normalized_scores(rows: list[dict[str, Any]]) -> None:
    severity = _minmax([float(row["severity_raw"]) for row in rows])
    consistency = _minmax([float(row["consistency_raw"]) for row in rows])
    efficiency = _minmax([float(row["efficiency_raw"]) for row in rows])
    for row, severity_n, consistency_n, efficiency_n in zip(
        rows, severity, consistency, efficiency, strict=True
    ):
        row["severity_norm"] = severity_n
        row["consistency_norm"] = consistency_n
        row["efficiency_norm"] = efficiency_n
        row["composite_score"] = (
            SCORE_WEIGHTS["severity"] * severity_n
            + SCORE_WEIGHTS["consistency"] * consistency_n
            + SCORE_WEIGHTS["efficiency"] * efficiency_n
        )


def _add_group_rank(
    rows: list[dict[str, Any]],
    column: str,
    group_columns: tuple[str, ...],
) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_columns)
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: -float(row["composite_score"]))
        for rank, row in enumerate(group_rows, start=1):
            row[column] = rank


def _build_contract_ablation(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("object_set", "torsion_displace", "gaussian", "directed_vs_gaussian"),
        ("cost_map", "torsion_displace", "gaussian", "directed_vs_gaussian"),
        ("cost_map", "torsion_swirl", "swirl_illegal", "legal_vs_boundary_warp_swirl"),
    ]
    for representation, legal_method, violating_method, comparison in pairs:
        for magnitude in MAGNITUDES:
            legal = _find_row(leaderboard, representation, legal_method, magnitude)
            violating = _find_row(leaderboard, representation, violating_method, magnitude)
            rows.append(
                _comparison_record(
                    representation=representation,
                    magnitude=magnitude,
                    comparison=comparison,
                    first_role="legal",
                    first=legal,
                    second_role="violating",
                    second=violating,
                )
            )
    return rows


def _build_directedness_ablation(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in ("object_set", "cost_map"):
        for magnitude in MAGNITUDES:
            directed = _find_row(leaderboard, representation, "torsion_displace", magnitude)
            random = _find_row(leaderboard, representation, "random_warp", magnitude)
            rows.append(
                _comparison_record(
                    representation=representation,
                    magnitude=magnitude,
                    comparison="directed_vs_random",
                    first_role="directed",
                    first=directed,
                    second_role="random",
                    second=random,
                )
            )
    return rows


def _comparison_record(
    *,
    representation: str,
    magnitude: str,
    comparison: str,
    first_role: str,
    first: dict[str, Any],
    second_role: str,
    second: dict[str, Any],
) -> dict[str, Any]:
    return {
        "representation": representation,
        "magnitude": magnitude,
        "comparison": comparison,
        f"{first_role}_method": first["method"],
        f"{second_role}_method": second["method"],
        f"{first_role}_collision_rate": float(first["collision_rate"]),
        f"{second_role}_collision_rate": float(second["collision_rate"]),
        "collision_rate_gap_second_minus_first": float(second["collision_rate"])
        - float(first["collision_rate"]),
        f"{first_role}_mean_min_ttc": float(first["mean_min_ttc"]),
        f"{second_role}_mean_min_ttc": float(second["mean_min_ttc"]),
        "mean_min_ttc_gap_second_minus_first_s": float(second["mean_min_ttc"])
        - float(first["mean_min_ttc"]),
        f"{first_role}_std_min_ttc": float(first["std_min_ttc"]),
        f"{second_role}_std_min_ttc": float(second["std_min_ttc"]),
        "std_min_ttc_gap_second_minus_first_s": float(second["std_min_ttc"])
        - float(first["std_min_ttc"]),
        f"{first_role}_worst5pct_min_ttc": float(first["worst5pct_min_ttc"]),
        f"{second_role}_worst5pct_min_ttc": float(second["worst5pct_min_ttc"]),
        f"{first_role}_mean_realized_budget": float(first["mean_realized_budget"]),
        f"{second_role}_mean_realized_budget": float(second["mean_realized_budget"]),
        f"{first_role}_severity_raw": float(first["severity_raw"]),
        f"{second_role}_severity_raw": float(second["severity_raw"]),
        "severity_gap_first_minus_second": float(first["severity_raw"])
        - float(second["severity_raw"]),
    }


def _load_unified_runs(
    *,
    synthetic_path: Path,
    costmap_path: Path,
    illegal_path: Path,
) -> pd.DataFrame:
    synthetic = _read_runs(
        synthetic_path,
        representation="object_set",
        method_map=OBJECT_METHOD_MAP,
        ttc_column="min_ttc",
    )
    costmap = _read_runs(
        costmap_path,
        representation="cost_map",
        method_map=COSTMAP_METHOD_MAP,
        ttc_column="min_ttc_censored",
    )
    illegal = _read_runs(
        illegal_path,
        representation="cost_map",
        method_map=COSTMAP_METHOD_MAP,
        ttc_column="min_ttc_censored",
    )
    return pd.concat([synthetic, costmap, illegal], ignore_index=True)


def _read_runs(
    path: Path,
    *,
    representation: str,
    method_map: dict[str, str],
    ttc_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    df = pd.read_csv(path)
    out = df.copy()
    out["representation"] = representation
    out["source_method"] = out["method"].astype(str)
    out["method"] = out["source_method"].map(method_map).fillna(out["source_method"])
    if ttc_column not in out.columns:
        ttc_column = "min_ttc"
    out["min_ttc_plot"] = pd.to_numeric(out[ttc_column], errors="coerce")
    if representation == "cost_map":
        out["min_ttc_plot"] = out["min_ttc_plot"].clip(lower=0.0, upper=TTC_CENSOR_S)
    out["mean_realized_budget"] = pd.to_numeric(out["mean_realized_budget"], errors="coerce")
    out["collision"] = pd.to_numeric(out["collision"], errors="coerce")
    return out


def _make_figure8(contract_rows: list[dict[str, Any]], path: Path) -> None:
    high = [row for row in contract_rows if row["magnitude"] == "high"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=False)
    fig.suptitle("Figure 8. Contract preservation ablation at high matched budget", fontsize=13)

    consistency_groups = [
        ("Object-set", _select_contract(high, "object_set", "directed_vs_gaussian")),
        ("Cost-map", _select_contract(high, "cost_map", "directed_vs_gaussian")),
    ]
    x = np.arange(len(consistency_groups), dtype=np.float64)
    width = 0.34
    legal_std = np.array([row["legal_std_min_ttc"] for _, row in consistency_groups])
    violating_std = np.array([row["violating_std_min_ttc"] for _, row in consistency_groups])
    axes[0].bar(x - width / 2, legal_std, width, label="directed legal", color="#2b6cb0")
    axes[0].bar(x + width / 2, violating_std, width, label="Gaussian", color="#d69e2e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name for name, _ in consistency_groups])
    axes[0].set_ylabel("std(min-TTC) across seeds/scenarios (s)")
    axes[0].set_title("Consistency")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    swirl = _select_contract(high, "cost_map", "legal_vs_boundary_warp_swirl")
    methods = ["Legal swirl", "Boundary-warp swirl"]
    collision = [swirl["legal_collision_rate"], swirl["violating_collision_rate"]]
    std = [swirl["legal_std_min_ttc"], swirl["violating_std_min_ttc"]]
    x2 = np.arange(2, dtype=np.float64)
    axes[1].bar(x2, collision, color=["#2f855a", "#c53030"], alpha=0.82)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(methods, rotation=0)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("collision rate")
    axes[1].set_title("Boundary contract violation")
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, (coll, spread) in enumerate(zip(collision, std, strict=True)):
        axes[1].text(
            idx,
            min(coll + 0.04, 0.98),
            f"std={spread:.3f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.text(
        0.5,
        0.02,
        "Legal operators keep the representation contract valid; Gaussian and boundary-warping swirl are "
        "contract-violating ablations shown for attribution/validity contrast.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.28)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _make_figure9(runs: pd.DataFrame, path: Path) -> None:
    high = runs[runs["magnitude"] == "high"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=False, constrained_layout=False)
    fig.suptitle("Figure 9. Severe and consistent degradation from directed semantic displacement", fontsize=13)

    panel_methods = {
        "object_set": ("clean", "torsion_displace", "random_warp", "gaussian"),
        "cost_map": ("clean", "torsion_displace", "random_warp", "gaussian", "swirl_illegal"),
    }
    colors = {
        "clean": "#4a5568",
        "torsion_displace": "#c53030",
        "random_warp": "#2f855a",
        "gaussian": "#d69e2e",
        "swirl_illegal": "#805ad5",
    }
    for ax, representation in zip(axes, ("object_set", "cost_map"), strict=True):
        rep = high[high["representation"] == representation]
        methods = panel_methods[representation]
        data = [
            rep[rep["method"] == method]["min_ttc_plot"].dropna().to_numpy(dtype=np.float64)
            for method in methods
        ]
        positions = np.arange(1, len(methods) + 1)
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#1a202c", "linewidth": 1.4},
            whiskerprops={"color": "#4a5568", "linewidth": 1.0},
            capprops={"color": "#4a5568", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#edf2f7",
                "markeredgecolor": "#4a5568",
                "alpha": 0.8,
            },
        )
        for patch, method in zip(box["boxes"], methods, strict=True):
            patch.set_facecolor(colors[method])
            patch.set_edgecolor(colors[method])
            patch.set_alpha(0.28)
            patch.set_linewidth(1.4)
        for position, method, values in zip(positions, methods, data, strict=True):
            if values.size == 0:
                continue
            mean_value = float(np.mean(values))
            ax.scatter(
                position,
                mean_value,
                marker="D",
                s=30,
                facecolor=colors[method],
                edgecolor="#1a202c",
                linewidth=0.6,
                zorder=4,
            )
        ax.set_title(representation.replace("_", " ").title())
        ax.set_xticks(positions)
        ax.set_xticklabels([_short_label(method) for method in methods], fontsize=8)
        ax.set_ylabel("min-TTC across scenarios/seeds (s); lower is worse")
        ax.grid(True, axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.02,
        "High-budget distributions over 3 scenarios x 30 seeds. Diamonds mark means. "
        "Cost-map TTC is capped at 5 s, matching the Phase 2b summaries.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.23, wspace=0.25)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _short_label(method: str) -> str:
    return {
        "clean": "Clean",
        "torsion_displace": "Directed\nlegal",
        "random_warp": "Random\nlegal",
        "gaussian": "Gaussian\ninvalid",
        "swirl_illegal": "Boundary\ninvalid",
    }.get(method, method)


def _select_contract(
    rows: list[dict[str, Any]],
    representation: str,
    comparison: str,
) -> dict[str, Any]:
    for row in rows:
        if row["representation"] == representation and row["comparison"] == comparison:
            return row
    raise KeyError((representation, comparison))


def _print_leaderboard(rows: list[dict[str, Any]]) -> None:
    print(
        "\nComposite formula: 0.55*minmax(severity_raw) + "
        "0.20*minmax(severity_raw/(1+std_min_ttc)) + "
        "0.25*minmax(severity_raw/mean_realized_budget)."
    )
    print(
        "severity_raw = 0.5*TTC-drop-fraction vs clean + "
        "0.5*collision-rate-increase vs clean; normalization is global over the unified table."
    )
    print("\nUnified leaderboard: top rows")
    _print_table(
        [
            {
                "rank": row["rank_overall"],
                "rep": row["representation"],
                "method": row["method"],
                "status": row["contract_status"],
                "mag": row["magnitude"],
                "coll": f"{float(row['collision_rate']):.3f}",
                "mean_ttc": f"{float(row['mean_min_ttc']):.3f}",
                "std": f"{float(row['std_min_ttc']):.3f}",
                "budget": f"{float(row['mean_realized_budget']):.3f}",
                "score": f"{float(row['composite_score']):.3f}",
            }
            for row in rows[:16]
        ]
    )
    print("\nWinners by representation (highest composite row)")
    for representation in ("object_set", "cost_map"):
        rep_rows = [row for row in rows if row["representation"] == representation]
        winner = max(rep_rows, key=lambda row: float(row["composite_score"]))
        print(
            f"{representation}: {winner['method']} ({winner['contract_status']}), "
            f"magnitude={winner['magnitude']}, score={float(winner['composite_score']):.3f}, "
            f"collision={float(winner['collision_rate']):.3f}, "
            f"mean_min_ttc={float(winner['mean_min_ttc']):.3f}, "
            f"std={float(winner['std_min_ttc']):.3f}, "
            f"budget={float(winner['mean_realized_budget']):.3f}"
        )
    overall = max(rows, key=lambda row: float(row["composite_score"]))
    print(
        f"Overall: {overall['representation']} / {overall['method']} "
        f"({overall['contract_status']}), magnitude={overall['magnitude']}, "
        f"score={float(overall['composite_score']):.3f}"
    )


def _print_ablation_results(
    contract: list[dict[str, Any]],
    directedness: list[dict[str, Any]],
) -> None:
    high_contract = [row for row in contract if row["magnitude"] == "high"]
    print("\nAblation A, high magnitude")
    _print_table(
        [
            {
                "rep": row["representation"],
                "comparison": row["comparison"],
                "legal_coll": f"{float(row['legal_collision_rate']):.3f}",
                "viol_coll": f"{float(row['violating_collision_rate']):.3f}",
                "legal_std": f"{float(row['legal_std_min_ttc']):.3f}",
                "viol_std": f"{float(row['violating_std_min_ttc']):.3f}",
                "std_gap": f"{float(row['std_min_ttc_gap_second_minus_first_s']):.3f}",
                "legal_budget": f"{float(row['legal_mean_realized_budget']):.3f}",
                "viol_budget": f"{float(row['violating_mean_realized_budget']):.3f}",
            }
            for row in high_contract
        ]
    )
    high_directed = [row for row in directedness if row["magnitude"] == "high"]
    print("\nAblation B, high magnitude")
    _print_table(
        [
            {
                "rep": row["representation"],
                "directed_coll": f"{float(row['directed_collision_rate']):.3f}",
                "random_coll": f"{float(row['random_collision_rate']):.3f}",
                "ttc_gap_rand-dir": f"{float(row['mean_min_ttc_gap_second_minus_first_s']):.3f}",
                "std_gap_rand-dir": f"{float(row['std_min_ttc_gap_second_minus_first_s']):.3f}",
                "directed_budget": f"{float(row['directed_mean_realized_budget']):.3f}",
                "random_budget": f"{float(row['random_mean_realized_budget']):.3f}",
            }
            for row in high_directed
        ]
    )


def _find_row(
    rows: list[dict[str, Any]],
    representation: str,
    method: str,
    magnitude: str,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["representation"] == representation
            and row["method"] == method
            and row["magnitude"] == magnitude
        ):
            return row
    raise KeyError((representation, method, magnitude))


def _sort_cost_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, name in enumerate(SCENARIOS)}
    magnitude_order = {name: idx for idx, name in enumerate(MAGNITUDES)}
    return sorted(
        records,
        key=lambda row: (
            scenario_order[str(row["scenario_id"])],
            magnitude_order[str(row["magnitude"])],
            int(row["seed"]),
        ),
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _minmax(values: Iterable[float]) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high <= low + EPS:
        return [0.0 for _ in arr]
    return [float((value - low) / (high - low)) for value in arr]


def _finite_metric(value: Any, *, cap: float) -> float:
    out = float(value)
    if math.isnan(out):
        return cap
    if math.isinf(out):
        return cap if out > 0.0 else 0.0
    return float(np.clip(out, 0.0, cap))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


if __name__ == "__main__":
    main()
