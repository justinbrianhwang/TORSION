"""Create Phase 2b cost-map figures.

Figure 6 is regenerated directly from a representative run so the road/lane
boundary topology fix is visible.  Figure 7 is generated from the full raw
cost-map sweep CSV and shows magnitude response with bootstrap confidence
bands.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from torsion.metrics.statistics import summarize_safety_group
from torsion.scenarios.costmap_runner import (
    COSTMAP_METHODS,
    CostMapRunnerConfig,
    run_costmap_closed_loop,
)

FIGURE_DIR = Path("results/figures")
RUNS_PATH = Path("results/metrics/costmap_runs.csv")
MAGNITUDES = ("low", "medium", "high")
TTC_CENSOR_S = 5.0
METHOD_LABELS = {
    "clean": "clean",
    "cost_translate": "cost translate",
    "gaussian_cost": "gaussian cost",
    "random_warp_cost": "random warp cost",
    "torsion_swirl": "torsion swirl",
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure6_path = FIGURE_DIR / "figure6_costmap_overlay.png"
    make_figure6(figure6_path)
    print(f"Wrote {figure6_path}")

    if RUNS_PATH.exists():
        figure7_path = FIGURE_DIR / "figure7_costmap_magnitude_response.png"
        make_figure7(figure7_path, RUNS_PATH)
        print(f"Wrote {figure7_path}")
    else:
        print(f"Skipped Figure 7 because {RUNS_PATH} does not exist")


def make_figure6(path: Path) -> None:
    result = run_costmap_closed_loop(
        CostMapRunnerConfig(scenario="cut_in", method="torsion_swirl", magnitude="medium", seed=0)
    )
    row = _representative_frame(result.trace)
    clean = np.asarray(row["clean_cost_grid"], dtype=np.float64)
    warped = np.asarray(row["warped_cost_grid"], dtype=np.float64)
    diff = warped - clean
    spec = row["cost_grid_spec"]
    extent = [spec["x_min_m"], spec["x_max_m"], spec["y_min_m"], spec["y_max_m"]]
    clean_path = np.asarray(row["clean_reference_path"]["path_xy"], dtype=np.float64)
    warped_path = np.asarray(row["chosen_path"]["path_xy"], dtype=np.float64)

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.8))
    fig.suptitle(
        f"Figure 6. Topology-preserving cost-map twist (cut_in, frame {row['frame']})",
        fontsize=13,
    )

    _plot_cost(axes[0], clean, extent, "Clean cost map")
    _plot_cost(axes[1], warped, extent, "TORSION obstacle/free-space warp")

    vmax = max(float(np.max(np.abs(diff))), 1e-6)
    im = axes[2].imshow(
        diff,
        origin="upper",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[2].set_title("Difference heatmap")
    _format_axis(axes[2])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)

    _plot_cost(axes[3], warped, extent, "Chosen path overlay")
    axes[3].plot(
        clean_path[:, 0],
        clean_path[:, 1],
        color="#2b6cb0",
        linewidth=2.2,
        label="clean choice",
    )
    axes[3].plot(
        warped_path[:, 0],
        warped_path[:, 1],
        color="#c53030",
        linewidth=2.2,
        linestyle="--",
        label="faulted choice",
    )
    axes[3].legend(loc="upper right", fontsize=8)
    fig.text(
        0.5,
        0.045,
        "Road/lane-boundary cells are copied from the clean map; only the obstacle/free-space component is perturbed.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.20, wspace=0.35)

    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_figure7(path: Path, runs_path: Path = RUNS_PATH) -> None:
    records = _read_runs(runs_path)
    summaries = _aggregate_by_method_magnitude(records)
    x = np.arange(len(MAGNITUDES), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    fig.suptitle(
        "Figure 7. Budget-matched topology-preserving cost-map magnitude response",
        fontsize=13,
    )

    for method in COSTMAP_METHODS:
        rows = [summaries[(method, magnitude)] for magnitude in MAGNITUDES]
        label = METHOD_LABELS.get(method, method)
        collision = np.array([row["collision_rate"] for row in rows], dtype=np.float64)
        collision_low = np.array([row["collision_rate_ci_low"] for row in rows], dtype=np.float64)
        collision_high = np.array([row["collision_rate_ci_high"] for row in rows], dtype=np.float64)
        mean_ttc = np.array([row["mean_min_ttc"] for row in rows], dtype=np.float64)
        ttc_low = np.array([row["mean_min_ttc_ci_low"] for row in rows], dtype=np.float64)
        ttc_high = np.array([row["mean_min_ttc_ci_high"] for row in rows], dtype=np.float64)

        axes[0].plot(x, collision, marker="o", linewidth=2.0, label=label)
        axes[0].fill_between(x, collision_low, collision_high, alpha=0.14)
        axes[1].plot(x, mean_ttc, marker="o", linewidth=2.0, label=label)
        axes[1].fill_between(x, ttc_low, ttc_high, alpha=0.14)

    axes[0].set_title("Collision rate")
    axes[0].set_ylabel("rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Mean min-TTC")
    axes[1].set_ylabel("seconds, capped at 5 s")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(MAGNITUDES)
        ax.set_xlabel("magnitude / target path budget")
        ax.grid(True, color="#d8dee9", linewidth=0.7, alpha=0.8)
    axes[1].legend(loc="best", fontsize=8)
    fig.text(
        0.5,
        0.045,
        "Non-clean methods are calibrated to equal realized path L2 budget; road/lane-boundary topology is preserved.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.20, wspace=0.25)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _aggregate_by_method_magnitude(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for method in COSTMAP_METHODS:
        for magnitude in MAGNITUDES:
            group = [
                row
                for row in records
                if row["method"] == method and row["magnitude"] == magnitude
            ]
            summary = summarize_safety_group(
                collision=[float(row["collision"]) for row in group],
                min_ttc=[float(row["min_ttc_censored"]) for row in group],
                realized_budget=[float(row["mean_realized_budget"]) for row in group],
            )
            out[(method, magnitude)] = summary
    return out


def _read_runs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["collision"] = float(row["collision"])
        row["mean_realized_budget"] = _parse_float(row["mean_realized_budget"], default=0.0)
        min_ttc = _parse_float(row["min_ttc"], default=TTC_CENSOR_S)
        row["min_ttc_censored"] = min(max(min_ttc, 0.0), TTC_CENSOR_S)
    return rows


def _parse_float(value: Any, *, default: float) -> float:
    if value in ("", None):
        return default
    out = float(value)
    if math.isnan(out):
        return default
    if math.isinf(out):
        return default if out > 0.0 else 0.0
    return out


def _representative_frame(trace: tuple[dict, ...]) -> dict:
    active = [row for row in trace if row["fault_active"]]
    if not active:
        return trace[0]
    return max(active, key=lambda row: float(row["realized_path_deviation_m"]))


def _plot_cost(ax: plt.Axes, grid: np.ndarray, extent: list[float], title: str) -> None:
    im = ax.imshow(
        grid,
        origin="upper",
        extent=extent,
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_title(title)
    _format_axis(ax)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def _format_axis(ax: plt.Axes) -> None:
    ax.axhline(1.75, color="#1a202c", linewidth=0.6, alpha=0.55)
    ax.axhline(-1.75, color="#1a202c", linewidth=0.6, alpha=0.55)
    ax.axhline(0.0, color="#4a5568", linewidth=0.5, alpha=0.35)
    ax.set_xlim(-2.0, 44.0)
    ax.set_ylim(-6.2, 6.2)
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.grid(False)


if __name__ == "__main__":
    main()
