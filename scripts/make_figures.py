"""Create paper-style figures for the synthetic Phase 2a FAIR harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torsion.metrics.statistics import summarize_safety_group
from torsion.scenarios.predict import constant_velocity_predict
from torsion.scenarios.synthetic_runner import RunnerConfig, run_synthetic_closed_loop
from torsion.scenarios.synthetic_scenarios import get_scenario

METRICS_CSV = Path("results/metrics/synthetic_runs.csv")
FIGURE_DIR = Path("results/figures")
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


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig3_path = FIGURE_DIR / "figure3_error_propagation.png"
    fig4_path = FIGURE_DIR / "figure4_magnitude_response.png"
    fig5_path = FIGURE_DIR / "figure5_consistency.png"
    make_figure3(fig3_path)
    make_figure4(fig4_path)
    make_figure5(fig5_path)
    print(f"Wrote {fig3_path}")
    print(f"Wrote {fig4_path}")
    print(f"Wrote {fig5_path}")


def make_figure3(path: Path) -> None:
    result = run_synthetic_closed_loop(
        RunnerConfig(scenario="cut_in", method="torsion_swirl", magnitude="medium", seed=0)
    )
    clean_result = run_synthetic_closed_loop(
        RunnerConfig(scenario="cut_in", method="clean", magnitude="medium", seed=0)
    )
    scenario = get_scenario("cut_in", seed=0)
    snapshot_frame = 0
    gt_snapshot = scenario.object_set(snapshot_frame)
    clean_prediction = constant_velocity_predict(gt_snapshot, horizon_s=3.0, dt=scenario.dt)
    corrupt_prediction = result.trace[snapshot_frame]["predictions"][0]

    times = np.array([row["time_s"] for row in result.trace], dtype=np.float64)
    gt_yaw = np.rad2deg([row["gt_actors"][0]["yaw"] for row in result.trace])
    perceived_yaw = np.rad2deg([row["perceived_actors"][0]["yaw"] for row in result.trace])
    ttc = np.array([row["actual_ttc_s"] for row in result.trace], dtype=np.float64)
    ttc_plot = np.where(np.isfinite(ttc), ttc, np.nan)
    brake = np.array([row["control"]["brake"] for row in result.trace], dtype=np.float64)
    clearance = np.array([row["min_actor_distance_m"] for row in result.trace], dtype=np.float64)

    gt_actor_x = np.array([row["gt_actors"][0]["x"] for row in result.trace], dtype=np.float64)
    gt_actor_y = np.array([row["gt_actors"][0]["y"] for row in result.trace], dtype=np.float64)
    ego_x = np.array([row["ego"]["x"] for row in result.trace], dtype=np.float64)
    clean_ego_x = np.array([row["ego"]["x"] for row in clean_result.trace], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 10.0), constrained_layout=True)
    fig.suptitle(
        "Figure 3. Error propagation from object-set torsion (method: torsion_swirl)",
        fontsize=14,
    )

    axes[0].plot(times, gt_yaw, label="GT yaw", color="#2b6cb0", linewidth=2)
    axes[0].plot(times, perceived_yaw, label="Corrupted yaw", color="#c53030", linewidth=2)
    axes[0].axvspan(0.0, 3.0, color="#f6ad55", alpha=0.18, label="fault active")
    axes[0].set_ylabel("Actor yaw (deg)")
    axes[0].legend(loc="best", ncols=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    clean_traj = clean_prediction.trajectories[0]
    axes[1].plot(gt_actor_x, gt_actor_y, color="#2f855a", linewidth=2, label="GT actor path")
    axes[1].plot(clean_traj.xy[:, 0], clean_traj.xy[:, 1], "--", color="#2b6cb0", label="Clean prediction")
    axes[1].plot(corrupt_prediction["x"], corrupt_prediction["y"], "--", color="#c53030", label="TORSION prediction")
    axes[1].plot(ego_x, np.zeros_like(ego_x), color="#1a202c", linewidth=1.5, label="Ego path")
    axes[1].plot(clean_ego_x, np.zeros_like(clean_ego_x), color="#718096", linewidth=1, alpha=0.6, label="Clean ego ref")
    axes[1].axhline(1.2, color="#a0aec0", linewidth=0.8)
    axes[1].axhline(-1.2, color="#a0aec0", linewidth=0.8)
    axes[1].set_ylabel("Lateral y (m)")
    axes[1].set_xlabel("Longitudinal x (m)")
    axes[1].set_ylim(-1.8, 4.2)
    axes[1].legend(loc="best", ncols=2, fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(times, ttc_plot, color="#c53030", linewidth=2)
    axes[2].axhline(3.05, color="#805ad5", linestyle="--", linewidth=1, label="brake threshold")
    min_idx = int(np.nanargmin(ttc_plot))
    axes[2].scatter(times[min_idx], ttc_plot[min_idx], color="#1a202c", zorder=4, label="min TTC")
    axes[2].set_ylabel("Actual TTC (s)")
    axes[2].set_ylim(bottom=0.0)
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    ax_brake = axes[3]
    ax_clearance = ax_brake.twinx()
    ax_brake.step(times, brake, where="post", color="#c53030", linewidth=2, label="brake command")
    ax_clearance.plot(times, clearance, color="#2b6cb0", linewidth=2, label="clearance")
    ax_clearance.axhline(0.0, color="#1a202c", linestyle="--", linewidth=1)
    ax_brake.set_xlabel("Time (s)")
    ax_brake.set_ylabel("Brake")
    ax_clearance.set_ylabel("BBox clearance (m)")
    ax_brake.set_ylim(-0.02, 1.02)
    lines_a, labels_a = ax_brake.get_legend_handles_labels()
    lines_b, labels_b = ax_clearance.get_legend_handles_labels()
    ax_brake.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=8)
    ax_brake.grid(True, alpha=0.25)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figure4(path: Path) -> None:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(
            f"{METRICS_CSV} does not exist; run scripts/run_synthetic_experiment.py first"
        )

    df = pd.read_csv(METRICS_CSV)
    grouped = _summarize_method_magnitude(df)
    plot_rows = _with_magnitude_positions(grouped)
    colors = _method_colors()
    labels = _method_labels()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.84, wspace=0.24)
    fig.suptitle("Figure 4. FAIR magnitude-response curve", fontsize=14)

    for method in METHODS:
        rows = plot_rows[plot_rows["method"] == method].sort_values("magnitude_numeric")
        x = rows["magnitude_numeric"].to_numpy(dtype=np.float64)
        collision_rate = rows["collision_rate"].to_numpy(dtype=np.float64)
        collision_yerr = _asymmetric_yerr(rows, "collision_rate")
        mean_min_ttc = rows["mean_min_ttc"].to_numpy(dtype=np.float64)
        ttc_yerr = _asymmetric_yerr(rows, "mean_min_ttc")

        axes[0].errorbar(
            x,
            collision_rate,
            yerr=collision_yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            elinewidth=1.2,
            color=colors[method],
            label=labels[method],
        )
        axes[1].errorbar(
            x,
            mean_min_ttc,
            yerr=ttc_yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            elinewidth=1.2,
            color=colors[method],
            label=labels[method],
        )

    axes[0].set_xlabel("Magnitude level")
    axes[0].set_ylabel("Collision rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_xticks(range(len(MAGNITUDES)), MAGNITUDES)

    axes[1].set_xlabel("Magnitude level")
    axes[1].set_ylabel("Mean min TTC (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xticks(range(len(MAGNITUDES)), MAGNITUDES)
    axes[1].legend(loc="best", fontsize=8)
    fig.text(
        0.5,
        0.01,
        (
            "Budget matched on the target actor and active window using mean realized "
            "prediction L2 shift.\n"
            "Section 8.1 position levels define low/medium/high target budgets: "
            "0.2/0.5/1.0 m."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figure5(path: Path) -> None:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(
            f"{METRICS_CSV} does not exist; run scripts/run_synthetic_experiment.py first"
        )

    df = pd.read_csv(METRICS_CSV)
    high = df[df["magnitude"] == "high"].copy()
    colors = _method_colors()
    labels = _method_labels(multiline=True)

    fig, axes = plt.subplots(
        1,
        len(SCENARIOS),
        figsize=(11.4, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle("Figure 5. High-magnitude min-TTC consistency and tail risk", fontsize=14)

    for ax, scenario in zip(np.ravel(axes), SCENARIOS, strict=True):
        scenario_rows = high[high["scenario_id"] == scenario]
        data = [
            scenario_rows[scenario_rows["method"] == method]["min_ttc"].to_numpy(
                dtype=np.float64
            )
            for method in METHODS
        ]
        positions = np.arange(1, len(METHODS) + 1)
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#1a202c", "linewidth": 1.5},
            whiskerprops={"color": "#4a5568", "linewidth": 1.0},
            capprops={"color": "#4a5568", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#a0aec0",
                "markeredgecolor": "#4a5568",
                "alpha": 0.7,
            },
        )
        for patch, method in zip(box["boxes"], METHODS, strict=True):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.28)
            patch.set_edgecolor(colors[method])
            patch.set_linewidth(1.5)

        for position, method, values in zip(positions, METHODS, data, strict=True):
            p5 = float(np.percentile(values, 5.0))
            ax.scatter(
                position,
                p5,
                marker="D",
                s=34,
                facecolor=colors[method],
                edgecolor="#1a202c",
                linewidth=0.6,
                zorder=4,
            )
            ax.hlines(
                p5,
                position - 0.23,
                position + 0.23,
                colors=colors[method],
                linewidth=1.5,
                zorder=3,
            )

        ax.set_title(scenario.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[method] for method in METHODS], fontsize=8)
        ax.set_xlabel("Method")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Min TTC across seeds (s); lower is worse")
    axes[-1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="#1a202c",
                markerfacecolor="#edf2f7",
                linestyle="",
                markersize=6,
                label="5th percentile",
            )
        ],
        loc="best",
        fontsize=8,
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _with_magnitude_positions(grouped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in grouped.to_dict(orient="records"):
        magnitude = str(row["magnitude"])
        rows.append(
            {
                **row,
                "magnitude_numeric": MAGNITUDES.index(magnitude),
            }
        )
    return pd.DataFrame.from_records(rows)


def _method_colors() -> dict[str, str]:
    return {
        "clean": "#2b6cb0",
        "gaussian_matched": "#718096",
        "random_warp": "#2f855a",
        "torsion_translate": "#c53030",
        "torsion_swirl": "#805ad5",
        "torsion_curl": "#d69e2e",
        "torsion_combined": "#319795",
    }


def _method_labels(*, multiline: bool = False) -> dict[str, str]:
    sep = "\n" if multiline else " "
    return {
        "clean": "Clean",
        "gaussian_matched": f"Gaussian{sep}matched",
        "random_warp": f"Random{sep}warp",
        "torsion_translate": f"Torsion{sep}translate",
        "torsion_swirl": f"Torsion{sep}swirl",
        "torsion_curl": f"Torsion{sep}curl",
        "torsion_combined": f"Torsion{sep}combined",
    }


def _summarize_method_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, magnitude), group in df.groupby(["method", "magnitude"], sort=False):
        record = {"method": str(method), "magnitude": str(magnitude)}
        record.update(
            summarize_safety_group(
                collision=group["collision"],
                min_ttc=group["min_ttc"],
                realized_budget=group["mean_realized_budget"],
            )
        )
        rows.append(record)

    out = pd.DataFrame.from_records(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHODS, ordered=True)
    out["magnitude"] = pd.Categorical(out["magnitude"], categories=MAGNITUDES, ordered=True)
    out = out.sort_values(["method", "magnitude"]).reset_index(drop=True)
    out["method"] = out["method"].astype(str)
    out["magnitude"] = out["magnitude"].astype(str)
    return out


def _asymmetric_yerr(rows: pd.DataFrame, metric: str) -> np.ndarray:
    value = rows[metric].to_numpy(dtype=np.float64)
    low = rows[f"{metric}_ci_low"].to_numpy(dtype=np.float64)
    high = rows[f"{metric}_ci_high"].to_numpy(dtype=np.float64)
    return np.vstack([np.maximum(value - low, 0.0), np.maximum(high - value, 0.0)])


if __name__ == "__main__":
    main()
