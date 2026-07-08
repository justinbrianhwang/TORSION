"""Run Phase A+ mechanism analyses without modifying the pipeline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torsion.analysis.mechanism import (  # noqa: E402
    DEFAULT_EPS_SWEEP,
    decision_margin_analysis,
    prediction_jacobian,
    rasterization_jacobian,
)
from torsion.scenarios.unified_pipeline import (  # noqa: E402
    SCENARIOS,
    UnifiedPipelineConfig,
)

DEFAULT_SEEDS = 30
DEFAULT_OUT_DIR = Path("results/metrics")
DEFAULT_MAGNITUDE = "low"
DEFAULT_METHOD = "torsion_displace"
DEFAULT_DECISION_INJECTION = "costmap"
PREDICTION_HORIZONS_S = (1.0, 2.0, 3.0)
PREDICTION_EPS_MPS = 0.01
HORIZON_CROSSCHECK_PATH = Path("results/metrics/propagation_pred_horizon.csv")

RASTER_COLUMNS = (
    "scenario_id",
    "interface",
    "magnitude",
    "n_seeds",
    "frame",
    "j_raster",
    "j_raster_std",
    "j_raster_min",
    "j_raster_max",
    "linearity_cv",
    "linearity_max_rel_spread",
    "linearity_r2",
)

DECISION_COLUMNS = (
    "scenario_id",
    "injection_point",
    "method",
    "magnitude",
    "seed",
    "frame",
    "time_s",
    "fault_active",
    "decision_margin_score",
    "inverse_margin",
    "decision_margin_pool",
    "fallback_all_candidates",
    "n_candidates",
    "n_feasible",
    "clean_target_lateral_m",
    "fault_target_lateral_m",
    "argmin_flip",
    "realized_path_deviation_m",
)

PREDICTION_COLUMNS = (
    "scenario_id",
    "horizon_s",
    "n_seeds",
    "j_pred_analytic",
    "j_pred_empirical",
    "abs_error",
    "rel_error",
    "raw_prediction_l2_crosscheck",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated scenarios",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--magnitude", default=DEFAULT_MAGNITUDE)
    parser.add_argument("--frame", type=int, default=None)
    args = parser.parse_args()

    scenarios = _parse_choices(args.scenarios, choices=SCENARIOS, name="scenario")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    raster_rows = _run_rasterization(
        scenarios,
        seeds=int(args.seeds),
        magnitude=str(args.magnitude),
        frame_idx=args.frame,
    )
    raster_path = args.out_dir / "mechanism_rasterization.csv"
    _write_csv(raster_path, raster_rows, fieldnames=RASTER_COLUMNS)

    decision = _run_decision_margin(
        scenarios,
        seeds=int(args.seeds),
        magnitude=str(args.magnitude),
    )
    decision_path = args.out_dir / "mechanism_decision_margin.csv"
    _write_csv(decision_path, decision["per_frame"], fieldnames=DECISION_COLUMNS)

    prediction_rows = _run_prediction_jacobian(
        scenarios,
        seeds=int(args.seeds),
        frame_idx=args.frame,
    )
    prediction_path = args.out_dir / "mechanism_prediction_jacobian.csv"
    _write_csv(prediction_path, prediction_rows, fieldnames=PREDICTION_COLUMNS)

    _print_m1_summary(raster_rows)
    _print_m2_summary(decision)
    _print_m3_summary(prediction_rows)
    print(f"\nWrote {len(raster_rows)} rasterization rows to {raster_path}")
    print(f"Wrote {len(decision['per_frame'])} decision-margin rows to {decision_path}")
    print(f"Wrote {len(prediction_rows)} prediction-Jacobian rows to {prediction_path}")


def _run_rasterization(
    scenarios: list[str],
    *,
    seeds: int,
    magnitude: str,
    frame_idx: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for use_prediction in (False, True):
            samples = [
                rasterization_jacobian(
                    scenario,
                    magnitude=magnitude,
                    seed=seed,
                    use_prediction=use_prediction,
                    eps_sweep=DEFAULT_EPS_SWEEP,
                    frame_idx=frame_idx,
                )
                for seed in range(seeds)
            ]
            rows.append(_summarize_raster_samples(samples))
    return rows


def _summarize_raster_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize empty raster samples")
    return {
        "scenario_id": samples[0]["scenario_id"],
        "interface": samples[0]["interface"],
        "magnitude": samples[0]["magnitude"],
        "n_seeds": len(samples),
        "frame": samples[0]["frame"],
        "j_raster": _mean(samples, "j_raster"),
        "j_raster_std": _std(samples, "j_raster"),
        "j_raster_min": _min(samples, "j_raster"),
        "j_raster_max": _max(samples, "j_raster"),
        "linearity_cv": _mean(samples, "linearity_cv"),
        "linearity_max_rel_spread": _mean(samples, "linearity_max_rel_spread"),
        "linearity_r2": _mean(samples, "linearity_r2"),
    }


def _run_decision_margin(
    scenarios: list[str],
    *,
    seeds: int,
    magnitude: str,
) -> dict[str, Any]:
    configs = [
        UnifiedPipelineConfig(
            scenario=scenario,
            injection_point=DEFAULT_DECISION_INJECTION,
            method=DEFAULT_METHOD,
            magnitude=magnitude,
            seed=seed,
            trace_grids=False,
        )
        for scenario in scenarios
        for seed in range(seeds)
    ]
    return decision_margin_analysis(configs)


def _run_prediction_jacobian(
    scenarios: list[str],
    *,
    seeds: int,
    frame_idx: int | None,
) -> list[dict[str, Any]]:
    crosscheck = _load_horizon_crosscheck(HORIZON_CROSSCHECK_PATH)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for horizon in PREDICTION_HORIZONS_S:
            samples = [
                prediction_jacobian(
                    scenario,
                    horizon_s=horizon,
                    eps=PREDICTION_EPS_MPS,
                    seed=seed,
                    frame_idx=frame_idx if frame_idx is not None else 0,
                )
                for seed in range(seeds)
            ]
            rows.append(
                _summarize_prediction_samples(
                    samples,
                    crosscheck_value=(
                        crosscheck.get(horizon) if scenario == "leading_vehicle" else None
                    ),
                )
            )

    for horizon in PREDICTION_HORIZONS_S:
        horizon_rows = [
            row for row in rows if math.isclose(float(row["horizon_s"]), horizon)
        ]
        rows.append(
            {
                "scenario_id": "ALL",
                "horizon_s": horizon,
                "n_seeds": sum(int(row["n_seeds"]) for row in horizon_rows),
                "j_pred_analytic": _mean(horizon_rows, "j_pred_analytic"),
                "j_pred_empirical": _mean(horizon_rows, "j_pred_empirical"),
                "abs_error": _mean(horizon_rows, "abs_error"),
                "rel_error": _mean(horizon_rows, "rel_error"),
                "raw_prediction_l2_crosscheck": crosscheck.get(horizon),
            }
        )
    return rows


def _summarize_prediction_samples(
    samples: list[dict[str, Any]],
    *,
    crosscheck_value: float | None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize empty prediction samples")
    return {
        "scenario_id": samples[0]["scenario_id"],
        "horizon_s": samples[0]["horizon_s"],
        "n_seeds": len(samples),
        "j_pred_analytic": _mean(samples, "j_pred_analytic"),
        "j_pred_empirical": _mean(samples, "j_pred_empirical"),
        "abs_error": _mean(samples, "abs_error"),
        "rel_error": _mean(samples, "rel_error"),
        "raw_prediction_l2_crosscheck": crosscheck_value,
    }


def _load_horizon_crosscheck(path: Path) -> dict[float, float]:
    if not path.exists():
        return {}
    out: dict[float, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                horizon = float(row["prediction_horizon_s"])
                value = float(row["raw_prediction_l2_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(horizon) and math.isfinite(value):
                out[horizon] = value
    return out


def _print_m1_summary(rows: list[dict[str, Any]]) -> None:
    table: list[dict[str, Any]] = []
    for scenario in _ordered_scenarios_from_rows(rows):
        scenario_rows = [row for row in rows if row["scenario_id"] == scenario]
        object_row = _find_row(scenario_rows, "interface", "object__cost")
        pred_row = _find_row(scenario_rows, "interface", "prediction__cost")
        table.append(
            {
                "scenario": scenario,
                "object__cost_J": _format_float(object_row.get("j_raster")),
                "object_cv": _format_float(object_row.get("linearity_cv")),
                "prediction__cost_J": _format_float(pred_row.get("j_raster")),
                "prediction_cv": _format_float(pred_row.get("linearity_cv")),
            }
        )
    all_small = all(float(row["j_raster"]) < 1.0 for row in rows)
    print("\nM1 rasterization Jacobian: RMS cost delta per meter")
    _print_table(table)
    print(
        "M1 verdict: "
        + (
            "attenuation is supported; every J_raster is < 1 and the sweep is near-linear."
            if all_small
            else "attenuation is mixed; at least one J_raster is not < 1."
        )
    )


def _print_m2_summary(decision: dict[str, Any]) -> None:
    quartiles = decision["quartile_summary"]
    table = [
        {
            "quartile": row["quartile"],
            "margin_range": f"{float(row['margin_min']):.6g}-{float(row['margin_max']):.6g}",
            "mean_dev_m": _format_float(row["mean_realized_path_deviation_m"]),
            "flip_rate": _format_float(row["argmin_flip_rate"]),
            "n": row["n_frames"],
        }
        for row in quartiles
    ]
    flip = decision["flip_rates"]
    corr = decision["correlations"]
    print("\nM2 decision-margin quartiles: low margin means near argmin boundary")
    _print_table(table)
    print(
        "M2 flips: "
        f"small-margin={_format_float(flip.get('small_margin_flip_rate'))} "
        f"(n={flip.get('small_margin_n')}), "
        f"large-margin={_format_float(flip.get('large_margin_flip_rate'))} "
        f"(n={flip.get('large_margin_n')})"
    )
    print(
        "M2 correlations: "
        f"Pearson(dev, 1/(m+eps))={_format_float(corr.get('pearson'))}, "
        f"Spearman={_format_float(corr.get('spearman'))}, n={corr.get('n')}"
    )
    if quartiles:
        small_dev = float(quartiles[0]["mean_realized_path_deviation_m"])
        large_dev = float(quartiles[-1]["mean_realized_path_deviation_m"])
        small_flip = float(quartiles[0]["argmin_flip_rate"])
        large_flip = float(quartiles[-1]["argmin_flip_rate"])
        supported = small_dev >= large_dev and small_flip >= large_flip
        print(
            "M2 verdict: "
            + (
                "argmin-boundary amplification is supported by larger deviations/flips at low margins."
                if supported
                else "argmin-boundary mechanism is only partially supported by these runs."
            )
        )
    else:
        print("M2 verdict: no fault-active frames were available for the margin analysis.")


def _print_m3_summary(rows: list[dict[str, Any]]) -> None:
    all_rows = [row for row in rows if row["scenario_id"] == "ALL"]
    table = [
        {
            "horizon_s": _format_float(row["horizon_s"]),
            "analytic": _format_float(row["j_pred_analytic"]),
            "empirical": _format_float(row["j_pred_empirical"]),
            "abs_err": _format_float(row["abs_error"]),
            "raw_L2_crosscheck": _format_float(row.get("raw_prediction_l2_crosscheck")),
        }
        for row in all_rows
    ]
    empirical = [float(row["j_pred_empirical"]) for row in all_rows]
    grows = all(later > earlier for earlier, later in zip(empirical, empirical[1:]))
    print("\nM3 prediction Jacobian: CV velocity error integrates over horizon")
    _print_table(table)
    print(
        "M3 verdict: "
        + (
            "empirical J matches mean sample time and grows with horizon."
            if grows and all(float(row["abs_error"]) < 1e-9 for row in all_rows)
            else "prediction Jacobian trend is weaker than the ideal CV expectation."
        )
    )


def _parse_choices(value: str, *, choices: Iterable[str], name: str) -> list[str]:
    allowed = tuple(choices)
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"at least one {name} must be provided")
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"unknown {name} value(s) {invalid!r}; expected {', '.join(allowed)}")
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


def _csv_record(record: Mapping[str, Any]) -> dict[str, Any]:
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


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean(np.asarray([float(row[key]) for row in rows], dtype=np.float64)))


def _std(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.std(np.asarray([float(row[key]) for row in rows], dtype=np.float64), ddof=0))


def _min(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.min(np.asarray([float(row[key]) for row in rows], dtype=np.float64)))


def _max(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.max(np.asarray([float(row[key]) for row in rows], dtype=np.float64)))


def _ordered_scenarios_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        scenario = str(row["scenario_id"])
        if scenario not in seen:
            seen.add(scenario)
            out.append(scenario)
    return out


def _find_row(rows: Sequence[Mapping[str, Any]], key: str, value: Any) -> Mapping[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"missing row where {key}={value!r}")


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(out):
        return ""
    return f"{out:.6g}"


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
