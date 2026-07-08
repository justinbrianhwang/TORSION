"""Data-grounded failure taxonomy for prediction-enabled unified runs.

This module is analysis-only.  It classifies already-computed unified pipeline
traces into propagation paths of the form

    fault origin -> dominant signature/interface -> failure mode

The signature is a planner switch when the chosen planner lateral target flips
on any fault-active frame; otherwise it is the largest defined raw interface
gain above one, or ``attenuated`` when no interface amplifies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

from torsion.analysis.mechanism import decision_margin_analysis
from torsion.analysis.propagation import (
    TTC_CENSOR_S,
    censor_ttc,
    compute_normalized_stage_errors,
    compute_stage_errors,
    interface_gains,
    propagation_depth_from_raw,
)
from torsion.scenarios.costmap_runner import CostMapPlannerConfig

PREDICTION_INTERFACES: tuple[str, ...] = (
    "object__prediction",
    "prediction__cost",
    "cost__plan",
    "plan__control",
    "control__safety",
)
FAILURE_MODES: tuple[str, ...] = (
    "collision",
    "off_road",
    "lane_departure",
    "hard_brake",
    "near_miss",
    "safe",
)
FAULT_ORIGINS: tuple[str, ...] = ("object", "prediction", "costmap", "none")
PLANNER_SWITCH_SIGNATURE = "planner_switch"
ATTENUATED_SIGNATURE = "attenuated"
RAW_GAIN_AMPLIFICATION_THRESHOLD = 1.0
NEAR_MISS_TTC_S = 1.5
SCENARIO_SPECIFIC_FREQ_THRESHOLD = 50.0
SCENARIO_SPECIFIC_MIN_COUNT = 2

_INTERFACE_STAGES: dict[str, tuple[str, str]] = {
    "object__prediction": ("object", "prediction"),
    "prediction__cost": ("prediction", "cost"),
    "cost__plan": ("cost", "plan"),
    "plan__control": ("plan", "control"),
    "control__safety": ("control", "safety"),
}


def classify_run(fault_result: Any, clean_result: Any) -> dict[str, Any]:
    """Classify one fault/clean paired unified run.

    Failure modes are assigned in the documented priority order:
    collision, off-road, lane departure, hard brake, near miss, then safe.
    Frequencies are not inferred here; this function only reports the measured
    path components for one run.
    """

    fault_origin = _result_injection_point(fault_result)
    use_prediction = _result_uses_prediction(fault_result)
    stage_error = compute_stage_errors(fault_result, clean_result)
    normalized = compute_normalized_stage_errors(stage_error, clean_result)
    defined_gains = interface_gains(
        normalized,
        fault_origin,
        use_prediction=use_prediction,
    )
    raw_gains = _raw_interface_gains(stage_error, defined_gains)
    dominant_interface, dominant_raw_gain = _dominant_interface(raw_gains)
    planner_gateway = (
        dominant_interface == "cost__plan"
        and dominant_raw_gain is not None
        and float(dominant_raw_gain) > RAW_GAIN_AMPLIFICATION_THRESHOLD
    )
    argmin_flipped = _argmin_flipped(fault_result)
    hard_brake_response = _has_hard_brake_response(fault_result)
    failure_mode = _failure_mode(
        fault_result,
        hard_brake_response=hard_brake_response,
    )
    min_ttc_censored = _min_ttc_censored(fault_result)
    _, deepest_stage = propagation_depth_from_raw(
        stage_error,
        fault_origin,
        use_prediction=use_prediction,
    )
    signature = (
        PLANNER_SWITCH_SIGNATURE if argmin_flipped else dominant_interface
    )

    record: dict[str, Any] = {
        "scenario": _result_scenario(fault_result),
        "scenario_id": _result_scenario(fault_result),
        "fault_origin": fault_origin,
        "signature": signature,
        "dominant_interface": dominant_interface,
        "dominant_raw_gain": dominant_raw_gain,
        "planner_gateway": bool(planner_gateway),
        "argmin_flipped": bool(argmin_flipped),
        "failure_mode": failure_mode,
        "method": _result_method(fault_result),
        "magnitude": _result_magnitude(fault_result),
        "seed": _result_seed(fault_result),
        "min_ttc": min_ttc_censored,
        "min_ttc_censored": min_ttc_censored,
        "collision": _has_collision(fault_result),
        "reach_safety": deepest_stage == "safety",
        "hard_brake_response": bool(hard_brake_response),
    }
    for interface in PREDICTION_INTERFACES:
        record[f"{interface}_raw_gain"] = raw_gains.get(interface)
    return record


def build_taxonomy(run_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate classified runs into counted propagation paths.

    ``freq_overall`` and ``freq_within_origin`` are percentages in the range
    0..100.  Empty inputs produce empty outputs rather than synthetic rows.
    """

    records = [dict(record) for record in run_records]
    total = len(records)
    if total == 0:
        return {"paths": [], "per_origin": {}, "scenario_specific": []}

    origin_counts: Counter[str] = Counter(str(row["fault_origin"]) for row in records)
    path_counts: Counter[tuple[str, str, str]] = Counter(
        (
            str(row["fault_origin"]),
            str(row["signature"]),
            str(row["failure_mode"]),
        )
        for row in records
    )

    paths = [
        {
            "fault_origin": fault_origin,
            "signature": signature,
            "failure_mode": failure_mode,
            "count": int(count),
            "freq_overall": _percent(count, total),
            "freq_within_origin": _percent(count, origin_counts[fault_origin]),
        }
        for (fault_origin, signature, failure_mode), count in path_counts.items()
    ]
    paths.sort(key=_path_sort_key)

    per_origin: dict[str, dict[str, Any]] = {}
    for fault_origin in _sorted_origins(origin_counts):
        n_origin = origin_counts[fault_origin]
        origin_paths = [
            row for row in paths if row["fault_origin"] == fault_origin
        ]
        dominant = origin_paths[0]
        origin_records = [
            row for row in records if str(row["fault_origin"]) == fault_origin
        ]
        collisions = sum(1 for row in origin_records if _truthy_bool(row.get("collision")))
        per_origin[fault_origin] = {
            "fault_origin": fault_origin,
            "n_runs": int(n_origin),
            "signature": dominant["signature"],
            "failure_mode": dominant["failure_mode"],
            "count": int(dominant["count"]),
            "freq_within_origin": float(dominant["freq_within_origin"]),
            "collision_rate": float(collisions / n_origin) if n_origin else 0.0,
        }

    return {
        "paths": paths,
        "per_origin": per_origin,
        "scenario_specific": _scenario_specific_paths(records),
    }


def _raw_interface_gains(
    stage_error: Any,
    defined_gains: Mapping[str, float | None],
) -> dict[str, float | None]:
    values = {
        "object": float(stage_error.object_shift_m),
        "prediction": float(stage_error.prediction_l2),
        "cost": float(stage_error.cost_l2),
        "plan": float(stage_error.plan_dev_m),
        "control": float(stage_error.control_dev),
        "safety": float(stage_error.safety_drop_s),
    }
    out: dict[str, float | None] = {}
    for interface in PREDICTION_INTERFACES:
        if defined_gains.get(interface) is None:
            out[interface] = None
            continue
        upstream, downstream = _INTERFACE_STAGES[interface]
        out[interface] = _safe_ratio(values[downstream], values[upstream])
    return out


def _dominant_interface(
    raw_gains: Mapping[str, float | None],
) -> tuple[str, float | None]:
    best_interface = ATTENUATED_SIGNATURE
    best_gain: float | None = None
    for interface in PREDICTION_INTERFACES:
        gain = raw_gains.get(interface)
        if gain is None:
            continue
        value = float(gain)
        if (
            math.isfinite(value)
            and value > RAW_GAIN_AMPLIFICATION_THRESHOLD
            and (best_gain is None or value > best_gain)
        ):
            best_interface = interface
            best_gain = value
    return best_interface, best_gain


def _failure_mode(result: Any, *, hard_brake_response: bool) -> str:
    if _has_collision(result):
        return "collision"
    trace = list(getattr(result, "trace", ()))
    if any(_truthy_bool(row.get("off_road")) for row in trace):
        return "off_road"
    if any(_truthy_bool(row.get("lane_departure")) for row in trace):
        return "lane_departure"
    if hard_brake_response:
        return "hard_brake"
    if _min_ttc_censored(result) < NEAR_MISS_TTC_S:
        return "near_miss"
    return "safe"


def _has_hard_brake_response(result: Any) -> bool:
    brake_threshold = 0.9 * _brake_accel_mps2(result)
    for row in getattr(result, "trace", ()):
        if not _truthy_bool(row.get("fault_active")):
            continue
        control = _mapping_or_empty(row.get("control"))
        chosen = _mapping_or_empty(row.get("chosen_path"))
        reason = f"{control.get('reason', '')} {chosen.get('reason', '')}".lower()
        if "brake" in reason:
            return True
        accel = control.get("accel_mps2", chosen.get("accel_mps2"))
        if _is_finite_number(accel) and float(accel) <= brake_threshold:
            return True
    return False


def _argmin_flipped(result: Any) -> bool:
    analysis = decision_margin_analysis(result)
    return any(bool(row["argmin_flip"]) for row in analysis["per_frame"])


def _scenario_specific_paths(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cell_counts: Counter[tuple[str, str]] = Counter()
    path_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in records:
        scenario = str(row.get("scenario", row.get("scenario_id", "")))
        fault_origin = str(row["fault_origin"])
        signature = str(row["signature"])
        failure_mode = str(row["failure_mode"])
        cell_counts[(scenario, fault_origin)] += 1
        path_counts[(scenario, fault_origin, signature, failure_mode)] += 1

    out: list[dict[str, Any]] = []
    for (scenario, fault_origin, signature, failure_mode), count in path_counts.items():
        cell_total = cell_counts[(scenario, fault_origin)]
        freq = _percent(count, cell_total)
        if count < SCENARIO_SPECIFIC_MIN_COUNT or freq < SCENARIO_SPECIFIC_FREQ_THRESHOLD:
            continue
        out.append(
            {
                "scenario": scenario,
                "fault_origin": fault_origin,
                "signature": signature,
                "failure_mode": failure_mode,
                "count": int(count),
                "cell_total": int(cell_total),
                "freq_within_scenario_origin": freq,
            }
        )
    out.sort(
        key=lambda row: (
            -float(row["freq_within_scenario_origin"]),
            -int(row["count"]),
            str(row["scenario"]),
            _origin_rank(str(row["fault_origin"])),
            str(row["signature"]),
            _failure_rank(str(row["failure_mode"])),
        )
    )
    return out


def _min_ttc_censored(result: Any) -> float:
    values = [
        censor_ttc(row["actual_ttc_s"], cap=TTC_CENSOR_S)
        for row in getattr(result, "trace", ())
        if "actual_ttc_s" in row
    ]
    if values:
        return float(min(values))
    summary = _summary(result)
    if "min_ttc" in summary:
        return censor_ttc(summary["min_ttc"], cap=TTC_CENSOR_S)
    return float(TTC_CENSOR_S)


def _has_collision(result: Any) -> bool:
    summary = _summary(result)
    if _truthy_bool(summary.get("collision")):
        return True
    return any(_truthy_bool(row.get("collision")) for row in getattr(result, "trace", ()))


def _result_scenario(result: Any) -> str:
    cfg = getattr(result, "config", None)
    if cfg is not None and hasattr(cfg, "scenario"):
        return str(getattr(cfg, "scenario"))
    return str(_summary(result).get("scenario_id", ""))


def _result_method(result: Any) -> str:
    cfg = getattr(result, "config", None)
    if cfg is None:
        return ""
    if hasattr(cfg, "method_key"):
        return str(getattr(cfg, "method_key"))
    return str(getattr(cfg, "method", ""))


def _result_magnitude(result: Any) -> str:
    cfg = getattr(result, "config", None)
    if cfg is None:
        return ""
    if hasattr(cfg, "magnitude_key"):
        return str(getattr(cfg, "magnitude_key"))
    return str(getattr(cfg, "magnitude", ""))


def _result_seed(result: Any) -> int:
    cfg = getattr(result, "config", None)
    if cfg is None:
        return 0
    return int(getattr(cfg, "seed", 0))


def _result_injection_point(result: Any) -> str:
    cfg = getattr(result, "config", None)
    if cfg is None:
        return "none"
    if hasattr(cfg, "injection_key"):
        return _injection_key(str(getattr(cfg, "injection_key")))
    return _injection_key(str(getattr(cfg, "injection_point", "none")))


def _result_uses_prediction(result: Any) -> bool:
    cfg = getattr(result, "config", None)
    if cfg is not None and bool(getattr(cfg, "use_prediction", False)):
        return True
    if _result_injection_point(result) == "prediction":
        return True
    for row in getattr(result, "trace", ()):
        value = row.get("prediction_traj_l2_delta")
        if _is_finite_number(value) and abs(float(value)) > 1e-12:
            return True
    return False


def _brake_accel_mps2(result: Any) -> float:
    cfg = getattr(result, "config", None)
    planner = getattr(cfg, "planner", None) if cfg is not None else None
    value = getattr(
        planner,
        "brake_accel_mps2",
        CostMapPlannerConfig().brake_accel_mps2,
    )
    if not _is_finite_number(value):
        return float(CostMapPlannerConfig().brake_accel_mps2)
    return float(value)


def _summary(result: Any) -> Mapping[str, Any]:
    summary = getattr(result, "summary", {})
    return summary if isinstance(summary, Mapping) else {}


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator / denominator)


def _percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(100.0 * count / total)


def _path_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["count"]),
        _origin_rank(str(row["fault_origin"])),
        str(row["signature"]),
        _failure_rank(str(row["failure_mode"])),
    )


def _sorted_origins(counts: Counter[str]) -> list[str]:
    return sorted(counts, key=lambda item: (_origin_rank(item), item))


def _origin_rank(value: str) -> int:
    try:
        return FAULT_ORIGINS.index(value)
    except ValueError:
        return len(FAULT_ORIGINS)


def _failure_rank(value: str) -> int:
    try:
        return FAILURE_MODES.index(value)
    except ValueError:
        return len(FAILURE_MODES)


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _truthy_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "none"}:
            return False
        if normalized in {"1", "true", "yes"}:
            return True
    return bool(value)


def _injection_key(value: str) -> str:
    key = str(value).lower().strip().replace("-", "_")
    if key in {"object", "object_set", "objects", "stage_a"}:
        return "object"
    if key in {"prediction", "predict", "stage_p"}:
        return "prediction"
    if key in {"costmap", "cost_map", "cost", "stage_b"}:
        return "costmap"
    if key in {"none", "clean"}:
        return "none"
    raise ValueError(f"unknown injection point {value!r}")


__all__ = [
    "ATTENUATED_SIGNATURE",
    "FAILURE_MODES",
    "NEAR_MISS_TTC_S",
    "PLANNER_SWITCH_SIGNATURE",
    "PREDICTION_INTERFACES",
    "build_taxonomy",
    "classify_run",
]
