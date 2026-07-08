"""Cross-representation fault-propagation metrics for unified traces.

The functions in this module are analysis-only: they consume
``UnifiedRunResult.trace`` rows that already exist and do not alter any
operators, injection logic, or closed-loop execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from torsion.operators.costmap import COST_MAX, COST_MIN
from torsion.scenarios.costmap_runner import CostMapPlannerConfig

TTC_CENSOR_S = 5.0
EPS_RATIO = 1e-12
EPS_OBJECT_M = 0.1
EPS_PREDICTION_M = 0.1
EPS_COST = 1e-12
PLAN_L_REF_M = 1.0
CONTROL_L_REF_M = 1.0
A_REF_MPS2 = abs(float(CostMapPlannerConfig().hard_brake_accel_mps2))

STAGE_ORDER = ("object", "cost", "plan", "control", "safety")
STAGE_ORDER_PRED = ("object", "prediction", "cost", "plan", "control", "safety")
INJECTION_TO_STAGE = {
    "object": "object",
    "prediction": "prediction",
    "costmap": "cost",
    "none": "",
}

# Physically-motivated detectability thresholds for whether a stage carries a
# fault-induced raw error. These are a modeling choice stated in the paper.
STAGE_DETECT_FLOORS: dict[str, float] = {
    "object": 0.05,
    "prediction": 0.05,
    "cost": 0.02,
    "plan": 0.05,
    "control": 0.02,
    "safety": 0.10,
}


@dataclass(frozen=True)
class StageError:
    """Raw per-stage errors aggregated over fault-active frames."""

    object_shift_m: float
    cost_l2: float
    plan_dev_m: float
    control_dev: float
    safety_drop_s: float
    collision_delta: float
    prediction_l2: float = 0.0


@dataclass(frozen=True)
class StageScales:
    """Characteristic scales used to normalize stage errors."""

    object_shift_m: float
    cost_l2: float
    plan_dev_m: float
    control_dev: float
    safety_drop_s: float
    collision_delta: float
    prediction_l2: float = EPS_PREDICTION_M


@dataclass(frozen=True)
class NormalizedStageError:
    """Unitless stage errors under the fixed Phase A normalization scheme."""

    object_shift: float
    cost: float
    plan: float
    control: float
    safety: float
    collision_delta: float
    prediction: float = 0.0


@dataclass(frozen=True)
class RecoveryTime:
    """Post-fault lateral recovery result."""

    recovery_time_s: float
    recovered: bool


def compute_stage_errors(fault_result: Any, clean_result: Any) -> StageError:
    """Compute raw stage errors from one fault run and its clean twin."""

    active_rows = _active_rows(fault_result)
    object_shift = _mean_trace_value(active_rows, "object_position_shift_m")
    prediction_l2 = _mean_trace_value(
        active_rows,
        "prediction_traj_l2_delta",
        default=0.0,
    )
    cost_l2 = _mean_trace_value(active_rows, "cost_grid_l2_delta")
    plan_dev = _mean_trace_value(active_rows, "realized_path_deviation_m")
    control_dev = _mean_float(_control_frame_deviation(row) for row in active_rows)
    clean_min_ttc = _min_ttc_censored(clean_result)
    fault_min_ttc = _min_ttc_censored(fault_result)
    clean_collision = 1.0 if _has_collision(clean_result) else 0.0
    fault_collision = 1.0 if _has_collision(fault_result) else 0.0
    return StageError(
        object_shift_m=object_shift,
        cost_l2=cost_l2,
        plan_dev_m=plan_dev,
        control_dev=control_dev,
        safety_drop_s=float(clean_min_ttc - fault_min_ttc),
        collision_delta=float(fault_collision - clean_collision),
        prediction_l2=prediction_l2,
    )


def compute_stage_scales(clean_result: Any) -> StageScales:
    """Return clean-twin scales for unitless cross-stage comparisons."""

    # Object normalization uses the clean target actor's nominal frame-to-frame
    # motion variation. A 0.1 m floor avoids singular ratios for static or
    # constant-speed actors, and is the documented fixed fallback reference.
    object_scale = max(_target_motion_std_m(clean_result), EPS_OBJECT_M)

    # Prediction errors are measured in meters over the CV horizon; 0.1 m is the
    # fixed reference epsilon matching the object-stage fallback floor.
    prediction_scale = EPS_PREDICTION_M

    # Cost maps are clipped to [COST_MIN, COST_MAX], so the fixed dynamic range
    # is the only meaningful clean reference; clean cost deltas are near zero.
    cost_scale = max(float(COST_MAX - COST_MIN), EPS_COST)

    # Plan error is measured in meters; 1.0 m is the fixed lateral/path budget
    # reference used for Phase A comparability.
    plan_scale = PLAN_L_REF_M

    # control_dev is already dimensionless:
    # |delta accel| / hard-brake magnitude + |delta lateral target| / 1.0 m.
    control_scale = 1.0

    # Safety is normalized by the same min-TTC censoring cap used by the unified
    # pipeline, so a full 5 s drop maps to a unit normalized loss.
    safety_scale = TTC_CENSOR_S

    # Collision delta is binary and already unitless.
    collision_scale = 1.0
    return StageScales(
        object_shift_m=float(object_scale),
        cost_l2=float(cost_scale),
        plan_dev_m=float(plan_scale),
        control_dev=float(control_scale),
        safety_drop_s=float(safety_scale),
        collision_delta=float(collision_scale),
        prediction_l2=float(prediction_scale),
    )


def compute_normalized_stage_errors(
    stage_error: StageError,
    clean_result: Any,
) -> NormalizedStageError:
    """Normalize raw stage errors by the Phase A clean-reference scales."""

    scales = compute_stage_scales(clean_result)
    return NormalizedStageError(
        object_shift=float(stage_error.object_shift_m / max(scales.object_shift_m, EPS_RATIO)),
        prediction=float(stage_error.prediction_l2 / max(scales.prediction_l2, EPS_RATIO)),
        cost=float(stage_error.cost_l2 / max(scales.cost_l2, EPS_RATIO)),
        plan=float(stage_error.plan_dev_m / max(scales.plan_dev_m, EPS_RATIO)),
        control=float(stage_error.control_dev / max(scales.control_dev, EPS_RATIO)),
        safety=float(stage_error.safety_drop_s / max(scales.safety_drop_s, EPS_RATIO)),
        collision_delta=float(
            stage_error.collision_delta / max(scales.collision_delta, EPS_RATIO)
        ),
    )


def fault_amplification_ratio(normalized: NormalizedStageError) -> float | None:
    """Return FAR = normalized safety loss divided by normalized plan error."""

    return _safe_ratio(normalized.safety, normalized.plan)


def source_fault_amplification_ratio(
    normalized: NormalizedStageError,
    injection_point: str,
    *,
    use_prediction: bool | None = None,
) -> float | None:
    """Return source FAR using the injected representation as denominator."""

    injection_key = _injection_key(injection_point)
    if injection_key == "object":
        return _safe_ratio(normalized.safety, normalized.object_shift)
    if injection_key == "prediction":
        return _safe_ratio(normalized.safety, normalized.prediction)
    if injection_key == "costmap":
        return _safe_ratio(normalized.safety, normalized.cost)
    return None


def interface_gains(
    normalized: NormalizedStageError,
    injection_point: str,
    *,
    use_prediction: bool | None = None,
) -> dict[str, float | None]:
    """Return the downstream interface-gain vector for one paired run."""

    injection_key = _injection_key(injection_point)
    prediction_enabled = (
        bool(use_prediction)
        if use_prediction is not None
        else injection_key == "prediction" or abs(float(normalized.prediction)) > EPS_RATIO
    )
    if prediction_enabled:
        return {
            "object__cost": None,
            "object__prediction": (
                _safe_ratio(normalized.prediction, normalized.object_shift)
                if injection_key == "object"
                else None
            ),
            "prediction__cost": (
                _safe_ratio(normalized.cost, normalized.prediction)
                if injection_key in {"object", "prediction"}
                else None
            ),
            "cost__plan": _safe_ratio(normalized.plan, normalized.cost),
            "plan__control": _safe_ratio(normalized.control, normalized.plan),
            "control__safety": _safe_ratio(normalized.safety, normalized.control),
        }
    return {
        "object__cost": (
            _safe_ratio(normalized.cost, normalized.object_shift)
            if injection_key == "object"
            else None
        ),
        "object__prediction": None,
        "prediction__cost": None,
        "cost__plan": _safe_ratio(normalized.plan, normalized.cost),
        "plan__control": _safe_ratio(normalized.control, normalized.plan),
        "control__safety": _safe_ratio(normalized.safety, normalized.control),
    }


def attenuation_ratios(
    gains: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Return max(0, 1 - gain) for each defined interface gain."""

    return {
        name: None if gain is None else float(max(0.0, 1.0 - gain))
        for name, gain in gains.items()
    }


def propagation_depth(
    fault_result: Any,
    clean_result: Any,
    *,
    floors: Mapping[str, float] | None = None,
) -> tuple[int, str | None]:
    """Return consecutive downstream stages exceeding raw detectability floors."""

    stage_error = compute_stage_errors(fault_result, clean_result)
    return propagation_depth_from_raw(
        stage_error,
        _result_injection_point(fault_result),
        floors=floors,
        use_prediction=_result_uses_prediction(fault_result),
    )


def propagation_depth_from_raw(
    stage_error: StageError,
    injection_point: str,
    *,
    floors: Mapping[str, float] | None = None,
    use_prediction: bool | None = None,
) -> tuple[int, str | None]:
    """Return propagation depth from precomputed raw stage errors."""

    injection_key = _injection_key(injection_point)
    injection_stage = INJECTION_TO_STAGE.get(injection_key, "")
    if not injection_stage:
        return 0, None
    stage_order = _stage_order_for_raw(
        stage_error,
        injection_key,
        use_prediction=use_prediction,
    )

    stage_values = {
        "object": stage_error.object_shift_m,
        "prediction": stage_error.prediction_l2,
        "cost": stage_error.cost_l2,
        "plan": stage_error.plan_dev_m,
        "control": stage_error.control_dev,
        "safety": stage_error.safety_drop_s,
    }
    detect_floors = _stage_detect_floors(floors, stage_order=stage_order)
    start = stage_order.index(injection_stage)
    depth = 0
    deepest: str | None = None
    for stage in stage_order[start:]:
        value = float(stage_values[stage])
        if not math.isfinite(value) or value <= detect_floors[stage]:
            break
        depth += 1
        deepest = stage
    return depth, deepest


def critical_interface_score(
    safety_drop_s: Iterable[float],
    plan_dev_m: Iterable[float],
) -> float | None:
    """Return paired safety drop per unit matched plan budget."""

    drops = _finite_array(safety_drop_s, name="safety_drop_s")
    budgets = _finite_array(plan_dev_m, name="plan_dev_m")
    if drops.size != budgets.size:
        raise ValueError("safety_drop_s and plan_dev_m must have equal length")
    denominator = float(np.mean(budgets))
    if denominator <= EPS_RATIO:
        return None
    numerator = float(np.mean(np.maximum(drops, 0.0)))
    return float(numerator / denominator)


def recovery_time(
    fault_result: Any,
    clean_result: Any,
    *,
    eps_recover_m: float = 0.2,
) -> RecoveryTime:
    """Return time after fault end until lateral offset matches the clean twin."""

    if eps_recover_m < 0.0 or not math.isfinite(float(eps_recover_m)):
        raise ValueError("eps_recover_m must be non-negative and finite")
    if not _active_rows(fault_result):
        return RecoveryTime(recovery_time_s=0.0, recovered=True)

    fault_rows = list(getattr(fault_result, "trace"))
    clean_rows = list(getattr(clean_result, "trace"))
    frame_to_clean = {int(row["frame"]): row for row in clean_rows}
    aligned: list[tuple[float, float]] = []
    for fault_row in fault_rows:
        clean_row = frame_to_clean.get(int(fault_row["frame"]))
        if clean_row is None:
            continue
        diff = abs(_ego_y(fault_row) - _ego_y(clean_row))
        aligned.append((float(fault_row["time_s"]), float(diff)))

    if not aligned:
        return RecoveryTime(recovery_time_s=0.0, recovered=False)

    fault_end = _fault_end_time_s(fault_result)
    post = [(time_s, diff) for time_s, diff in aligned if time_s >= fault_end]
    last_time = float(aligned[-1][0])
    if not post:
        remaining = max(0.0, last_time - fault_end)
        return RecoveryTime(recovery_time_s=float(remaining), recovered=False)

    within = np.asarray([diff <= eps_recover_m for _, diff in post], dtype=bool)
    suffix_ok = np.flip(np.cumprod(np.flip(within).astype(np.int64))).astype(bool)
    for (time_s, _), ok in zip(post, suffix_ok, strict=True):
        if ok:
            return RecoveryTime(
                recovery_time_s=float(max(0.0, time_s - fault_end)),
                recovered=True,
            )

    remaining = max(0.0, last_time - fault_end)
    return RecoveryTime(recovery_time_s=float(remaining), recovered=False)


def compute_propagation_metrics(
    fault_result: Any,
    clean_result: Any,
    *,
    floors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return one flat metric record for a paired fault/clean run."""

    stage_error = compute_stage_errors(fault_result, clean_result)
    scales = compute_stage_scales(clean_result)
    normalized = compute_normalized_stage_errors(stage_error, clean_result)
    uses_prediction = _result_uses_prediction(fault_result)
    gains = interface_gains(
        normalized,
        _result_injection_point(fault_result),
        use_prediction=uses_prediction,
    )
    depth, deepest_stage = propagation_depth_from_raw(
        stage_error,
        _result_injection_point(fault_result),
        floors=floors,
        use_prediction=uses_prediction,
    )
    recovery = recovery_time(fault_result, clean_result)
    record = {
        **_result_identity(fault_result),
        **_prefix_mapping("raw_", asdict(stage_error)),
        **_prefix_mapping("scale_", asdict(scales)),
        "n_object": normalized.object_shift,
        "n_prediction": normalized.prediction,
        "n_cost": normalized.cost,
        "n_plan": normalized.plan,
        "n_control": normalized.control,
        "n_safety": normalized.safety,
        "n_collision_delta": normalized.collision_delta,
        "object__cost_gain": gains["object__cost"],
        "object__prediction_gain": gains["object__prediction"],
        "prediction__cost_gain": gains["prediction__cost"],
        "cost__plan_gain": gains["cost__plan"],
        "plan__control_gain": gains["plan__control"],
        "control__safety_gain": gains["control__safety"],
        "far": fault_amplification_ratio(normalized),
        "far_source": source_fault_amplification_ratio(
            normalized,
            _result_injection_point(fault_result),
            use_prediction=uses_prediction,
        ),
        "propagation_depth": depth,
        "deepest_stage": deepest_stage,
        "reached_safety": deepest_stage == "safety",
        "recovery_time_s": recovery.recovery_time_s,
        "recovered": recovery.recovered,
    }
    return record


def censor_ttc(value: Any, cap: float = TTC_CENSOR_S) -> float:
    """Return min-TTC censored like the unified pipeline's finite metric helper."""

    out = float(value)
    if math.isnan(out):
        return float(cap)
    if math.isinf(out):
        return float(cap if out > 0.0 else 0.0)
    return float(np.clip(out, 0.0, cap))


def _active_rows(result: Any) -> list[Mapping[str, Any]]:
    return [row for row in getattr(result, "trace") if bool(row.get("fault_active", False))]


def _mean_trace_value(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    default: float | None = None,
) -> float:
    values: list[float] = []
    for row in rows:
        if key not in row:
            if default is None:
                raise KeyError(key)
            values.append(float(default))
        else:
            values.append(float(row[key]))
    return _mean_float(values)


def _mean_float(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("metric values must be finite")
    return float(np.mean(array))


def _control_frame_deviation(row: Mapping[str, Any]) -> float:
    chosen = _mapping(row["chosen_path"], name="chosen_path")
    clean = _mapping(row["clean_reference_path"], name="clean_reference_path")
    accel = abs(_mapping_float(chosen, "accel_mps2") - _mapping_float(clean, "accel_mps2"))
    lateral = abs(
        _mapping_float(chosen, "target_lateral_m")
        - _mapping_float(clean, "target_lateral_m")
    )
    return float(accel / max(A_REF_MPS2, EPS_RATIO) + lateral / CONTROL_L_REF_M)


def _min_ttc_censored(result: Any) -> float:
    trace = list(getattr(result, "trace"))
    if not trace:
        return TTC_CENSOR_S
    values = [censor_ttc(row["actual_ttc_s"], cap=TTC_CENSOR_S) for row in trace]
    return float(np.min(np.asarray(values, dtype=np.float64)))


def _has_collision(result: Any) -> bool:
    summary = getattr(result, "summary", {})
    if "collision" in summary:
        return bool(summary["collision"])
    return any(bool(row["collision"]) for row in getattr(result, "trace"))


def _target_motion_std_m(clean_result: Any) -> float:
    target_actor = getattr(clean_result, "summary", {}).get("target_actor")
    positions: list[tuple[float, float]] = []
    for row in getattr(clean_result, "trace"):
        objects = row.get("clean_object_set", ())
        target = _find_object_record(objects, target_actor)
        if target is None:
            continue
        positions.append((float(target["x"]), float(target["y"])))
    if len(positions) < 2:
        return EPS_OBJECT_M
    array = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(np.diff(array, axis=0), axis=1)
    if distances.size == 0 or not np.all(np.isfinite(distances)):
        return EPS_OBJECT_M
    return float(np.std(distances, ddof=0))


def _find_object_record(
    records: Any,
    target_actor: Any,
) -> Mapping[str, Any] | None:
    if target_actor is None:
        return None
    for record in records:
        if record.get("track_id") == target_actor:
            return record
    return None


def _result_identity(result: Any) -> dict[str, Any]:
    cfg = getattr(result, "config")
    return {
        "scenario_id": str(getattr(cfg, "scenario")),
        "injection_point": _result_injection_point(result),
        "method": str(getattr(cfg, "method_key", getattr(cfg, "method", ""))),
        "magnitude": str(getattr(cfg, "magnitude_key", getattr(cfg, "magnitude", ""))),
        "seed": int(getattr(cfg, "seed")),
    }


def _result_injection_point(result: Any) -> str:
    cfg = getattr(result, "config")
    value = getattr(cfg, "injection_key", getattr(cfg, "injection_point", "none"))
    return _injection_key(str(value))


def _result_uses_prediction(result: Any) -> bool:
    cfg = getattr(result, "config", None)
    if cfg is not None and bool(getattr(cfg, "use_prediction", False)):
        return True
    if _result_injection_point(result) == "prediction":
        return True
    for row in getattr(result, "trace", ()):
        value = row.get("prediction_traj_l2_delta")
        if value is not None and abs(float(value)) > EPS_RATIO:
            return True
    return False


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


def _stage_order_for_raw(
    stage_error: StageError,
    injection_key: str,
    *,
    use_prediction: bool | None,
) -> tuple[str, ...]:
    if injection_key == "prediction":
        return STAGE_ORDER_PRED
    if use_prediction is not None:
        return STAGE_ORDER_PRED if use_prediction else STAGE_ORDER
    if abs(float(stage_error.prediction_l2)) > EPS_RATIO:
        return STAGE_ORDER_PRED
    return STAGE_ORDER


def _stage_detect_floors(
    floors: Mapping[str, float] | None,
    *,
    stage_order: tuple[str, ...],
) -> dict[str, float]:
    source: dict[str, float] = dict(STAGE_DETECT_FLOORS)
    if floors is not None:
        unknown = set(floors) - set(STAGE_DETECT_FLOORS)
        if unknown:
            raise ValueError(f"unknown detectability floor stage(s) {sorted(unknown)!r}")
        source.update(floors)
    out: dict[str, float] = {}
    for stage in stage_order:
        value = float(source[stage])
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("detectability floors must be non-negative and finite")
        out[stage] = value
    return out


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if abs(float(denominator)) <= EPS_RATIO:
        return None
    return float(numerator / denominator)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _mapping_float(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _ego_y(row: Mapping[str, Any]) -> float:
    ego = _mapping(row["ego"], name="ego")
    return _mapping_float(ego, "y")


def _fault_end_time_s(result: Any) -> float:
    summary = getattr(result, "summary", {})
    if "fault_end_time_s" in summary:
        return float(summary["fault_end_time_s"])
    active = _active_rows(result)
    if not active:
        return 0.0
    return float(active[-1]["time_s"])


def _prefix_mapping(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _finite_array(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
