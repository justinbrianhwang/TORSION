"""Transfer-function characterization for unified pipeline interfaces.

This module is analysis-only.  It treats each representation boundary as a
stage transfer element and measures raw physical gain as downstream stage error
divided by upstream stage error.  Linearity is judged by gain constancy across
input magnitudes, not by products of normalized gains.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from torsion.analysis.propagation import EPS_RATIO, StageError, compute_stage_errors

INTERFACE_ORDER: tuple[str, ...] = (
    "object__prediction",
    "prediction__cost",
    "cost__plan",
    "plan__control",
    "control__safety",
)

INTERFACE_STAGES: dict[str, tuple[str, str]] = {
    "object__prediction": ("object", "prediction"),
    "prediction__cost": ("prediction", "cost"),
    "cost__plan": ("cost", "plan"),
    "plan__control": ("plan", "control"),
    "control__safety": ("control", "safety"),
}

STAGE_ORDER = ("object", "cost", "plan", "control", "safety")
STAGE_ORDER_PRED = ("object", "prediction", "cost", "plan", "control", "safety")
INJECTION_TO_STAGE = {
    "object": "object",
    "prediction": "prediction",
    "costmap": "cost",
    "none": "",
}

DEFAULT_LINEAR_CV_THRESHOLD = 0.15
DEFAULT_LINEAR_NORMALIZED_SLOPE_THRESHOLD = 0.25


def interface_raw_gains(
    fault_result: Any,
    clean_result: Any,
) -> dict[str, float | None]:
    """Return raw downstream/upstream gains for one paired fault/clean run.

    Gains are reported only for interfaces at or downstream of the injection
    point, and only when the upstream raw error is nonzero.  The prediction
    interfaces are unavailable when the run did not use the prediction stage.
    """

    stage_error = compute_stage_errors(fault_result, clean_result)
    injection_key = _result_injection_point(fault_result)
    injection_stage = INJECTION_TO_STAGE.get(injection_key, "")
    stage_order = _stage_order_for_result(fault_result, stage_error)
    raw_values = _stage_error_values(stage_error)

    out: dict[str, float | None] = {name: None for name in INTERFACE_ORDER}
    if not injection_stage or injection_stage not in stage_order:
        return out

    injection_index = stage_order.index(injection_stage)
    for interface in INTERFACE_ORDER:
        upstream, downstream = INTERFACE_STAGES[interface]
        if upstream not in stage_order or downstream not in stage_order:
            continue
        if stage_order.index(upstream) < injection_index:
            continue
        out[interface] = _safe_ratio(raw_values[downstream], raw_values[upstream])
    return out


def characterize_linearity(
    gain_by_budget: Mapping[float, float | None],
    *,
    cv_threshold: float = DEFAULT_LINEAR_CV_THRESHOLD,
    normalized_slope_threshold: float = DEFAULT_LINEAR_NORMALIZED_SLOPE_THRESHOLD,
    eps: float = EPS_RATIO,
) -> dict[str, Any]:
    """Summarize whether gain is approximately constant across budgets.

    The default documented rule is ``linear`` when ``gain_cv < 0.15`` and
    ``abs(normalized_slope) < 0.25``.  The normalized slope is
    ``gain_slope * mean_budget / mean_gain``.  CV uses ``abs(mean_gain)`` in the
    denominator so it remains nonnegative for signed gains.
    """

    _validate_threshold(cv_threshold, name="cv_threshold")
    _validate_threshold(normalized_slope_threshold, name="normalized_slope_threshold")
    if eps <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite")

    pairs = _finite_pairs(gain_by_budget)
    if not pairs:
        return {
            "mean_gain": None,
            "gain_cv": None,
            "gain_slope": None,
            "normalized_slope": None,
            "norm_slope": None,
            "monotonic": False,
            "verdict": "nonlinear",
        }

    budgets = np.asarray([budget for budget, _ in pairs], dtype=np.float64)
    gains = np.asarray([gain for _, gain in pairs], dtype=np.float64)
    mean_gain = float(np.mean(gains))
    mean_budget = float(np.mean(budgets))
    mean_abs = abs(mean_gain)
    gain_cv = (
        float(np.std(gains, ddof=0) / mean_abs)
        if mean_abs > float(eps)
        else None
    )
    gain_slope = _ols_slope(budgets, gains, eps=float(eps))
    normalized_slope = (
        float(gain_slope * mean_budget / mean_gain)
        if gain_slope is not None and mean_abs > float(eps)
        else None
    )
    monotonic = _is_monotonic(gains)
    verdict = (
        "linear"
        if (
            len(pairs) >= 2
            and gain_cv is not None
            and normalized_slope is not None
            and gain_cv < float(cv_threshold)
            and abs(normalized_slope) < float(normalized_slope_threshold)
        )
        else "nonlinear"
    )
    return {
        "mean_gain": mean_gain,
        "gain_cv": gain_cv,
        "gain_slope": gain_slope,
        "normalized_slope": normalized_slope,
        "norm_slope": normalized_slope,
        "monotonic": monotonic,
        "verdict": verdict,
    }


def _stage_error_values(stage_error: StageError) -> dict[str, float]:
    return {
        "object": float(stage_error.object_shift_m),
        "prediction": float(stage_error.prediction_l2),
        "cost": float(stage_error.cost_l2),
        "plan": float(stage_error.plan_dev_m),
        "control": float(stage_error.control_dev),
        "safety": float(stage_error.safety_drop_s),
    }


def _result_injection_point(result: Any) -> str:
    cfg = getattr(result, "config", None)
    value = "none" if cfg is None else getattr(cfg, "injection_key", None)
    if value is None and cfg is not None:
        value = getattr(cfg, "injection_point", "none")
    return _injection_key(str(value))


def _result_uses_prediction(result: Any, stage_error: StageError) -> bool:
    cfg = getattr(result, "config", None)
    if cfg is not None and bool(getattr(cfg, "use_prediction", False)):
        return True
    if _result_injection_point(result) == "prediction":
        return True
    return abs(float(stage_error.prediction_l2)) > EPS_RATIO


def _stage_order_for_result(result: Any, stage_error: StageError) -> tuple[str, ...]:
    return STAGE_ORDER_PRED if _result_uses_prediction(result, stage_error) else STAGE_ORDER


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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if abs(float(denominator)) <= EPS_RATIO:
        return None
    return float(numerator / denominator)


def _finite_pairs(
    gain_by_budget: Mapping[float, float | None],
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for budget, gain in gain_by_budget.items():
        if gain is None:
            continue
        budget_value = float(budget)
        gain_value = float(gain)
        if math.isfinite(budget_value) and math.isfinite(gain_value):
            pairs.append((budget_value, gain_value))
    return sorted(pairs, key=lambda item: item[0])


def _ols_slope(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps: float,
) -> float | None:
    if x.size < 2:
        return None
    x_centered = x - float(np.mean(x))
    denominator = float(np.sum(x_centered * x_centered))
    if denominator <= eps:
        return None
    y_centered = y - float(np.mean(y))
    return float(np.sum(x_centered * y_centered) / denominator)


def _is_monotonic(values: np.ndarray) -> bool:
    if values.size < 2:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs >= -EPS_RATIO) or np.all(diffs <= EPS_RATIO))


def _validate_threshold(value: float, *, name: str) -> None:
    if float(value) < 0.0 or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be non-negative and finite")
