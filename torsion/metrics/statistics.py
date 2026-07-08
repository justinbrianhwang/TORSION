"""Statistical summaries for synthetic TORSION experiment groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

Statistic = Callable[[np.ndarray], float]


def mean_statistic(values: np.ndarray) -> float:
    """Return the arithmetic mean as a plain float."""

    return float(np.mean(values))


def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Statistic = mean_statistic,
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a 1D sample.

    The function is deterministic for a fixed ``seed`` and intentionally avoids
    global random state so experiment summaries are reproducible.
    """

    sample = _as_1d_float_array(values, name="values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    if sample.size == 1:
        value = float(statistic(sample))
        return value, value

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(n_resamples, sample.size))
    resamples = sample[indices]
    if statistic is mean_statistic or statistic is np.mean:
        statistics = np.mean(resamples, axis=1)
    else:
        statistics = np.apply_along_axis(statistic, 1, resamples)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(statistics, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(low), float(high)


def population_std(values: ArrayLike) -> float:
    """Population standard deviation for a seed distribution."""

    sample = _as_1d_float_array(values, name="values")
    return float(np.std(sample, ddof=0))


def iqr(values: ArrayLike) -> float:
    """Interquartile range using NumPy's default linear percentile method."""

    sample = _as_1d_float_array(values, name="values")
    q25, q75 = np.percentile(sample, [25.0, 75.0])
    return float(q75 - q25)


def percentile(values: ArrayLike, q: float) -> float:
    """Return percentile ``q`` for a finite 1D sample."""

    sample = _as_1d_float_array(values, name="values")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    return float(np.percentile(sample, q))


def worst_case(values: ArrayLike) -> float:
    """Return the minimum value, used for worst-case min-TTC."""

    sample = _as_1d_float_array(values, name="values")
    return float(np.min(sample))


def summarize_safety_group(
    *,
    collision: ArrayLike,
    min_ttc: ArrayLike,
    realized_budget: ArrayLike,
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize one scenario/method/magnitude group across seeds."""

    collision_values = _as_1d_float_array(collision, name="collision")
    min_ttc_values = _as_1d_float_array(min_ttc, name="min_ttc")
    budget_values = _as_1d_float_array(realized_budget, name="realized_budget")

    if not (
        collision_values.size == min_ttc_values.size == budget_values.size
    ):
        raise ValueError("collision, min_ttc, and realized_budget must have equal length")

    collision_low, collision_high = bootstrap_ci(
        collision_values,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    ttc_low, ttc_high = bootstrap_ci(
        min_ttc_values,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )

    return {
        "n_runs": int(min_ttc_values.size),
        "collision_rate": float(np.mean(collision_values)),
        "collision_rate_ci_low": collision_low,
        "collision_rate_ci_high": collision_high,
        "mean_min_ttc": float(np.mean(min_ttc_values)),
        "mean_min_ttc_ci_low": ttc_low,
        "mean_min_ttc_ci_high": ttc_high,
        "std_min_ttc": population_std(min_ttc_values),
        "iqr_min_ttc": iqr(min_ttc_values),
        "worst5pct_min_ttc": percentile(min_ttc_values, 5.0),
        "worst_case_min_ttc": worst_case(min_ttc_values),
        "mean_realized_budget": float(np.mean(budget_values)),
    }


def summarize_paired_delta(
    baseline: ArrayLike,
    treatment: ArrayLike,
    *,
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize paired ``baseline - treatment`` effects.

    Positive deltas mean the treatment reduced the metric relative to the
    paired baseline.  For min-TTC, positive values therefore indicate safety
    degradation.
    """

    baseline_values = _as_1d_float_array(baseline, name="baseline")
    treatment_values = _as_1d_float_array(treatment, name="treatment")
    if baseline_values.size != treatment_values.size:
        raise ValueError("baseline and treatment must have equal length")

    deltas = baseline_values - treatment_values
    low, high = bootstrap_ci(
        deltas,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "paired_n": int(deltas.size),
        "paired_mean_delta": float(np.mean(deltas)),
        "paired_delta_ci_low": low,
        "paired_delta_ci_high": high,
        "paired_std_delta": population_std(deltas),
    }


def _as_1d_float_array(values: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64)
    else:
        array = np.asarray(list(values), dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
