import math

import pytest

from torsion.metrics.statistics import (
    bootstrap_ci,
    iqr,
    percentile,
    population_std,
    summarize_paired_delta,
    summarize_safety_group,
    worst_case,
)


def test_bootstrap_ci_known_array_contains_mean_with_expected_bounds() -> None:
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=5_000, seed=7)

    assert 1.0 <= low < 2.5
    assert 2.5 < high <= 4.0


def test_bootstrap_ci_is_exact_for_constant_array() -> None:
    low, high = bootstrap_ci([2.0, 2.0, 2.0, 2.0], n_resamples=1_000, seed=11)

    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)


def test_distribution_metrics_match_hand_computed_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert population_std(values) == pytest.approx(math.sqrt(1.25))
    assert iqr(values) == pytest.approx(1.5)
    assert percentile(values, 5.0) == pytest.approx(1.15)
    assert worst_case(values) == pytest.approx(1.0)


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    first = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0], n_resamples=1_000, seed=42)
    second = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0], n_resamples=1_000, seed=42)

    assert second == first


def test_summarize_safety_group_reports_spread_and_tail_metrics() -> None:
    summary = summarize_safety_group(
        collision=[0, 1, 0, 1],
        min_ttc=[1.0, 2.0, 3.0, 4.0],
        realized_budget=[0.4, 0.5, 0.6, 0.5],
        n_resamples=1_000,
        seed=3,
    )

    assert summary["n_runs"] == 4
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["mean_min_ttc"] == pytest.approx(2.5)
    assert summary["std_min_ttc"] == pytest.approx(math.sqrt(1.25))
    assert summary["iqr_min_ttc"] == pytest.approx(1.5)
    assert summary["worst5pct_min_ttc"] == pytest.approx(1.15)
    assert summary["worst_case_min_ttc"] == pytest.approx(1.0)
    assert summary["mean_realized_budget"] == pytest.approx(0.5)


def test_summarize_paired_delta_reports_mean_and_non_degenerate_ci() -> None:
    summary = summarize_paired_delta(
        baseline=[4.0, 3.5, 3.0, 2.5, 2.0],
        treatment=[3.0, 3.2, 2.0, 2.3, 1.0],
        n_resamples=1_000,
        seed=5,
    )

    assert summary["paired_n"] == 5
    assert summary["paired_mean_delta"] == pytest.approx(0.7)
    assert summary["paired_std_delta"] > 0.0
    assert summary["paired_delta_ci_high"] > summary["paired_delta_ci_low"]
