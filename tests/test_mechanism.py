from __future__ import annotations

import math
from typing import Any

import pytest

from torsion.analysis.mechanism import (
    decision_margin_analysis,
    prediction_jacobian,
    rasterization_jacobian,
)
from torsion.scenarios.unified_pipeline import UnifiedPipelineConfig, run_unified_pipeline


def test_clean_run_has_zero_argmin_flips() -> None:
    clean = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario="cut_in",
            injection_point="none",
            method="clean",
            magnitude="medium",
            seed=0,
            steps=12,
            duration_frames=4,
            trace_grids=False,
        )
    )

    analysis = decision_margin_analysis(clean, fault_active_only=False)
    rows = analysis["per_frame"]

    assert rows
    assert all(not bool(row["argmin_flip"]) for row in rows)
    assert all(float(row["realized_path_deviation_m"]) == pytest.approx(0.0) for row in rows)


def test_decision_margin_is_computable_for_fault_frames() -> None:
    fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario="leading_vehicle",
            injection_point="costmap",
            method="torsion_displace",
            magnitude="high",
            seed=1,
            steps=12,
            duration_frames=4,
            trace_grids=False,
        )
    )

    analysis = decision_margin_analysis(fault)
    rows = analysis["per_frame"]

    assert rows
    assert analysis["quartile_summary"]
    assert analysis["correlations"]["n"] == len(rows)
    assert all(math.isfinite(float(row["decision_margin_score"])) for row in rows)
    assert all(int(row["n_candidates"]) >= 2 for row in rows)


def test_mechanism_analysis_is_deterministic_for_same_configs() -> None:
    configs: list[Any] = [
        UnifiedPipelineConfig(
            scenario="pedestrian_crossing",
            injection_point="costmap",
            method="torsion_displace",
            magnitude="medium",
            seed=2,
            steps=10,
            duration_frames=3,
            trace_grids=False,
        )
    ]

    first = decision_margin_analysis(configs)
    second = decision_margin_analysis(configs)

    assert first == second


def test_rasterization_jacobian_is_positive_and_attenuating() -> None:
    object_cost = rasterization_jacobian(
        "leading_vehicle",
        magnitude="medium",
        seed=0,
        use_prediction=False,
        eps_sweep=(0.01, 0.02, 0.05),
    )
    prediction_cost = rasterization_jacobian(
        "leading_vehicle",
        magnitude="medium",
        seed=0,
        use_prediction=True,
        eps_sweep=(0.01, 0.02, 0.05),
    )

    for row in (object_cost, prediction_cost):
        assert row["j_raster"] > 0.0
        assert row["j_raster"] < 1.0
        assert math.isfinite(float(row["linearity_cv"]))


def test_prediction_jacobian_matches_mean_sample_time() -> None:
    result = prediction_jacobian(
        "cut_in",
        horizon_s=2.0,
        eps=0.01,
        seed=0,
    )

    assert result["j_pred_empirical"] == pytest.approx(result["j_pred_analytic"], abs=1e-12)
    assert result["j_pred_analytic"] == pytest.approx(1.0)
