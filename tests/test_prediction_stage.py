from __future__ import annotations

import math
from typing import Any

import pytest

from torsion.analysis.propagation import compute_propagation_metrics
from torsion.scenarios.unified_pipeline import UnifiedPipelineConfig, run_unified_pipeline


def test_use_prediction_false_matches_legacy_default_summary() -> None:
    common: dict[str, Any] = {
        "scenario": "cut_in",
        "injection_point": "object",
        "method": "torsion_displace",
        "magnitude": "medium",
        "seed": 0,
        "steps": 12,
        "duration_frames": 6,
        "trace_grids": False,
    }

    legacy = run_unified_pipeline(UnifiedPipelineConfig(**common))
    explicit_false = run_unified_pipeline(
        UnifiedPipelineConfig(use_prediction=False, **common)
    )

    assert explicit_false.summary == legacy.summary
    assert explicit_false.trace == legacy.trace
    assert all(row["prediction_traj_l2_delta"] == pytest.approx(0.0) for row in legacy.trace)


def test_prediction_clean_run_has_zero_prediction_error() -> None:
    result = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario="leading_vehicle",
            injection_point="none",
            method="clean",
            magnitude="high",
            seed=1,
            steps=12,
            trace_grids=False,
            use_prediction=True,
        )
    )

    assert all(
        row["prediction_traj_l2_delta"] == pytest.approx(0.0) for row in result.trace
    )


def test_object_fault_with_prediction_has_nonzero_prediction_delta() -> None:
    result = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario="leading_vehicle",
            injection_point="object",
            method="torsion_displace",
            magnitude="high",
            seed=2,
            steps=18,
            duration_frames=8,
            trace_grids=False,
            use_prediction=True,
        )
    )

    active = [row for row in result.trace if row["fault_active"]]
    assert active
    assert max(float(row["prediction_traj_l2_delta"]) for row in active) > 0.0


def test_prediction_injection_yields_finite_metrics() -> None:
    common: dict[str, Any] = {
        "scenario": "leading_vehicle",
        "magnitude": "high",
        "seed": 3,
        "steps": 40,
        "duration_frames": 20,
        "trace_grids": False,
        "use_prediction": True,
    }
    clean = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    )
    fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            injection_point="prediction",
            method="torsion_displace",
            **common,
        )
    )
    metrics = compute_propagation_metrics(fault, clean)

    assert metrics["raw_prediction_l2"] > 0.0
    assert metrics["object__prediction_gain"] is None
    assert metrics["prediction__cost_gain"] is not None
    assert math.isfinite(float(metrics["prediction__cost_gain"]))
    assert metrics["far"] is None or math.isfinite(float(metrics["far"]))


def test_prediction_metrics_are_deterministic_for_same_config() -> None:
    common: dict[str, Any] = {
        "scenario": "leading_vehicle",
        "magnitude": "high",
        "seed": 4,
        "steps": 30,
        "duration_frames": 15,
        "trace_grids": False,
        "use_prediction": True,
    }
    clean_cfg = UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    fault_cfg = UnifiedPipelineConfig(
        injection_point="prediction",
        method="random_warp",
        **common,
    )

    first = compute_propagation_metrics(
        run_unified_pipeline(fault_cfg),
        run_unified_pipeline(clean_cfg),
    )
    second = compute_propagation_metrics(
        run_unified_pipeline(fault_cfg),
        run_unified_pipeline(clean_cfg),
    )

    assert first == second
