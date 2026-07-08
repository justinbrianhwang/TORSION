from __future__ import annotations

import math
from typing import Any

import pytest

from torsion.analysis.transfer_function import (
    INTERFACE_ORDER,
    characterize_linearity,
    interface_raw_gains,
)
from torsion.scenarios.unified_pipeline import (
    MAGNITUDE_PATH_BUDGETS_M,
    UnifiedPipelineConfig,
    _target_path_budget,
    run_unified_pipeline,
)


def test_target_path_budget_default_none_reproduces_magnitude_budget() -> None:
    for magnitude, budget in MAGNITUDE_PATH_BUDGETS_M.items():
        cfg = UnifiedPipelineConfig(
            injection_point="object",
            method="torsion_displace",
            magnitude=magnitude,
            target_path_budget_m=None,
        )
        assert _target_path_budget(cfg) == pytest.approx(budget)

    overridden = UnifiedPipelineConfig(
        injection_point="object",
        method="torsion_displace",
        magnitude="low",
        target_path_budget_m=0.35,
    )
    clean = UnifiedPipelineConfig(
        injection_point="none",
        method="clean",
        magnitude="high",
        target_path_budget_m=0.35,
    )
    assert _target_path_budget(overridden) == pytest.approx(0.35)
    assert _target_path_budget(clean) == pytest.approx(0.0)


def test_interface_raw_gains_are_deterministic_for_same_config() -> None:
    common: dict[str, Any] = {
        "scenario": "leading_vehicle",
        "magnitude": "medium",
        "seed": 6,
        "steps": 12,
        "duration_frames": 6,
        "trace_grids": False,
        "use_prediction": True,
    }
    clean_cfg = UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    fault_cfg = UnifiedPipelineConfig(
        injection_point="object",
        method="torsion_displace",
        target_path_budget_m=0.2,
        **common,
    )

    first = interface_raw_gains(
        run_unified_pipeline(fault_cfg),
        run_unified_pipeline(clean_cfg),
    )
    second = interface_raw_gains(
        run_unified_pipeline(fault_cfg),
        run_unified_pipeline(clean_cfg),
    )

    assert first == second
    assert tuple(first) == INTERFACE_ORDER


def test_constant_gain_series_is_linear() -> None:
    summary = characterize_linearity({0.05: 2.0, 0.1: 2.0, 0.2: 2.0, 0.5: 2.0})

    assert summary["verdict"] == "linear"
    assert summary["mean_gain"] == pytest.approx(2.0)
    assert summary["gain_cv"] == pytest.approx(0.0)
    assert summary["norm_slope"] == pytest.approx(0.0)
    assert summary["monotonic"]


def test_threshold_ramp_gain_series_is_nonlinear() -> None:
    summary = characterize_linearity(
        {
            0.05: 0.1,
            0.1: 0.1,
            0.2: 0.8,
            0.35: 2.0,
            0.5: 3.4,
        }
    )

    assert summary["verdict"] == "nonlinear"
    assert summary["gain_cv"] is not None
    assert summary["norm_slope"] is not None
    assert math.isfinite(float(summary["gain_cv"]))
    assert math.isfinite(float(summary["norm_slope"]))


def test_interface_raw_gains_returns_none_for_ill_defined_interfaces() -> None:
    common: dict[str, Any] = {
        "scenario": "cut_in",
        "magnitude": "medium",
        "seed": 4,
        "steps": 10,
        "duration_frames": 5,
        "trace_grids": False,
        "use_prediction": True,
    }
    clean = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    )
    costmap_fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            injection_point="costmap",
            method="torsion_displace",
            target_path_budget_m=0.2,
            **common,
        )
    )

    assert all(value is None for value in interface_raw_gains(clean, clean).values())

    gains = interface_raw_gains(costmap_fault, clean)
    assert gains["object__prediction"] is None
    assert gains["prediction__cost"] is None
    assert tuple(gains) == INTERFACE_ORDER
