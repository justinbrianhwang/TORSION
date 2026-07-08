from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torsion.analysis.failure_taxonomy import build_taxonomy, classify_run
from torsion.scenarios.costmap_runner import CostMapPlannerConfig
from torsion.scenarios.unified_pipeline import UnifiedPipelineConfig, run_unified_pipeline


def test_clean_run_classifies_safe_without_argmin_flip() -> None:
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
            use_prediction=True,
        )
    )

    record = classify_run(clean, clean)

    assert record["failure_mode"] == "safe"
    assert record["dominant_interface"] == "attenuated"
    assert record["signature"] == "attenuated"
    assert not record["argmin_flipped"]


def test_failure_taxonomy_classification_is_deterministic() -> None:
    common: dict[str, Any] = {
        "scenario": "leading_vehicle",
        "magnitude": "high",
        "seed": 4,
        "steps": 20,
        "duration_frames": 8,
        "trace_grids": False,
        "use_prediction": True,
    }
    clean_cfg = UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    fault_cfg = UnifiedPipelineConfig(
        injection_point="prediction",
        method="torsion_displace",
        **common,
    )

    first = classify_run(
        run_unified_pipeline(fault_cfg),
        run_unified_pipeline(clean_cfg),
    )
    second = classify_run(
        run_unified_pipeline(fault_cfg),
        run_unified_pipeline(clean_cfg),
    )

    assert first == second


def test_failure_mode_priority_keeps_collision_above_lane_departure() -> None:
    clean = _fake_result(collision=False, off_road=False, lane_departure=False)
    fault = _fake_result(collision=True, off_road=True, lane_departure=True)

    record = classify_run(fault, clean)

    assert record["failure_mode"] == "collision"
    assert record["collision"]


def test_build_taxonomy_counts_paths_and_per_origin_collision_rate() -> None:
    rows = [
        _classified("object", "planner_switch", "lane_departure", collision=False),
        _classified("object", "planner_switch", "lane_departure", collision=False),
        _classified("object", "attenuated", "collision", collision=True),
        _classified("prediction", "control__safety", "collision", collision=True),
    ]

    taxonomy = build_taxonomy(rows)

    top = taxonomy["paths"][0]
    assert top["fault_origin"] == "object"
    assert top["signature"] == "planner_switch"
    assert top["failure_mode"] == "lane_departure"
    assert top["count"] == 2
    assert top["freq_overall"] == pytest.approx(50.0)
    assert top["freq_within_origin"] == pytest.approx(100.0 * 2.0 / 3.0)
    assert taxonomy["per_origin"]["object"]["collision_rate"] == pytest.approx(1.0 / 3.0)


def _fake_result(
    *,
    collision: bool,
    off_road: bool,
    lane_departure: bool,
) -> SimpleNamespace:
    config = SimpleNamespace(
        scenario="unit",
        injection_point="object",
        method="torsion_displace",
        magnitude="high",
        seed=0,
        use_prediction=True,
        planner=CostMapPlannerConfig(),
    )
    row = {
        "frame": 0,
        "time_s": 0.0,
        "fault_active": True,
        "injection_point": "object",
        "object_position_shift_m": 0.2 if collision else 0.0,
        "prediction_traj_l2_delta": 0.3 if collision else 0.0,
        "cost_grid_l2_delta": 0.1 if collision else 0.0,
        "realized_path_deviation_m": 0.2 if collision else 0.0,
        "actual_ttc_s": 0.5 if collision else 5.0,
        "collision": collision,
        "off_road": off_road,
        "lane_departure": lane_departure,
        "control": {
            "accel_mps2": -8.0 if collision else 0.0,
            "reason": "hard_brake_no_collision_free_path" if collision else "track_low_cost_path",
        },
        "chosen_path": _plan(0.2 if collision else 0.0, accel=-8.0 if collision else 0.0),
        "clean_reference_path": _plan(0.0),
        "clean_object_set": [{"track_id": "target", "x": 0.0, "y": 0.0}],
    }
    return SimpleNamespace(
        config=config,
        trace=(row,),
        summary={
            "collision": collision,
            "target_actor": "target",
            "min_ttc": row["actual_ttc_s"],
        },
    )


def _plan(target_lateral_m: float, *, accel: float = 0.0) -> dict[str, Any]:
    return {
        "target_lateral_m": float(target_lateral_m),
        "accel_mps2": float(accel),
        "reason": "track_low_cost_path",
        "alternatives": [
            {"score": 0.0, "collision_free": True},
            {"score": 1.0, "collision_free": True},
        ],
    }


def _classified(
    fault_origin: str,
    signature: str,
    failure_mode: str,
    *,
    collision: bool,
) -> dict[str, Any]:
    return {
        "scenario": "unit",
        "fault_origin": fault_origin,
        "signature": signature,
        "dominant_interface": signature,
        "argmin_flipped": signature == "planner_switch",
        "failure_mode": failure_mode,
        "magnitude": "medium",
        "seed": 0,
        "min_ttc": 0.0 if collision else 5.0,
        "collision": collision,
        "reach_safety": collision,
    }
