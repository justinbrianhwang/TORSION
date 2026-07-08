import pytest

from torsion.scenarios.unified_pipeline import UnifiedPipelineConfig, run_unified_pipeline


def test_unified_pipeline_runs_and_records_both_stages() -> None:
    result = run_unified_pipeline(
        UnifiedPipelineConfig(
            scenario="cut_in",
            injection_point="object",
            method="torsion_displace",
            magnitude="medium",
            seed=0,
            steps=4,
            duration_frames=2,
        )
    )

    assert len(result.trace) == 4
    first = result.trace[0]
    assert "clean_object_set" in first
    assert "stage_a_object_set" in first
    assert "clean_cost_grid" in first
    assert "stage_b_cost_grid" in first
    assert "final_cost_grid" in first
    assert "clean_reference_path" in first
    assert "chosen_path" in first
    assert "control" in first
    assert "actual_ttc_s" in first


def test_unified_pipeline_determinism_same_config_same_result() -> None:
    config = UnifiedPipelineConfig(
        scenario="pedestrian_crossing",
        injection_point="costmap",
        method="torsion_displace",
        magnitude="medium",
        seed=7,
        steps=8,
        duration_frames=4,
        trace_grids=False,
    )

    assert run_unified_pipeline(config) == run_unified_pipeline(config)


def test_unified_same_seed_pairs_clean_and_faulted_instances() -> None:
    common = {
        "scenario": "leading_vehicle",
        "magnitude": "medium",
        "seed": 11,
        "steps": 5,
        "duration_frames": 3,
        "trace_grids": False,
    }

    clean = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    )
    object_fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            injection_point="object",
            method="torsion_displace",
            **common,
        )
    )
    costmap_fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            injection_point="costmap",
            method="torsion_displace",
            **common,
        )
    )

    assert clean.trace[0]["gt_actors"] == object_fault.trace[0]["gt_actors"]
    assert clean.trace[0]["gt_actors"] == costmap_fault.trace[0]["gt_actors"]
    assert clean.trace[0]["clean_object_set"] == object_fault.trace[0]["clean_object_set"]
    assert clean.trace[0]["clean_object_set"] == costmap_fault.trace[0]["clean_object_set"]


def test_unified_budget_matching_is_comparable_across_injection_points() -> None:
    common = {
        "scenario": "cut_in",
        "method": "torsion_displace",
        "magnitude": "medium",
        "seed": 2,
        "steps": 16,
        "duration_frames": 8,
        "trace_grids": False,
    }

    object_result = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="object", **common)
    )
    costmap_result = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="costmap", **common)
    )

    object_budget = object_result.summary["mean_realized_budget"]
    costmap_budget = costmap_result.summary["mean_realized_budget"]
    assert object_budget == pytest.approx(costmap_budget, abs=0.08)


def test_object_and_costmap_injection_both_change_the_plan_nontrivially() -> None:
    common = {
        "scenario": "cut_in",
        "method": "torsion_displace",
        "magnitude": "medium",
        "seed": 3,
        "steps": 16,
        "duration_frames": 8,
        "trace_grids": False,
    }

    object_result = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="object", **common)
    )
    costmap_result = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="costmap", **common)
    )

    assert object_result.summary["mean_realized_budget"] > 0.05
    assert costmap_result.summary["mean_realized_budget"] > 0.05


def test_pedestrian_is_sensitive_to_at_least_one_injection_point() -> None:
    common = {
        "scenario": "pedestrian_crossing",
        "magnitude": "high",
        "seed": 5,
        "steps": 60,
        "trace_grids": False,
    }

    clean = run_unified_pipeline(
        UnifiedPipelineConfig(injection_point="none", method="clean", **common)
    )
    object_fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            injection_point="object",
            method="torsion_displace",
            **common,
        )
    )
    costmap_fault = run_unified_pipeline(
        UnifiedPipelineConfig(
            injection_point="costmap",
            method="torsion_displace",
            **common,
        )
    )

    assert clean.summary["min_ttc"] > 0.0
    assert not clean.summary["collision"]
    assert any(
        result.summary["min_ttc"] < clean.summary["min_ttc"] - 1e-3
        for result in (object_fault, costmap_fault)
    )
