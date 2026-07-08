from __future__ import annotations

import numpy as np
import pytest

from torsion.scenarios.costmap_runner import (
    CostMapPlan,
    CostMapPlanner,
    CostMapPlannerConfig,
    CostMapSpec,
    PotentialFieldPlanner,
    build_planner,
)
from torsion.scenarios.planner import EgoState
from torsion.scenarios.unified_pipeline import UnifiedPipelineConfig


def test_potential_field_planner_returns_valid_in_bounds_plan() -> None:
    spec = _spec()
    grid = np.full(spec.shape, 0.03, dtype=np.float64)
    planner = PotentialFieldPlanner(_planner_config())
    ego = EgoState(x=0.0, y=0.0, yaw=0.0, speed=8.0)

    plan = planner.plan(ego, grid, spec)

    assert isinstance(plan, CostMapPlan)
    assert plan.path_xy.shape == (planner.config.path_samples, 2)
    assert np.all(plan.path_xy[:, 0] >= spec.x_min_m)
    assert np.all(plan.path_xy[:, 0] <= spec.x_max_m)
    assert np.all(plan.path_xy[:, 1] >= spec.y_min_m)
    assert np.all(plan.path_xy[:, 1] <= spec.y_max_m)
    assert np.all(np.abs(ego.y + plan.path_xy[:, 1]) <= planner.config.road_half_width_m)
    assert plan.alternatives == ()


def test_potential_field_planner_drives_straight_on_clean_cost_grid() -> None:
    spec = _spec()
    grid = np.full(spec.shape, 0.03, dtype=np.float64)
    planner = PotentialFieldPlanner(_planner_config())

    plan = planner.plan(EgoState(x=0.0, y=0.0, yaw=0.0, speed=8.0), grid, spec)

    assert abs(plan.target_lateral_m) < 1e-12
    assert np.max(np.abs(plan.path_xy[:, 1])) < 1e-12


def test_potential_field_planner_steers_away_from_lateral_obstacle() -> None:
    spec = _spec()
    local_x, local_y = spec.metric_mesh()
    grid = np.full(spec.shape, 0.03, dtype=np.float64)
    obstacle_left = np.exp(
        -0.5 * (((local_x - 24.0) / 4.0) ** 2 + ((local_y - 1.2) / 1.0) ** 2)
    )
    grid = np.clip(grid + 0.95 * obstacle_left, 0.0, 1.0)
    planner = PotentialFieldPlanner(_planner_config())

    plan = planner.plan(EgoState(x=0.0, y=0.0, yaw=0.0, speed=8.0), grid, spec)

    assert plan.target_lateral_m < -0.05
    assert plan.path_xy[-1, 1] == pytest.approx(plan.target_lateral_m)


def test_build_planner_factory_and_default_unified_planner_type() -> None:
    cfg = _planner_config()

    assert isinstance(build_planner("sampling", cfg), CostMapPlanner)
    assert isinstance(build_planner("potential_field", cfg), PotentialFieldPlanner)
    assert isinstance(build_planner("field", cfg), PotentialFieldPlanner)
    assert UnifiedPipelineConfig().planner_type == "sampling"
    with pytest.raises(ValueError, match="planner_type"):
        build_planner("unknown", cfg)


def _spec() -> CostMapSpec:
    return CostMapSpec(
        resolution_m=0.25,
        x_min_m=-2.0,
        x_max_m=36.0,
        y_min_m=-5.0,
        y_max_m=5.0,
    )


def _planner_config() -> CostMapPlannerConfig:
    return CostMapPlannerConfig(
        target_speed_mps=8.0,
        path_samples=41,
        lateral_targets_m=(-3.0, 3.0),
    )
