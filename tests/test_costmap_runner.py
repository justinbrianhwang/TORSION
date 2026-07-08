import numpy as np
import pytest

from torsion.operators.costmap import directional_obstacle_inflation
from torsion.scenarios.costmap_runner import (
    COSTMAP_FAULT_METHODS,
    CostMapPlanner,
    CostMapPlannerConfig,
    CostMapRunnerConfig,
    CostMapSpec,
    road_boundary_mask,
    run_costmap_closed_loop,
)
from torsion.scenarios.planner import EgoState


def test_costmap_planner_picks_low_cost_corridor() -> None:
    spec = CostMapSpec(resolution_m=0.5, x_min_m=-2.0, x_max_m=32.0, y_min_m=-5.0, y_max_m=5.0)
    local_x, local_y = spec.metric_mesh()
    grid = np.full(spec.shape, 0.35, dtype=np.float64)
    grid[np.abs(local_y - 1.6) < 0.65] = 0.02
    grid[(local_x < 2.0) & (np.abs(local_y) < 0.8)] = 0.02
    planner = CostMapPlanner(
        CostMapPlannerConfig(
            target_speed_mps=10.0,
            lateral_targets_m=(0.0, -1.6, 1.6),
            path_samples=35,
        )
    )

    plan = planner.plan(EgoState(x=0.0, y=0.0, yaw=0.0, speed=10.0), grid, spec)

    assert plan.target_lateral_m == pytest.approx(1.6)
    assert plan.mean_cost < 0.16


def test_costmap_planner_avoids_high_cost_blob_directly_ahead() -> None:
    spec = CostMapSpec(resolution_m=0.5, x_min_m=-2.0, x_max_m=36.0, y_min_m=-5.0, y_max_m=5.0)
    grid = np.full(spec.shape, 0.03, dtype=np.float64)
    center = spec.metric_to_grid(np.array([14.0, 0.0], dtype=np.float64))
    grid = directional_obstacle_inflation(
        grid,
        center=(float(center[0]), float(center[1])),
        beta=0.98,
        cov=np.diag([8.0, 8.0]),
    )
    planner = CostMapPlanner(CostMapPlannerConfig(target_speed_mps=10.0, path_samples=41))

    first = planner.plan(EgoState(x=0.0, y=0.0, yaw=0.0, speed=10.0), grid, spec)
    second = planner.plan(EgoState(x=0.0, y=0.0, yaw=0.0, speed=10.0), grid, spec)

    assert abs(first.target_lateral_m) >= 1.6
    assert first.collision_free
    assert np.allclose(first.path_xy, second.path_xy)
    assert first.score == pytest.approx(second.score)


def test_costmap_runner_determinism_same_config_same_result() -> None:
    config = CostMapRunnerConfig(
        scenario="cut_in",
        method="torsion_swirl",
        magnitude="medium",
        seed=4,
        steps=12,
        duration_frames=6,
    )

    first = run_costmap_closed_loop(config)
    second = run_costmap_closed_loop(config)

    assert first == second


def test_costmap_fault_methods_preserve_road_boundary_cells() -> None:
    for method in COSTMAP_FAULT_METHODS:
        result = run_costmap_closed_loop(
            CostMapRunnerConfig(
                scenario="cut_in",
                method=method,  # type: ignore[arg-type]
                magnitude="medium",
                seed=1,
                steps=4,
                duration_frames=1,
            )
        )
        row = next(row for row in result.trace if row["fault_active"])
        spec_record = row["cost_grid_spec"]
        spec = CostMapSpec(
            resolution_m=spec_record["resolution_m"],
            x_min_m=spec_record["x_min_m"],
            x_max_m=spec_record["x_max_m"],
            y_min_m=spec_record["y_min_m"],
            y_max_m=spec_record["y_max_m"],
        )
        ego = EgoState(**row["ego"])
        mask = road_boundary_mask(ego, spec, CostMapPlannerConfig())
        clean = np.asarray(row["clean_cost_grid"], dtype=np.float64)
        warped = np.asarray(row["warped_cost_grid"], dtype=np.float64)

        assert np.array_equal(warped[mask], clean[mask])


def test_costmap_fault_methods_match_realized_path_budget() -> None:
    target_budget = 0.5
    budgets = {}
    for method in COSTMAP_FAULT_METHODS:
        result = run_costmap_closed_loop(
            CostMapRunnerConfig(
                scenario="cut_in",
                method=method,  # type: ignore[arg-type]
                magnitude="medium",
                seed=2,
                steps=28,
                duration_frames=8,
            )
        )
        budgets[method] = result.summary["mean_realized_budget"]

    for budget in budgets.values():
        assert budget == pytest.approx(target_budget, abs=0.24)
    assert max(budgets.values()) - min(budgets.values()) <= 0.24


def test_costmap_random_baselines_are_deterministic() -> None:
    for method in ("gaussian_cost", "random_warp_cost", "cost_translate"):
        config = CostMapRunnerConfig(
            scenario="pedestrian_crossing",
            method=method,  # type: ignore[arg-type]
            magnitude="low",
            seed=5,
            steps=12,
            duration_frames=5,
        )

        assert run_costmap_closed_loop(config) == run_costmap_closed_loop(config)
