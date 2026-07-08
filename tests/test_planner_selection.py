import numpy as np
import pytest

from torsion.scenarios.costmap_runner import CostMapPlanner, CostMapPlannerConfig, CostMapSpec
from torsion.scenarios.planner import EgoState


def test_softmax_low_temperature_matches_argmin() -> None:
    ego, grid, spec = _flat_problem()
    common = {
        "target_speed_mps": 8.0,
        "path_samples": 41,
        "lateral_targets_m": (0.0, 1.0, 2.0),
    }

    argmin = CostMapPlanner(CostMapPlannerConfig(**common)).plan(ego, grid, spec)
    soft = CostMapPlanner(
        CostMapPlannerConfig(
            **common,
            selection_mode="softmax",
            selection_temperature=1e-9,
        )
    ).plan(ego, grid, spec)

    assert soft.target_lateral_m == pytest.approx(argmin.target_lateral_m, abs=1e-12)
    assert np.allclose(soft.path_xy, argmin.path_xy, atol=1e-12)
    assert soft.reason == argmin.reason
    assert soft.alternatives == argmin.alternatives


def test_softmax_temperature_controls_scalar_lateral_target() -> None:
    ego, grid, spec = _flat_problem()
    cfg = CostMapPlannerConfig(
        target_speed_mps=8.0,
        path_samples=41,
        lateral_targets_m=(0.0, 1.0, 2.0),
        selection_mode="softmax",
        selection_temperature=5.0,
    )

    plan = CostMapPlanner(cfg).plan(ego, grid, spec)
    expected_target = _softmax_target(plan.alternatives, tau=cfg.selection_temperature)

    assert plan.target_lateral_m == pytest.approx(expected_target)
    assert plan.target_lateral_m > 0.75
    assert plan.path_xy[-1, 1] == pytest.approx(plan.target_lateral_m)
    assert plan.score >= 0.0


def test_default_argmin_matches_explicit_argmin() -> None:
    ego, grid, spec = _flat_problem()
    common = {
        "target_speed_mps": 8.0,
        "path_samples": 41,
        "lateral_targets_m": (0.0, 1.0, 2.0),
    }

    default = CostMapPlanner(CostMapPlannerConfig(**common)).plan(ego, grid, spec)
    explicit = CostMapPlanner(
        CostMapPlannerConfig(
            **common,
            selection_mode="argmin",
            selection_temperature=0.0,
        )
    ).plan(ego, grid, spec)

    assert default.to_record() == explicit.to_record()


def test_softmax_selection_is_deterministic() -> None:
    ego, grid, spec = _flat_problem()
    planner = CostMapPlanner(
        CostMapPlannerConfig(
            target_speed_mps=8.0,
            path_samples=41,
            lateral_targets_m=(0.0, 1.0, 2.0),
            selection_mode="softmax",
            selection_temperature=0.1,
        )
    )

    first = planner.plan(ego, grid, spec)
    second = planner.plan(ego, grid, spec)

    assert first.to_record() == second.to_record()


def test_softmax_requires_positive_temperature() -> None:
    with pytest.raises(ValueError, match="selection_temperature"):
        CostMapPlannerConfig(selection_mode="softmax", selection_temperature=0.0)


def _flat_problem() -> tuple[EgoState, np.ndarray, CostMapSpec]:
    spec = CostMapSpec(
        resolution_m=0.5,
        x_min_m=-2.0,
        x_max_m=32.0,
        y_min_m=-6.0,
        y_max_m=6.0,
    )
    grid = np.zeros(spec.shape, dtype=np.float64)
    ego = EgoState(x=0.0, y=0.0, yaw=0.0, speed=8.0)
    return ego, grid, spec


def _softmax_target(alternatives: tuple[dict[str, object], ...], *, tau: float) -> float:
    scores = np.asarray([float(row["score"]) for row in alternatives], dtype=np.float64)
    targets = np.asarray(
        [float(row["target_lateral_m"]) for row in alternatives],
        dtype=np.float64,
    )
    weights = np.exp(-(scores - float(np.min(scores))) / float(tau))
    weights /= float(np.sum(weights))
    return float(np.sum(weights * targets))
