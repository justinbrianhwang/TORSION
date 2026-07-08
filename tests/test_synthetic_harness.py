import pytest

from torsion.metrics.statistics import summarize_safety_group
from torsion.operators.object import ObjectSet
from torsion.scenarios.planner import EgoState, PlannerConfig, ReactivePlanner
from torsion.scenarios.predict import constant_velocity_predict
from torsion.scenarios.synthetic_runner import RunnerConfig, run_synthetic_closed_loop
from torsion.scenarios.synthetic_scenarios import get_scenario


def test_constant_velocity_prediction_is_exact() -> None:
    objects = ObjectSet(
        x=[1.0],
        y=[2.0],
        z=[0.0],
        w=[2.0],
        h=[1.5],
        l=[4.0],
        yaw=[0.0],
        v=[[3.0, -1.0]],
        cls=["vehicle"],
        conf=[0.9],
        track_id=["actor"],
    )

    prediction = constant_velocity_predict(objects, horizon_s=0.3, dt=0.1)
    trajectory = prediction.trajectories[0]

    assert trajectory.xy[:, 0].tolist() == pytest.approx([1.0, 1.3, 1.6, 1.9])
    assert trajectory.xy[:, 1].tolist() == pytest.approx([2.0, 1.9, 1.8, 1.7])


def test_planner_brakes_for_predicted_actor_in_ego_path() -> None:
    objects = ObjectSet(
        x=[18.0],
        y=[0.0],
        z=[0.0],
        w=[2.0],
        h=[1.5],
        l=[4.5],
        yaw=[0.0],
        v=[[0.0, 0.0]],
        cls=["vehicle"],
        conf=[1.0],
        track_id=["lead"],
    )
    planner = ReactivePlanner(PlannerConfig(target_speed_mps=12.0))

    command = planner.plan(
        EgoState(x=0.0, y=0.0, yaw=0.0, speed=12.0),
        constant_velocity_predict(objects, horizon_s=3.0, dt=0.1),
    )

    assert command.brake > 0.0
    assert command.accel_mps2 < 0.0


def test_planner_does_not_brake_on_clean_free_road() -> None:
    objects = ObjectSet.from_records([])
    planner = ReactivePlanner(PlannerConfig(target_speed_mps=12.0))

    command = planner.plan(
        EgoState(x=0.0, y=0.0, yaw=0.0, speed=12.0),
        constant_velocity_predict(objects, horizon_s=3.0, dt=0.1),
    )

    assert command.brake == pytest.approx(0.0)
    assert command.accel_mps2 == pytest.approx(0.0)


def test_runner_determinism_same_config_same_summary() -> None:
    config = RunnerConfig(
        scenario="cut_in",
        method="torsion_swirl",
        magnitude="medium",
        seed=2,
    )

    first = run_synthetic_closed_loop(config)
    second = run_synthetic_closed_loop(config)

    assert first == second


def test_scenario_sampling_is_deterministic_per_seed() -> None:
    first = get_scenario("cut_in", seed=12)
    second = get_scenario("cut_in", seed=12)
    different = get_scenario("cut_in", seed=13)

    assert first.sample_parameters == second.sample_parameters
    assert first.actor_records(3) == second.actor_records(3)
    assert first.actor_records(3, observed=True) == second.actor_records(3, observed=True)
    assert first.sample_parameters != different.sample_parameters


def test_sampled_group_has_non_degenerate_ci() -> None:
    values = []
    for seed in range(8):
        result = run_synthetic_closed_loop(
            RunnerConfig(
                scenario="leading_vehicle",
                method="clean",
                magnitude="medium",
                seed=seed,
                steps=35,
            )
        )
        values.append(min(float(result.summary["min_ttc"]), 5.0))

    summary = summarize_safety_group(
        collision=[0.0 for _ in values],
        min_ttc=values,
        realized_budget=[0.0 for _ in values],
        n_resamples=500,
        seed=9,
    )

    assert summary["std_min_ttc"] > 0.0
    assert summary["mean_min_ttc_ci_high"] > summary["mean_min_ttc_ci_low"]


def test_matched_methods_have_similar_realized_prediction_budget() -> None:
    methods = (
        "gaussian_matched",
        "random_warp",
        "torsion_translate",
        "torsion_swirl",
        "torsion_curl",
        "torsion_combined",
    )
    scenarios = ("cut_in", "leading_vehicle", "pedestrian_crossing")
    budgets: dict[str, float] = {}

    for method in methods:
        values = []
        for scenario in scenarios:
            for seed in range(10):
                result = run_synthetic_closed_loop(
                    RunnerConfig(
                        scenario=scenario,
                        method=method,  # type: ignore[arg-type]
                        magnitude="medium",
                        seed=seed,
                    )
                )
                values.append(result.summary["mean_realized_budget"])
        budgets[method] = sum(values) / len(values)

    reference = 0.5
    for budget in budgets.values():
        assert budget == pytest.approx(reference, rel=0.25)


def test_pedestrian_clean_is_safe_near_miss() -> None:
    result = run_synthetic_closed_loop(
        RunnerConfig(scenario="pedestrian_crossing", method="clean", magnitude="medium")
    )

    assert not result.summary["collision"]
    assert result.summary["min_ttc"] > 0.0
