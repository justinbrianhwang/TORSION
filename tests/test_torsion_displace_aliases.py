from torsion.scenarios.costmap_runner import CostMapRunnerConfig, run_costmap_closed_loop
from torsion.scenarios.synthetic_runner import RunnerConfig, run_synthetic_closed_loop


def test_object_torsion_displace_alias_matches_translate_operator() -> None:
    alias = RunnerConfig(
        scenario="cut_in",
        method="torsion_displace",
        magnitude="medium",
        seed=3,
        steps=12,
        duration_frames=6,
    )
    legacy = RunnerConfig(
        scenario="cut_in",
        method="torsion_translate",
        magnitude="medium",
        seed=3,
        steps=12,
        duration_frames=6,
    )

    assert alias.operator_name == legacy.operator_name
    assert alias.method_key == "torsion_displace"
    assert run_synthetic_closed_loop(alias).summary == run_synthetic_closed_loop(legacy).summary


def test_costmap_torsion_displace_alias_matches_translate_operator() -> None:
    alias = CostMapRunnerConfig(
        scenario="cut_in",
        method="torsion_displace",
        magnitude="medium",
        seed=3,
        steps=12,
        duration_frames=6,
    )
    legacy = CostMapRunnerConfig(
        scenario="cut_in",
        method="cost_translate",
        magnitude="medium",
        seed=3,
        steps=12,
        duration_frames=6,
    )

    assert alias.operator_name == legacy.operator_name
    assert alias.method_key == "torsion_displace"
    assert run_costmap_closed_loop(alias).summary == run_costmap_closed_loop(legacy).summary
