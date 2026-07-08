from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

from torsion.data.nuplan_adapter import list_logs
from scripts.run_nuplan_propagation import (
    NuPlanPropagationConfig,
    _safe_ratio,
    run_experiment,
)

VAL_ROOT = Path("Dataset/nuplan-v1.0_val/data/cache/public_set_val")
MAPS_ROOT = Path("Dataset/nuplan-maps-v1.0")


def test_nuplan_propagation_one_real_log_smoke_and_determinism(tmp_path: Path) -> None:
    if not MAPS_ROOT.exists():
        pytest.skip(f"nuPlan maps not found under {MAPS_ROOT}")
    db_path = _first_real_val_db()
    cfg = NuPlanPropagationConfig(
        logs_root=db_path,
        maps_root=MAPS_ROOT,
        n_frames=1,
        categories=("FOLLOWING", "INTERSECTION", "LANE_CHANGE"),
        injections=("object", "costmap"),
        methods=("torsion_displace",),
        planners=("sampling",),
        magnitudes_m=(0.5,),
        out_dir=tmp_path / "first",
        max_logs=1,
        bootstrap_resamples=25,
    )

    first = run_experiment(cfg)
    if not first.runs:
        pytest.skip("no eligible tagged nuPlan frames with an in-path target in the first validation log")

    completed = [row for row in first.runs if row["status"] == "completed"]
    assert completed
    assert first.runs_path.is_file()
    assert first.summary_path.is_file()

    for row in completed:
        assert math.isfinite(float(row["cost_l2"]))
        assert math.isfinite(float(row["plan_dev"]))
        assert math.isfinite(float(row["safety1_plan_dev"]))
        assert math.isfinite(float(row["clean_min_dist_m"]))
        assert math.isfinite(float(row["fault_min_dist_m"]))
        assert "safety2_mindist_drop" in row
        assert "safety2_ttc_drop" in row
        if row["injection"] == "object":
            assert math.isfinite(float(row["object_shift_m"]))
            assert row["object_shift_m"] == pytest.approx(0.5)
            assert row["object__cost_gain"] is not None
        if row["injection"] == "costmap":
            assert row["object_shift_m"] == pytest.approx(0.0)
            assert row["object__cost_gain"] is None

    second = run_experiment(replace(cfg, out_dir=tmp_path / "second"))
    assert first.runs == second.runs
    assert first.summary == second.summary


def test_gain_denominator_guard_returns_none() -> None:
    assert _safe_ratio(1.0, 0.0) is None
    assert _safe_ratio(1.0, 1.0e-13) is None
    assert _safe_ratio(2.0, 4.0) == pytest.approx(0.5)


def _first_real_val_db() -> Path:
    logs = list_logs(VAL_ROOT)
    if not logs:
        pytest.skip(f"nuPlan validation logs not found under {VAL_ROOT}")
    return logs[0]
