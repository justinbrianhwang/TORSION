from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from torsion.data.nuscenes_adapter import (
    category_to_coarse_class,
    global_to_ego_xy,
    load_scenes,
    load_tables,
)
from torsion.operators.object import ObjectSet

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nuscenes_mini"
ROTATED_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nuscenes_mini_rotated"


def test_load_tables_reads_fixture_metadata() -> None:
    tables = load_tables(FIXTURE_DIR)

    assert {
        "calibrated_sensor",
        "category",
        "ego_pose",
        "instance",
        "sample",
        "sample_annotation",
        "sample_data",
        "scene",
        "sensor",
    }.issubset(tables)
    assert len(tables["scene"]) == 1
    assert len(tables["sample"]) == 3
    assert len(tables["sample_annotation"]) == 6


def test_straight_fixture_builds_object_sets_and_preserves_tracks() -> None:
    scene = load_scenes(FIXTURE_DIR)[0]

    assert scene.scene_token == "scene_token_1"
    assert scene.name == "fixture_scene_straight"
    assert scene.dt_s == pytest.approx(0.5)
    assert scene.ego_trajectory.flags.writeable is False
    assert [frame.sample_token for frame in scene.frames] == ["sample_0", "sample_1", "sample_2"]
    np.testing.assert_allclose(scene.ego_at(0), np.array([0.0, 0.0, 0.0]))

    for frame in scene.frames:
        objects = frame.object_set
        assert isinstance(objects, ObjectSet)
        assert len(objects) == 2
        assert set(objects.track_id.tolist()) == {"inst_car", "inst_pedestrian"}
        assert np.all((objects.conf >= 0.0) & (objects.conf <= 1.0))

    frame0 = scene.frames[0]
    car_idx = _object_index(frame0.object_set, "inst_car")
    ped_idx = _object_index(frame0.object_set, "inst_pedestrian")

    assert frame0.ego_yaw == pytest.approx(0.0)
    assert frame0.object_set.x[car_idx] == pytest.approx(10.0)
    assert frame0.object_set.y[car_idx] == pytest.approx(0.0)
    np.testing.assert_allclose(frame0.object_set.v[car_idx], np.array([5.0, 0.0]))
    assert frame0.object_set.cls[car_idx] == "car"
    assert frame0.object_set.track_id[car_idx] == "inst_car"
    assert frame0.object_set.conf[car_idx] == pytest.approx(0.8)
    assert frame0.object_set.w[car_idx] == pytest.approx(2.0)
    assert frame0.object_set.h[car_idx] == pytest.approx(1.5)
    assert frame0.object_set.l[car_idx] == pytest.approx(4.0)

    assert frame0.object_set.cls[ped_idx] == "pedestrian"
    assert frame0.object_set.x[ped_idx] == pytest.approx(0.0)
    assert frame0.object_set.y[ped_idx] == pytest.approx(4.0)

    car_track_ids = [
        frame.object_set.track_id[_object_index(frame.object_set, "inst_car")]
        for frame in scene.frames
    ]
    assert car_track_ids == ["inst_car", "inst_car", "inst_car"]


def test_rotated_ego_fixture_rotates_positions_yaw_and_velocity() -> None:
    scene = load_scenes(ROTATED_FIXTURE_DIR)[0]
    frame0 = scene.frames[0]
    car_idx = _object_index(frame0.object_set, "inst_car")

    assert frame0.ego_yaw == pytest.approx(np.pi / 2.0)
    np.testing.assert_allclose(
        global_to_ego_xy([10.0, 0.0], [0.0, 0.0], np.pi / 2.0),
        np.array([0.0, -10.0]),
        atol=1.0e-12,
    )
    assert frame0.object_set.x[car_idx] == pytest.approx(0.0, abs=1.0e-12)
    assert frame0.object_set.y[car_idx] == pytest.approx(-10.0)
    assert frame0.object_set.yaw[car_idx] == pytest.approx(-np.pi / 2.0)
    np.testing.assert_allclose(
        frame0.object_set.v[car_idx],
        np.array([0.0, -5.0]),
        atol=1.0e-12,
    )


def test_category_mapping_keeps_known_cost_map_inputs_safe() -> None:
    assert category_to_coarse_class("vehicle.car") == "car"
    assert category_to_coarse_class("vehicle.truck") == "truck"
    assert category_to_coarse_class("vehicle.bus.rigid") == "truck"
    assert category_to_coarse_class("vehicle.bicycle") == "cyclist"
    assert category_to_coarse_class("human.pedestrian.adult") == "pedestrian"
    assert category_to_coarse_class("movable_object.trafficcone") == "other"


def test_load_scenes_is_deterministic() -> None:
    first = _snapshot(load_scenes(FIXTURE_DIR))
    second = _snapshot(load_scenes(FIXTURE_DIR))

    assert first == second


def _object_index(objects: ObjectSet, track_id: str) -> int:
    matches = np.flatnonzero(objects.track_id == track_id)
    assert matches.size == 1
    return int(matches[0])


def _snapshot(scenes: list[Any]) -> tuple[Any, ...]:
    rows = []
    for scene in scenes:
        frame_rows = []
        for frame in scene.frames:
            objects = frame.object_set
            object_rows = []
            for idx in range(len(objects)):
                object_rows.append(
                    (
                        str(objects.track_id[idx]),
                        str(objects.cls[idx]),
                        round(float(objects.x[idx]), 12),
                        round(float(objects.y[idx]), 12),
                        round(float(objects.yaw[idx]), 12),
                        round(float(objects.v[idx, 0]), 12),
                        round(float(objects.v[idx, 1]), 12),
                        round(float(objects.conf[idx]), 12),
                    )
                )
            frame_rows.append(
                (
                    frame.sample_token,
                    round(frame.time_s, 12),
                    round(float(frame.ego_xy[0]), 12),
                    round(float(frame.ego_xy[1]), 12),
                    round(float(frame.ego_yaw), 12),
                    tuple(object_rows),
                )
            )
        rows.append(
            (
                scene.scene_token,
                scene.name,
                round(float(scene.dt_s), 12),
                tuple(frame_rows),
            )
        )
    return tuple(rows)
