from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from torsion.data.nuplan_adapter import (
    COARSE_CLASSES,
    find_frames_with_tag,
    list_logs,
    load_log,
    select_target_actor,
)
from torsion.operators.object import ObjectSet

VAL_ROOT = Path("Dataset/nuplan-v1.0_val/data/cache/public_set_val")


def test_tiny_sqlite_fixture_transforms_agent_to_ego_local(tmp_path: Path) -> None:
    db_path, track_id = _write_tiny_nuplan_db(tmp_path / "tiny.db")

    assert list_logs(tmp_path) == [db_path]
    log = load_log(db_path)

    assert log.dt_s == pytest.approx(0.0)
    assert log.scene_names == ("tiny_scene",)
    assert find_frames_with_tag(log, "stationary") == [0]
    assert len(log.frames) == 1
    np.testing.assert_allclose(log.ego_at(0), np.array([0.0, 0.0, 0.0]))

    frame = log.frames[0]
    objects = frame.object_set
    assert isinstance(objects, ObjectSet)
    assert len(objects) == 1
    assert objects.track_id[0] == track_id
    assert objects.cls[0] == "car"
    assert objects.x[0] == pytest.approx(10.0)
    assert objects.y[0] == pytest.approx(0.0)
    assert objects.yaw[0] == pytest.approx(0.0)
    np.testing.assert_allclose(objects.v[0], np.array([2.0, 0.0]))
    assert select_target_actor(frame, frame.ego_xy) == track_id


def test_real_val_log_loads_object_sets_deterministically() -> None:
    db_path = _first_real_val_db()

    log = load_log(db_path)
    repeat = load_log(db_path)

    assert log.dt_s == pytest.approx(0.05, abs=1.0e-4)
    assert len(log.frames) > 0
    assert log.ego_trajectory.shape == (len(log.frames), 3)
    assert log.ego_trajectory.flags.writeable is False
    _assert_logs_equal(log, repeat)

    nonempty_frames = 0
    observed_classes: set[str] = set()
    for frame in log.frames:
        objects = frame.object_set
        assert isinstance(objects, ObjectSet)
        frame_classes = {str(cls) for cls in objects.cls.tolist()}
        assert frame_classes <= COARSE_CLASSES
        observed_classes.update(frame_classes)
        if len(objects) == 0:
            continue

        nonempty_frames += 1
        assert np.all(np.abs(objects.x) <= 60.0 + 1.0e-9)
        assert np.all(np.abs(objects.y) <= 60.0 + 1.0e-9)
        assert np.all((objects.conf >= 0.05) & (objects.conf <= 1.0))

    assert nonempty_frames > 0
    assert observed_classes
    assert observed_classes <= COARSE_CLASSES

    targetable_frames = [
        frame for frame in log.frames if select_target_actor(frame, frame.ego_xy) is not None
    ]
    assert targetable_frames
    busy_frame = max(targetable_frames, key=lambda frame: len(frame.object_set))
    target = select_target_actor(busy_frame, busy_frame.ego_xy)
    assert target in {str(track_id) for track_id in busy_frame.object_set.track_id.tolist()}


def _first_real_val_db() -> Path:
    logs = list_logs(VAL_ROOT)
    if not logs:
        pytest.skip(f"nuPlan validation logs not found under {VAL_ROOT}")
    return logs[0]


def _write_tiny_nuplan_db(db_path: Path) -> tuple[Path, str]:
    frame_token = b"frame001"
    ego_token = b"ego00001"
    scene_token = b"scene001"
    box_token = b"box00001"
    track_token = b"track001"
    category_token = b"cat00001"
    tag_token = b"tag00001"

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE lidar_pc (
                token BLOB PRIMARY KEY,
                next_token BLOB,
                prev_token BLOB,
                ego_pose_token BLOB NOT NULL,
                scene_token BLOB,
                timestamp INTEGER
            );
            CREATE TABLE ego_pose (
                token BLOB PRIMARY KEY,
                timestamp INTEGER,
                x FLOAT,
                y FLOAT,
                z FLOAT,
                qw FLOAT,
                qx FLOAT,
                qy FLOAT,
                qz FLOAT,
                vx FLOAT,
                vy FLOAT,
                vz FLOAT
            );
            CREATE TABLE lidar_box (
                token BLOB PRIMARY KEY,
                lidar_pc_token BLOB NOT NULL,
                track_token BLOB NOT NULL,
                x FLOAT,
                y FLOAT,
                z FLOAT,
                width FLOAT,
                length FLOAT,
                height FLOAT,
                vx FLOAT,
                vy FLOAT,
                vz FLOAT,
                yaw FLOAT,
                confidence FLOAT
            );
            CREATE TABLE track (
                token BLOB PRIMARY KEY,
                category_token BLOB NOT NULL,
                width FLOAT,
                length FLOAT,
                height FLOAT
            );
            CREATE TABLE category (
                token BLOB PRIMARY KEY,
                name VARCHAR(64),
                description TEXT
            );
            CREATE TABLE scene (
                token BLOB PRIMARY KEY,
                log_token BLOB,
                name TEXT,
                goal_ego_pose_token BLOB,
                roadblock_ids TEXT
            );
            CREATE TABLE scenario_tag (
                token BLOB PRIMARY KEY,
                lidar_pc_token BLOB NOT NULL,
                type TEXT,
                agent_track_token BLOB
            );
            """
        )
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, NULL, NULL, ?, ?, ?)",
            (frame_token, ego_token, scene_token, 0),
        )
        connection.execute(
            "INSERT INTO ego_pose VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ego_token, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        connection.execute(
            "INSERT INTO category VALUES (?, ?, ?)",
            (category_token, "vehicle", "Vehicle"),
        )
        connection.execute(
            "INSERT INTO track VALUES (?, ?, ?, ?, ?)",
            (track_token, category_token, 2.0, 4.0, 1.5),
        )
        connection.execute(
            "INSERT INTO lidar_box VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                box_token,
                frame_token,
                track_token,
                10.0,
                0.0,
                0.0,
                2.0,
                4.0,
                1.5,
                2.0,
                0.0,
                0.0,
                0.0,
                0.9,
            ),
        )
        connection.execute(
            "INSERT INTO scene VALUES (?, ?, ?, NULL, ?)",
            (scene_token, b"log0001", "tiny_scene", "[]"),
        )
        connection.execute(
            "INSERT INTO scenario_tag VALUES (?, ?, ?, ?)",
            (tag_token, frame_token, "stationary", track_token),
        )
        connection.commit()
    finally:
        connection.close()

    return db_path, track_token.hex()


def _assert_logs_equal(first: Any, second: Any) -> None:
    assert first.db_path == second.db_path
    assert first.dt_s == second.dt_s
    assert first.scene_names == second.scene_names
    np.testing.assert_allclose(first.ego_trajectory, second.ego_trajectory)
    assert len(first.frames) == len(second.frames)

    for left, right in zip(first.frames, second.frames, strict=True):
        assert left.frame_token == right.frame_token
        assert left.time_s == right.time_s
        assert left.ego_xy == right.ego_xy
        assert left.ego_yaw == right.ego_yaw
        assert left.ego_v == right.ego_v
        assert left.scenario_tags == right.scenario_tags
        _assert_object_sets_equal(left.object_set, right.object_set)


def _assert_object_sets_equal(left: ObjectSet, right: ObjectSet) -> None:
    np.testing.assert_allclose(left.x, right.x)
    np.testing.assert_allclose(left.y, right.y)
    np.testing.assert_allclose(left.z, right.z)
    np.testing.assert_allclose(left.w, right.w)
    np.testing.assert_allclose(left.h, right.h)
    np.testing.assert_allclose(left.l, right.l)
    np.testing.assert_allclose(left.yaw, right.yaw)
    np.testing.assert_allclose(left.v, right.v)
    np.testing.assert_allclose(left.conf, right.conf)
    np.testing.assert_array_equal(left.cls, right.cls)
    np.testing.assert_array_equal(left.track_id, right.track_id)
