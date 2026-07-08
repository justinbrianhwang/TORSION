from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("pyproj")

from shapely.geometry import Polygon

from torsion.data.nuplan_adapter import list_logs, load_log
from torsion.data.nuplan_map import (
    epsg_for_location,
    geometry_from_gpkg_binary,
    load_map,
    map_path_for_log,
    build_map_road_cost_grid,
)
from torsion.scenarios.costmap_runner import CostMapPlannerConfig, CostMapSpec

VAL_ROOT = Path("Dataset/nuplan-v1.0_val/data/cache/public_set_val")
MAPS_ROOT = Path("Dataset/nuplan-maps-v1.0")


def test_tiny_gpkg_binary_polygon_builds_drivable_cost_grid(tmp_path: Path) -> None:
    gpkg_path = tmp_path / "tiny_map.gpkg"
    square = Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
    blob = _gpkg_blob(square, envelope_code=1)
    _write_tiny_gpkg(gpkg_path, blob)

    parsed = geometry_from_gpkg_binary(blob)
    assert parsed.equals_exact(square, tolerance=0.0)

    nmap = load_map(gpkg_path, "EPSG:4326", layers=("lanes_polygons", "boundaries"))
    spec = CostMapSpec(resolution_m=0.5, x_min_m=-2.0, x_max_m=2.0, y_min_m=-2.0, y_max_m=2.0)
    grid = build_map_road_cost_grid(
        nmap,
        ego_global_xy=(0.0, 0.0),
        ego_yaw=0.0,
        spec=spec,
        planner_config=CostMapPlannerConfig(),
    )

    assert grid.shape == spec.shape
    assert grid.dtype == np.float64
    assert float(np.min(grid)) >= 0.0
    assert float(np.max(grid)) <= 1.0

    ego_col, ego_row = np.rint(spec.metric_to_grid(np.array([0.0, 0.0]))).astype(int)
    assert grid[ego_row, ego_col] < 0.04
    assert grid[0, 0] == pytest.approx(1.0)


def test_real_vegas_map_contains_trajectory_and_grid_is_deterministic() -> None:
    db_path = _first_real_vegas_db()
    gpkg_path = map_path_for_log(db_path, MAPS_ROOT)
    epsg = _epsg_for_db(db_path)

    log = load_log(db_path)
    nmap = load_map(gpkg_path, epsg)

    lane_bounds = np.asarray([geometry.bounds for geometry in nmap.lane_geometries], dtype=np.float64)
    min_x, min_y = np.min(lane_bounds[:, :2], axis=0)
    max_x, max_y = np.max(lane_bounds[:, 2:], axis=0)
    trajectory = log.ego_trajectory[:, :2]

    assert np.all(trajectory[:, 0] >= min_x)
    assert np.all(trajectory[:, 0] <= max_x)
    assert np.all(trajectory[:, 1] >= min_y)
    assert np.all(trajectory[:, 1] <= max_y)

    frame_idx = _first_drivable_frame(nmap, log.ego_trajectory)
    ego_xy = tuple(float(value) for value in log.ego_trajectory[frame_idx, :2])
    ego_yaw = float(log.ego_trajectory[frame_idx, 2])
    spec = CostMapSpec()
    cfg = CostMapPlannerConfig()

    first = build_map_road_cost_grid(nmap, ego_xy, ego_yaw, spec, cfg)
    second = build_map_road_cost_grid(nmap, ego_xy, ego_yaw, spec, cfg)

    assert first.shape == spec.shape
    assert float(np.min(first)) >= 0.0
    assert float(np.max(first)) <= 1.0
    np.testing.assert_array_equal(first, second)

    ego_col, ego_row = np.rint(spec.metric_to_grid(np.array([0.0, 0.0]))).astype(int)
    assert first[ego_row, ego_col] < 0.20
    assert np.any(first >= 0.95)


def _first_real_vegas_db() -> Path:
    if not MAPS_ROOT.exists():
        pytest.skip(f"nuPlan maps not found under {MAPS_ROOT}")
    for db_path in list_logs(VAL_ROOT):
        metadata = _log_metadata(db_path)
        if metadata["location"] == "las_vegas" and metadata["map_version"] == "us-nv-las-vegas-strip":
            return db_path
    pytest.skip(f"Las Vegas nuPlan validation logs not found under {VAL_ROOT}")


def _epsg_for_db(db_path: Path) -> str:
    metadata = _log_metadata(db_path)
    return epsg_for_location(str(metadata["map_version"] or metadata["location"]))


def _log_metadata(db_path: Path) -> dict[str, str]:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
    finally:
        connection.close()
    if row is None:
        raise AssertionError(f"nuPlan DB has no log row: {db_path}")
    return {"location": str(row[0]), "map_version": str(row[1])}


def _first_drivable_frame(nmap, ego_trajectory: np.ndarray) -> int:
    spec = CostMapSpec()
    cfg = CostMapPlannerConfig()
    ego_col, ego_row = np.rint(spec.metric_to_grid(np.array([0.0, 0.0]))).astype(int)
    stride = max(1, len(ego_trajectory) // 20)
    for frame_idx in range(0, len(ego_trajectory), stride):
        grid = build_map_road_cost_grid(
            nmap,
            ego_global_xy=tuple(float(value) for value in ego_trajectory[frame_idx, :2]),
            ego_yaw=float(ego_trajectory[frame_idx, 2]),
            spec=spec,
            planner_config=cfg,
        )
        if float(grid[ego_row, ego_col]) < 0.20 and np.any(grid >= 0.95):
            return frame_idx
    for frame_idx, pose in enumerate(ego_trajectory):
        grid = build_map_road_cost_grid(
            nmap,
            ego_global_xy=tuple(float(value) for value in pose[:2]),
            ego_yaw=float(pose[2]),
            spec=spec,
            planner_config=cfg,
        )
        if float(grid[ego_row, ego_col]) < 0.20 and np.any(grid >= 0.95):
            return frame_idx
    raise AssertionError("no ego trajectory frame produced a drivable ego cell with off-road cells")


def _write_tiny_gpkg(gpkg_path: Path, lane_blob: bytes) -> None:
    connection = sqlite3.connect(gpkg_path)
    try:
        connection.executescript(
            """
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL
            );
            CREATE TABLE lanes_polygons (
                fid INTEGER PRIMARY KEY,
                geom POLYGON
            );
            CREATE TABLE boundaries (
                fid INTEGER PRIMARY KEY,
                geom LINESTRING
            );
            INSERT INTO gpkg_geometry_columns VALUES
                ('lanes_polygons', 'geom', 'POLYGON', 4326, 0, 0),
                ('boundaries', 'geom', 'LINESTRING', 4326, 0, 0);
            """
        )
        connection.execute("INSERT INTO lanes_polygons VALUES (?, ?)", (1, lane_blob))
        connection.commit()
    finally:
        connection.close()


def _gpkg_blob(geometry, *, envelope_code: int) -> bytes:
    min_x, min_y, max_x, max_y = geometry.bounds
    envelope = b""
    if envelope_code == 1:
        envelope = struct.pack("<dddd", min_x, max_x, min_y, max_y)
    elif envelope_code != 0:
        raise ValueError("test helper only implements envelope codes 0 and 1")
    flags = 1 | (envelope_code << 1)
    return b"GP" + bytes([0, flags]) + (4326).to_bytes(4, "little", signed=True) + envelope + geometry.wkb
