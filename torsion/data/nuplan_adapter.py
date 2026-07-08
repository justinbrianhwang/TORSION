"""nuPlan SQLite log loader for TORSION ObjectSet frames.

The adapter reads nuPlan ``.db`` logs directly with ``sqlite3``. Agent boxes
are emitted in the ego-local frame used by TORSION: ``x`` is forward and ``y``
is left.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.data._geometry import (
    global_to_ego_velocity,
    global_to_ego_xy,
    quaternion_yaw,
    wrap_angle,
)
from torsion.operators.object import ObjectSet

DEFAULT_MAX_ABS_XY_M = 60.0
MICROSECONDS_TO_SECONDS = 1.0e-6
STATIC_NUPLAN_CATEGORIES = frozenset(
    {"traffic_cone", "barrier", "czone_sign", "generic_object"}
)
COARSE_CLASSES = frozenset({"car", "cyclist", "pedestrian", "other"})
DYNAMIC_TARGET_CLASSES = frozenset({"car", "cyclist", "pedestrian"})


@dataclass(frozen=True)
class NuPlanFrame:
    """One nuPlan lidar frame converted to ego-local TORSION objects."""

    frame_token: str
    time_s: float
    ego_xy: tuple[float, float]
    ego_yaw: float
    ego_v: tuple[float, float]
    object_set: ObjectSet
    scenario_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.frame_token:
            raise ValueError("frame_token must be non-empty")
        if not np.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        _vector2(self.ego_xy, "ego_xy")
        _vector2(self.ego_v, "ego_v")
        if not np.isfinite(self.ego_yaw):
            raise ValueError("ego_yaw must be finite")
        if not isinstance(self.object_set, ObjectSet):
            raise TypeError("object_set must be an ObjectSet")
        object.__setattr__(self, "scenario_tags", tuple(str(tag) for tag in self.scenario_tags))


@dataclass(frozen=True)
class NuPlanLog:
    """A nuPlan SQLite log as a deterministic sequence of ObjectSet frames."""

    db_path: Path
    dt_s: float
    frames: tuple[NuPlanFrame, ...]
    ego_trajectory: NDArray[np.float64]
    scene_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "db_path", Path(self.db_path))
        frames = tuple(self.frames)
        object.__setattr__(self, "frames", frames)

        ego = np.asarray(self.ego_trajectory, dtype=np.float64)
        if ego.ndim != 2 or ego.shape[1] != 3:
            raise ValueError("ego_trajectory must have shape (N, 3)")
        if ego.shape[0] != len(frames):
            raise ValueError("ego_trajectory length must match frame count")
        if not np.all(np.isfinite(ego)):
            raise ValueError("ego_trajectory must contain only finite values")
        ego = ego.copy()
        ego.setflags(write=False)
        object.__setattr__(self, "ego_trajectory", ego)

        if not np.isfinite(self.dt_s) or self.dt_s < 0.0:
            raise ValueError("dt_s must be finite and non-negative")
        object.__setattr__(self, "scene_names", tuple(str(name) for name in self.scene_names))

    def object_set(self, i: int) -> ObjectSet:
        """Return the ObjectSet for frame ``i``."""

        return self.frames[int(i)].object_set

    def ego_at(self, i: int) -> NDArray[np.float64]:
        """Return ego ``[x, y, yaw]`` in the global frame for frame ``i``."""

        return self.ego_trajectory[int(i)].copy()


@dataclass(frozen=True)
class _FrameContext:
    token_key: Hashable
    frame_token: str
    time_s: float
    ego_xy: tuple[float, float]
    ego_z: float
    ego_yaw: float
    ego_v: tuple[float, float]


def load_log(
    db_path: str | Path,
    *,
    max_frames: int | None = None,
    subsample_hz: float | None = None,
    max_abs_xy_m: float | None = DEFAULT_MAX_ABS_XY_M,
    min_confidence: float = 0.0,
    include_static: bool = True,
) -> NuPlanLog:
    """Load a nuPlan ``.db`` log as ego-local ObjectSet frames."""

    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"nuPlan log not found: {path}")
    if max_frames is not None and int(max_frames) < 0:
        raise ValueError("max_frames must be non-negative")
    max_range = _resolve_max_abs_xy(max_abs_xy_m)
    min_conf = _resolve_min_confidence(min_confidence)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        frame_rows = _read_frame_rows(connection)
        selected_rows = _subsample_frame_rows(frame_rows, subsample_hz=subsample_hz)
        if max_frames is not None:
            selected_rows = selected_rows[: int(max_frames)]

        contexts = [_frame_context(row) for row in selected_rows]
        selected_tokens = {context.token_key for context in contexts}
        scene_names = _read_scene_names(connection)
        tags_by_frame = _read_scenario_tags(connection, selected_tokens=selected_tokens)
        records_by_frame = _read_box_records(
            connection,
            contexts=contexts,
            selected_tokens=selected_tokens,
            max_abs_xy_m=max_range,
            min_confidence=min_conf,
            include_static=include_static,
        )
    finally:
        connection.close()

    frames: list[NuPlanFrame] = []
    ego_rows: list[list[float]] = []
    for context in contexts:
        records = records_by_frame.get(context.token_key, [])
        records.sort(key=lambda row: (float(np.hypot(row["x"], row["y"])), str(row["track_id"])))
        frame = NuPlanFrame(
            frame_token=context.frame_token,
            time_s=context.time_s,
            ego_xy=context.ego_xy,
            ego_yaw=context.ego_yaw,
            ego_v=context.ego_v,
            object_set=ObjectSet.from_records(records),
            scenario_tags=tuple(sorted(tags_by_frame.get(context.token_key, ()))),
        )
        frames.append(frame)
        ego_rows.append([context.ego_xy[0], context.ego_xy[1], context.ego_yaw])

    times = np.array([frame.time_s for frame in frames], dtype=np.float64)
    dt_s = _median_positive_diff(times)
    ego_trajectory = np.asarray(ego_rows, dtype=np.float64).reshape(len(frames), 3)
    return NuPlanLog(
        db_path=path,
        dt_s=dt_s,
        frames=tuple(frames),
        ego_trajectory=ego_trajectory,
        scene_names=scene_names,
    )


def list_logs(root_dir: str | Path) -> list[Path]:
    """Return nuPlan ``.db`` logs under ``root_dir`` in deterministic order."""

    root = Path(root_dir)
    if root.is_file():
        return [root] if root.suffix.lower() == ".db" else []
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.db") if path.is_file())


def find_frames_with_tag(log: NuPlanLog, tag: str) -> list[int]:
    """Return frame indices whose scenario tags contain ``tag`` exactly."""

    wanted = str(tag)
    return [idx for idx, frame in enumerate(log.frames) if wanted in frame.scenario_tags]


def select_target_actor(frame: NuPlanFrame, ego: Any = None) -> str | None:
    """Select the nearest in-path actor ahead, with deterministic tie-breaking."""

    del ego
    objects = frame.object_set
    if len(objects) == 0:
        return None

    dynamic = np.isin(objects.cls, np.array(sorted(DYNAMIC_TARGET_CLASSES), dtype=object))
    ahead = (objects.x > 0.0) & dynamic
    in_corridor = ahead & (np.abs(objects.y) <= 2.0)
    candidates = np.flatnonzero(in_corridor)
    if candidates.size == 0:
        candidates = np.flatnonzero(ahead)
    if candidates.size == 0:
        return None

    chosen = min(
        (int(idx) for idx in candidates),
        key=lambda idx: (
            float(np.hypot(objects.x[idx], objects.y[idx])),
            str(objects.track_id[idx]),
        ),
    )
    return str(objects.track_id[chosen])


def category_to_coarse_class(category_name: str | None) -> str:
    """Map a nuPlan category name to the coarse class used by TORSION."""

    name = _normalize_category(category_name)
    if name == "vehicle":
        return "car"
    if name == "bicycle":
        return "cyclist"
    if name == "pedestrian":
        return "pedestrian"
    return "other"


def _read_frame_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    query = """
        SELECT
            lp.token AS frame_token,
            lp.timestamp AS frame_timestamp,
            ep.x AS ego_x,
            ep.y AS ego_y,
            ep.z AS ego_z,
            ep.qw AS ego_qw,
            ep.qx AS ego_qx,
            ep.qy AS ego_qy,
            ep.qz AS ego_qz,
            ep.vx AS ego_vx,
            ep.vy AS ego_vy
        FROM lidar_pc AS lp
        JOIN ego_pose AS ep
            ON ep.token = lp.ego_pose_token
        ORDER BY lp.timestamp, lp.token
    """
    return list(connection.execute(query))


def _subsample_frame_rows(
    rows: list[sqlite3.Row],
    *,
    subsample_hz: float | None,
) -> list[sqlite3.Row]:
    if subsample_hz is None:
        return list(rows)
    hz = _finite_float(subsample_hz, "subsample_hz")
    if hz <= 0.0:
        raise ValueError("subsample_hz must be positive")
    if len(rows) <= 1:
        return list(rows)

    times = np.array(
        [_timestamp_to_seconds(_finite_float(row["frame_timestamp"], "lidar_pc.timestamp")) for row in rows],
        dtype=np.float64,
    )
    raw_dt_s = _median_positive_diff(times)
    if raw_dt_s <= 0.0:
        stride = 1
    else:
        stride = max(1, int(round((1.0 / hz) / raw_dt_s)))
    return list(rows[::stride])


def _frame_context(row: sqlite3.Row) -> _FrameContext:
    ego_xy = (
        _finite_float(row["ego_x"], "ego_pose.x"),
        _finite_float(row["ego_y"], "ego_pose.y"),
    )
    ego_z = _finite_float(row["ego_z"], "ego_pose.z")
    ego_yaw = quaternion_yaw(
        (
            _finite_float(row["ego_qw"], "ego_pose.qw"),
            _finite_float(row["ego_qx"], "ego_pose.qx"),
            _finite_float(row["ego_qy"], "ego_pose.qy"),
            _finite_float(row["ego_qz"], "ego_pose.qz"),
        )
    )
    token_key = _token_key(row["frame_token"], "lidar_pc.token")
    return _FrameContext(
        token_key=token_key,
        frame_token=_token_to_str(token_key),
        time_s=_timestamp_to_seconds(_finite_float(row["frame_timestamp"], "lidar_pc.timestamp")),
        ego_xy=ego_xy,
        ego_z=ego_z,
        ego_yaw=ego_yaw,
        ego_v=(
            _finite_float(row["ego_vx"], "ego_pose.vx"),
            _finite_float(row["ego_vy"], "ego_pose.vy"),
        ),
    )


def _read_scene_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute("SELECT token, name FROM scene ORDER BY name, token").fetchall()
    names = []
    for row in rows:
        name = row["name"]
        if name is None or str(name) == "":
            name = _token_to_str(_token_key(row["token"], "scene.token"))
        names.append(str(name))
    return tuple(names)


def _read_scenario_tags(
    connection: sqlite3.Connection,
    *,
    selected_tokens: set[Hashable],
) -> dict[Hashable, tuple[str, ...]]:
    tags: dict[Hashable, set[str]] = defaultdict(set)
    query = """
        SELECT lidar_pc_token, type
        FROM scenario_tag
        ORDER BY type, token
    """
    for row in connection.execute(query):
        token_key = _token_key(row["lidar_pc_token"], "scenario_tag.lidar_pc_token")
        if token_key not in selected_tokens:
            continue
        tag_type = row["type"]
        if tag_type is not None and str(tag_type) != "":
            tags[token_key].add(str(tag_type))
    return {token: tuple(sorted(values)) for token, values in tags.items()}


def _read_box_records(
    connection: sqlite3.Connection,
    *,
    contexts: Iterable[_FrameContext],
    selected_tokens: set[Hashable],
    max_abs_xy_m: float | None,
    min_confidence: float,
    include_static: bool,
) -> dict[Hashable, list[dict[str, Any]]]:
    context_by_token = {context.token_key: context for context in contexts}
    records: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    query = """
        SELECT
            b.lidar_pc_token AS lidar_pc_token,
            b.track_token AS track_token,
            b.x AS box_x,
            b.y AS box_y,
            b.z AS box_z,
            b.width AS box_width,
            b.length AS box_length,
            b.height AS box_height,
            b.vx AS box_vx,
            b.vy AS box_vy,
            b.yaw AS box_yaw,
            b.confidence AS confidence,
            c.name AS category_name
        FROM lidar_box AS b
        LEFT JOIN track AS t
            ON t.token = b.track_token
        LEFT JOIN category AS c
            ON c.token = t.category_token
        ORDER BY b.lidar_pc_token, b.token
    """
    for row in connection.execute(query):
        frame_key = _token_key(row["lidar_pc_token"], "lidar_box.lidar_pc_token")
        if frame_key not in selected_tokens:
            continue
        context = context_by_token[frame_key]
        record = _box_record(
            row,
            context=context,
            max_abs_xy_m=max_abs_xy_m,
            min_confidence=min_confidence,
            include_static=include_static,
        )
        if record is not None:
            records[frame_key].append(record)
    return dict(records)


def _box_record(
    row: sqlite3.Row,
    *,
    context: _FrameContext,
    max_abs_xy_m: float | None,
    min_confidence: float,
    include_static: bool,
) -> dict[str, Any] | None:
    raw_category = _normalize_category(row["category_name"])
    if not include_static and raw_category in STATIC_NUPLAN_CATEGORIES:
        return None

    confidence = _finite_float(row["confidence"], "lidar_box.confidence")
    if confidence < min_confidence:
        return None

    local_xy = global_to_ego_xy(
        (
            _finite_float(row["box_x"], "lidar_box.x"),
            _finite_float(row["box_y"], "lidar_box.y"),
        ),
        context.ego_xy,
        context.ego_yaw,
    )
    if max_abs_xy_m is not None and (
        abs(float(local_xy[0])) > max_abs_xy_m or abs(float(local_xy[1])) > max_abs_xy_m
    ):
        return None

    velocity_local = global_to_ego_velocity(
        (
            _finite_float(row["box_vx"], "lidar_box.vx"),
            _finite_float(row["box_vy"], "lidar_box.vy"),
        ),
        context.ego_yaw,
    )

    width = _positive_float(row["box_width"], "lidar_box.width")
    length = _positive_float(row["box_length"], "lidar_box.length")
    height = _positive_float(row["box_height"], "lidar_box.height")
    track_key = _token_key(row["track_token"], "lidar_box.track_token")
    return {
        "track_id": _token_to_str(track_key),
        "cls": category_to_coarse_class(raw_category),
        "raw_category": raw_category,
        "x": float(local_xy[0]),
        "y": float(local_xy[1]),
        "z": _finite_float(row["box_z"], "lidar_box.z") - context.ego_z,
        "w": width,
        "h": height,
        "l": length,
        "yaw": wrap_angle(_finite_float(row["box_yaw"], "lidar_box.yaw") - context.ego_yaw),
        "vx": float(velocity_local[0]),
        "vy": float(velocity_local[1]),
        "conf": float(np.clip(confidence, 0.05, 1.0)),
    }


def _normalize_category(category_name: str | None) -> str:
    if category_name is None:
        return "unknown"
    return str(category_name).strip().lower()


def _token_key(value: Any, name: str) -> Hashable:
    if isinstance(value, bytes):
        if not value:
            raise ValueError(f"{name} must be a non-empty token")
        return value
    if isinstance(value, bytearray):
        data = bytes(value)
        if not data:
            raise ValueError(f"{name} must be a non-empty token")
        return data
    if isinstance(value, memoryview):
        data = value.tobytes()
        if not data:
            raise ValueError(f"{name} must be a non-empty token")
        return data
    if value is None or str(value) == "":
        raise ValueError(f"{name} must be a non-empty token")
    return str(value)


def _token_to_str(value: Hashable) -> str:
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _timestamp_to_seconds(timestamp: float) -> float:
    return float(timestamp) * MICROSECONDS_TO_SECONDS


def _median_positive_diff(values: NDArray[np.float64]) -> float:
    if values.size <= 1:
        return 0.0
    diffs = np.diff(values)
    positive = diffs[diffs > 0.0]
    return float(np.median(positive)) if positive.size else 0.0


def _resolve_max_abs_xy(max_abs_xy_m: float | None) -> float | None:
    if max_abs_xy_m is None:
        return None
    value = _finite_float(max_abs_xy_m, "max_abs_xy_m")
    if value <= 0.0:
        raise ValueError("max_abs_xy_m must be positive")
    return value


def _resolve_min_confidence(min_confidence: float) -> float:
    value = _finite_float(min_confidence, "min_confidence")
    if value < 0.0:
        raise ValueError("min_confidence must be non-negative")
    return value


def _positive_float(value: Any, name: str) -> float:
    out = _finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _vector2(value: Iterable[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr.copy()
