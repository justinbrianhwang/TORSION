"""nuScenes metadata loader for TORSION ObjectSet frames.

The adapter reads nuScenes JSON metadata tables directly, without the
``nuscenes-devkit``. Annotation centers and velocities are emitted in the
ego-local frame used by TORSION: ``x`` is forward and ``y`` is left.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torsion.data._geometry import (
    global_to_ego_velocity,
    global_to_ego_xy,
    quaternion_yaw,
    rotate_xy,
    wrap_angle,
)
from torsion.operators.object import ObjectSet

DEFAULT_MAX_ABS_XY_M = 60.0
LIDAR_TOP_CHANNEL = "LIDAR_TOP"
MICROSECONDS_TO_SECONDS = 1.0e-6


@dataclass(frozen=True)
class NuScenesFrame:
    """One nuScenes keyframe converted to ego-local TORSION objects."""

    sample_token: str
    time_s: float
    ego_xy: tuple[float, float]
    ego_yaw: float
    object_set: ObjectSet


@dataclass(frozen=True)
class NuScenesSceneData:
    """A nuScenes scene as a deterministic sequence of ObjectSet frames."""

    scene_token: str
    name: str
    dt_s: float
    frames: tuple[NuScenesFrame, ...]
    ego_trajectory: NDArray[np.float64]

    def __post_init__(self) -> None:
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

    def object_set(self, i: int) -> ObjectSet:
        """Return the ObjectSet for frame ``i``."""

        return self.frames[int(i)].object_set

    def ego_at(self, i: int) -> NDArray[np.float64]:
        """Return ego ``[x, y, yaw]`` in the global frame for frame ``i``."""

        return self.ego_trajectory[int(i)].copy()


def load_tables(meta_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read all JSON metadata tables present in ``meta_dir``.

    Each table is returned under its file stem, e.g. ``sample.json`` becomes
    ``tables["sample"]``. Optional nuScenes tables may be absent; callers that
    need a table validate it explicitly.
    """

    root = Path(meta_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"nuScenes metadata directory not found: {root}")

    tables: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"{path.name} must contain a JSON list")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{path.name} must contain only JSON objects")
        tables[path.stem] = rows
    return tables


def load_scenes(
    meta_dir: str | Path,
    version: str | None = None,
    max_scenes: int | None = None,
    *,
    max_abs_xy_m: float | None = DEFAULT_MAX_ABS_XY_M,
) -> list[NuScenesSceneData]:
    """Load nuScenes scenes from metadata JSON tables.

    ``version`` may name a child metadata directory such as ``v1.0-trainval``.
    ``max_abs_xy_m`` filters objects whose ego-local ``|x|`` or ``|y|`` exceeds
    the configured range; pass ``None`` to disable the range filter.
    """

    root = _metadata_root(meta_dir, version)
    tables = load_tables(root)
    _require_tables(tables, ("scene", "sample", "sample_annotation", "ego_pose", "sample_data"))

    if max_scenes is not None and max_scenes < 0:
        raise ValueError("max_scenes must be non-negative")
    max_range = _resolve_max_abs_xy(max_abs_xy_m)

    samples_by_token = _index_by_token(tables["sample"], "sample")
    ego_pose_by_token = _index_by_token(tables["ego_pose"], "ego_pose")
    instance_by_token = _index_by_token(tables.get("instance", []), "instance")
    category_by_token = {
        token: str(row.get("name", "unknown"))
        for token, row in _index_by_token(tables.get("category", []), "category").items()
    }
    calibrated_by_token = _index_by_token(
        tables.get("calibrated_sensor", []), "calibrated_sensor"
    )
    sensor_by_token = _index_by_token(tables.get("sensor", []), "sensor")

    lidar_calibrated_tokens = _lidar_top_calibrated_tokens(
        calibrated_by_token=calibrated_by_token,
        sensor_by_token=sensor_by_token,
    )
    sample_data_by_sample = _group_sample_data(tables["sample_data"])
    annotations_by_sample, annotations_by_instance = _group_annotations(
        tables["sample_annotation"],
        samples_by_token=samples_by_token,
    )
    velocity_by_annotation = _annotation_velocities(
        annotations_by_instance,
        samples_by_token=samples_by_token,
    )

    scenes = sorted(
        tables["scene"],
        key=lambda row: (str(row.get("name", "")), str(row.get("token", ""))),
    )
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    return [
        _build_scene(
            scene,
            samples_by_token=samples_by_token,
            annotations_by_sample=annotations_by_sample,
            velocity_by_annotation=velocity_by_annotation,
            ego_pose_by_token=ego_pose_by_token,
            sample_data_by_sample=sample_data_by_sample,
            lidar_calibrated_tokens=lidar_calibrated_tokens,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
            max_abs_xy_m=max_range,
        )
        for scene in scenes
    ]


def category_to_coarse_class(category_name: str | None) -> str:
    """Map a nuScenes category name to the coarse class used by TORSION."""

    name = "" if category_name is None else str(category_name).strip().lower()
    if name == "vehicle.car":
        return "car"
    if (
        name.startswith("vehicle.truck")
        or name.startswith("vehicle.bus")
        or name.startswith("vehicle.trailer")
        or name.startswith("vehicle.construction")
    ):
        return "truck"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    if name in {"vehicle.bicycle", "vehicle.motorcycle"}:
        return "cyclist"
    return "other"


def confidence_from_lidar_points(num_lidar_pts: Any) -> float:
    """Saturating confidence proxy from lidar returns."""

    try:
        points = float(num_lidar_pts)
    except (TypeError, ValueError):
        points = 0.0
    if not np.isfinite(points):
        points = 0.0
    return float(np.clip(points / 50.0, 0.05, 1.0))


def _build_scene(
    scene: Mapping[str, Any],
    *,
    samples_by_token: Mapping[str, Mapping[str, Any]],
    annotations_by_sample: Mapping[str, list[Mapping[str, Any]]],
    velocity_by_annotation: Mapping[str, NDArray[np.float64]],
    ego_pose_by_token: Mapping[str, Mapping[str, Any]],
    sample_data_by_sample: Mapping[str, list[Mapping[str, Any]]],
    lidar_calibrated_tokens: set[str],
    instance_by_token: Mapping[str, Mapping[str, Any]],
    category_by_token: Mapping[str, str],
    max_abs_xy_m: float | None,
) -> NuScenesSceneData:
    scene_token = _field_str(scene, "token", "scene")
    scene_name = str(scene.get("name", scene_token))
    samples = _scene_samples(scene, samples_by_token)

    frames: list[NuScenesFrame] = []
    ego_rows: list[list[float]] = []
    for sample in samples:
        sample_token = _field_str(sample, "token", "sample")
        sample_time_s = _sample_time_s(sample)
        ego_pose = _ego_pose_for_sample(
            sample_token,
            ego_pose_by_token=ego_pose_by_token,
            sample_data_by_sample=sample_data_by_sample,
            lidar_calibrated_tokens=lidar_calibrated_tokens,
        )
        ego_translation = _vector3(ego_pose.get("translation"), "ego_pose.translation")
        ego_xy = (float(ego_translation[0]), float(ego_translation[1]))
        ego_z = float(ego_translation[2])
        ego_yaw = quaternion_yaw(_require_field(ego_pose, "rotation", "ego_pose"))

        records = []
        for annotation in annotations_by_sample.get(sample_token, []):
            record = _annotation_record(
                annotation,
                ego_xy=ego_xy,
                ego_z=ego_z,
                ego_yaw=ego_yaw,
                velocity_by_annotation=velocity_by_annotation,
                instance_by_token=instance_by_token,
                category_by_token=category_by_token,
            )
            if max_abs_xy_m is not None and (
                abs(record["x"]) > max_abs_xy_m or abs(record["y"]) > max_abs_xy_m
            ):
                continue
            records.append(record)

        records.sort(key=lambda row: (float(np.hypot(row["x"], row["y"])), str(row["track_id"])))
        object_set = ObjectSet.from_records(records)
        frames.append(
            NuScenesFrame(
                sample_token=sample_token,
                time_s=sample_time_s,
                ego_xy=ego_xy,
                ego_yaw=ego_yaw,
                object_set=object_set,
            )
        )
        ego_rows.append([ego_xy[0], ego_xy[1], ego_yaw])

    times = np.array([frame.time_s for frame in frames], dtype=np.float64)
    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0.0]
    dt_s = float(np.median(positive_diffs)) if positive_diffs.size else 0.0
    ego_trajectory = np.asarray(ego_rows, dtype=np.float64).reshape(len(frames), 3)
    return NuScenesSceneData(
        scene_token=scene_token,
        name=scene_name,
        dt_s=dt_s,
        frames=tuple(frames),
        ego_trajectory=ego_trajectory,
    )


def _annotation_record(
    annotation: Mapping[str, Any],
    *,
    ego_xy: tuple[float, float],
    ego_z: float,
    ego_yaw: float,
    velocity_by_annotation: Mapping[str, NDArray[np.float64]],
    instance_by_token: Mapping[str, Mapping[str, Any]],
    category_by_token: Mapping[str, str],
) -> dict[str, Any]:
    annotation_token = _field_str(annotation, "token", "sample_annotation")
    instance_token = _field_str(annotation, "instance_token", "sample_annotation")
    translation = _vector3(annotation.get("translation"), "sample_annotation.translation")
    size = _bbox_size(annotation.get("size"), "sample_annotation.size")
    local_xy = global_to_ego_xy(translation[:2], ego_xy, ego_yaw)
    annotation_yaw = quaternion_yaw(_require_field(annotation, "rotation", "sample_annotation"))
    velocity_global = velocity_by_annotation.get(
        annotation_token, np.zeros(2, dtype=np.float64)
    )
    velocity_local = global_to_ego_velocity(velocity_global, ego_yaw)
    raw_category = _annotation_category(
        annotation,
        instance_by_token=instance_by_token,
        category_by_token=category_by_token,
    )

    return {
        "track_id": instance_token,
        "cls": category_to_coarse_class(raw_category),
        "raw_category": raw_category,
        "x": float(local_xy[0]),
        "y": float(local_xy[1]),
        "z": float(translation[2] - ego_z),
        "w": float(size[0]),
        "h": float(size[2]),
        "l": float(size[1]),
        "yaw": wrap_angle(annotation_yaw - ego_yaw),
        "vx": float(velocity_local[0]),
        "vy": float(velocity_local[1]),
        "conf": confidence_from_lidar_points(annotation.get("num_lidar_pts", 0)),
    }


def _annotation_category(
    annotation: Mapping[str, Any],
    *,
    instance_by_token: Mapping[str, Mapping[str, Any]],
    category_by_token: Mapping[str, str],
) -> str:
    direct = annotation.get("category_name")
    if direct:
        return str(direct)

    instance = instance_by_token.get(str(annotation.get("instance_token", "")))
    if instance is None:
        return "unknown"
    category_token = instance.get("category_token")
    if category_token is None:
        return "unknown"
    return category_by_token.get(str(category_token), "unknown")


def _annotation_velocities(
    annotations_by_instance: Mapping[str, list[Mapping[str, Any]]],
    *,
    samples_by_token: Mapping[str, Mapping[str, Any]],
) -> dict[str, NDArray[np.float64]]:
    velocities: dict[str, NDArray[np.float64]] = {}
    for annotations in annotations_by_instance.values():
        rows = []
        for annotation in annotations:
            sample_token = _field_str(annotation, "sample_token", "sample_annotation")
            sample = samples_by_token[sample_token]
            rows.append(
                (
                    _sample_time_s(sample),
                    _field_str(annotation, "token", "sample_annotation"),
                    _vector2(
                        _vector3(
                            annotation.get("translation"), "sample_annotation.translation"
                        )[:2],
                        "sample_annotation.translation.xy",
                    ),
                )
            )

        rows.sort(key=lambda row: (row[0], row[1]))
        if len(rows) == 1:
            velocities[rows[0][1]] = np.zeros(2, dtype=np.float64)
            continue

        for i, (_, annotation_token, _) in enumerate(rows):
            if i == 0:
                before = rows[0]
                after = rows[1]
            elif i == len(rows) - 1:
                before = rows[-2]
                after = rows[-1]
            else:
                before = rows[i - 1]
                after = rows[i + 1]

            dt_s = after[0] - before[0]
            if dt_s <= 0.0:
                velocity = np.zeros(2, dtype=np.float64)
            else:
                velocity = (after[2] - before[2]) / dt_s
            velocities[annotation_token] = velocity.astype(np.float64, copy=True)
    return velocities


def _scene_samples(
    scene: Mapping[str, Any],
    samples_by_token: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    scene_token = _field_str(scene, "token", "scene")
    first_token = str(scene.get("first_sample_token", ""))
    if first_token:
        samples = []
        seen: set[str] = set()
        current = first_token
        while current:
            if current in seen:
                raise ValueError(f"sample chain for scene {scene_token!r} contains a cycle")
            seen.add(current)
            try:
                sample = samples_by_token[current]
            except KeyError as exc:
                raise ValueError(
                    f"scene {scene_token!r} references missing sample {current!r}"
                ) from exc
            if str(sample.get("scene_token", scene_token)) != scene_token:
                raise ValueError(
                    f"sample {current!r} belongs to a different scene than {scene_token!r}"
                )
            samples.append(sample)
            current = str(sample.get("next", ""))
        return samples

    samples = [
        sample for sample in samples_by_token.values() if str(sample.get("scene_token", "")) == scene_token
    ]
    return sorted(samples, key=lambda sample: (_sample_time_s(sample), str(sample.get("token", ""))))


def _ego_pose_for_sample(
    sample_token: str,
    *,
    ego_pose_by_token: Mapping[str, Mapping[str, Any]],
    sample_data_by_sample: Mapping[str, list[Mapping[str, Any]]],
    lidar_calibrated_tokens: set[str],
) -> Mapping[str, Any]:
    candidates = sample_data_by_sample.get(sample_token, [])
    lidar_candidates = [
        row
        for row in candidates
        if bool(row.get("is_key_frame", False))
        and str(row.get("calibrated_sensor_token", "")) in lidar_calibrated_tokens
    ]
    if not lidar_candidates and lidar_calibrated_tokens:
        lidar_candidates = [
            row
            for row in candidates
            if str(row.get("calibrated_sensor_token", "")) in lidar_calibrated_tokens
        ]

    if lidar_candidates:
        chosen = sorted(lidar_candidates, key=_sample_data_sort_key)[0]
    else:
        # Fixtures or reduced metadata may omit sensor/calibration tables. If
        # exactly one keyframe sample_data row remains, its ego pose is
        # unambiguous and schema-compatible.
        keyframes = [
            row
            for row in candidates
            if bool(row.get("is_key_frame", False)) and row.get("ego_pose_token") is not None
        ]
        pose_rows = keyframes or [row for row in candidates if row.get("ego_pose_token") is not None]
        if len(pose_rows) != 1:
            raise ValueError(
                f"could not identify a unique LIDAR_TOP ego pose for sample {sample_token!r}"
            )
        chosen = pose_rows[0]

    ego_pose_token = _field_str(chosen, "ego_pose_token", "sample_data")
    try:
        return ego_pose_by_token[ego_pose_token]
    except KeyError as exc:
        raise ValueError(
            f"sample_data references missing ego_pose {ego_pose_token!r}"
        ) from exc


def _lidar_top_calibrated_tokens(
    *,
    calibrated_by_token: Mapping[str, Mapping[str, Any]],
    sensor_by_token: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    tokens: set[str] = set()
    for token, calibrated in calibrated_by_token.items():
        sensor = sensor_by_token.get(str(calibrated.get("sensor_token", "")))
        if sensor is not None and str(sensor.get("channel", "")) == LIDAR_TOP_CHANNEL:
            tokens.add(token)
    return tokens


def _group_sample_data(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _field_str(row, "token", "sample_data")
        grouped[_field_str(row, "sample_token", "sample_data")].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=_sample_data_sort_key)
    return dict(grouped)


def _group_annotations(
    rows: Iterable[Mapping[str, Any]],
    *,
    samples_by_token: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, list[Mapping[str, Any]]]]:
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_instance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _field_str(row, "token", "sample_annotation")
        sample_token = _field_str(row, "sample_token", "sample_annotation")
        instance_token = _field_str(row, "instance_token", "sample_annotation")
        if sample_token not in samples_by_token:
            raise ValueError(f"annotation references missing sample {sample_token!r}")
        by_sample[sample_token].append(row)
        by_instance[instance_token].append(row)

    for grouped in (by_sample, by_instance):
        for group_rows in grouped.values():
            group_rows.sort(
                key=lambda row: (
                    _sample_time_s(samples_by_token[_field_str(row, "sample_token", "sample_annotation")]),
                    _field_str(row, "token", "sample_annotation"),
                )
            )
    return dict(by_sample), dict(by_instance)


def _metadata_root(meta_dir: str | Path, version: str | None) -> Path:
    root = Path(meta_dir)
    if version is None:
        return root
    if root.name == version:
        return root
    candidate = root / version
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"nuScenes version directory not found: {candidate}")


def _require_tables(tables: Mapping[str, Any], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in tables]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"missing required nuScenes metadata table(s): {joined}")


def _index_by_token(
    rows: Iterable[Mapping[str, Any]],
    table_name: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        token = _field_str(row, "token", table_name)
        if token in indexed:
            raise ValueError(f"{table_name} contains duplicate token {token!r}")
        indexed[token] = row
    return indexed


def _sample_time_s(sample: Mapping[str, Any]) -> float:
    timestamp = _require_field(sample, "timestamp", "sample")
    try:
        value = float(timestamp) * MICROSECONDS_TO_SECONDS
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample timestamp must be numeric: {timestamp!r}") from exc
    if not np.isfinite(value):
        raise ValueError("sample timestamp must be finite")
    return value


def _sample_data_sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
    timestamp = row.get("timestamp", 0)
    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        timestamp_value = 0.0
    return (timestamp_value, str(row.get("token", "")))


def _require_field(row: Mapping[str, Any], key: str, table_name: str) -> Any:
    if key not in row:
        raise ValueError(f"{table_name} row is missing required field {key!r}")
    return row[key]


def _field_str(row: Mapping[str, Any], key: str, table_name: str) -> str:
    value = _require_field(row, key, table_name)
    if value is None or str(value) == "":
        raise ValueError(f"{table_name}.{key} must be a non-empty token")
    return str(value)


def _vector2(value: Any, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr.copy()


def _vector3(value: Any, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 3D vector")
    return arr.copy()


def _bbox_size(value: Any, name: str) -> NDArray[np.float64]:
    size = _vector3(value, name)
    if np.any(size <= 0.0):
        raise ValueError(f"{name} dimensions must be positive")
    return size


def _resolve_max_abs_xy(max_abs_xy_m: float | None) -> float | None:
    if max_abs_xy_m is None:
        return None
    value = float(max_abs_xy_m)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("max_abs_xy_m must be positive and finite")
    return value
