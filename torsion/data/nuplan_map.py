"""nuPlan GeoPackage map adapter for ego-local road cost priors.

The adapter intentionally avoids GDAL/geopandas/nuPlan devkit dependencies. It
reads feature layers directly from the map ``.gpkg`` SQLite file, strips the
GeoPackage Binary header, reprojects lon/lat geometry to the log's local UTM
CRS, and rasterizes nearby lane polygons into the TORSION cost-map convention.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree

from torsion.data._geometry import rotate_xy
from torsion.operators.costmap import COST_MAX, COST_MIN

LOCATION_TO_EPSG: dict[str, str] = {
    "las_vegas": "EPSG:32611",
    "las-vegas": "EPSG:32611",
    "us-nv-las-vegas-strip": "EPSG:32611",
    "boston": "EPSG:32619",
    "us-ma-boston": "EPSG:32619",
    "pittsburgh": "EPSG:32617",
    "pittsburgh_hazelwood": "EPSG:32617",
    "pittsburgh-hazelwood": "EPSG:32617",
    "us-pa-pittsburgh-hazelwood": "EPSG:32617",
    "singapore": "EPSG:32648",
    "one_north": "EPSG:32648",
    "one-north": "EPSG:32648",
    "sg-one-north": "EPSG:32648",
}

DEFAULT_LAYERS: tuple[str, ...] = (
    "lanes_polygons",
    "gen_lane_connectors_scaled_width_polygons",
    "boundaries",
    "crosswalks",
    "walkways",
    "generic_drivable_areas",
)

# The drivable surface is NOT `lanes_polygons` alone. nuPlan stores the area a
# vehicle traverses *through an intersection* in a separate lane-connector layer;
# inside an intersection the ego is outside every `lanes_polygons` polygon. Taking
# lanes only therefore marks the entire intersection as hard off-road, which makes
# every planner candidate collide and leaves the planner permanently in its
# fallback pool. Both layers together are the drivable surface.
LANE_LAYERS: tuple[str, ...] = (
    "lanes_polygons",
    "gen_lane_connectors_scaled_width_polygons",
)
LANE_LAYER = "lanes_polygons"  # retained for backwards compatibility
BOUNDARY_LAYER = "boundaries"
SOURCE_CRS = "EPSG:4326"
DRIVABLE_BASE_COST = 0.03
BOUNDARY_RIDGE_COST = 0.08
BOUNDARY_RIDGE_SIGMA_M = 0.22
_GPKG_ENVELOPE_SIZES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


@dataclass(frozen=True)
class NuPlanMap:
    """Reprojected nuPlan map layers and spatial indexes."""

    gpkg_path: Path
    epsg: str
    layers: Mapping[str, tuple[BaseGeometry, ...]]
    lane_geometries: tuple[BaseGeometry, ...]
    lane_tree: STRtree
    boundary_geometries: tuple[BaseGeometry, ...]
    boundary_tree: STRtree | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpkg_path", Path(self.gpkg_path))
        object.__setattr__(self, "epsg", _normalize_epsg(self.epsg))
        object.__setattr__(self, "lane_geometries", tuple(self.lane_geometries))
        object.__setattr__(self, "boundary_geometries", tuple(self.boundary_geometries))
        object.__setattr__(
            self,
            "layers",
            MappingProxyType({str(k): tuple(v) for k, v in self.layers.items()}),
        )
        if not self.lane_geometries:
            raise ValueError("NuPlanMap requires at least one lane polygon")


def epsg_for_location(location_or_map_version: str) -> str:
    """Return the UTM EPSG code for a nuPlan location or map-version key."""

    key = str(location_or_map_version).strip().lower()
    for candidate in (key, key.replace("_", "-"), key.replace("-", "_")):
        epsg = LOCATION_TO_EPSG.get(candidate)
        if epsg is not None:
            return epsg
    valid = ", ".join(sorted(LOCATION_TO_EPSG))
    raise ValueError(f"unknown nuPlan location/map_version {location_or_map_version!r}; expected one of {valid}")


def map_path_for_log(log_row_or_db: Any, maps_root: str | Path) -> Path:
    """Resolve the ``map.gpkg`` path for a nuPlan log row, DB path, or DB connection."""

    metadata = _log_metadata(log_row_or_db)
    map_version = _metadata_value(metadata, "map_version")
    location = _metadata_value(metadata, "location")
    map_key = str(map_version or location or "").strip()
    if not map_key:
        raise ValueError("log metadata must include map_version or location")

    root = Path(maps_root)
    candidates = (
        root / map_key,
        root / "nuplan-maps-v1.0" / map_key,
    )
    for base in candidates:
        direct = base / "map.gpkg"
        if direct.is_file():
            return direct
        if base.is_dir():
            for child in sorted(base.iterdir(), key=lambda path: path.name):
                candidate = child / "map.gpkg"
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(f"map.gpkg for {map_key!r} not found under {root}")


def geometry_from_gpkg_binary(blob: bytes | bytearray | memoryview) -> BaseGeometry:
    """Parse a GeoPackage Binary geometry blob into a shapely geometry."""

    return wkb.loads(_strip_gpkg_binary_header(blob))


def load_map(
    gpkg_path: str | Path,
    epsg: str,
    layers: Sequence[str] = DEFAULT_LAYERS,
) -> NuPlanMap:
    """Load and reproject nuPlan map feature layers once, with an LRU cache."""

    path = Path(gpkg_path)
    if not path.is_file():
        raise FileNotFoundError(f"nuPlan map not found: {path}")
    layer_tuple = tuple(dict.fromkeys(str(layer) for layer in layers))
    missing = tuple(layer for layer in LANE_LAYERS if layer not in layer_tuple)
    if missing:
        layer_tuple = (*missing, *layer_tuple)
    return _load_map_cached(str(path.resolve()), _normalize_epsg(epsg), layer_tuple)


def build_map_road_cost_grid(
    nmap: NuPlanMap,
    ego_global_xy: Iterable[float],
    ego_yaw: float,
    spec: Any,
    planner_config: Any,
) -> NDArray[np.float64]:
    """Rasterize map lanes/boundaries into an ego-local TORSION road cost grid.

    Modeling choice: cells strictly inside nearby lane polygons receive the same
    low baseline as the synthetic road prior; cells outside those polygons are
    hard off-road cost. Nearby map boundary lines add a Gaussian ridge on top of
    drivable cells, matching the synthetic lane-boundary scale.
    """

    del planner_config
    ego_xy = _vector2(ego_global_xy, "ego_global_xy")
    yaw = _finite_float(ego_yaw, "ego_yaw")
    _, _, global_x, global_y = _grid_local_and_global(spec, ego_xy, yaw)
    grid_box = _global_box_for_spec(spec, ego_xy, yaw, margin_m=float(spec.resolution_m))

    lane_indices = _query_tree(nmap.lane_tree, grid_box)
    if lane_indices.size == 0:
        return np.full(tuple(spec.shape), COST_MAX, dtype=np.float64)

    lane_union = _union_geometries(_take_geometries(nmap.lane_geometries, lane_indices))
    if lane_union.is_empty:
        return np.full(tuple(spec.shape), COST_MAX, dtype=np.float64)

    shapely.prepare(lane_union)
    flat_x = np.asarray(global_x, dtype=np.float64).ravel()
    flat_y = np.asarray(global_y, dtype=np.float64).ravel()
    drivable_flat = np.asarray(shapely.contains_xy(lane_union, flat_x, flat_y), dtype=bool)
    if hasattr(shapely, "intersects_xy"):
        drivable_flat |= np.asarray(shapely.intersects_xy(lane_union, flat_x, flat_y), dtype=bool)
    drivable = drivable_flat.reshape(tuple(spec.shape))

    cost = np.full(tuple(spec.shape), COST_MAX, dtype=np.float64)
    cost[drivable] = DRIVABLE_BASE_COST

    boundary_distance = _geometry_distance_grid(lane_union.boundary, global_x, global_y)
    ridge = BOUNDARY_RIDGE_COST * np.exp(
        -0.5 * (boundary_distance / BOUNDARY_RIDGE_SIGMA_M) ** 2
    )
    cost[drivable] += ridge[drivable]

    return np.clip(cost, COST_MIN, COST_MAX).astype(np.float64, copy=False)


def road_boundary_mask_from_map(
    nmap: NuPlanMap,
    ego_global_xy: Iterable[float],
    ego_yaw: float,
    spec: Any,
    planner_config: Any | None = None,
) -> NDArray[np.bool_]:
    """Return cells near map boundary lines for cost-map topology preservation."""

    del planner_config
    ego_xy = _vector2(ego_global_xy, "ego_global_xy")
    yaw = _finite_float(ego_yaw, "ego_yaw")
    _, _, global_x, global_y = _grid_local_and_global(spec, ego_xy, yaw)
    tolerance = max(0.5 * float(spec.resolution_m), 0.24)
    grid_box = _global_box_for_spec(spec, ego_xy, yaw, margin_m=tolerance)
    lane_indices = _query_tree(nmap.lane_tree, grid_box)
    if lane_indices.size == 0:
        return np.zeros(tuple(spec.shape), dtype=bool)
    lane_union = _union_geometries(_take_geometries(nmap.lane_geometries, lane_indices))
    distance = _geometry_distance_grid(lane_union.boundary, global_x, global_y)
    return (distance <= tolerance).astype(bool, copy=False)


@lru_cache(maxsize=8)
def _load_map_cached(path_str: str, epsg: str, layers: tuple[str, ...]) -> NuPlanMap:
    path = Path(path_str)
    transformer = Transformer.from_crs(SOURCE_CRS, epsg, always_xy=True)
    layers_by_name: dict[str, tuple[BaseGeometry, ...]] = {}

    connection = sqlite3.connect(path)
    try:
        for layer in layers:
            layers_by_name[layer] = _read_layer_geometries(connection, layer, transformer)
    finally:
        connection.close()

    # Drivable surface = lane polygons + intersection lane-connector polygons.
    lane_geometries = tuple(
        geometry
        for layer in LANE_LAYERS
        for geometry in layers_by_name.get(layer, ())
    )
    if not lane_geometries:
        raise ValueError(
            f"none of the drivable layers {LANE_LAYERS!r} are present in {path}"
        )
    boundary_geometries = layers_by_name.get(BOUNDARY_LAYER, ())
    return NuPlanMap(
        gpkg_path=path,
        epsg=epsg,
        layers=layers_by_name,
        lane_geometries=lane_geometries,
        lane_tree=STRtree(lane_geometries),
        boundary_geometries=boundary_geometries,
        boundary_tree=STRtree(boundary_geometries) if boundary_geometries else None,
    )


def _read_layer_geometries(
    connection: sqlite3.Connection,
    layer: str,
    transformer: Transformer,
) -> tuple[BaseGeometry, ...]:
    if not _table_exists(connection, layer):
        return ()
    geom_column = _geometry_column(connection, layer)
    if geom_column is None:
        return ()

    query = (
        f"SELECT {_quote_identifier(geom_column)} "
        f"FROM {_quote_identifier(layer)} "
        "WHERE "
        f"{_quote_identifier(geom_column)} IS NOT NULL "
        "ORDER BY fid"
    )
    geometries: list[BaseGeometry] = []
    for (blob,) in connection.execute(query):
        geometry = geometry_from_gpkg_binary(blob)
        if geometry.is_empty:
            continue
        reprojected = transform_geometry(transformer.transform, geometry)
        if not reprojected.is_empty:
            geometries.append(reprojected)
    return tuple(geometries)


def _strip_gpkg_binary_header(blob: bytes | bytearray | memoryview) -> bytes:
    data = bytes(blob)
    if len(data) < 8:
        raise ValueError("GeoPackage geometry blob is too short")
    if data[:2] != b"GP":
        raise ValueError("GeoPackage geometry blob must start with b'GP'")
    envelope_code = (data[3] >> 1) & 0b111
    try:
        envelope_size = _GPKG_ENVELOPE_SIZES[envelope_code]
    except KeyError as exc:
        raise ValueError(f"unsupported GeoPackage envelope code {envelope_code}") from exc
    header_size = 8 + envelope_size
    if len(data) <= header_size:
        raise ValueError("GeoPackage geometry blob has no WKB payload")
    return data[header_size:]


def _grid_local_and_global(
    spec: Any,
    ego_xy: NDArray[np.float64],
    ego_yaw: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    local_x, local_y = spec.metric_mesh()
    local_x = np.asarray(local_x, dtype=np.float64)
    local_y = np.asarray(local_y, dtype=np.float64)
    if local_x.shape != tuple(spec.shape) or local_y.shape != tuple(spec.shape):
        raise ValueError("spec.metric_mesh() must match spec.shape")

    c = float(np.cos(ego_yaw))
    s = float(np.sin(ego_yaw))
    global_x = ego_xy[0] + c * local_x - s * local_y
    global_y = ego_xy[1] + s * local_x + c * local_y
    return local_x, local_y, global_x, global_y


def _global_box_for_spec(spec: Any, ego_xy: NDArray[np.float64], ego_yaw: float, *, margin_m: float) -> BaseGeometry:
    corners = (
        (float(spec.x_min_m), float(spec.y_min_m)),
        (float(spec.x_min_m), float(spec.y_max_m)),
        (float(spec.x_max_m), float(spec.y_min_m)),
        (float(spec.x_max_m), float(spec.y_max_m)),
    )
    global_corners = np.asarray([ego_xy + rotate_xy(corner, ego_yaw) for corner in corners], dtype=np.float64)
    min_xy = np.min(global_corners, axis=0) - float(margin_m)
    max_xy = np.max(global_corners, axis=0) + float(margin_m)
    return box(float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1]))


def _geometry_distance_grid(
    geometry: BaseGeometry,
    global_x: NDArray[np.float64],
    global_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    points = np.column_stack(
        (
            np.asarray(global_x, dtype=np.float64).ravel(),
            np.asarray(global_y, dtype=np.float64).ravel(),
        )
    )
    distances = np.asarray(
        shapely.distance(shapely.points(points[:, 0], points[:, 1]), geometry),
        dtype=np.float64,
    )
    return distances.reshape(global_x.shape)


def _query_tree(tree: STRtree, geometry: BaseGeometry) -> NDArray[np.int64]:
    indices = np.asarray(tree.query(geometry), dtype=np.int64)
    return indices.reshape(-1)


def _take_geometries(
    geometries: Sequence[BaseGeometry],
    indices: NDArray[np.int64],
) -> tuple[BaseGeometry, ...]:
    return tuple(geometries[int(index)] for index in indices)


def _union_geometries(geometries: Sequence[BaseGeometry]) -> BaseGeometry:
    if len(geometries) == 1:
        return geometries[0]
    return shapely.union_all(np.asarray(tuple(geometries), dtype=object))


def _log_metadata(log_row_or_db: Any) -> Mapping[str, Any]:
    if isinstance(log_row_or_db, sqlite3.Connection):
        return _read_log_metadata(log_row_or_db)
    if isinstance(log_row_or_db, (str, Path)):
        path = Path(log_row_or_db)
        if path.is_file():
            connection = sqlite3.connect(path)
            try:
                return _read_log_metadata(connection)
            finally:
                connection.close()
    if hasattr(log_row_or_db, "db_path"):
        return _log_metadata(Path(log_row_or_db.db_path))
    if isinstance(log_row_or_db, Mapping):
        return log_row_or_db
    if hasattr(log_row_or_db, "keys"):
        return {str(key): log_row_or_db[key] for key in log_row_or_db.keys()}
    return {
        "location": getattr(log_row_or_db, "location", None),
        "map_version": getattr(log_row_or_db, "map_version", None),
    }


def _read_log_metadata(connection: sqlite3.Connection) -> Mapping[str, Any]:
    cursor = connection.execute("SELECT location, map_version FROM log LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        raise ValueError("nuPlan DB has no log row")
    columns = [str(description[0]) for description in cursor.description]
    return dict(zip(columns, row, strict=True))


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    lowered = {str(name).lower(): value for name, value in metadata.items()}
    return lowered.get(key.lower())


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _geometry_column(connection: sqlite3.Connection, table_name: str) -> str | None:
    if _table_exists(connection, "gpkg_geometry_columns"):
        row = connection.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        if row is not None:
            return str(row[0])

    rows = connection.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    for row in rows:
        if str(row[1]).lower() == "geom":
            return str(row[1])
    for row in rows:
        column_type = str(row[2]).upper()
        if column_type in {"GEOMETRY", "POLYGON", "LINESTRING", "MULTIPOLYGON", "MULTILINESTRING"}:
            return str(row[1])
    return None


def _quote_identifier(identifier: str) -> str:
    text = str(identifier)
    if not text or any(not (char.isalnum() or char == "_") for char in text):
        raise ValueError(f"unsafe SQLite identifier {identifier!r}")
    return '"' + text.replace('"', '""') + '"'


def _normalize_epsg(epsg: str) -> str:
    text = str(epsg).strip().upper()
    if text.isdecimal():
        return f"EPSG:{text}"
    if text.startswith("EPSG:") and text[5:].isdecimal():
        return text
    raise ValueError(f"epsg must be an EPSG code, got {epsg!r}")


def _vector2(value: Iterable[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return arr.copy()


def _finite_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


__all__ = [
    "DEFAULT_LAYERS",
    "LOCATION_TO_EPSG",
    "NuPlanMap",
    "build_map_road_cost_grid",
    "epsg_for_location",
    "geometry_from_gpkg_binary",
    "load_map",
    "map_path_for_log",
    "road_boundary_mask_from_map",
]
