"""CARLA closed-loop validation harness for object-set semantic faults.

The runner attaches to the currently loaded CARLA world and never reloads or
switches maps.  For the leading-vehicle scenario it enumerates the active
map's driving waypoints at 2 m spacing, then walks forward from each candidate
to find starts whose lane remains non-junction and near-straight for at least
60 m.  Ego spawns at the selected start and the lead vehicle spawns ahead in
the same lane.  For the cut-in scenario it starts from those same straight
lane candidates, then accepts only candidates whose immediate left or right
driving lane is parallel, similarly straight, and separated by a plausible lane
width.  Ego starts in the base lane while the cut-in vehicle starts slightly
ahead in the adjacent lane and follows a smooth lateral script into ego's lane.
Only the controller's object-set perception is perturbed; CARLA physics and
ground-truth metrics always use the unmodified actors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from torsion.metrics.safety import min_actor_distance, min_ttc
from torsion.metrics.statistics import summarize_safety_group
from torsion.operators.object import ObjectSet, position_torsion
from torsion.operators.temporal import is_active

MethodName = Literal["clean", "torsion_displace", "random_warp"]
ScenarioName = Literal["leading_vehicle", "cut_in"]

CARLA_MAGNITUDES_M: dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}
METHODS: tuple[MethodName, ...] = ("clean", "torsion_displace", "random_warp")
MAGNITUDES: tuple[str, ...] = ("low", "medium", "high")
ROLE_PREFIX = "TORSION_CARLA_VALIDATION"
DEFAULT_RESULTS_CSV = Path("results/metrics/carla_runs.csv")
DEFAULT_SUMMARY_CSV = Path("results/metrics/carla_summary.csv")
DEFAULT_FIGURE = Path("results/figures/figure15_carla_validation.png")


@dataclass(frozen=True)
class CarlaEpisodeConfig:
    """Configuration for one real CARLA closed-loop episode."""

    scenario: ScenarioName = "leading_vehicle"
    method: MethodName = "clean"
    magnitude: str = "medium"
    seed: int = 0
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 60.0
    map_name: str = "current"
    fixed_delta_seconds: float = 0.05
    max_ticks: int = 240
    temporal_pattern: str = "burst"
    start_frame: int = 15
    duration_frames: int = 105
    target_speed_mps: float = 13.5
    lead_cruise_speed_mps: float = 9.5
    initial_gap_m: float = 28.0
    lane_gate_m: float = 1.15
    ttc_horizon_s: float = 5.0

    @property
    def run_id(self) -> str:
        return (
            f"carla_{self.scenario}_{self.method}_{self.magnitude}_"
            f"seed{self.seed}_{self.map_name}"
        )

    @property
    def displacement_budget_m(self) -> float:
        if self.method == "clean":
            return 0.0
        return _magnitude_to_meters(self.magnitude)


@dataclass(frozen=True)
class CarlaRunResult:
    """Trace and summary for one CARLA episode."""

    config: CarlaEpisodeConfig
    trace: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = {
            "run_id": self.config.run_id,
            "simulator": "carla",
            "simulator_version": "0.9.16",
            "scenario_id": self.config.scenario,
            "map_name": self.config.map_name,
            "seed": self.config.seed,
            "method": self.config.method,
            "magnitude": self.config.magnitude,
            "magnitude_m": _magnitude_to_meters(self.config.magnitude),
            "injection_point": "object_set" if self.config.method != "clean" else "none",
            "operator": self.config.method if self.config.method != "clean" else "none",
            "temporal_pattern": self.config.temporal_pattern,
            "start_frame": self.config.start_frame,
            "duration_frames": self.config.duration_frames,
            "fixed_delta_seconds": self.config.fixed_delta_seconds,
            "max_ticks": self.config.max_ticks,
            "lane_gate_m": self.config.lane_gate_m,
        }
        record.update(self.summary)
        return record


def _magnitude_to_meters(magnitude: str) -> float:
    key = str(magnitude).strip()
    if key in CARLA_MAGNITUDES_M:
        return float(CARLA_MAGNITUDES_M[key])
    try:
        value = float(key)
    except ValueError as exc:
        valid = ", ".join((*MAGNITUDES, "<meters>"))
        raise ValueError(f"unknown CARLA magnitude {magnitude!r}; expected {valid}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"CARLA magnitude must be a non-negative meter value, got {magnitude!r}")
    return value


@dataclass(frozen=True)
class LaneCandidate:
    """A straight lane segment selected from current-map generated waypoints."""

    waypoint: Any
    length_m: float
    max_yaw_change_deg: float
    road_id: int
    lane_id: int


@dataclass(frozen=True)
class CutInLaneCandidate:
    """A straight ego lane with a parallel adjacent lane for cut-in episodes."""

    ego_lane: LaneCandidate
    adjacent_waypoint: Any
    adjacent_length_m: float
    adjacent_max_yaw_change_deg: float
    adjacent_road_id: int
    adjacent_lane_id: int
    adjacent_side: str
    lateral_separation_m: float


@dataclass(frozen=True)
class EpisodeSpawn:
    """Spawn and scenario parameters sampled from a seed."""

    ego_transform: Any
    lead_transform: Any
    route_forward: np.ndarray
    lane_candidate: LaneCandidate
    initial_gap_m: float
    lead_brake_frame: int
    lead_brake_deceleration: float


@dataclass(frozen=True)
class CutInSpawn:
    """Spawn and merge-script parameters sampled from a seed."""

    ego_transform: Any
    cut_in_transform: Any
    route_forward: np.ndarray
    lane_candidate: LaneCandidate
    adjacent_start_waypoint: Any
    adjacent_side: str
    adjacent_road_id: int
    adjacent_lane_id: int
    adjacent_straight_length_m: float
    adjacent_max_yaw_change_deg: float
    adjacent_lateral_separation_m: float
    initial_gap_m: float
    merge_start_frame: int
    merge_duration_frames: int
    cut_in_speed_mps: float


@dataclass(frozen=True)
class ControlDecision:
    """Controller output plus interpretable intermediate values."""

    throttle: float
    brake: float
    steer: float
    perceived_gap_m: float
    perceived_lateral_m: float
    perceived_ttc_s: float
    desired_accel_mps2: float
    lead_considered: bool


def run_carla_sweep(
    configs: Iterable[CarlaEpisodeConfig],
    *,
    metrics_csv: Path = DEFAULT_RESULTS_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    figure_path: Path = DEFAULT_FIGURE,
) -> tuple[list[CarlaRunResult], list[dict[str, Any]]]:
    """Run a group of episodes in one synchronous CARLA session."""

    cfgs = list(configs)
    if not cfgs:
        raise ValueError("at least one CARLA episode config is required")

    carla = _import_carla()
    client = carla.Client(cfgs[0].host, int(cfgs[0].port))
    client.set_timeout(float(cfgs[0].timeout_s))

    world: Any | None = None
    results: list[CarlaRunResult] = []

    try:
        world = client.get_world()
        current_map_name = _current_map_label(world)
        cfgs = _configs_for_current_map(cfgs, current_map_name)
        _destroy_leftover_torsion_actors(world)
        _configure_synchronous_world(world, cfgs[0].fixed_delta_seconds)
        world.tick(seconds=cfgs[0].timeout_s)

        lane_candidates = _find_straight_lane_candidates(world.get_map())
        if not lane_candidates:
            raise RuntimeError(
                f"no straight lane segment long enough was found in current map {current_map_name}"
            )
        cut_in_candidates = _find_cut_in_lane_candidates(world.get_map(), lane_candidates)

        for cfg in cfgs:
            if abs(cfg.fixed_delta_seconds - cfgs[0].fixed_delta_seconds) > 1e-12:
                raise ValueError("run_carla_sweep expects one fixed_delta_seconds value")
            spawned: list[Any] = []
            try:
                result, spawned = _run_episode_in_world(
                    carla,
                    world,
                    cfg,
                    lane_candidates=lane_candidates,
                    cut_in_candidates=cut_in_candidates,
                )
                results.append(result)
            except Exception as exc:
                if _is_carla_connection_error(exc):
                    raise RuntimeError(f"CARLA connection dropped during {cfg.run_id}: {exc}") from exc
                results.append(_failed_run_result(cfg, exc))
            finally:
                _destroy_spawned(spawned)
            try:
                world.tick(seconds=cfg.timeout_s)
            except Exception as exc:
                if _is_carla_connection_error(exc):
                    raise RuntimeError(
                        f"CARLA connection dropped while settling after {cfg.run_id}: {exc}"
                    ) from exc
    finally:
        if world is not None:
            try:
                _destroy_leftover_torsion_actors(world)
            finally:
                _restore_async_world(world)

    records = [result.to_record() for result in results]
    write_records_csv(records, metrics_csv)
    summary = summarize_records(records)
    write_records_csv(summary, summary_csv)
    make_carla_validation_figure(summary, figure_path)
    return results, summary


def run_carla_episode(config: CarlaEpisodeConfig | Mapping[str, Any]) -> CarlaRunResult:
    """Run one CARLA episode, restoring async mode and destroying spawned actors."""

    cfg = config if isinstance(config, CarlaEpisodeConfig) else CarlaEpisodeConfig(**dict(config))
    results, _ = run_carla_sweep([cfg])
    return results[0]


def _current_map_label(world: Any) -> str:
    raw_name = str(getattr(world.get_map(), "name", "current"))
    return Path(raw_name.replace("\\", "/")).name or raw_name


def _configs_for_current_map(
    configs: list[CarlaEpisodeConfig],
    current_map_name: str,
) -> list[CarlaEpisodeConfig]:
    current = Path(str(current_map_name).replace("\\", "/")).name
    updated: list[CarlaEpisodeConfig] = []
    for cfg in configs:
        requested = Path(str(cfg.map_name).replace("\\", "/")).name
        if requested and requested != "current" and requested != current:
            raise ValueError(
                "CARLA is already running on current map "
                f"{current!r}; refusing to reload or switch to requested map {cfg.map_name!r}"
            )
        updated.append(replace(cfg, map_name=current))
    return updated


def _failed_run_result(cfg: CarlaEpisodeConfig, exc: Exception) -> CarlaRunResult:
    summary = {
        "connected": True,
        "episode_failed": True,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "collision": False,
        "collision_frame": None,
        "collision_time_s": float("nan"),
        "collision_event_count": 0,
        "min_ttc": float("nan"),
        "min_actor_distance": float("nan"),
        "min_center_distance": float("nan"),
        "brake_reaction_delay_s": float("nan"),
        "route_progress_m": float("nan"),
        "route_completion": float("nan"),
        "ticks_run": 0,
        "lead_brake_frame": None,
        "lead_brake_time_s": float("nan"),
        "response_event": "",
        "response_event_frame": None,
        "response_event_time_s": float("nan"),
        "fault_active_frames": 0,
        "target_realized_budget_m": cfg.displacement_budget_m,
        "mean_realized_budget": float("nan"),
        "max_realized_budget": float("nan"),
        "lane_road_id": None,
        "lane_id": None,
        "lane_straight_length_m": float("nan"),
        "lane_max_yaw_change_deg": float("nan"),
    }
    return CarlaRunResult(config=cfg, trace=(), summary=summary)


def _empty_summary_row(method: str, magnitude: str, *, attempted: int) -> dict[str, Any]:
    return {
        "method": method,
        "magnitude": magnitude,
        "magnitude_m": CARLA_MAGNITUDES_M[magnitude],
        "n_attempted": attempted,
        "n_failed": attempted,
        "n_runs": 0,
        "collision_rate": float("nan"),
        "collision_rate_ci_low": float("nan"),
        "collision_rate_ci_high": float("nan"),
        "mean_min_ttc": float("nan"),
        "mean_min_ttc_ci_low": float("nan"),
        "mean_min_ttc_ci_high": float("nan"),
        "std_min_ttc": float("nan"),
        "iqr_min_ttc": float("nan"),
        "worst5pct_min_ttc": float("nan"),
        "worst_case_min_ttc": float("nan"),
        "mean_realized_budget": float("nan"),
        "mean_min_actor_distance": float("nan"),
        "mean_route_progress_m": float("nan"),
        "mean_route_completion": float("nan"),
        "mean_brake_reaction_delay_s": float("nan"),
        "n_collisions": 0,
    }


def _record_failed(row: Mapping[str, Any]) -> bool:
    value = row.get("episode_failed", False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_carla_connection_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "connection refused",
        "connection reset",
        "connection dropped",
        "failed to connect",
        "unable to connect",
        "rpc",
        "time-out",
        "timeout",
        "timed out",
    )
    return any(marker in text for marker in markers)


def write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write dictionaries to a CSV with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-episode CARLA records by method and magnitude."""

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for magnitude in MAGNITUDES:
            attempted = [
                row
                for row in records
                if row.get("method") == method and row.get("magnitude") == magnitude
            ]
            if not attempted:
                continue
            group = [
                row
                for row in attempted
                if not _record_failed(row) and _is_finite_number(row.get("min_ttc"))
            ]
            if not group:
                rows.append(_empty_summary_row(method, magnitude, attempted=len(attempted)))
                continue
            min_ttc_values = [float(row["min_ttc"]) for row in group]
            collision_values = [1.0 if row["collision"] else 0.0 for row in group]
            budget_values = [float(row["mean_realized_budget"]) for row in group]
            safe_summary = summarize_safety_group(
                collision=collision_values,
                min_ttc=min_ttc_values,
                realized_budget=budget_values,
                n_resamples=2000,
                seed=17,
            )
            rows.append(
                {
                    "method": method,
                    "magnitude": magnitude,
                    "magnitude_m": CARLA_MAGNITUDES_M[magnitude],
                    "n_attempted": len(attempted),
                    "n_failed": len(attempted) - len(group),
                    **safe_summary,
                    "mean_min_actor_distance": float(
                        np.mean([float(row["min_actor_distance"]) for row in group])
                    ),
                    "mean_route_progress_m": float(
                        np.mean([float(row["route_progress_m"]) for row in group])
                    ),
                    "mean_route_completion": float(
                        np.mean([float(row["route_completion"]) for row in group])
                    ),
                    "mean_brake_reaction_delay_s": _nanmean(
                        [float(row["brake_reaction_delay_s"]) for row in group]
                    ),
                    "n_collisions": int(sum(collision_values)),
                }
            )
    return rows


def make_carla_validation_figure(summary: list[dict[str, Any]], path: Path) -> None:
    """Create Figure 15: CARLA collision rate and min-TTC by magnitude."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "clean": "#2b6cb0",
        "torsion_displace": "#c53030",
        "random_warp": "#2f855a",
    }
    labels = {
        "clean": "Clean",
        "torsion_displace": "Directed displacement",
        "random_warp": "Random direction",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.84, wspace=0.28)
    fig.suptitle("Figure 15. Real CARLA object-set semantic fault validation", fontsize=13)

    x = np.arange(len(MAGNITUDES), dtype=np.float64)
    offsets = {"clean": -0.08, "torsion_displace": 0.0, "random_warp": 0.08}

    for method in METHODS:
        rows = []
        for magnitude in MAGNITUDES:
            matching = [
                row
                for row in summary
                if row["method"] == method
                and row["magnitude"] == magnitude
                and int(row.get("n_runs", 0)) > 0
            ]
            if matching:
                rows.append(matching[0])
        if not rows:
            continue
        row_x = np.array([MAGNITUDES.index(str(row["magnitude"])) for row in rows], dtype=np.float64)
        collision = np.array([float(row["collision_rate"]) for row in rows], dtype=np.float64)
        collision_yerr = _asymmetric_yerr(rows, "collision_rate")
        min_ttc_mean = np.array([float(row["mean_min_ttc"]) for row in rows], dtype=np.float64)
        ttc_yerr = _asymmetric_yerr(rows, "mean_min_ttc")

        axes[0].errorbar(
            row_x + offsets[method],
            collision,
            yerr=collision_yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            elinewidth=1.1,
            color=colors[method],
            label=labels[method],
        )
        axes[1].errorbar(
            row_x + offsets[method],
            min_ttc_mean,
            yerr=ttc_yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            elinewidth=1.1,
            color=colors[method],
            label=labels[method],
        )

    axes[0].set_xlabel("Magnitude budget")
    axes[0].set_ylabel("Collision rate")
    axes[0].set_xticks(x, [f"{name}\n{CARLA_MAGNITUDES_M[name]:.1f} m" for name in MAGNITUDES])
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Magnitude budget")
    axes[1].set_ylabel("Mean min-TTC (s, 5 s horizon)")
    axes[1].set_xticks(x, [f"{name}\n{CARLA_MAGNITUDES_M[name]:.1f} m" for name in MAGNITUDES])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    fig.text(
        0.5,
        0.03,
        "Caption: real CARLA 0.9.16, Town10HD current-map closed loop. Ego consumes ground-truth "
        "object-set perception; faults perturb only the perceived lead-vehicle position.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_episode_in_world(
    carla: Any,
    world: Any,
    cfg: CarlaEpisodeConfig,
    *,
    lane_candidates: list[LaneCandidate],
    cut_in_candidates: list[CutInLaneCandidate] | None = None,
) -> tuple[CarlaRunResult, list[Any]]:
    if cfg.scenario == "leading_vehicle":
        return _run_leading_vehicle_episode_in_world(
            carla,
            world,
            cfg,
            lane_candidates=lane_candidates,
        )
    if cfg.scenario == "cut_in":
        return _run_cut_in_episode_in_world(
            carla,
            world,
            cfg,
            cut_in_candidates=cut_in_candidates or [],
        )
    raise ValueError(f"unsupported CARLA scenario {cfg.scenario!r}")


def _run_leading_vehicle_episode_in_world(
    carla: Any,
    world: Any,
    cfg: CarlaEpisodeConfig,
    *,
    lane_candidates: list[LaneCandidate],
) -> tuple[CarlaRunResult, list[Any]]:

    spawned: list[Any] = []
    collision_events: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed("episode", cfg.seed, cfg.scenario))

    try:
        spawn: EpisodeSpawn | None = None
        ego = None
        lead = None
        spawn_errors: list[str] = []
        for _ in range(min(25, len(lane_candidates))):
            candidate_spawn = _sample_episode_spawn(carla, lane_candidates, cfg, rng)
            try:
                ego, lead = _spawn_ego_and_lead(carla, world, cfg, candidate_spawn)
                spawn = candidate_spawn
                spawned.extend([ego, lead])
                break
            except Exception as exc:
                spawn_errors.append(str(exc))
                _destroy_spawned([ego, lead])
                ego = None
                lead = None
        if spawn is None or ego is None or lead is None:
            detail = "; ".join(spawn_errors[-3:]) if spawn_errors else "no spawn attempts made"
            raise RuntimeError(f"failed to spawn CARLA ego/lead pair after retries: {detail}")

        collision_sensor = _attach_collision_sensor(carla, world, ego, cfg, collision_events)
        spawned.append(collision_sensor)
        world.tick(seconds=cfg.timeout_s)

        _set_initial_velocity(ego, spawn.route_forward, cfg.lead_cruise_speed_mps)
        _set_initial_velocity(lead, spawn.route_forward, cfg.lead_cruise_speed_mps)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        world.tick(seconds=cfg.timeout_s)

        first_collision_frame: int | None = None
        first_ego_brake_frame: int | None = None
        active_shifts: list[float] = []
        first_ego_location = ego.get_location()

        random_angle = float(
            np.random.default_rng(_stable_seed("random_warp", cfg.seed, cfg.magnitude)).uniform(
                -math.pi,
                math.pi,
            )
        )

        for frame_idx in range(cfg.max_ticks):
            time_s = frame_idx * cfg.fixed_delta_seconds
            lead_braking = frame_idx >= spawn.lead_brake_frame
            lead_control = _lead_control(carla, lead, world.get_map(), cfg, lead_braking)

            clean_objects = _lead_object_set(lead)
            perceived_objects, shift_m = _apply_perception_fault(
                clean_objects,
                cfg=cfg,
                frame_idx=frame_idx,
                ego=ego,
                random_angle=random_angle,
            )
            if _fault_active(cfg, frame_idx):
                active_shifts.append(shift_m)

            decision = _ego_controller(
                carla,
                ego,
                perceived_objects,
                world.get_map(),
                cfg,
            )
            if lead_braking and first_ego_brake_frame is None and decision.brake > 0.05:
                first_ego_brake_frame = frame_idx

            lead.apply_control(lead_control)
            ego.apply_control(
                carla.VehicleControl(
                    throttle=decision.throttle,
                    brake=decision.brake,
                    steer=decision.steer,
                )
            )
            world.tick(seconds=cfg.timeout_s)

            if collision_events and first_collision_frame is None:
                first_collision_frame = frame_idx

            true_objects = _lead_object_set(lead)
            ego_state = _actor_state_record(ego)
            lead_state = _actor_state_record(lead)
            ego_velocity = ego.get_velocity()
            ttc_raw = min_ttc(
                (ego_state["x"], ego_state["y"]),
                (ego_velocity.x, ego_velocity.y),
                true_objects,
                ego_width=_actor_width(ego),
                ego_length=_actor_length(ego),
                horizon_s=cfg.ttc_horizon_s,
            )
            ttc_value = cfg.ttc_horizon_s if not math.isfinite(ttc_raw) else min(ttc_raw, cfg.ttc_horizon_s)
            distance = min_actor_distance(
                (ego_state["x"], ego_state["y"]),
                true_objects,
                clearance=True,
                ego_width=_actor_width(ego),
                ego_length=_actor_length(ego),
            )
            center_distance = min_actor_distance(
                (ego_state["x"], ego_state["y"]),
                true_objects,
                clearance=False,
            )
            route_progress_m = _route_progress_m(first_ego_location, ego.get_location(), spawn.route_forward)

            trace.append(
                {
                    "frame": frame_idx,
                    "time_s": float(time_s),
                    "fault_active": _fault_active(cfg, frame_idx),
                    "applied_position_shift_m": float(shift_m),
                    "lead_braking": lead_braking,
                    "lead_brake_frame": spawn.lead_brake_frame,
                    "ego": ego_state,
                    "lead": lead_state,
                    "control": asdict(decision),
                    "lead_control": {
                        "throttle": float(lead_control.throttle),
                        "brake": float(lead_control.brake),
                        "steer": float(lead_control.steer),
                    },
                    "true_ttc_s": float(ttc_value),
                    "true_ttc_raw_s": float(ttc_raw) if math.isfinite(ttc_raw) else float("inf"),
                    "min_actor_distance_m": float(distance),
                    "center_distance_m": float(center_distance),
                    "collision": bool(collision_events),
                    "route_progress_m": float(route_progress_m),
                }
            )

            if first_collision_frame is not None and frame_idx >= first_collision_frame + 8:
                break

        summary = _summarize_episode(
            cfg,
            trace=trace,
            collision_events=collision_events,
            first_collision_frame=first_collision_frame,
            first_ego_brake_frame=first_ego_brake_frame,
            response_event_frame=spawn.lead_brake_frame,
            response_event_name="lead_brake",
            lead_brake_frame=spawn.lead_brake_frame,
            active_shifts=active_shifts,
            lane_candidate=spawn.lane_candidate,
        )
        return CarlaRunResult(config=cfg, trace=tuple(trace), summary=summary), spawned
    except Exception:
        _destroy_spawned(spawned)
        raise


def _run_cut_in_episode_in_world(
    carla: Any,
    world: Any,
    cfg: CarlaEpisodeConfig,
    *,
    cut_in_candidates: list[CutInLaneCandidate],
) -> tuple[CarlaRunResult, list[Any]]:
    if not cut_in_candidates:
        raise RuntimeError("no straight adjacent-lane segment was found for cut_in on the current map")

    spawned: list[Any] = []
    collision_events: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed("episode", cfg.seed, cfg.scenario))

    try:
        spawn: CutInSpawn | None = None
        ego = None
        cut_in = None
        spawn_errors: list[str] = []
        for _ in range(min(25, len(cut_in_candidates))):
            candidate_spawn = _sample_cut_in_spawn(carla, cut_in_candidates, cfg, rng)
            try:
                ego, cut_in = _spawn_ego_and_cut_in(carla, world, cfg, candidate_spawn)
                spawn = candidate_spawn
                spawned.extend([ego, cut_in])
                break
            except Exception as exc:
                spawn_errors.append(str(exc))
                _destroy_spawned([ego, cut_in])
                ego = None
                cut_in = None
        if spawn is None or ego is None or cut_in is None:
            detail = "; ".join(spawn_errors[-3:]) if spawn_errors else "no spawn attempts made"
            raise RuntimeError(f"failed to spawn CARLA ego/cut-in pair after retries: {detail}")

        collision_sensor = _attach_collision_sensor(carla, world, ego, cfg, collision_events)
        spawned.append(collision_sensor)
        world.tick(seconds=cfg.timeout_s)

        _set_initial_velocity(ego, spawn.route_forward, cfg.lead_cruise_speed_mps)
        _set_initial_velocity(cut_in, spawn.route_forward, spawn.cut_in_speed_mps)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        cut_in.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        world.tick(seconds=cfg.timeout_s)

        first_collision_frame: int | None = None
        first_ego_brake_frame: int | None = None
        active_shifts: list[float] = []
        first_ego_location = ego.get_location()

        random_angle = float(
            np.random.default_rng(_stable_seed("random_warp", cfg.seed, cfg.magnitude)).uniform(
                -math.pi,
                math.pi,
            )
        )

        for frame_idx in range(cfg.max_ticks):
            time_s = frame_idx * cfg.fixed_delta_seconds
            cut_in_transform, cut_in_velocity, merge_alpha = _cut_in_script_state(
                carla,
                spawn,
                cfg,
                frame_idx,
            )
            cut_in.set_transform(cut_in_transform)
            cut_in.set_target_velocity(cut_in_velocity)

            clean_objects = _lead_object_set(cut_in)
            perceived_objects, shift_m = _apply_perception_fault(
                clean_objects,
                cfg=cfg,
                frame_idx=frame_idx,
                ego=ego,
                random_angle=random_angle,
            )
            if _fault_active(cfg, frame_idx):
                active_shifts.append(shift_m)

            decision = _ego_controller(
                carla,
                ego,
                perceived_objects,
                world.get_map(),
                cfg,
            )
            if (
                frame_idx >= spawn.merge_start_frame
                and first_ego_brake_frame is None
                and decision.brake > 0.05
            ):
                first_ego_brake_frame = frame_idx

            cut_in.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
            ego.apply_control(
                carla.VehicleControl(
                    throttle=decision.throttle,
                    brake=decision.brake,
                    steer=decision.steer,
                )
            )
            world.tick(seconds=cfg.timeout_s)

            if collision_events and first_collision_frame is None:
                first_collision_frame = frame_idx

            true_objects = _lead_object_set(cut_in)
            ego_state = _actor_state_record(ego)
            cut_in_state = _actor_state_record(cut_in)
            ego_velocity = ego.get_velocity()
            ttc_raw = min_ttc(
                (ego_state["x"], ego_state["y"]),
                (ego_velocity.x, ego_velocity.y),
                true_objects,
                ego_width=_actor_width(ego),
                ego_length=_actor_length(ego),
                horizon_s=cfg.ttc_horizon_s,
            )
            ttc_value = cfg.ttc_horizon_s if not math.isfinite(ttc_raw) else min(ttc_raw, cfg.ttc_horizon_s)
            distance = min_actor_distance(
                (ego_state["x"], ego_state["y"]),
                true_objects,
                clearance=True,
                ego_width=_actor_width(ego),
                ego_length=_actor_length(ego),
            )
            center_distance = min_actor_distance(
                (ego_state["x"], ego_state["y"]),
                true_objects,
                clearance=False,
            )
            route_progress_m = _route_progress_m(first_ego_location, ego.get_location(), spawn.route_forward)

            trace.append(
                {
                    "frame": frame_idx,
                    "time_s": float(time_s),
                    "fault_active": _fault_active(cfg, frame_idx),
                    "applied_position_shift_m": float(shift_m),
                    "lead_braking": False,
                    "lead_brake_frame": None,
                    "cut_in_merge_start_frame": spawn.merge_start_frame,
                    "cut_in_merge_duration_frames": spawn.merge_duration_frames,
                    "cut_in_merge_alpha": float(merge_alpha),
                    "ego": ego_state,
                    "cut_in": cut_in_state,
                    "control": asdict(decision),
                    "cut_in_control": {
                        "scripted_speed_mps": float(spawn.cut_in_speed_mps),
                        "vx": float(cut_in_velocity.x),
                        "vy": float(cut_in_velocity.y),
                    },
                    "true_ttc_s": float(ttc_value),
                    "true_ttc_raw_s": float(ttc_raw) if math.isfinite(ttc_raw) else float("inf"),
                    "min_actor_distance_m": float(distance),
                    "center_distance_m": float(center_distance),
                    "collision": bool(collision_events),
                    "route_progress_m": float(route_progress_m),
                }
            )

            if first_collision_frame is not None and frame_idx >= first_collision_frame + 8:
                break

        summary = _summarize_episode(
            cfg,
            trace=trace,
            collision_events=collision_events,
            first_collision_frame=first_collision_frame,
            first_ego_brake_frame=first_ego_brake_frame,
            response_event_frame=spawn.merge_start_frame,
            response_event_name="cut_in_merge_start",
            lead_brake_frame=None,
            active_shifts=active_shifts,
            lane_candidate=spawn.lane_candidate,
            extra_summary={
                "cut_in_adjacent_side": spawn.adjacent_side,
                "cut_in_adjacent_road_id": spawn.adjacent_road_id,
                "cut_in_adjacent_lane_id": spawn.adjacent_lane_id,
                "cut_in_adjacent_straight_length_m": spawn.adjacent_straight_length_m,
                "cut_in_adjacent_max_yaw_change_deg": spawn.adjacent_max_yaw_change_deg,
                "cut_in_lateral_separation_m": spawn.adjacent_lateral_separation_m,
                "cut_in_merge_start_frame": spawn.merge_start_frame,
                "cut_in_merge_duration_frames": spawn.merge_duration_frames,
                "cut_in_speed_mps": spawn.cut_in_speed_mps,
            },
        )
        return CarlaRunResult(config=cfg, trace=tuple(trace), summary=summary), spawned
    except Exception:
        _destroy_spawned(spawned)
        raise


def _summarize_episode(
    cfg: CarlaEpisodeConfig,
    *,
    trace: list[dict[str, Any]],
    collision_events: list[dict[str, Any]],
    first_collision_frame: int | None,
    first_ego_brake_frame: int | None,
    response_event_frame: int,
    response_event_name: str,
    lead_brake_frame: int | None,
    active_shifts: list[float],
    lane_candidate: LaneCandidate,
    extra_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not trace:
        raise RuntimeError("CARLA episode produced no trace rows")

    collision = bool(collision_events)
    collision_frame = first_collision_frame
    if collision and collision_frame is None:
        collision_frame = int(trace[-1]["frame"])
    brake_delay = (
        float("nan")
        if first_ego_brake_frame is None
        else (first_ego_brake_frame - response_event_frame) * cfg.fixed_delta_seconds
    )
    route_progress = float(trace[-1]["route_progress_m"])
    nominal_route_m = float(cfg.target_speed_mps * cfg.max_ticks * cfg.fixed_delta_seconds)
    summary = {
        "connected": True,
        "episode_failed": False,
        "error_type": "",
        "error": "",
        "collision": collision,
        "collision_frame": collision_frame,
        "collision_time_s": (
            float(collision_frame * cfg.fixed_delta_seconds)
            if collision_frame is not None
            else float("nan")
        ),
        "collision_event_count": len(collision_events),
        "min_ttc": float(min(float(row["true_ttc_s"]) for row in trace)),
        "min_actor_distance": float(min(float(row["min_actor_distance_m"]) for row in trace)),
        "min_center_distance": float(min(float(row["center_distance_m"]) for row in trace)),
        "brake_reaction_delay_s": brake_delay,
        "route_progress_m": route_progress,
        "route_completion": float(np.clip(route_progress / max(nominal_route_m, 1e-9), 0.0, 1.0)),
        "ticks_run": len(trace),
        "lead_brake_frame": lead_brake_frame,
        "lead_brake_time_s": (
            float(lead_brake_frame * cfg.fixed_delta_seconds)
            if lead_brake_frame is not None
            else float("nan")
        ),
        "response_event": response_event_name,
        "response_event_frame": response_event_frame,
        "response_event_time_s": float(response_event_frame * cfg.fixed_delta_seconds),
        "fault_active_frames": int(sum(bool(row["fault_active"]) for row in trace)),
        "target_realized_budget_m": cfg.displacement_budget_m,
        "mean_realized_budget": float(np.mean(active_shifts)) if active_shifts else 0.0,
        "max_realized_budget": float(max(active_shifts)) if active_shifts else 0.0,
        "lane_road_id": lane_candidate.road_id,
        "lane_id": lane_candidate.lane_id,
        "lane_straight_length_m": lane_candidate.length_m,
        "lane_max_yaw_change_deg": lane_candidate.max_yaw_change_deg,
    }
    if extra_summary:
        summary.update(extra_summary)
    return summary


def _apply_perception_fault(
    clean_objects: ObjectSet,
    *,
    cfg: CarlaEpisodeConfig,
    frame_idx: int,
    ego: Any,
    random_angle: float,
) -> tuple[ObjectSet, float]:
    if cfg.method == "clean" or not _fault_active(cfg, frame_idx):
        return clean_objects, 0.0

    shift_m = cfg.displacement_budget_m
    ego_transform = ego.get_transform()
    forward = _vector2_from_carla(ego_transform.get_forward_vector())
    right = _vector2_from_carla(ego_transform.get_right_vector())
    forward = _unit2(forward)
    right = _unit2(right)

    if cfg.method == "torsion_displace":
        direction = _unit2(0.80 * forward + 0.60 * right)
    elif cfg.method == "random_warp":
        direction = _unit2(math.cos(random_angle) * forward + math.sin(random_angle) * right)
    else:
        raise ValueError(f"unknown CARLA method {cfg.method!r}")
    delta = shift_m * direction

    out = position_torsion(
        clean_objects,
        dx=float(delta[0]),
        dy=float(delta[1]),
        track_ids=[clean_objects.track_id[0]],
        max_shift_m=max(2.0, shift_m + 1e-4),
    )
    realized = float(
        np.linalg.norm([out.x[0] - clean_objects.x[0], out.y[0] - clean_objects.y[0]])
    )
    return out, realized


def _fault_active(cfg: CarlaEpisodeConfig, frame_idx: int) -> bool:
    if cfg.method == "clean":
        return False
    return is_active(
        cfg.temporal_pattern,
        frame_idx,
        start_frame=cfg.start_frame,
        duration=cfg.duration_frames,
    )


def _ego_controller(
    carla: Any,
    ego: Any,
    perceived_objects: ObjectSet,
    carla_map: Any,
    cfg: CarlaEpisodeConfig,
) -> ControlDecision:
    ego_transform = ego.get_transform()
    ego_loc = ego_transform.location
    ego_velocity = ego.get_velocity()
    ego_speed = _speed_mps(ego)
    forward = _unit2(_vector2_from_carla(ego_transform.get_forward_vector()))
    right = _unit2(_vector2_from_carla(ego_transform.get_right_vector()))
    rel = np.array(
        [float(perceived_objects.x[0] - ego_loc.x), float(perceived_objects.y[0] - ego_loc.y)],
        dtype=np.float64,
    )
    perceived_forward = float(rel @ forward)
    perceived_lateral = float(rel @ right)
    gap = perceived_forward - 0.5 * _actor_length(ego) - 0.5 * float(perceived_objects.l[0])
    gap = max(gap, 0.1)
    lead_velocity = perceived_objects.v[0]
    ego_velocity_xy = np.array([ego_velocity.x, ego_velocity.y], dtype=np.float64)
    closing_speed = float((ego_velocity_xy - lead_velocity) @ forward)
    perceived_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
    lead_considered = perceived_forward > 0.0 and abs(perceived_lateral) <= cfg.lane_gate_m

    desired_accel = 1.25 * (cfg.target_speed_mps - ego_speed)
    if lead_considered:
        min_gap = 4.0
        time_headway = 1.15
        max_accel = 2.0
        comfortable_brake = 4.5
        dynamic_gap = min_gap + ego_speed * time_headway
        if closing_speed > 0.0:
            dynamic_gap += ego_speed * closing_speed / (2.0 * math.sqrt(max_accel * comfortable_brake))
        desired_accel = max_accel * (
            1.0
            - (ego_speed / max(cfg.target_speed_mps, 0.1)) ** 4
            - (dynamic_gap / gap) ** 2
        )
        if perceived_ttc < 1.2:
            desired_accel = min(desired_accel, -8.0)
        elif perceived_ttc < 2.0:
            desired_accel = min(desired_accel, -5.5)

    steer = _lane_keep_steer(ego, carla_map)
    throttle, brake = _accel_to_vehicle_control(desired_accel)
    return ControlDecision(
        throttle=throttle,
        brake=brake,
        steer=steer,
        perceived_gap_m=float(gap),
        perceived_lateral_m=float(perceived_lateral),
        perceived_ttc_s=float(perceived_ttc) if math.isfinite(perceived_ttc) else cfg.ttc_horizon_s,
        desired_accel_mps2=float(desired_accel),
        lead_considered=lead_considered,
    )


def _lead_control(
    carla: Any,
    lead: Any,
    carla_map: Any,
    cfg: CarlaEpisodeConfig,
    braking: bool,
) -> Any:
    steer = _lane_keep_steer(lead, carla_map)
    if braking:
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer)

    speed_error = cfg.lead_cruise_speed_mps - _speed_mps(lead)
    desired_accel = 1.1 * speed_error
    throttle, brake = _accel_to_vehicle_control(desired_accel, max_throttle=0.55)
    return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)


def _cut_in_script_state(
    carla: Any,
    spawn: CutInSpawn,
    cfg: CarlaEpisodeConfig,
    frame_idx: int,
) -> tuple[Any, Any, float]:
    time_s = frame_idx * cfg.fixed_delta_seconds
    merge_start_s = spawn.merge_start_frame * cfg.fixed_delta_seconds
    merge_duration_s = max(spawn.merge_duration_frames * cfg.fixed_delta_seconds, 1e-6)
    merge_u = float(np.clip((time_s - merge_start_s) / merge_duration_s, 0.0, 1.0))
    alpha = _smoothstep(merge_u)
    alpha_dot = 0.0
    if 0.0 < merge_u < 1.0:
        alpha_dot = (6.0 * merge_u * (1.0 - merge_u)) / merge_duration_s

    distance_m = spawn.initial_gap_m + spawn.cut_in_speed_mps * time_s
    adjacent_transform = _lane_transform_at_distance(carla, spawn.adjacent_start_waypoint, distance_m)
    ego_transform = _lane_transform_at_distance(carla, spawn.lane_candidate.waypoint, distance_m)
    adjacent_xy = np.array(
        [float(adjacent_transform.location.x), float(adjacent_transform.location.y)],
        dtype=np.float64,
    )
    ego_xy = np.array(
        [float(ego_transform.location.x), float(ego_transform.location.y)],
        dtype=np.float64,
    )
    lateral = ego_xy - adjacent_xy
    xy = adjacent_xy + alpha * lateral
    z = (1.0 - alpha) * float(adjacent_transform.location.z) + alpha * float(ego_transform.location.z)

    forward = _unit2(_vector2_from_carla(ego_transform.get_forward_vector()))
    velocity_xy = spawn.cut_in_speed_mps * forward + alpha_dot * lateral
    yaw = math.degrees(math.atan2(float(velocity_xy[1]), float(velocity_xy[0])))
    transform = carla.Transform(
        carla.Location(x=float(xy[0]), y=float(xy[1]), z=float(z)),
        carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0),
    )
    velocity = carla.Vector3D(x=float(velocity_xy[0]), y=float(velocity_xy[1]), z=0.0)
    return transform, velocity, alpha


def _lane_keep_steer(actor: Any, carla_map: Any) -> float:
    transform = actor.get_transform()
    location = transform.location
    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=_driving_lane_type(),
    )
    if waypoint is None:
        return 0.0
    next_waypoints = waypoint.next(8.0)
    target = _choose_waypoint(next_waypoints, waypoint)
    target_yaw = target.transform.rotation.yaw if target is not None else waypoint.transform.rotation.yaw
    yaw_error = math.radians(_angle_diff_deg(target_yaw, transform.rotation.yaw))
    right = waypoint.transform.get_right_vector()
    lateral = (
        (location.x - waypoint.transform.location.x) * right.x
        + (location.y - waypoint.transform.location.y) * right.y
    )
    steer = 0.85 * yaw_error - 0.12 * lateral
    return float(np.clip(steer, -0.35, 0.35))


def _accel_to_vehicle_control(
    accel_mps2: float,
    *,
    max_throttle: float = 0.75,
) -> tuple[float, float]:
    if accel_mps2 >= 0.0:
        return float(np.clip(accel_mps2 / 2.8, 0.0, max_throttle)), 0.0
    return 0.0, float(np.clip(-accel_mps2 / 7.0, 0.0, 1.0))


def _sample_episode_spawn(
    carla: Any,
    lane_candidates: list[LaneCandidate],
    cfg: CarlaEpisodeConfig,
    rng: np.random.Generator,
) -> EpisodeSpawn:
    candidate = lane_candidates[int(rng.integers(0, len(lane_candidates)))]
    gap = float(np.clip(cfg.initial_gap_m + rng.normal(0.0, 2.5), 23.0, 33.0))
    lead_wp = _waypoint_ahead(candidate.waypoint, gap)
    if lead_wp is None:
        raise RuntimeError("sampled lane candidate cannot place lead vehicle ahead")

    route_forward = _unit2(_vector2_from_carla(candidate.waypoint.transform.get_forward_vector()))
    brake_frame = int(np.clip(52 + rng.integers(-8, 9), 42, 66))
    decel = float(np.clip(7.0 + rng.normal(0.0, 0.35), 6.0, 8.0))
    return EpisodeSpawn(
        ego_transform=_spawn_transform(carla, candidate.waypoint),
        lead_transform=_spawn_transform(carla, lead_wp),
        route_forward=route_forward,
        lane_candidate=candidate,
        initial_gap_m=gap,
        lead_brake_frame=brake_frame,
        lead_brake_deceleration=decel,
    )


def _sample_cut_in_spawn(
    carla: Any,
    cut_in_candidates: list[CutInLaneCandidate],
    cfg: CarlaEpisodeConfig,
    rng: np.random.Generator,
) -> CutInSpawn:
    candidate = cut_in_candidates[int(rng.integers(0, len(cut_in_candidates)))]
    gap = float(np.clip(18.0 + rng.normal(0.0, 1.75), 14.5, 22.0))
    cut_in_wp = _waypoint_ahead(candidate.adjacent_waypoint, gap)
    if cut_in_wp is None:
        raise RuntimeError("sampled cut-in candidate cannot place vehicle ahead")

    route_forward = _unit2(_vector2_from_carla(candidate.ego_lane.waypoint.transform.get_forward_vector()))
    merge_start_frame = int(np.clip(22 + rng.integers(-5, 6), 15, 32))
    merge_duration_frames = int(np.clip(68 + rng.integers(-10, 11), 54, 84))
    cut_in_speed = float(np.clip(cfg.lead_cruise_speed_mps + rng.normal(-0.8, 0.35), 8.0, 10.0))
    return CutInSpawn(
        ego_transform=_spawn_transform(carla, candidate.ego_lane.waypoint),
        cut_in_transform=_spawn_transform(carla, cut_in_wp),
        route_forward=route_forward,
        lane_candidate=candidate.ego_lane,
        adjacent_start_waypoint=candidate.adjacent_waypoint,
        adjacent_side=candidate.adjacent_side,
        adjacent_road_id=candidate.adjacent_road_id,
        adjacent_lane_id=candidate.adjacent_lane_id,
        adjacent_straight_length_m=candidate.adjacent_length_m,
        adjacent_max_yaw_change_deg=candidate.adjacent_max_yaw_change_deg,
        adjacent_lateral_separation_m=candidate.lateral_separation_m,
        initial_gap_m=gap,
        merge_start_frame=merge_start_frame,
        merge_duration_frames=merge_duration_frames,
        cut_in_speed_mps=cut_in_speed,
    )


def _spawn_ego_and_lead(
    carla: Any,
    world: Any,
    cfg: CarlaEpisodeConfig,
    spawn: EpisodeSpawn,
) -> tuple[Any, Any]:
    blueprints = world.get_blueprint_library()
    ego_bp = _vehicle_blueprint(blueprints, "vehicle.tesla.model3")
    lead_bp = _vehicle_blueprint(blueprints, "vehicle.lincoln.mkz_2020")
    _set_role_name(ego_bp, f"{ROLE_PREFIX}_ego_{cfg.seed}_{cfg.method}_{cfg.magnitude}")
    _set_role_name(lead_bp, f"{ROLE_PREFIX}_lead_{cfg.seed}_{cfg.method}_{cfg.magnitude}")
    _set_color_if_available(ego_bp, "30,90,180")
    _set_color_if_available(lead_bp, "210,70,50")

    ego = world.try_spawn_actor(ego_bp, spawn.ego_transform)
    if ego is None:
        raise RuntimeError("failed to spawn CARLA ego vehicle")
    lead = world.try_spawn_actor(lead_bp, spawn.lead_transform)
    if lead is None:
        ego.destroy()
        raise RuntimeError("failed to spawn CARLA lead vehicle")

    ego.set_simulate_physics(True)
    lead.set_simulate_physics(True)
    return ego, lead


def _spawn_ego_and_cut_in(
    carla: Any,
    world: Any,
    cfg: CarlaEpisodeConfig,
    spawn: CutInSpawn,
) -> tuple[Any, Any]:
    blueprints = world.get_blueprint_library()
    ego_bp = _vehicle_blueprint(blueprints, "vehicle.tesla.model3")
    cut_in_bp = _vehicle_blueprint(blueprints, "vehicle.lincoln.mkz_2020")
    _set_role_name(ego_bp, f"{ROLE_PREFIX}_ego_{cfg.seed}_{cfg.method}_{cfg.magnitude}")
    _set_role_name(cut_in_bp, f"{ROLE_PREFIX}_cut_in_{cfg.seed}_{cfg.method}_{cfg.magnitude}")
    _set_color_if_available(ego_bp, "30,90,180")
    _set_color_if_available(cut_in_bp, "230,150,35")

    ego = world.try_spawn_actor(ego_bp, spawn.ego_transform)
    if ego is None:
        raise RuntimeError("failed to spawn CARLA ego vehicle")
    cut_in = world.try_spawn_actor(cut_in_bp, spawn.cut_in_transform)
    if cut_in is None:
        ego.destroy()
        raise RuntimeError("failed to spawn CARLA cut-in vehicle")

    ego.set_simulate_physics(True)
    cut_in.set_simulate_physics(True)
    return ego, cut_in


def _attach_collision_sensor(
    carla: Any,
    world: Any,
    ego: Any,
    cfg: CarlaEpisodeConfig,
    collision_events: list[dict[str, Any]],
) -> Any:
    sensor_bp = world.get_blueprint_library().find("sensor.other.collision")
    _set_role_name(sensor_bp, f"{ROLE_PREFIX}_collision_{cfg.seed}_{cfg.method}_{cfg.magnitude}")
    sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=ego)

    def _on_collision(event: Any) -> None:
        collision_events.append(
            {
                "frame": int(event.frame),
                "other_actor_id": int(event.other_actor.id),
                "other_actor_type": str(event.other_actor.type_id),
            }
        )

    sensor.listen(_on_collision)
    return sensor


def _lead_object_set(lead: Any) -> ObjectSet:
    transform = lead.get_transform()
    velocity = lead.get_velocity()
    bbox = lead.bounding_box
    return ObjectSet(
        x=[float(transform.location.x)],
        y=[float(transform.location.y)],
        z=[float(transform.location.z)],
        w=[float(2.0 * bbox.extent.y)],
        h=[float(2.0 * bbox.extent.z)],
        l=[float(2.0 * bbox.extent.x)],
        yaw=[float(math.radians(transform.rotation.yaw))],
        v=[[float(velocity.x), float(velocity.y)]],
        cls=["vehicle"],
        conf=[1.0],
        track_id=[int(lead.id)],
    )


def _actor_state_record(actor: Any) -> dict[str, float]:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    return {
        "id": int(actor.id),
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw_deg": float(transform.rotation.yaw),
        "vx": float(velocity.x),
        "vy": float(velocity.y),
        "vz": float(velocity.z),
        "speed_mps": _speed_mps(actor),
        "ax": float(acceleration.x),
        "ay": float(acceleration.y),
    }


def _find_straight_lane_candidates(
    carla_map: Any,
    *,
    min_length_m: float = 60.0,
    sample_distance_m: float = 2.0,
    yaw_tolerance_deg: float = 4.0,
) -> list[LaneCandidate]:
    """Find current-map lane starts that stay nearly straight for the scenario.

    CARLA's current map is sampled with ``generate_waypoints(2.0)``.  Each
    non-junction driving waypoint is walked forward along CARLA's lane graph in
    2 m steps.  A candidate is accepted only if every step remains a driving
    lane, stays out of junctions, and keeps heading within ``yaw_tolerance_deg``
    of the start heading for at least ``min_length_m``.
    """

    candidates: list[LaneCandidate] = []
    start_waypoints = carla_map.generate_waypoints(sample_distance_m)

    seen: set[tuple[int, int, int]] = set()
    for waypoint in start_waypoints:
        if waypoint is None or waypoint.is_junction or not _is_driving_waypoint(waypoint):
            continue
        key = (
            int(waypoint.road_id),
            int(waypoint.lane_id),
            int(round(float(getattr(waypoint, "s", 0.0)) / sample_distance_m)),
        )
        if key in seen:
            continue
        seen.add(key)
        walked = _walk_straight_lane(
            waypoint,
            min_length_m=min_length_m,
            step_m=sample_distance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
        )
        if walked is not None:
            length_m, max_yaw_change = walked
            candidates.append(
                LaneCandidate(
                    waypoint=waypoint,
                    length_m=length_m,
                    max_yaw_change_deg=max_yaw_change,
                    road_id=int(waypoint.road_id),
                    lane_id=int(waypoint.lane_id),
                )
            )

    candidates.sort(
        key=lambda item: (
            item.max_yaw_change_deg,
            -item.length_m,
            item.road_id,
            item.lane_id,
            float(item.waypoint.transform.location.x),
            float(item.waypoint.transform.location.y),
        )
    )
    return candidates[:160]


def _find_cut_in_lane_candidates(
    carla_map: Any,
    lane_candidates: list[LaneCandidate],
    *,
    min_length_m: float = 80.0,
    sample_distance_m: float = 2.0,
    yaw_tolerance_deg: float = 4.0,
    parallel_tolerance_deg: float = 8.0,
) -> list[CutInLaneCandidate]:
    """Find straight adjacent-lane starts for the cut-in scenario.

    The base lane is drawn from the current map's straight-lane candidates. For
    each base waypoint we query CARLA's immediate ``get_left_lane`` and
    ``get_right_lane`` neighbors.  A neighbor is accepted only when it is a
    non-junction driving lane on the same road, has nearly the same heading,
    starts at a similar longitudinal station, is separated by a plausible lane
    width, and can be walked forward as a straight lane for ``min_length_m``.
    This keeps the merge on a documented current-map multi-lane stretch without
    switching or reloading the map.
    """

    candidates: list[CutInLaneCandidate] = []
    seen: set[tuple[int, int, int, str]] = set()
    for base in lane_candidates:
        ego_walked = _walk_straight_lane(
            base.waypoint,
            min_length_m=min_length_m,
            step_m=sample_distance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
        )
        if ego_walked is None:
            continue
        ego_length_m, ego_max_yaw = ego_walked
        ego_lane = replace(
            base,
            length_m=ego_length_m,
            max_yaw_change_deg=max(base.max_yaw_change_deg, ego_max_yaw),
        )

        for side, getter_name in (("left", "get_left_lane"), ("right", "get_right_lane")):
            try:
                adjacent = getattr(base.waypoint, getter_name)()
            except Exception:
                adjacent = None
            if adjacent is None or adjacent.is_junction or not _is_driving_waypoint(adjacent):
                continue
            if int(adjacent.road_id) != int(base.waypoint.road_id):
                continue

            yaw_diff = abs(
                _angle_diff_deg(
                    adjacent.transform.rotation.yaw,
                    base.waypoint.transform.rotation.yaw,
                )
            )
            if yaw_diff > parallel_tolerance_deg:
                continue

            forward = _unit2(_vector2_from_carla(base.waypoint.transform.get_forward_vector()))
            rel = np.array(
                [
                    float(adjacent.transform.location.x - base.waypoint.transform.location.x),
                    float(adjacent.transform.location.y - base.waypoint.transform.location.y),
                ],
                dtype=np.float64,
            )
            longitudinal_offset = abs(float(rel @ forward))
            lateral_separation = float(np.linalg.norm(rel))
            if longitudinal_offset > 3.0 or not (2.4 <= lateral_separation <= 6.0):
                continue

            adjacent_walked = _walk_straight_lane(
                adjacent,
                min_length_m=min_length_m,
                step_m=sample_distance_m,
                yaw_tolerance_deg=yaw_tolerance_deg,
            )
            if adjacent_walked is None:
                continue
            adjacent_length_m, adjacent_max_yaw = adjacent_walked
            key = (
                int(base.road_id),
                int(base.lane_id),
                int(round(float(getattr(base.waypoint, "s", 0.0)) / sample_distance_m)),
                side,
            )
            if key in seen:
                continue
            seen.add(key)

            candidates.append(
                CutInLaneCandidate(
                    ego_lane=ego_lane,
                    adjacent_waypoint=adjacent,
                    adjacent_length_m=adjacent_length_m,
                    adjacent_max_yaw_change_deg=adjacent_max_yaw,
                    adjacent_road_id=int(adjacent.road_id),
                    adjacent_lane_id=int(adjacent.lane_id),
                    adjacent_side=side,
                    lateral_separation_m=lateral_separation,
                )
            )

    candidates.sort(
        key=lambda item: (
            max(item.ego_lane.max_yaw_change_deg, item.adjacent_max_yaw_change_deg),
            abs(item.lateral_separation_m - 3.6),
            -min(item.ego_lane.length_m, item.adjacent_length_m),
            item.ego_lane.road_id,
            item.ego_lane.lane_id,
            item.adjacent_side,
            float(item.ego_lane.waypoint.transform.location.x),
            float(item.ego_lane.waypoint.transform.location.y),
        )
    )
    return candidates[:120]


def _walk_straight_lane(
    start_waypoint: Any,
    *,
    min_length_m: float,
    step_m: float = 2.0,
    yaw_tolerance_deg: float = 4.0,
) -> tuple[float, float] | None:
    total = 0.0
    current = start_waypoint
    start_yaw = float(start_waypoint.transform.rotation.yaw)
    max_yaw_change = 0.0
    while total < min_length_m:
        next_waypoints = current.next(step_m)
        chosen = _choose_waypoint(next_waypoints, current)
        if chosen is None:
            return None
        if chosen.is_junction or not _is_driving_waypoint(chosen):
            return None
        yaw_change = abs(_angle_diff_deg(chosen.transform.rotation.yaw, start_yaw))
        if yaw_change > yaw_tolerance_deg:
            return None
        max_yaw_change = max(max_yaw_change, yaw_change)
        total += step_m
        current = chosen
    return total, max_yaw_change


def _choose_waypoint(next_waypoints: Iterable[Any], current: Any) -> Any | None:
    options = list(next_waypoints)
    if not options:
        return None
    same_lane = [
        wp
        for wp in options
        if int(wp.road_id) == int(current.road_id) and int(wp.lane_id) == int(current.lane_id)
    ]
    pool = same_lane if same_lane else options
    return min(
        pool,
        key=lambda wp: abs(_angle_diff_deg(wp.transform.rotation.yaw, current.transform.rotation.yaw)),
    )


def _waypoint_ahead(start: Any, distance_m: float, *, step_m: float = 2.0) -> Any | None:
    current = start
    remaining = float(distance_m)
    while remaining > 1e-6:
        step = min(step_m, remaining)
        chosen = _choose_waypoint(current.next(step), current)
        if chosen is None:
            return None
        current = chosen
        remaining -= step
    return current


def _spawn_transform(carla: Any, waypoint: Any) -> Any:
    transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
    transform.location.z += 0.45
    transform.rotation.pitch = 0.0
    transform.rotation.roll = 0.0
    return transform


def _lane_transform_at_distance(carla: Any, start_waypoint: Any, distance_m: float) -> Any:
    waypoint = _waypoint_ahead(start_waypoint, distance_m)
    if waypoint is not None:
        return _spawn_transform(carla, waypoint)

    base = start_waypoint.transform
    forward = base.get_forward_vector()
    location = carla.Location(
        x=float(base.location.x + forward.x * distance_m),
        y=float(base.location.y + forward.y * distance_m),
        z=float(base.location.z + 0.45),
    )
    return carla.Transform(
        location,
        carla.Rotation(pitch=0.0, yaw=float(base.rotation.yaw), roll=0.0),
    )


def _set_initial_velocity(actor: Any, route_forward: np.ndarray, speed_mps: float) -> None:
    carla = _import_carla()
    actor.set_target_velocity(
        carla.Vector3D(
            x=float(route_forward[0] * speed_mps),
            y=float(route_forward[1] * speed_mps),
            z=0.0,
        )
    )


def _route_progress_m(start_location: Any, current_location: Any, route_forward: np.ndarray) -> float:
    delta = np.array(
        [
            float(current_location.x - start_location.x),
            float(current_location.y - start_location.y),
        ],
        dtype=np.float64,
    )
    return max(float(delta @ route_forward), 0.0)


def _configure_synchronous_world(world: Any, fixed_delta_seconds: float) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(fixed_delta_seconds)
    settings.no_rendering_mode = False
    world.apply_settings(settings)


def _restore_async_world(world: Any) -> None:
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except Exception:
        pass


def _destroy_leftover_torsion_actors(world: Any) -> None:
    doomed = []
    for actor in world.get_actors():
        role_name = str(actor.attributes.get("role_name", ""))
        if role_name.startswith(ROLE_PREFIX):
            doomed.append(actor)
    _destroy_spawned(doomed)
    if doomed:
        world.tick(seconds=5.0)


def _destroy_spawned(actors: Iterable[Any]) -> None:
    for actor in reversed(list(actors)):
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except Exception:
            pass
    for actor in reversed(list(actors)):
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except Exception:
            pass


def _vehicle_blueprint(blueprints: Any, preferred_id: str) -> Any:
    try:
        return blueprints.find(preferred_id)
    except RuntimeError:
        vehicles = [
            bp
            for bp in blueprints.filter("vehicle.*")
            if int(str(bp.get_attribute("number_of_wheels"))) == 4
        ]
        if not vehicles:
            raise RuntimeError("no four-wheeled vehicle blueprint is available")
        return vehicles[0]


def _set_role_name(blueprint: Any, role_name: str) -> None:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name[:255])


def _set_color_if_available(blueprint: Any, color: str) -> None:
    if blueprint.has_attribute("color"):
        blueprint.set_attribute("color", color)


def _actor_length(actor: Any) -> float:
    return float(2.0 * actor.bounding_box.extent.x)


def _actor_width(actor: Any) -> float:
    return float(2.0 * actor.bounding_box.extent.y)


def _speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return float(math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))


def _vector2_from_carla(vector: Any) -> np.ndarray:
    return np.array([float(vector.x), float(vector.y)], dtype=np.float64)


def _unit2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vector / norm


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return float(value * value * (3.0 - 2.0 * value))


def _angle_diff_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _driving_lane_type() -> Any:
    carla = _import_carla()
    return carla.LaneType.Driving


def _is_driving_waypoint(waypoint: Any) -> bool:
    try:
        return bool(waypoint.lane_type == _driving_lane_type())
    except Exception:
        return True


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    value = 0x345678
    for byte in text.encode("utf-8"):
        value = ((value * 1_000_003) ^ byte) & 0xFFFFFFFF
    return int(value)


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _asymmetric_yerr(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    value = np.array([float(row[metric]) for row in rows], dtype=np.float64)
    low = np.array([float(row[f"{metric}_ci_low"]) for row in rows], dtype=np.float64)
    high = np.array([float(row[f"{metric}_ci_high"]) for row in rows], dtype=np.float64)
    return np.vstack([np.maximum(value - low, 0.0), np.maximum(high - value, 0.0)])


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return value


def _import_carla() -> Any:
    import carla

    return carla


def _build_default_configs(
    *,
    host: str,
    port: int,
    timeout_s: float,
    map_name: str,
    seeds: int,
    methods: Iterable[str],
    magnitudes: Iterable[str],
    max_ticks: int,
) -> list[CarlaEpisodeConfig]:
    return [
        CarlaEpisodeConfig(
            host=host,
            port=port,
            timeout_s=timeout_s,
            map_name=map_name,
            seed=seed,
            method=method,  # type: ignore[arg-type]
            magnitude=magnitude,
            max_ticks=max_ticks,
        )
        for method in methods
        for magnitude in magnitudes
        for seed in range(seeds)
    ]


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("method,magnitude,n,collision_rate,mean_min_ttc,mean_min_actor_distance")
    for row in summary:
        print(
            ",".join(
                [
                    str(row["method"]),
                    str(row["magnitude"]),
                    str(row["n_runs"]),
                    f"{float(row['collision_rate']):.3f}",
                    f"{float(row['mean_min_ttc']):.3f}",
                    f"{float(row['mean_min_actor_distance']):.3f}",
                ]
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--map",
        dest="map_name",
        default="current",
        help="Expected current map label only; the runner never reloads or switches maps.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-ticks", type=int, default=240)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--magnitudes", nargs="+", default=list(MAGNITUDES), choices=list(MAGNITUDES))
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)

    configs = _build_default_configs(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        map_name=args.map_name,
        seeds=args.seeds,
        methods=args.methods,
        magnitudes=args.magnitudes,
        max_ticks=args.max_ticks,
    )
    _, summary = run_carla_sweep(
        configs,
        metrics_csv=args.metrics_csv,
        summary_csv=args.summary_csv,
        figure_path=args.figure,
    )
    _print_summary(summary)
    print(f"Wrote {args.metrics_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
