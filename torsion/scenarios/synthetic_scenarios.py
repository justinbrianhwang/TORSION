"""Seeded stochastic scripted scenarios for the synthetic closed-loop harness.

Each factory accepts ``seed``.  ``seed=None`` returns the nominal deterministic
instance used by lightweight examples; an integer seed samples one concrete
scenario instance from the documented ranges below.  Runners pass their
experiment seed into the factory so clean and faulted runs with the same seed
share the same actor trajectories and the same per-frame observation noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from torsion.operators.object import ObjectSet
from torsion.scenarios.planner import EgoState


SCENARIO_INSTANCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "cut_in": {
        "ego_speed_mps": (11.3, 12.8),
        "actor_x0_m": (21.5, 27.0),
        "actor_y0_m": (3.0, 3.7),
        "actor_speed_mps": (4.9, 6.4),
        "actor_heading_rad": (float(np.deg2rad(-1.0)), float(np.deg2rad(1.0))),
        "cut_in_trigger_s": (0.2, 1.0),
        "merge_lateral_speed_mps": (-1.15, -0.75),
    },
    "leading_vehicle": {
        "ego_speed_mps": (11.0, 13.3),
        "actor_x0_m": (21.0, 29.0),
        "actor_y0_m": (-0.18, 0.18),
        "actor_speed_mps": (6.2, 8.2),
        "actor_heading_rad": (float(np.deg2rad(-0.7)), float(np.deg2rad(0.7))),
        "brake_trigger_s": (0.7, 1.6),
        "decel_mps2": (2.3, 3.5),
        "min_speed_mps": (1.6, 2.8),
    },
    "pedestrian_crossing": {
        "ego_speed_mps": (9.0, 10.0),
        "actor_x0_m": (25.5, 28.5),
        "actor_y0_m": (-5.0, -4.2),
        "actor_speed_mps": (1.8, 2.2),
        "actor_heading_rad": (float(np.deg2rad(86.0)), float(np.deg2rad(94.0))),
        "crossing_start_s": (0.6, 1.2),
    },
    "stopped_obstacle": {
        "ego_speed_mps": (11.0, 13.0),
        "actor_x0_m": (30.0, 38.0),
        "actor_y0_m": (-0.25, 0.25),
    },
    "oncoming_drift": {
        "ego_speed_mps": (10.5, 12.5),
        "actor_x0_m": (46.0, 56.0),
        "actor_y0_m": (3.2, 3.9),
        "actor_speed_mps": (8.0, 10.5),
        "drift_trigger_s": (0.4, 1.2),
        "drift_lateral_speed_mps": (-0.75, -0.45),
    },
    # Dense scene: the ego is boxed in by a lead vehicle and two flanking
    # vehicles.  Most planner candidates are infeasible, so the surviving cost
    # minimum is well separated -- the regime in which we predict argmin
    # switching should NOT occur (see the gateway-collapse hypothesis).
    "dense_traffic": {
        "ego_speed_mps": (10.0, 12.0),
        "lead_x0_m": (18.0, 24.0),
        "lead_speed_mps": (6.5, 8.5),
        "lead_decel_mps2": (1.8, 2.8),
        "left_x0_m": (8.0, 14.0),
        "left_speed_mps": (10.0, 12.5),
        "right_x0_m": (9.0, 15.0),
        "right_speed_mps": (9.5, 12.0),
        "lane_offset_m": (3.2, 3.6),
    },
}

OBSERVATION_NOISE_RANGES = {
    "xy_sigma_m": 0.04,
    "velocity_sigma_mps": 0.03,
    "yaw_sigma_rad": float(np.deg2rad(0.25)),
}


@dataclass(frozen=True)
class ScriptedActorTrajectory:
    """Ground-truth actor trajectory sampled on the scenario time grid."""

    track_id: Any
    cls: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    yaw: NDArray[np.float64]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]
    width: float
    height: float
    length: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        arrays = {}
        n: int | None = None
        for name in ("x", "y", "yaw", "vx", "vy"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array")
            if n is None:
                n = int(arr.shape[0])
            elif arr.shape[0] != n:
                raise ValueError(f"{name} length must match x length")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
            arr = arr.copy()
            arr.setflags(write=False)
            arrays[name] = arr

        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)

    @property
    def steps(self) -> int:
        return int(self.x.shape[0])

    def state_record(self, frame_idx: int) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "cls": self.cls,
            "x": float(self.x[frame_idx]),
            "y": float(self.y[frame_idx]),
            "z": 0.0,
            "w": self.width,
            "h": self.height,
            "l": self.length,
            "yaw": float(self.yaw[frame_idx]),
            "vx": float(self.vx[frame_idx]),
            "vy": float(self.vy[frame_idx]),
            "conf": self.confidence,
        }


@dataclass(frozen=True)
class SyntheticScenario:
    """Scripted straight-road world with fixed-dt actor trajectories.

    ``ground_truth_object_set`` exposes exact actor states for safety metrics.
    ``object_set`` exposes the deterministic noisy perception for this scenario
    seed and frame; this is what the planners and fault operators consume.
    """

    scenario_id: str
    dt: float
    steps: int
    ego_initial: EgoState
    actors: tuple[ScriptedActorTrajectory, ...]
    primary_actor_id: Any
    route_length_m: float
    description: str
    sample_seed: int | None = None
    sample_parameters: tuple[tuple[str, float], ...] = ()
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"]
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ]
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"]

    def __post_init__(self) -> None:
        if self.steps <= 1:
            raise ValueError("scenario must contain at least two steps")
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        for name in (
            "observation_noise_xy_sigma_m",
            "observation_noise_velocity_sigma_mps",
            "observation_noise_yaw_sigma_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for actor in self.actors:
            if actor.steps != self.steps:
                raise ValueError("all actor trajectories must match scenario steps")

    @property
    def times_s(self) -> NDArray[np.float64]:
        return np.arange(self.steps, dtype=np.float64) * self.dt

    def ground_truth_object_set(self, frame_idx: int) -> ObjectSet:
        frame = int(np.clip(frame_idx, 0, self.steps - 1))
        return ObjectSet.from_records(actor.state_record(frame) for actor in self.actors)

    def object_set(self, frame_idx: int) -> ObjectSet:
        clean = self.ground_truth_object_set(frame_idx)
        if self.sample_seed is None or len(clean) == 0:
            return clean
        if (
            self.observation_noise_xy_sigma_m == 0.0
            and self.observation_noise_velocity_sigma_mps == 0.0
            and self.observation_noise_yaw_sigma_rad == 0.0
        ):
            return clean

        frame = int(np.clip(frame_idx, 0, self.steps - 1))
        rng = np.random.default_rng(
            _stable_seed("observation", self.scenario_id, self.sample_seed, frame)
        )
        xy_noise = rng.normal(
            0.0, self.observation_noise_xy_sigma_m, size=(len(clean), 2)
        )
        velocity_noise = rng.normal(
            0.0, self.observation_noise_velocity_sigma_mps, size=clean.v.shape
        )
        yaw_noise = rng.normal(0.0, self.observation_noise_yaw_sigma_rad, size=len(clean))
        return clean.replace(
            x=clean.x + xy_noise[:, 0],
            y=clean.y + xy_noise[:, 1],
            yaw=_wrap_to_pi(clean.yaw + yaw_noise),
            v=clean.v + velocity_noise,
        )

    def actor_records(self, frame_idx: int, *, observed: bool = False) -> list[dict[str, Any]]:
        if observed:
            return _object_set_records(self.object_set(frame_idx))
        frame = int(np.clip(frame_idx, 0, self.steps - 1))
        return [actor.state_record(frame) for actor in self.actors]


ScenarioFactory = Callable[..., SyntheticScenario]


def cut_in(
    *,
    dt: float = 0.1,
    steps: int = 90,
    ego_speed_mps: float = 12.0,
    seed: int | None = None,
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"],
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ],
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"],
) -> SyntheticScenario:
    """Adjacent-lane vehicle merges into the ego lane ahead.

    Parameters are tuned so a clean CV prediction brakes early, while a
    medium yaw/velocity/position torsion delays the predicted lane conflict.
    """

    params = _cut_in_parameters(seed, ego_speed_mps=ego_speed_mps)
    y = np.empty(steps, dtype=np.float64)
    x = np.empty(steps, dtype=np.float64)
    vx = np.empty(steps, dtype=np.float64)
    vy = np.empty(steps, dtype=np.float64)
    actor_speed = params["actor_speed_mps"]
    heading = params["actor_heading_rad"]
    base_vx = actor_speed * float(np.cos(heading))
    base_vy = actor_speed * float(np.sin(heading))
    x[0] = params["actor_x0_m"]
    y[0] = params["actor_y0_m"]
    for frame in range(steps):
        if frame > 0:
            x[frame] = x[frame - 1] + vx[frame - 1] * dt
            y[frame] = y[frame - 1] + vy[frame - 1] * dt
        if frame * dt >= params["cut_in_trigger_s"] and y[frame] > 0.0:
            vx[frame] = base_vx
            vy[frame] = params["merge_lateral_speed_mps"]
        else:
            if y[frame] <= 0.0:
                y[frame] = 0.0
                vx[frame] = base_vx
                vy[frame] = 0.0
            else:
                vx[frame] = base_vx
                vy[frame] = base_vy
    yaw = _yaw_from_velocity(vx, vy)
    actor = ScriptedActorTrajectory(
        track_id="cut_in_vehicle",
        cls="vehicle",
        x=x,
        y=y,
        yaw=yaw,
        vx=vx,
        vy=vy,
        width=2.0,
        height=1.6,
        length=4.5,
    )
    return SyntheticScenario(
        scenario_id="cut_in",
        dt=dt,
        steps=steps,
        ego_initial=EgoState(x=0.0, y=0.0, yaw=0.0, speed=params["ego_speed_mps"]),
        actors=(actor,),
        primary_actor_id=actor.track_id,
        route_length_m=95.0,
        description="vehicle cuts in from the left lane ahead of ego",
        sample_seed=seed,
        sample_parameters=_parameter_tuple(params),
        observation_noise_xy_sigma_m=observation_noise_xy_sigma_m,
        observation_noise_velocity_sigma_mps=observation_noise_velocity_sigma_mps,
        observation_noise_yaw_sigma_rad=observation_noise_yaw_sigma_rad,
    )


def leading_vehicle(
    *,
    dt: float = 0.1,
    steps: int = 90,
    ego_speed_mps: float = 12.0,
    seed: int | None = None,
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"],
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ],
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"],
) -> SyntheticScenario:
    """Slow lead vehicle decelerates in the ego lane."""

    x = np.empty(steps, dtype=np.float64)
    params = _leading_vehicle_parameters(seed, ego_speed_mps=ego_speed_mps)
    y = np.empty(steps, dtype=np.float64)
    vx = np.empty(steps, dtype=np.float64)
    vy = np.empty(steps, dtype=np.float64)
    x[0] = params["actor_x0_m"]
    y[0] = params["actor_y0_m"]
    speed = params["actor_speed_mps"]
    heading = params["actor_heading_rad"]
    vx[0] = speed * float(np.cos(heading))
    vy[0] = speed * float(np.sin(heading))
    for frame in range(1, steps):
        t_prev = (frame - 1) * dt
        x[frame] = x[frame - 1] + vx[frame - 1] * dt
        y[frame] = y[frame - 1] + vy[frame - 1] * dt
        if t_prev >= params["brake_trigger_s"]:
            speed = max(params["min_speed_mps"], speed - params["decel_mps2"] * dt)
        vx[frame] = speed * float(np.cos(heading))
        vy[frame] = speed * float(np.sin(heading))
    yaw = _yaw_from_velocity(vx, vy)
    actor = ScriptedActorTrajectory(
        track_id="lead_vehicle",
        cls="vehicle",
        x=x,
        y=y,
        yaw=yaw,
        vx=vx,
        vy=vy,
        width=2.0,
        height=1.6,
        length=4.5,
    )
    return SyntheticScenario(
        scenario_id="leading_vehicle",
        dt=dt,
        steps=steps,
        ego_initial=EgoState(x=0.0, y=0.0, yaw=0.0, speed=params["ego_speed_mps"]),
        actors=(actor,),
        primary_actor_id=actor.track_id,
        route_length_m=95.0,
        description="lead vehicle slows in the ego lane",
        sample_seed=seed,
        sample_parameters=_parameter_tuple(params),
        observation_noise_xy_sigma_m=observation_noise_xy_sigma_m,
        observation_noise_velocity_sigma_mps=observation_noise_velocity_sigma_mps,
        observation_noise_yaw_sigma_rad=observation_noise_yaw_sigma_rad,
    )


def pedestrian_crossing(
    *,
    dt: float = 0.1,
    steps: int = 90,
    ego_speed_mps: float = 10.0,
    seed: int | None = None,
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"],
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ],
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"],
) -> SyntheticScenario:
    """Pedestrian crosses laterally through the ego lane.

    The seeded ranges place the pedestrian in a safe near-miss envelope: clean
    perception sees the crossing early enough to brake or steer, while moving
    the perceived crossing evidence away from the route can delay the response.
    """

    params = _pedestrian_crossing_parameters(seed, ego_speed_mps=ego_speed_mps)
    x = np.empty(steps, dtype=np.float64)
    y = np.empty(steps, dtype=np.float64)
    vx = np.empty(steps, dtype=np.float64)
    vy = np.empty(steps, dtype=np.float64)
    speed = params["actor_speed_mps"]
    heading = params["actor_heading_rad"]
    vx_value = speed * float(np.cos(heading))
    vy_value = speed * float(np.sin(heading))
    x[0] = params["actor_x0_m"]
    y[0] = params["actor_y0_m"]
    for frame in range(steps):
        if frame > 0:
            x[frame] = x[frame - 1] + vx[frame - 1] * dt
            y[frame] = y[frame - 1] + vy[frame - 1] * dt
        if frame * dt >= params["crossing_start_s"]:
            vx[frame] = vx_value
            vy[frame] = vy_value
        else:
            vx[frame] = 0.0
            vy[frame] = 0.0
    yaw = _yaw_from_velocity(vx, vy)
    actor = ScriptedActorTrajectory(
        track_id="crossing_pedestrian",
        cls="pedestrian",
        x=x,
        y=y,
        yaw=yaw,
        vx=vx,
        vy=vy,
        width=0.8,
        height=1.7,
        length=0.8,
    )
    return SyntheticScenario(
        scenario_id="pedestrian_crossing",
        dt=dt,
        steps=steps,
        ego_initial=EgoState(x=0.0, y=0.0, yaw=0.0, speed=params["ego_speed_mps"]),
        actors=(actor,),
        primary_actor_id=actor.track_id,
        route_length_m=80.0,
        description="pedestrian crosses from right to left through ego lane",
        sample_seed=seed,
        sample_parameters=_parameter_tuple(params),
        observation_noise_xy_sigma_m=observation_noise_xy_sigma_m,
        observation_noise_velocity_sigma_mps=observation_noise_velocity_sigma_mps,
        observation_noise_yaw_sigma_rad=observation_noise_yaw_sigma_rad,
    )


def stopped_obstacle(
    *,
    dt: float = 0.1,
    steps: int = 90,
    ego_speed_mps: float = 12.0,
    seed: int | None = None,
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"],
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ],
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"],
) -> SyntheticScenario:
    """Stationary vehicle blocking the ego lane.

    Unlike ``leading_vehicle`` the obstacle never moves, so a velocity fault
    carries no information and only a position fault can displace it.  This
    separates position-semantics from velocity-semantics in the taxonomy.
    """

    params = _sample_parameters("stopped_obstacle", seed) if seed is not None else {
        "ego_speed_mps": ego_speed_mps,
        "actor_x0_m": 34.0,
        "actor_y0_m": 0.0,
    }
    params.setdefault("ego_speed_mps", ego_speed_mps)

    x = np.full(steps, params["actor_x0_m"], dtype=np.float64)
    y = np.full(steps, params["actor_y0_m"], dtype=np.float64)
    vx = np.zeros(steps, dtype=np.float64)
    vy = np.zeros(steps, dtype=np.float64)
    actor = ScriptedActorTrajectory(
        track_id="stopped_vehicle",
        cls="vehicle",
        x=x,
        y=y,
        yaw=_yaw_from_velocity(vx, vy),
        vx=vx,
        vy=vy,
        width=2.0,
        height=1.6,
        length=4.5,
    )
    return SyntheticScenario(
        scenario_id="stopped_obstacle",
        dt=dt,
        steps=steps,
        ego_initial=EgoState(x=0.0, y=0.0, yaw=0.0, speed=params["ego_speed_mps"]),
        actors=(actor,),
        primary_actor_id=actor.track_id,
        route_length_m=95.0,
        description="stationary vehicle blocks the ego lane",
        sample_seed=seed,
        sample_parameters=_parameter_tuple(params),
        observation_noise_xy_sigma_m=observation_noise_xy_sigma_m,
        observation_noise_velocity_sigma_mps=observation_noise_velocity_sigma_mps,
        observation_noise_yaw_sigma_rad=observation_noise_yaw_sigma_rad,
    )


def oncoming_drift(
    *,
    dt: float = 0.1,
    steps: int = 90,
    ego_speed_mps: float = 11.5,
    seed: int | None = None,
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"],
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ],
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"],
) -> SyntheticScenario:
    """Oncoming vehicle drifts out of its lane toward the ego lane.

    Closing speed is the sum of both speeds, so the time-to-conflict is short
    and a heading/velocity fault matters more than in same-direction scenarios.
    """

    params = _sample_parameters("oncoming_drift", seed) if seed is not None else {
        "ego_speed_mps": ego_speed_mps,
        "actor_x0_m": 50.0,
        "actor_y0_m": 3.5,
        "actor_speed_mps": 9.0,
        "drift_trigger_s": 0.8,
        "drift_lateral_speed_mps": -0.6,
    }
    params.setdefault("ego_speed_mps", ego_speed_mps)

    x = np.empty(steps, dtype=np.float64)
    y = np.empty(steps, dtype=np.float64)
    vx = np.empty(steps, dtype=np.float64)
    vy = np.empty(steps, dtype=np.float64)
    x[0] = params["actor_x0_m"]
    y[0] = params["actor_y0_m"]
    base_vx = -params["actor_speed_mps"]  # travelling toward the ego
    for frame in range(steps):
        if frame > 0:
            x[frame] = x[frame - 1] + vx[frame - 1] * dt
            y[frame] = y[frame - 1] + vy[frame - 1] * dt
        vx[frame] = base_vx
        drifting = frame * dt >= params["drift_trigger_s"] and y[frame] > 0.0
        vy[frame] = params["drift_lateral_speed_mps"] if drifting else 0.0
    actor = ScriptedActorTrajectory(
        track_id="oncoming_vehicle",
        cls="vehicle",
        x=x,
        y=y,
        yaw=_yaw_from_velocity(vx, vy),
        vx=vx,
        vy=vy,
        width=2.0,
        height=1.6,
        length=4.5,
    )
    return SyntheticScenario(
        scenario_id="oncoming_drift",
        dt=dt,
        steps=steps,
        ego_initial=EgoState(x=0.0, y=0.0, yaw=0.0, speed=params["ego_speed_mps"]),
        actors=(actor,),
        primary_actor_id=actor.track_id,
        route_length_m=95.0,
        description="oncoming vehicle drifts toward the ego lane",
        sample_seed=seed,
        sample_parameters=_parameter_tuple(params),
        observation_noise_xy_sigma_m=observation_noise_xy_sigma_m,
        observation_noise_velocity_sigma_mps=observation_noise_velocity_sigma_mps,
        observation_noise_yaw_sigma_rad=observation_noise_yaw_sigma_rad,
    )


def dense_traffic(
    *,
    dt: float = 0.1,
    steps: int = 90,
    ego_speed_mps: float = 11.0,
    seed: int | None = None,
    observation_noise_xy_sigma_m: float = OBSERVATION_NOISE_RANGES["xy_sigma_m"],
    observation_noise_velocity_sigma_mps: float = OBSERVATION_NOISE_RANGES[
        "velocity_sigma_mps"
    ],
    observation_noise_yaw_sigma_rad: float = OBSERVATION_NOISE_RANGES["yaw_sigma_rad"],
) -> SyntheticScenario:
    """Ego boxed in by a braking lead vehicle and two flanking vehicles.

    This is the *dense* counterpart to the single-actor scenarios, and it exists
    to test the gateway-collapse hypothesis directly: if the planner-switch
    mechanism is a property of shallow, near-degenerate cost landscapes, then a
    scene where the flanking lanes are occupied -- and most candidates are
    therefore infeasible -- should show a *larger* decision margin and a *lower*
    argmin-flip rate than the sparse scenarios, under the same fault budget.
    """

    params = _sample_parameters("dense_traffic", seed) if seed is not None else {
        "ego_speed_mps": ego_speed_mps,
        "lead_x0_m": 21.0,
        "lead_speed_mps": 7.5,
        "lead_decel_mps2": 2.3,
        "left_x0_m": 11.0,
        "left_speed_mps": 11.2,
        "right_x0_m": 12.0,
        "right_speed_mps": 10.8,
        "lane_offset_m": 3.4,
    }
    params.setdefault("ego_speed_mps", ego_speed_mps)
    offset = params["lane_offset_m"]

    def _straight(x0: float, y0: float, speed: float, decel: float = 0.0):
        x = np.empty(steps, dtype=np.float64)
        y = np.full(steps, y0, dtype=np.float64)
        vx = np.empty(steps, dtype=np.float64)
        vy = np.zeros(steps, dtype=np.float64)
        x[0] = x0
        v = speed
        vx[0] = v
        for frame in range(1, steps):
            x[frame] = x[frame - 1] + vx[frame - 1] * dt
            if decel > 0.0 and (frame - 1) * dt >= 0.8:
                v = max(2.0, v - decel * dt)
            vx[frame] = v
        return x, y, vx, vy

    specs = [
        ("lead_vehicle", params["lead_x0_m"], 0.0, params["lead_speed_mps"], params["lead_decel_mps2"]),
        ("left_vehicle", params["left_x0_m"], offset, params["left_speed_mps"], 0.0),
        ("right_vehicle", params["right_x0_m"], -offset, params["right_speed_mps"], 0.0),
    ]
    actors = []
    for track_id, x0, y0, speed, decel in specs:
        x, y, vx, vy = _straight(x0, y0, speed, decel)
        actors.append(
            ScriptedActorTrajectory(
                track_id=track_id,
                cls="vehicle",
                x=x,
                y=y,
                yaw=_yaw_from_velocity(vx, vy),
                vx=vx,
                vy=vy,
                width=2.0,
                height=1.6,
                length=4.5,
            )
        )
    return SyntheticScenario(
        scenario_id="dense_traffic",
        dt=dt,
        steps=steps,
        ego_initial=EgoState(x=0.0, y=0.0, yaw=0.0, speed=params["ego_speed_mps"]),
        actors=tuple(actors),
        primary_actor_id="lead_vehicle",
        route_length_m=95.0,
        description="ego boxed in by a braking lead vehicle and two flanking vehicles",
        sample_seed=seed,
        sample_parameters=_parameter_tuple(params),
        observation_noise_xy_sigma_m=observation_noise_xy_sigma_m,
        observation_noise_velocity_sigma_mps=observation_noise_velocity_sigma_mps,
        observation_noise_yaw_sigma_rad=observation_noise_yaw_sigma_rad,
    )


SCENARIOS: dict[str, ScenarioFactory] = {
    "cut_in": cut_in,
    "leading_vehicle": leading_vehicle,
    "pedestrian_crossing": pedestrian_crossing,
    "stopped_obstacle": stopped_obstacle,
    "oncoming_drift": oncoming_drift,
    "dense_traffic": dense_traffic,
}

# Scenes with a single actor leave most planner candidates feasible (a shallow
# cost landscape); dense_traffic occupies the flanking lanes.  The propagation
# analysis uses this split to test the gateway-collapse hypothesis.
SPARSE_SCENARIOS = ("cut_in", "leading_vehicle", "pedestrian_crossing", "stopped_obstacle", "oncoming_drift")
DENSE_SCENARIOS = ("dense_traffic",)


def get_scenario(name: str, **kwargs: Any) -> SyntheticScenario:
    """Build a named scenario.

    Pass ``seed`` to sample a stochastic instance.  Reusing the same seed across
    clean and faulted runs gives a paired within-instance comparison.
    """

    key = name.lower().strip().replace("-", "_")
    try:
        return SCENARIOS[key](**kwargs)
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown synthetic scenario {name!r}; expected one of {valid}") from exc


def _yaw_from_velocity(vx: NDArray[np.float64], vy: NDArray[np.float64]) -> NDArray[np.float64]:
    yaw = np.arctan2(vy, vx)
    stopped = np.hypot(vx, vy) <= 1e-9
    if np.any(stopped):
        yaw = yaw.copy()
        yaw[stopped] = 0.0
    return yaw.astype(np.float64, copy=False)


def _cut_in_parameters(seed: int | None, *, ego_speed_mps: float) -> dict[str, float]:
    if seed is None:
        return {
            "ego_speed_mps": float(ego_speed_mps),
            "actor_x0_m": 24.0,
            "actor_y0_m": 3.4,
            "actor_speed_mps": 5.5,
            "actor_heading_rad": 0.0,
            "cut_in_trigger_s": 0.0,
            "merge_lateral_speed_mps": -0.9,
        }
    return _sample_parameters("cut_in", seed)


def _leading_vehicle_parameters(seed: int | None, *, ego_speed_mps: float) -> dict[str, float]:
    if seed is None:
        return {
            "ego_speed_mps": float(ego_speed_mps),
            "actor_x0_m": 25.0,
            "actor_y0_m": 0.0,
            "actor_speed_mps": 7.0,
            "actor_heading_rad": 0.0,
            "brake_trigger_s": 1.2,
            "decel_mps2": 2.8,
            "min_speed_mps": 2.0,
        }
    return _sample_parameters("leading_vehicle", seed)


def _pedestrian_crossing_parameters(
    seed: int | None, *, ego_speed_mps: float
) -> dict[str, float]:
    if seed is None:
        return {
            "ego_speed_mps": float(ego_speed_mps),
            "actor_x0_m": 24.8,
            "actor_y0_m": -3.7,
            "actor_speed_mps": 1.48,
            "actor_heading_rad": float(np.deg2rad(90.0)),
            "crossing_start_s": 0.0,
        }
    return _sample_parameters("pedestrian_crossing", seed)


def _sample_parameters(scenario_id: str, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(_stable_seed("scenario-instance", scenario_id, seed))
    ranges = SCENARIO_INSTANCE_RANGES[scenario_id]
    return {
        name: float(rng.uniform(low, high))
        for name, (low, high) in ranges.items()
    }


def _parameter_tuple(params: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((name, float(params[name])) for name in sorted(params))


def _object_set_records(objects: ObjectSet) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(objects)):
        rows.append(
            {
                "track_id": objects.track_id[idx],
                "cls": str(objects.cls[idx]),
                "x": float(objects.x[idx]),
                "y": float(objects.y[idx]),
                "z": float(objects.z[idx]),
                "w": float(objects.w[idx]),
                "h": float(objects.h[idx]),
                "l": float(objects.l[idx]),
                "yaw": float(objects.yaw[idx]),
                "vx": float(objects.v[idx, 0]),
                "vy": float(objects.v[idx, 1]),
                "conf": float(objects.conf[idx]),
            }
        )
    return rows


def _wrap_to_pi(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    value = 0xC2B2AE35
    for byte in text.encode("utf-8"):
        value = ((value * 1_000_003) ^ byte) & 0xFFFFFFFF
    return int(value)
