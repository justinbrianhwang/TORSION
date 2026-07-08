"""Run an incremental CARLA sweep against the already-running current world."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torsion.scenarios.carla_runner import (  # noqa: E402
    METHODS,
    CarlaEpisodeConfig,
    _configs_for_current_map,
    _current_map_label,
    _destroy_leftover_torsion_actors,
    _destroy_spawned,
    _failed_run_result,
    _find_cut_in_lane_candidates,
    _find_straight_lane_candidates,
    _import_carla,
    _is_carla_connection_error,
    _restore_async_world,
    _run_episode_in_world,
    _configure_synchronous_world,
)

SCENARIOS = ("leading_vehicle", "cut_in")
DEFAULT_MAGNITUDES_M = ("0.5", "1.0", "1.25", "1.5", "1.75", "2.0")
CSV_COLUMNS = (
    "scenario",
    "method",
    "magnitude",
    "seed",
    "collision",
    "min_ttc",
    "min_distance",
    "brake_reaction_s",
    "n_ticks",
    "status",
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        scenarios = _parse_choices(args.scenario or ["leading_vehicle"], choices=SCENARIOS, name="scenario")
        methods = _parse_choices([args.methods], choices=METHODS, name="method")
        magnitudes = _parse_magnitudes(args.magnitudes)
        if args.seeds <= 0:
            raise ValueError("--seeds must be positive")
    except ValueError as exc:
        parser.error(str(exc))

    configs = [
        CarlaEpisodeConfig(
            scenario=scenario,  # type: ignore[arg-type]
            method=method,  # type: ignore[arg-type]
            magnitude=magnitude,
            seed=seed,
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            max_ticks=args.max_ticks,
        )
        for scenario in scenarios
        for method in methods
        for magnitude in magnitudes
        for seed in range(args.seeds)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    append_with_header = args.append and args.out.exists() and args.out.stat().st_size > 0
    mode = "a" if args.append else "w"

    world: Any | None = None
    with args.out.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        if not append_with_header:
            writer.writeheader()
            _flush(handle)

        try:
            carla = _import_carla()
            client = carla.Client(args.host, int(args.port))
            client.set_timeout(float(args.timeout))
            world = client.get_world()
            current_map_name = _current_map_label(world)
            configs = _configs_for_current_map(configs, current_map_name)

            _destroy_leftover_torsion_actors(world)
            _configure_synchronous_world(world, configs[0].fixed_delta_seconds)
            world.tick(seconds=args.timeout)

            lane_candidates = _find_straight_lane_candidates(world.get_map())
            if not lane_candidates:
                raise RuntimeError(
                    f"no straight lane segment long enough was found in current map {current_map_name}"
                )
            cut_in_candidates = _find_cut_in_lane_candidates(world.get_map(), lane_candidates)

            total = len(configs)
            for idx, cfg in enumerate(configs, start=1):
                print(
                    "episode "
                    f"{idx}/{total}: scenario={cfg.scenario} method={cfg.method} "
                    f"magnitude={cfg.magnitude} seed={cfg.seed}",
                    flush=True,
                )
                spawned: list[Any] = []
                try:
                    result, spawned = _run_episode_in_world(
                        carla,
                        world,
                        cfg,
                        lane_candidates=lane_candidates,
                        cut_in_candidates=cut_in_candidates,
                    )
                    row = _result_row(result=result, status="completed")
                except Exception as exc:
                    if _is_carla_connection_error(exc):
                        print(
                            f"CARLA connection dropped during {cfg.run_id}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 2
                    result = _failed_run_result(cfg, exc)
                    row = _result_row(result=result, status="failed")
                    print(f"episode {idx}/{total} failed: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    _destroy_spawned(spawned)

                writer.writerow(row)
                _flush(handle)

                try:
                    world.tick(seconds=args.timeout)
                except Exception as exc:
                    if _is_carla_connection_error(exc):
                        print(
                            f"CARLA connection dropped while settling after {cfg.run_id}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 2
                    raise
        except Exception as exc:
            if _is_carla_connection_error(exc):
                print(f"CARLA connection failed or dropped: {exc}", file=sys.stderr, flush=True)
                return 2
            print(f"CARLA sweep setup failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        finally:
            if world is not None:
                try:
                    _destroy_leftover_torsion_actors(world)
                except Exception:
                    pass
                try:
                    _restore_async_world(world)
                except Exception:
                    pass

    print(f"Wrote {args.out}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name; repeat or pass comma list. Choices: leading_vehicle,cut_in.",
    )
    parser.add_argument(
        "--methods",
        default="clean,torsion_displace,random_warp",
        help="Comma list of methods: clean,torsion_displace,random_warp.",
    )
    parser.add_argument(
        "--magnitudes",
        default=",".join(DEFAULT_MAGNITUDES_M),
        help="Comma list of displacement budgets in meters.",
    )
    parser.add_argument("--seeds", type=int, default=15, help="Run seeds 0..N-1.")
    parser.add_argument("--out", type=Path, default=Path("results/metrics/carla_runs.csv"))
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV; write header only if the file is new or empty.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-ticks", type=int, default=240)
    return parser


def _parse_choices(
    values: Iterable[str],
    *,
    choices: Iterable[str],
    name: str,
) -> list[str]:
    allowed = tuple(choices)
    parsed = _split_csv(values)
    invalid = [value for value in parsed if value not in allowed]
    if invalid:
        raise ValueError(f"unknown {name} value(s) {invalid!r}; expected one of {', '.join(allowed)}")
    return parsed


def _parse_magnitudes(value: str) -> list[str]:
    parsed = _split_csv([value])
    for item in parsed:
        try:
            meters = float(item)
        except ValueError as exc:
            raise ValueError(f"magnitude {item!r} is not a meter value") from exc
        if not math.isfinite(meters) or meters < 0.0:
            raise ValueError(f"magnitude {item!r} must be non-negative and finite")
    return parsed


def _split_csv(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one value is required")
    return parsed


def _result_row(*, result: Any, status: str) -> dict[str, Any]:
    summary = result.summary
    return {
        "scenario": result.config.scenario,
        "method": result.config.method,
        "magnitude": result.config.magnitude,
        "seed": result.config.seed,
        "collision": bool(summary.get("collision", False)),
        "min_ttc": _csv_float(summary.get("min_ttc")),
        "min_distance": _csv_float(summary.get("min_actor_distance")),
        "brake_reaction_s": _csv_float(summary.get("brake_reaction_delay_s")),
        "n_ticks": int(summary.get("ticks_run") or 0),
        "status": status,
    }


def _csv_float(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(out):
        return ""
    return f"{out:.6g}"


def _flush(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
