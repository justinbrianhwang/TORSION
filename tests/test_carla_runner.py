from __future__ import annotations

import pytest


def test_carla_spawn_tick_destroy_if_available() -> None:
    try:
        import carla
    except ImportError as exc:
        pytest.skip(f"CARLA Python API is not installed: {exc}")

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)
    try:
        world = client.get_world()
    except RuntimeError as exc:
        pytest.skip(f"CARLA server is not reachable: {exc}")

    actor = None
    original = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        blueprint = blueprint_library.find("vehicle.tesla.model3")
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "TORSION_CARLA_VALIDATION_pytest")

        for transform in world.get_map().get_spawn_points():
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                break
        if actor is None:
            pytest.skip("CARLA connected, but no vehicle spawn point was available")

        world.tick(seconds=5.0)
        assert actor.is_alive
    finally:
        if actor is not None and actor.is_alive:
            actor.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        settings.no_rendering_mode = original.no_rendering_mode
        world.apply_settings(settings)
