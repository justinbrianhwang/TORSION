import numpy as np
import pytest

from torsion.operators.object import (
    ObjectSet,
    confidence_redistribution,
    position_torsion,
    resolve_magnitude,
    velocity_direction_torsion,
    yaw_torsion,
)


def sample_objects() -> ObjectSet:
    return ObjectSet(
        x=[5.0, 10.0, -3.0],
        y=[0.0, 1.0, 0.5],
        z=[0.0, 0.0, 0.0],
        w=[2.0, 2.1, 0.8],
        h=[1.5, 1.6, 1.7],
        l=[4.0, 4.2, 0.8],
        yaw=[0.0, 0.1, -0.2],
        v=[[5.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
        cls=["vehicle", "vehicle", "pedestrian"],
        conf=[0.9, 0.8, 0.7],
        track_id=[101, 102, 201],
    )


def assert_contract_preserved(before: ObjectSet, after: ObjectSet) -> None:
    assert len(after) == len(before)
    assert np.array_equal(after.cls, before.cls)
    assert np.array_equal(after.track_id, before.track_id)
    assert np.allclose(after.bbox_size, before.bbox_size)


@pytest.mark.parametrize(
    "operator",
    [
        lambda objects: position_torsion(objects, dx=0.2, dy=-0.1, target="all"),
        lambda objects: yaw_torsion(objects, d_yaw_deg=2.0, target="all"),
        lambda objects: velocity_direction_torsion(objects, theta_deg=5.0, target="all"),
        lambda objects: confidence_redistribution(objects, delta=0.05, target="all"),
    ],
)
def test_object_contract_invariant_under_every_operator(operator) -> None:
    objects = sample_objects()
    out = operator(objects)
    assert_contract_preserved(objects, out)


@pytest.mark.parametrize(
    ("level", "position_m", "angle_deg"),
    [
        ("low", 0.2, 2.0),
        ("medium", 0.5, 5.0),
        ("high", 1.0, 10.0),
    ],
)
def test_magnitude_levels_match_section_8_1(level, position_m, angle_deg) -> None:
    objects = sample_objects()
    target_id = [101]

    moved = position_torsion(objects, magnitude=level, track_ids=target_id, seed=7)
    shift = np.hypot(moved.x[0] - objects.x[0], moved.y[0] - objects.y[0])
    assert shift == pytest.approx(position_m)

    yawed = yaw_torsion(objects, magnitude=level, track_ids=target_id, seed=7)
    yaw_delta = abs((yawed.yaw[0] - objects.yaw[0] + np.pi) % (2.0 * np.pi) - np.pi)
    assert np.rad2deg(yaw_delta) == pytest.approx(angle_deg)

    rotated = velocity_direction_torsion(objects, magnitude=level, track_ids=target_id, seed=7)
    dot = float(objects.v[0] @ rotated.v[0])
    cross = float(objects.v[0, 0] * rotated.v[0, 1] - objects.v[0, 1] * rotated.v[0, 0])
    velocity_delta = abs(np.arctan2(cross, dot))
    assert np.rad2deg(velocity_delta) == pytest.approx(angle_deg)

    resolved = resolve_magnitude(level)
    assert resolved.position_shift_m == pytest.approx(position_m)
    assert resolved.yaw_shift_deg == pytest.approx(angle_deg)
    assert resolved.velocity_rotation_deg == pytest.approx(angle_deg)


def test_same_seed_identical_different_seed_changes_randomized_position() -> None:
    objects = sample_objects()
    a = position_torsion(objects, magnitude="medium", target="all", seed=123)
    b = position_torsion(objects, magnitude="medium", target="all", seed=123)
    c = position_torsion(objects, magnitude="medium", target="all", seed=456)

    assert np.allclose(a.x, b.x)
    assert np.allclose(a.y, b.y)
    assert not (np.allclose(a.x, c.x) and np.allclose(a.y, c.y))


def test_same_seed_identical_different_seed_changes_confidence() -> None:
    objects = sample_objects()
    a = confidence_redistribution(objects, magnitude="medium", target="all", seed=123)
    b = confidence_redistribution(objects, magnitude="medium", target="all", seed=123)
    c = confidence_redistribution(objects, magnitude="medium", target="all", seed=456)

    assert np.allclose(a.conf, b.conf)
    assert not np.allclose(a.conf, c.conf)


def test_velocity_direction_torsion_preserves_speed_magnitude() -> None:
    objects = sample_objects()
    out = velocity_direction_torsion(objects, magnitude="high", target="all", seed=5)

    assert np.allclose(np.linalg.norm(out.v, axis=1), np.linalg.norm(objects.v, axis=1))


def test_explicit_track_id_selector_only_changes_target() -> None:
    objects = sample_objects()
    out = yaw_torsion(objects, d_yaw_deg=5.0, track_ids=[102])

    assert out.yaw[0] == pytest.approx(objects.yaw[0])
    assert out.yaw[1] != pytest.approx(objects.yaw[1])
    assert out.yaw[2] == pytest.approx(objects.yaw[2])


def test_rejects_implausible_position_shift() -> None:
    objects = sample_objects()

    with pytest.raises(ValueError, match="position shift"):
        position_torsion(objects, dx=20.0, dy=0.0)
