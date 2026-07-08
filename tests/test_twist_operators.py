import numpy as np
import pytest

from torsion.operators.object import ObjectSet
from torsion.operators.twist import (
    scene_swirl_torsion,
    temporal_curl_angle,
    temporal_curl_torsion,
    twist_angles,
    twist_grid_inverse,
    twist_points,
)


def sample_objects() -> ObjectSet:
    return ObjectSet(
        x=[0.0, 2.0, 4.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        w=[2.0, 2.0, 0.8],
        h=[1.5, 1.5, 1.7],
        l=[4.0, 4.0, 0.8],
        yaw=[0.0, 0.1, -0.2],
        v=[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        cls=["vehicle", "vehicle", "pedestrian"],
        conf=[0.9, 0.8, 0.7],
        track_id=["egoish", "target", "ped"],
    )


def assert_contract_preserved(before: ObjectSet, after: ObjectSet) -> None:
    assert len(after) == len(before)
    assert np.array_equal(after.cls, before.cls)
    assert np.array_equal(after.track_id, before.track_id)
    assert np.allclose(after.bbox_size, before.bbox_size)


def test_twist_angle_varies_and_has_compactish_support() -> None:
    sigma = 2.0
    points = np.array(
        [
            [0.0, 0.0],
            [sigma, 0.0],
            [2.0 * sigma, 0.0],
            [8.0 * sigma, 0.0],
        ],
        dtype=np.float64,
    )

    theta = twist_angles(points, (0.0, 0.0), alpha=1.0, sigma=sigma)

    assert theta[0] == pytest.approx(0.0)
    assert theta[1] > theta[2]
    assert theta[1] != pytest.approx(theta[2])
    assert abs(theta[3]) < 1e-10


def test_twist_points_are_not_a_uniform_rotation() -> None:
    points = np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64)

    out = twist_points(points, (0.0, 0.0), alpha=0.5, sigma=2.0)
    local_angles = np.arctan2(out[:, 1], out[:, 0])

    assert local_angles[0] != pytest.approx(local_angles[1])


def test_twist_grid_inverse_recovers_source_coordinates() -> None:
    points = np.array([[[2.0, 0.0], [0.0, 2.0]], [[4.0, 0.0], [1.0, 1.0]]])
    warped = twist_points(points, (0.0, 0.0), alpha=0.4, sigma=2.5)

    recovered = twist_grid_inverse(warped, (0.0, 0.0), alpha=0.4, sigma=2.5)

    assert recovered == pytest.approx(points)


def test_scene_swirl_preserves_contract_and_rotates_attached_vectors_locally() -> None:
    objects = sample_objects()
    out = scene_swirl_torsion(
        objects,
        pivot=(0.0, 0.0),
        alpha=0.5,
        sigma=2.0,
        target="all",
    )
    repeat = scene_swirl_torsion(
        objects,
        pivot=(0.0, 0.0),
        alpha=0.5,
        sigma=2.0,
        target="all",
    )
    theta = twist_angles(np.array([[2.0, 0.0]], dtype=np.float64), (0.0, 0.0), 0.5, 2.0)[0]

    assert_contract_preserved(objects, out)
    assert np.allclose(out.xy, repeat.xy)
    assert np.allclose(out.yaw, repeat.yaw)
    assert np.allclose(out.v, repeat.v)
    assert np.allclose(objects.x, [0.0, 2.0, 4.0])
    assert out.yaw[0] == pytest.approx(objects.yaw[0])
    assert out.yaw[1] == pytest.approx(objects.yaw[1] + theta)
    assert out.v[1, 0] == pytest.approx(2.0 * np.cos(theta))
    assert out.v[1, 1] == pytest.approx(2.0 * np.sin(theta))


def test_temporal_curl_angle_increases_within_window_and_zero_outside() -> None:
    alpha = 0.5

    assert temporal_curl_angle(-1, alpha, 5) == pytest.approx(0.0)
    assert temporal_curl_angle(0, alpha, 5) == pytest.approx(0.0)
    assert temporal_curl_angle(1, alpha, 5) == pytest.approx(0.1)
    assert temporal_curl_angle(4, alpha, 5) == pytest.approx(0.4)
    assert temporal_curl_angle(5, alpha, 5) == pytest.approx(0.0)


def test_temporal_curl_torsion_rotates_target_velocity_only_inside_window() -> None:
    objects = sample_objects()

    out = temporal_curl_torsion(
        objects,
        target="track_ids",
        track_ids=["target"],
        alpha=0.5,
        T_window=5,
        t_offset=3,
    )
    outside = temporal_curl_torsion(
        objects,
        target="track_ids",
        track_ids=["target"],
        alpha=0.5,
        T_window=5,
        t_offset=5,
    )
    theta = temporal_curl_angle(3, 0.5, 5)

    assert_contract_preserved(objects, out)
    assert out.v[1, 0] == pytest.approx(2.0 * np.cos(theta))
    assert out.v[1, 1] == pytest.approx(2.0 * np.sin(theta))
    assert np.allclose(outside.xy, objects.xy)
    assert np.allclose(outside.yaw, objects.yaw)
    assert np.allclose(outside.v, objects.v)
