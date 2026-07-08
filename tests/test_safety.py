import numpy as np
import pytest

from torsion.metrics.safety import has_bbox_collision, min_actor_distance, min_ttc
from torsion.operators.object import ObjectSet


def safety_objects() -> ObjectSet:
    return ObjectSet(
        x=[10.0, 50.0],
        y=[0.0, 10.0],
        z=[0.0, 0.0],
        w=[2.0, 2.0],
        h=[1.5, 1.5],
        l=[4.0, 4.0],
        yaw=[0.0, 0.0],
        v=[[0.0, 0.0], [0.0, 0.0]],
        cls=["vehicle", "vehicle"],
        conf=[0.9, 0.8],
        track_id=[1, 2],
    )


def test_min_actor_distance_center_distance() -> None:
    assert min_actor_distance((0.0, 0.0), safety_objects()) == pytest.approx(10.0)


def test_min_ttc_constant_velocity() -> None:
    ttc = min_ttc(
        (0.0, 0.0),
        (5.0, 0.0),
        safety_objects(),
        ego_width=2.0,
        ego_length=4.0,
        track_ids=[1],
    )
    expected_radius = np.sqrt(2.0**2 + 4.0**2)
    assert ttc == pytest.approx((10.0 - expected_radius) / 5.0)


def test_bbox_collision_overlap() -> None:
    objects = safety_objects().replace(x=[1.0, 50.0], y=[0.0, 10.0])

    assert has_bbox_collision(0.0, 0.0, 0.0, 2.0, 4.0, objects)
    assert not has_bbox_collision(0.0, 0.0, 0.0, 2.0, 4.0, safety_objects(), track_ids=[2])
