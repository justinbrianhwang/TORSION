import pytest

from torsion.operators.temporal import is_active


def test_single_frame_activation() -> None:
    assert not is_active("single-frame", 9, start_frame=10)
    assert is_active("single-frame", 10, start_frame=10)
    assert not is_active("single-frame", 11, start_frame=10)


def test_burst_activation_start_inclusive_end_exclusive() -> None:
    active = [is_active("burst", frame, start_frame=10, duration=3) for frame in range(8, 15)]
    assert active == [False, False, True, True, True, False, False]


def test_persistent_activation() -> None:
    assert not is_active("persistent", 4, start_frame=5)
    assert is_active("persistent", 5, start_frame=5)
    assert is_active("persistent", 500, start_frame=5)


@pytest.mark.parametrize("pattern", ["drift", "actor-locked"])
def test_future_temporal_patterns_are_todo(pattern) -> None:
    with pytest.raises(NotImplementedError):
        is_active(pattern, 10, start_frame=0, duration=5)
