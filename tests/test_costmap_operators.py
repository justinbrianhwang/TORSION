import numpy as np
import pytest

from torsion.operators.costmap import (
    directional_obstacle_inflation,
    false_free_space,
    false_wall,
    gaussian_cost_noise,
    lane_boundary_shear,
    obstacle_deflation,
    spatial_cost_warp,
    translate_cost_field,
)


def sample_grid() -> np.ndarray:
    grid = np.full((41, 41), 0.05, dtype=np.float64)
    grid[17:23, 25:31] = 0.82
    grid[10:12, 5:36] = 0.35
    return grid


def test_spatial_cost_warp_preserves_range_is_local_and_non_mutating() -> None:
    grid = sample_grid()
    before = grid.copy()

    warped = spatial_cost_warp(grid, pivot=(20.0, 20.0), alpha=0.8, sigma=4.0)
    repeat = spatial_cost_warp(grid, pivot=(20.0, 20.0), alpha=0.8, sigma=4.0)

    assert np.array_equal(grid, before)
    assert warped.min() >= 0.0
    assert warped.max() <= 1.0
    assert np.allclose(warped, repeat)
    assert warped[0, 0] == pytest.approx(grid[0, 0])
    assert warped[40, 40] == pytest.approx(grid[40, 40])
    assert not np.allclose(warped[14:28, 14:34], grid[14:28, 14:34])


def test_gaussian_inflation_and_deflation_are_local_deterministic_and_clipped() -> None:
    grid = sample_grid()
    grid[20, 20] = 0.95
    before = grid.copy()
    cov = np.array([[5.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    inflated = directional_obstacle_inflation(grid, center=(20.0, 20.0), beta=0.5, cov=cov)
    repeat = directional_obstacle_inflation(grid, center=(20.0, 20.0), beta=0.5, cov=cov)
    deflated = obstacle_deflation(grid, center=(20.0, 20.0), beta=0.5, cov=cov)

    assert np.array_equal(grid, before)
    assert inflated.min() >= 0.0
    assert inflated.max() <= 1.0
    assert deflated.min() >= 0.0
    assert deflated.max() <= 1.0
    assert np.allclose(inflated, repeat)
    assert inflated[0, 0] == pytest.approx(grid[0, 0])
    assert deflated[0, 0] == pytest.approx(grid[0, 0])
    assert inflated[20, 20] >= grid[20, 20]
    assert deflated[20, 20] <= grid[20, 20]


def test_lane_boundary_shear_and_false_regions_preserve_contract() -> None:
    grid = sample_grid()
    sheared = lane_boundary_shear(grid, s0=20.0, kappa=0.08, sigma=4.0)

    assert sheared.shape == grid.shape
    assert sheared.min() >= 0.0
    assert sheared.max() <= 1.0
    assert sheared[:, 0].tolist() == pytest.approx(grid[:, 0].tolist())
    assert not np.allclose(sheared[:, 15:26], grid[:, 15:26])

    free = false_free_space(grid, (18.0, 18.0, 22.0, 22.0))
    wall = false_wall(grid, (18.0, 18.0, 22.0, 22.0))

    assert free[20, 20] == pytest.approx(0.0)
    assert wall[20, 20] == pytest.approx(1.0)
    assert free[0, 0] == pytest.approx(grid[0, 0])
    assert wall[0, 0] == pytest.approx(grid[0, 0])


def test_cost_operators_preserve_fixed_boundary_cells() -> None:
    grid = sample_grid()
    fixed = np.zeros_like(grid, dtype=bool)
    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True
    fixed[10:12, 5:36] = True
    cov = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    noise = np.ones_like(grid, dtype=np.float64)

    outputs = (
        spatial_cost_warp(grid, pivot=(20.0, 20.0), alpha=0.8, sigma=4.0, fixed_mask=fixed),
        translate_cost_field(grid, shift_xy=(3.0, -2.0), fixed_mask=fixed),
        gaussian_cost_noise(grid, noise, scale=0.5, fixed_mask=fixed),
        directional_obstacle_inflation(
            grid, center=(20.0, 20.0), beta=0.5, cov=cov, fixed_mask=fixed
        ),
        obstacle_deflation(grid, center=(20.0, 20.0), beta=0.5, cov=cov, fixed_mask=fixed),
        lane_boundary_shear(grid, s0=20.0, kappa=0.08, sigma=4.0, fixed_mask=fixed),
        false_free_space(grid, (0.0, 0.0, 40.0, 40.0), fixed_mask=fixed),
        false_wall(grid, (0.0, 0.0, 40.0, 40.0), fixed_mask=fixed),
    )

    for out in outputs:
        assert np.array_equal(out[fixed], grid[fixed])
