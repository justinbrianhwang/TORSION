from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torsion.models.interfuser_wrapper import (
    BEV_HOOK_MODULE,
    HIGH_RES_BEV_HOOK_MODULE,
    build_synthetic_inputs,
    load_interfuser,
    output_shapes,
)
from torsion.operators.bev import bev_translate, bev_twist, gaussian_feature, random_warp_feature


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="InterFuser integration tests require CUDA",
)


@pytest.fixture(scope="module")
def interfuser():
    if not Path("weights/interfuser.pth").exists():
        pytest.skip("weights/interfuser.pth is not present")
    return load_interfuser(device="cuda")


@pytest.fixture(scope="module")
def synthetic_inputs(interfuser):
    return build_synthetic_inputs(batch_size=1, device=interfuser.device, seed=123)


def test_interfuser_model_builds_and_loads_cleanly(interfuser) -> None:
    report = interfuser.load_report

    assert report.state_dict_keys == 1132
    assert report.missing_count == 0
    assert report.unexpected_count == 0
    assert report.checkpoint_arch == "mvt_baseline"
    assert not interfuser.model.training
    assert next(interfuser.model.parameters()).is_cuda


def test_interfuser_forward_runs_on_gpu(interfuser, synthetic_inputs) -> None:
    outputs = interfuser.forward(synthetic_inputs)

    assert output_shapes(outputs) == {
        "traffic": (1, 400, 7),
        "waypoints": (1, 10, 2),
        "is_junction": (1, 2),
        "traffic_light_state": (1, 2),
        "stop_sign": (1, 2),
        "traffic_feature": (1, 400, 256),
    }
    assert all(tensor.is_cuda for tensor in outputs)


def test_bev_hook_replaces_feature_and_zero_hook_is_noop(interfuser, synthetic_inputs) -> None:
    clean_outputs = interfuser.forward(synthetic_inputs)
    noop = interfuser.run_with_bev_perturbation(synthetic_inputs, lambda feature: feature)
    replaced = interfuser.run_with_bev_perturbation(
        synthetic_inputs,
        lambda feature: torch.zeros_like(feature),
    )

    assert noop.hook_fired
    assert replaced.hook_fired
    assert noop.feature_shape == (1, 256, 7, 7)
    assert replaced.feature_shape == (1, 256, 7, 7)
    assert BEV_HOOK_MODULE == "lidar_patch_embed"
    assert all(torch.equal(clean, hooked) for clean, hooked in zip(clean_outputs, noop.outputs))
    assert torch.linalg.vector_norm(replaced.outputs[0] - clean_outputs[0]).item() > 0.0


def test_high_resolution_bev_hook_resolves_and_noop_is_noop(interfuser, synthetic_inputs) -> None:
    clean_outputs = interfuser.forward(synthetic_inputs)
    noop = interfuser.run_with_bev_perturbation(
        synthetic_inputs,
        lambda feature: feature,
        hook_module=HIGH_RES_BEV_HOOK_MODULE,
    )

    assert noop.hook_fired
    assert noop.feature_shape == (1, 128, 28, 28)
    assert all(torch.equal(clean, hooked) for clean, hooked in zip(clean_outputs, noop.outputs))


def test_bev_twist_preserves_shape_and_is_deterministic() -> None:
    feature = torch.arange(2 * 3 * 7 * 7, device="cuda", dtype=torch.float32).reshape(2, 3, 7, 7)

    out1 = bev_twist(feature, pivot=(3.0, 3.0), alpha=0.7, sigma=2.0)
    out2 = bev_twist(feature, pivot=(3.0, 3.0), alpha=0.7, sigma=2.0)

    assert out1.shape == feature.shape
    assert torch.allclose(out1, out2)


def test_new_bev_operators_preserve_shape_and_are_deterministic() -> None:
    feature = torch.arange(2 * 3 * 9 * 11, device="cuda", dtype=torch.float32).reshape(2, 3, 9, 11)

    translated1 = bev_translate(feature, shift_xy=(1.25, -0.75))
    translated2 = bev_translate(feature, shift_xy=(1.25, -0.75))
    gaussian1 = gaussian_feature(feature, sigma=0.2, seed=7)
    gaussian2 = gaussian_feature(feature, sigma=0.2, seed=7)
    random1 = random_warp_feature(feature, alpha=0.8, sigma=3.0, seed=11)
    random2 = random_warp_feature(feature, alpha=0.8, sigma=3.0, seed=11)

    assert translated1.shape == feature.shape
    assert gaussian1.shape == feature.shape
    assert random1.shape == feature.shape
    assert torch.allclose(translated1, translated2)
    assert torch.allclose(gaussian1, gaussian2)
    assert torch.allclose(random1, random2)
