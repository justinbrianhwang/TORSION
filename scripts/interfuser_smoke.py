from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torsion.models.interfuser_wrapper import build_synthetic_inputs, load_interfuser, output_shapes
from torsion.operators.bev import bev_twist


def main() -> int:
    parser = argparse.ArgumentParser(description="InterFuser GPU load/forward/hook smoke test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    wrapper = load_interfuser(device="cuda")
    inputs = build_synthetic_inputs(batch_size=args.batch_size, device=wrapper.device, seed=args.seed)

    clean_outputs = wrapper.forward(inputs)
    noop_result = wrapper.run_with_bev_perturbation(inputs, lambda feature: feature)
    twist_result = wrapper.run_with_bev_perturbation(
        inputs,
        lambda feature: bev_twist(feature, alpha=args.alpha, sigma=args.sigma),
    )

    clean_traffic = clean_outputs[0]
    noop_traffic = noop_result.outputs[0]
    twist_traffic = twist_result.outputs[0]
    zero_exact = all(torch.equal(clean, noop) for clean, noop in zip(clean_outputs, noop_result.outputs))
    zero_l2 = torch.linalg.vector_norm(noop_traffic - clean_traffic).item()
    twist_l2 = torch.linalg.vector_norm(twist_traffic - clean_traffic).item()

    report = wrapper.load_report
    print(
        "state_dict_load: "
        f"arch={report.checkpoint_arch} keys={report.state_dict_keys} "
        f"missing={report.missing_count} unexpected={report.unexpected_count}"
    )
    print("forward_outputs:")
    for name, shape in output_shapes(clean_outputs).items():
        print(f"  {name}: {shape}")
    print(f"bev_hook: module=lidar_patch_embed feature_shape={noop_result.feature_shape}")
    print(f"zero_hook: fired={noop_result.hook_fired} exact={zero_exact} traffic_l2={zero_l2:.12g}")
    print(
        "twist_hook: "
        f"fired={twist_result.hook_fired} feature_shape={twist_result.feature_shape} "
        f"traffic_l2={twist_l2:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
