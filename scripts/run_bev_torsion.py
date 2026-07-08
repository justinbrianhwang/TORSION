"""Phase 4b BEV-feature torsion experiment for vendored InterFuser.

This is an open-loop feature-sensitivity study. It does not simulate CARLA
closed-loop behavior; it measures downstream InterFuser output changes caused
by matched feature-space perturbation budgets.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torsion.models.interfuser_wrapper import (
    BEV_HOOK_DOCUMENTED_SHAPES,
    BEV_HOOK_MODULE,
    HIGH_RES_BEV_HOOK_MODULE,
    InterFuserWrapper,
    build_synthetic_inputs,
    load_interfuser,
    outputs_to_dict,
)
from torsion.operators.bev import bev_translate, bev_twist, gaussian_feature, random_warp_feature


HOOKS = (
    ("patch7", BEV_HOOK_MODULE),
    ("layer2_28", HIGH_RES_BEV_HOOK_MODULE),
)
OPERATORS = ("bev_twist", "bev_translate", "gaussian_feature", "random_warp_feature")
LEVEL_FRACTIONS = {"low": 0.01, "medium": 0.03, "high": 0.06}
DEFAULT_INPUT_SHIFT = (16.0, 0.0)


@dataclass
class Case:
    seed: int
    inputs: dict[str, torch.Tensor]
    clean_outputs: tuple[torch.Tensor, ...]
    feature: torch.Tensor


@dataclass
class Calibration:
    strength: float
    perturbed_feature: torch.Tensor
    realized_budget: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-inputs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--input-shift-x", type=float, default=DEFAULT_INPUT_SHIFT[0])
    parser.add_argument("--input-shift-y", type=float, default=DEFAULT_INPUT_SHIFT[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_inputs < 20:
        raise ValueError("--num-inputs must be at least 20 for the fair comparison")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the InterFuser BEV experiment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    wrapper = load_interfuser(device=args.device)

    summary: dict[str, Any] = {
        "num_inputs": args.num_inputs,
        "seed_start": args.seed_start,
        "budget_fractions": LEVEL_FRACTIONS,
        "hooks": {},
        "equivariance": {},
        "notes": [
            "Open-loop InterFuser feature-sensitivity study.",
            "Feature budgets are matched by ||perturbed_feature - clean_feature||_2.",
            "No closed-loop CARLA TTC or intervention metric is reported.",
        ],
    }

    all_records: list[dict[str, Any]] = []
    cases_by_hook: dict[str, list[Case]] = {}
    budgets_by_hook: dict[str, dict[str, float]] = {}

    for hook_label, hook_module in HOOKS:
        cases = collect_cases(wrapper, hook_module, args.num_inputs, args.seed_start)
        cases_by_hook[hook_label] = cases
        feature_norms = torch.tensor(
            [torch.linalg.vector_norm(case.feature).item() for case in cases],
            dtype=torch.float64,
        )
        median_norm = float(torch.median(feature_norms).item())
        level_budgets = {level: fraction * median_norm for level, fraction in LEVEL_FRACTIONS.items()}
        budgets_by_hook[hook_label] = level_budgets

        records = run_hook_comparison(hook_label, hook_module, wrapper, cases, level_budgets)
        all_records.extend(records)
        summary["hooks"][hook_label] = {
            "module": hook_module,
            "documented_shape": BEV_HOOK_DOCUMENTED_SHAPES[hook_module],
            "observed_shape": list(cases[0].feature.shape),
            "median_clean_feature_l2": median_norm,
            "target_budgets": level_budgets,
            "records": summarize_records(records),
            "directedness": directedness_summary(records),
        }

        summary["equivariance"][hook_label] = translation_equivariance_probe(
            wrapper,
            hook_module,
            cases,
            input_shift_xy=(float(args.input_shift_x), float(args.input_shift_y)),
        )

    figure10 = args.output_dir / "figure10_bev_feature_overlay.png"
    figure11 = args.output_dir / "figure11_bev_sensitivity.png"
    make_overlay_figure(
        wrapper,
        cases_by_hook["layer2_28"][0],
        HIGH_RES_BEV_HOOK_MODULE,
        budgets_by_hook["layer2_28"]["medium"],
        figure10,
    )
    make_sensitivity_figure(all_records, figure11)

    summary["figures"] = {
        "figure10_bev_feature_overlay": str(figure10),
        "figure11_bev_sensitivity": str(figure11),
    }

    json_path = args.output_dir / "bev_feature_torsion_phase4b.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    print_summary_table(summary)


def collect_cases(
    wrapper: InterFuserWrapper,
    hook_module: str,
    num_inputs: int,
    seed_start: int,
) -> list[Case]:
    cases: list[Case] = []
    for offset in range(num_inputs):
        seed = seed_start + offset
        inputs = build_synthetic_inputs(batch_size=1, device=wrapper.device, seed=seed)
        result = wrapper.run_with_bev_perturbation(
            inputs,
            lambda feature: feature,
            hook_module=hook_module,
            capture_feature=True,
        )
        if not result.hook_fired or result.clean_feature is None:
            raise RuntimeError(f"hook {hook_module!r} did not capture a feature")
        cases.append(
            Case(
                seed=seed,
                inputs=inputs,
                clean_outputs=tuple(output.detach().clone() for output in result.outputs),
                feature=result.clean_feature,
            )
        )
    return cases


def run_hook_comparison(
    hook_label: str,
    hook_module: str,
    wrapper: InterFuserWrapper,
    cases: list[Case],
    level_budgets: dict[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        for operator_index, operator_name in enumerate(OPERATORS):
            operator_seed = 100_000 + 997 * case.seed + operator_index
            for level, target_budget in level_budgets.items():
                calibration = calibrate_operator(
                    case.feature,
                    operator_name,
                    target_budget,
                    seed=operator_seed,
                )
                result = wrapper.run_with_bev_perturbation(
                    case.inputs,
                    lambda feature, name=operator_name, strength=calibration.strength, seed=operator_seed: apply_operator(
                        feature,
                        name,
                        strength,
                        seed=seed,
                    ),
                    hook_module=hook_module,
                )
                metrics = output_delta_metrics(case.clean_outputs, result.outputs)
                budget = max(calibration.realized_budget, 1e-12)
                records.append(
                    {
                        "hook": hook_label,
                        "hook_module": hook_module,
                        "seed": case.seed,
                        "operator": operator_name,
                        "level": level,
                        "target_feature_budget": target_budget,
                        "realized_feature_budget": calibration.realized_budget,
                        "strength": calibration.strength,
                        "waypoint_l2": metrics["waypoint_l2"],
                        "traffic_l2": metrics["traffic_l2"],
                        "waypoint_per_budget": metrics["waypoint_l2"] / budget,
                        "traffic_per_budget": metrics["traffic_l2"] / budget,
                    }
                )
    return records


def calibrate_operator(
    feature: torch.Tensor,
    operator_name: str,
    target_budget: float,
    *,
    seed: int,
) -> Calibration:
    if target_budget <= 0.0:
        return Calibration(0.0, feature.clone(), 0.0)

    high = initial_strength(operator_name, feature, target_budget)
    max_strength = max_strength_for(operator_name, feature)
    low = 0.0
    best_feature = apply_operator(feature, operator_name, high, seed=seed)
    best_budget = feature_delta_l2(feature, best_feature)

    for _ in range(24):
        if best_budget >= target_budget or high >= max_strength:
            break
        low = high
        high = min(high * 2.0, max_strength)
        best_feature = apply_operator(feature, operator_name, high, seed=seed)
        best_budget = feature_delta_l2(feature, best_feature)

    if best_budget < target_budget:
        return Calibration(high, best_feature, best_budget)

    for _ in range(24):
        mid = 0.5 * (low + high)
        candidate = apply_operator(feature, operator_name, mid, seed=seed)
        budget = feature_delta_l2(feature, candidate)
        if budget < target_budget:
            low = mid
        else:
            high = mid
            best_feature = candidate
            best_budget = budget

    return Calibration(high, best_feature, best_budget)


def apply_operator(
    feature: torch.Tensor,
    operator_name: str,
    strength: float,
    *,
    seed: int,
) -> torch.Tensor:
    height, width = feature.shape[-2:]
    spatial_sigma = 0.35 * float(max(height, width))
    if operator_name == "bev_twist":
        return bev_twist(
            feature,
            pivot=((width - 1) / 2.0, (height - 1) / 2.0),
            alpha=float(strength),
            sigma=spatial_sigma,
        )
    if operator_name == "bev_translate":
        return bev_translate(feature, shift_xy=(float(strength), 0.0))
    if operator_name == "gaussian_feature":
        return gaussian_feature(feature, sigma=float(strength), seed=seed)
    if operator_name == "random_warp_feature":
        return random_warp_feature(feature, alpha=float(strength), sigma=spatial_sigma, seed=seed)
    raise ValueError(f"unknown operator {operator_name!r}")


def initial_strength(operator_name: str, feature: torch.Tensor, target_budget: float) -> float:
    if operator_name == "gaussian_feature":
        return max(target_budget / math.sqrt(float(feature.numel())), 1e-6)
    if operator_name == "bev_translate":
        return 0.05
    return 0.05


def max_strength_for(operator_name: str, feature: torch.Tensor) -> float:
    if operator_name == "gaussian_feature":
        return 100.0
    if operator_name == "bev_translate":
        return 0.75 * float(max(feature.shape[-2:]))
    return 32.0


def feature_delta_l2(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(perturbed - clean).item())


def output_delta_metrics(
    clean_outputs: tuple[torch.Tensor, ...],
    perturbed_outputs: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    clean = outputs_to_dict(clean_outputs)
    perturbed = outputs_to_dict(perturbed_outputs)
    return {
        "waypoint_l2": float(torch.linalg.vector_norm(perturbed["waypoints"] - clean["waypoints"]).item()),
        "traffic_l2": float(torch.linalg.vector_norm(perturbed["traffic"] - clean["traffic"]).item()),
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["level"], record["operator"])].append(record)

    out: list[dict[str, Any]] = []
    for level in LEVEL_FRACTIONS:
        for operator_name in OPERATORS:
            rows = grouped[(level, operator_name)]
            out.append(
                {
                    "level": level,
                    "operator": operator_name,
                    "realized_feature_budget": mean_std(rows, "realized_feature_budget"),
                    "waypoint_l2": mean_std(rows, "waypoint_l2"),
                    "traffic_l2": mean_std(rows, "traffic_l2"),
                    "waypoint_per_budget": mean_std(rows, "waypoint_per_budget"),
                    "traffic_per_budget": mean_std(rows, "traffic_per_budget"),
                    "strength": mean_std(rows, "strength"),
                }
            )
    return out


def mean_std(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)
    std = float(values.std(unbiased=True).item()) if len(values) > 1 else 0.0
    return {"mean": float(values.mean().item()), "std": std}


def directedness_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = summarize_records(records)
    by_level_operator = {(row["level"], row["operator"]): row for row in summary}
    out: list[dict[str, Any]] = []
    for level in LEVEL_FRACTIONS:
        translate = by_level_operator[(level, "bev_translate")]["waypoint_per_budget"]["mean"]
        twist = by_level_operator[(level, "bev_twist")]["waypoint_per_budget"]["mean"]
        random_warp = by_level_operator[(level, "random_warp_feature")]["waypoint_per_budget"]["mean"]
        out.append(
            {
                "level": level,
                "translate_over_twist_waypoint_per_budget": translate / max(twist, 1e-12),
                "translate_over_random_warp_waypoint_per_budget": translate / max(random_warp, 1e-12),
            }
        )
    return out


def translation_equivariance_probe(
    wrapper: InterFuserWrapper,
    hook_module: str,
    cases: list[Case],
    *,
    input_shift_xy: tuple[float, float],
) -> dict[str, Any]:
    waypoint_errors: list[float] = []
    traffic_errors: list[float] = []
    waypoint_input_changes: list[float] = []
    traffic_input_changes: list[float] = []

    for case in cases:
        shifted_inputs = {key: value for key, value in case.inputs.items()}
        shifted_inputs["lidar"] = bev_translate(case.inputs["lidar"], shift_xy=input_shift_xy)
        input_shift_outputs = wrapper.forward(shifted_inputs)

        feature_shift_xy = (
            input_shift_xy[0] * case.feature.shape[-1] / case.inputs["lidar"].shape[-1],
            input_shift_xy[1] * case.feature.shape[-2] / case.inputs["lidar"].shape[-2],
        )
        feature_shift_outputs = wrapper.run_with_bev_perturbation(
            case.inputs,
            lambda feature, shift=feature_shift_xy: bev_translate(feature, shift_xy=shift),
            hook_module=hook_module,
        ).outputs

        clean = outputs_to_dict(case.clean_outputs)
        input_shift = outputs_to_dict(input_shift_outputs)
        feature_shift = outputs_to_dict(feature_shift_outputs)
        waypoint_input_delta = input_shift["waypoints"] - clean["waypoints"]
        waypoint_feature_delta = feature_shift["waypoints"] - clean["waypoints"]
        traffic_input_delta = input_shift["traffic"] - clean["traffic"]
        traffic_feature_delta = feature_shift["traffic"] - clean["traffic"]

        waypoint_den = float(torch.linalg.vector_norm(waypoint_input_delta).item())
        traffic_den = float(torch.linalg.vector_norm(traffic_input_delta).item())
        waypoint_errors.append(
            float(torch.linalg.vector_norm(waypoint_feature_delta - waypoint_input_delta).item())
            / max(waypoint_den, 1e-12)
        )
        traffic_errors.append(
            float(torch.linalg.vector_norm(traffic_feature_delta - traffic_input_delta).item())
            / max(traffic_den, 1e-12)
        )
        waypoint_input_changes.append(waypoint_den)
        traffic_input_changes.append(traffic_den)

    return {
        "input_shift_xy_lidar_pixels": list(input_shift_xy),
        "waypoint_relative_error": list_mean_std(waypoint_errors),
        "traffic_relative_error": list_mean_std(traffic_errors),
        "waypoint_input_shift_l2": list_mean_std(waypoint_input_changes),
        "traffic_input_shift_l2": list_mean_std(traffic_input_changes),
    }


def list_mean_std(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    std = float(tensor.std(unbiased=True).item()) if len(values) > 1 else 0.0
    return {"mean": float(tensor.mean().item()), "std": std}


def make_overlay_figure(
    wrapper: InterFuserWrapper,
    case: Case,
    hook_module: str,
    target_budget: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    twist = calibrate_operator(case.feature, "bev_twist", target_budget, seed=77)
    translate = calibrate_operator(case.feature, "bev_translate", target_budget, seed=78)
    twist_outputs = wrapper.run_with_bev_perturbation(
        case.inputs,
        lambda feature: apply_operator(feature, "bev_twist", twist.strength, seed=77),
        hook_module=hook_module,
    ).outputs
    translate_outputs = wrapper.run_with_bev_perturbation(
        case.inputs,
        lambda feature: apply_operator(feature, "bev_translate", translate.strength, seed=78),
        hook_module=hook_module,
    ).outputs

    feature = case.feature[0]
    channel = int(feature.flatten(1).std(dim=1).argmax().item())
    maps = [
        ("clean", feature[channel]),
        ("twisted", twist.perturbed_feature[0, channel]),
        ("translated", translate.perturbed_feature[0, channel]),
    ]
    vmin = min(float(item[1].min().item()) for item in maps)
    vmax = max(float(item[1].max().item()) for item in maps)

    clean_wp = outputs_to_dict(case.clean_outputs)["waypoints"][0].detach().cpu()
    twist_wp = outputs_to_dict(twist_outputs)["waypoints"][0].detach().cpu()
    translate_wp = outputs_to_dict(translate_outputs)["waypoints"][0].detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (title, image) in zip(axes[0], maps, strict=True):
        ax.imshow(image.detach().cpu(), cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"{title} ch {channel}")
        ax.set_xticks([])
        ax.set_yticks([])

    plot_waypoints(axes[1, 0], clean_wp, twist_wp, "clean vs twist", "twist")
    plot_waypoints(axes[1, 1], clean_wp, translate_wp, "clean vs translate", "translate")
    axes[1, 2].plot(clean_wp[:, 0], clean_wp[:, 1], "o-", label="clean", linewidth=2)
    axes[1, 2].plot(twist_wp[:, 0], twist_wp[:, 1], "o-", label="twist")
    axes[1, 2].plot(translate_wp[:, 0], translate_wp[:, 1], "o-", label="translate")
    axes[1, 2].set_title("waypoints")
    axes[1, 2].axis("equal")
    axes[1, 2].grid(True, alpha=0.25)
    axes[1, 2].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_waypoints(ax: Any, clean_wp: torch.Tensor, perturbed_wp: torch.Tensor, title: str, label: str) -> None:
    ax.plot(clean_wp[:, 0], clean_wp[:, 1], "o-", label="clean", linewidth=2)
    ax.plot(perturbed_wp[:, 0], perturbed_wp[:, 1], "o-", label=label)
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def make_sensitivity_figure(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, hook in zip(axes, ("patch7", "layer2_28"), strict=True):
        hook_records = [record for record in records if record["hook"] == hook]
        summary = summarize_records(hook_records)
        for operator_name in OPERATORS:
            rows = [row for row in summary if row["operator"] == operator_name]
            x = [row["realized_feature_budget"]["mean"] for row in rows]
            y = [row["waypoint_l2"]["mean"] for row in rows]
            yerr = [row["waypoint_l2"]["std"] for row in rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=operator_name)
        ax.set_title(hook)
        ax.set_xlabel("feature budget ||dZ||2")
        ax.set_ylabel("waypoint L2 change")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary_table(summary: dict[str, Any]) -> None:
    for hook_label, hook in summary["hooks"].items():
        print(f"\n{hook_label} ({hook['module']}, observed {hook['observed_shape']})")
        print("level,operator,wp_per_budget_mean,wp_per_budget_std,traffic_per_budget_mean,traffic_per_budget_std")
        for row in hook["records"]:
            wp = row["waypoint_per_budget"]
            traffic = row["traffic_per_budget"]
            print(
                f"{row['level']},{row['operator']},"
                f"{wp['mean']:.8g},{wp['std']:.8g},"
                f"{traffic['mean']:.8g},{traffic['std']:.8g}"
            )
        eq = summary["equivariance"][hook_label]
        print(
            "equivariance waypoint_rel_error="
            f"{eq['waypoint_relative_error']['mean']:.8g}+/-{eq['waypoint_relative_error']['std']:.8g}, "
            "traffic_rel_error="
            f"{eq['traffic_relative_error']['mean']:.8g}+/-{eq['traffic_relative_error']['std']:.8g}"
        )


if __name__ == "__main__":
    main()
