"""Thin integration wrapper for the vendored InterFuser model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from collections.abc import Callable, Mapping

import torch


OUTPUT_NAMES = (
    "traffic",
    "waypoints",
    "is_junction",
    "traffic_light_state",
    "stop_sign",
    "traffic_feature",
)
BEV_HOOK_MODULE = "lidar_patch_embed"
HIGH_RES_BEV_HOOK_MODULE = "lidar_backbone.layer2"
BEV_HOOK_MODULES = (
    BEV_HOOK_MODULE,
    HIGH_RES_BEV_HOOK_MODULE,
)
BEV_HOOK_DOCUMENTED_SHAPES = {
    BEV_HOOK_MODULE: "B,256,7,7 projected LiDAR token grid",
    HIGH_RES_BEV_HOOK_MODULE: "B,128,28,28 LiDAR ResNet layer2 feature map",
}


@dataclass(frozen=True)
class StateDictLoadReport:
    """Summary of the checkpoint load operation."""

    checkpoint_path: Path
    checkpoint_arch: str | None
    state_dict_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    @property
    def missing_count(self) -> int:
        return len(self.missing_keys)

    @property
    def unexpected_count(self) -> int:
        return len(self.unexpected_keys)


@dataclass(frozen=True)
class BevHookResult:
    """Outputs and metadata from a BEV-feature perturbation forward pass."""

    outputs: tuple[torch.Tensor, ...]
    hook_fired: bool
    feature_shape: tuple[int, ...] | None
    clean_feature: torch.Tensor | None = None


class InterFuserWrapper:
    """Loaded InterFuser model plus the BEV perturbation hook."""

    def __init__(self, model: torch.nn.Module, load_report: StateDictLoadReport, device: torch.device):
        self.model = model
        self.load_report = load_report
        self.device = device

    @torch.inference_mode()
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(self.model(inputs))

    @torch.inference_mode()
    def run_with_bev_perturbation(
        self,
        inputs: Mapping[str, torch.Tensor],
        fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        hook_module: str = BEV_HOOK_MODULE,
        capture_feature: bool = False,
    ) -> BevHookResult:
        return run_with_bev_perturbation(
            self.model,
            inputs,
            fn,
            hook_module=hook_module,
            capture_feature=capture_feature,
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendored_interfuser_path(root: Path | None = None) -> Path:
    root = repo_root() if root is None else root
    return root / "third_party" / "InterFuser" / "interfuser"


def checkpoint_path(root: Path | None = None) -> Path:
    root = repo_root() if root is None else root
    return root / "weights" / "interfuser.pth"


def load_interfuser(
    *,
    root: Path | None = None,
    weights_path: Path | None = None,
    device: str | torch.device = "cuda",
    strict: bool = True,
) -> InterFuserWrapper:
    """Build ``interfuser_baseline()``, load local weights, and move to GPU eval mode."""

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to load InterFuser on GPU")

    root = repo_root() if root is None else Path(root)
    weights_path = checkpoint_path(root) if weights_path is None else Path(weights_path)

    _ensure_vendored_timm_namespace(root)
    _disable_vendored_pretrained_downloads()

    from timm.models.interfuser import interfuser_baseline

    model = interfuser_baseline()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    report = StateDictLoadReport(
        checkpoint_path=weights_path,
        checkpoint_arch=checkpoint.get("arch"),
        state_dict_keys=len(state_dict),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
    if strict and (report.missing_keys or report.unexpected_keys):
        raise RuntimeError(
            "InterFuser checkpoint did not load cleanly: "
            f"{report.missing_count} missing, {report.unexpected_count} unexpected"
        )

    model.to(target_device)
    model.eval()
    return InterFuserWrapper(model=model, load_report=report, device=target_device)


def build_synthetic_inputs(
    *,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 0,
    speed: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Create a valid InterFuser batch with deterministic synthetic tensors."""

    target_device = torch.device(device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)

    def randn(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=target_device) * scale

    rgb = randn((batch_size, 3, 224, 224), 0.05)
    rgb_left = randn((batch_size, 3, 128, 128), 0.05)
    rgb_right = randn((batch_size, 3, 128, 128), 0.05)
    rgb_center = randn((batch_size, 3, 128, 128), 0.05)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 224, device=target_device),
        torch.linspace(-1.0, 1.0, 224, device=target_device),
        indexing="ij",
    )
    obstacle1_x = -0.45 + 0.30 * torch.rand((batch_size, 1, 1), generator=generator, device=target_device)
    obstacle1_y = -0.25 + 0.50 * torch.rand((batch_size, 1, 1), generator=generator, device=target_device)
    obstacle2_x = 0.05 + 0.45 * torch.rand((batch_size, 1, 1), generator=generator, device=target_device)
    obstacle2_y = -0.35 + 0.70 * torch.rand((batch_size, 1, 1), generator=generator, device=target_device)
    lidar = torch.stack(
        (
            torch.exp(-8.0 * ((xx.unsqueeze(0) - obstacle1_x).square() + (yy.unsqueeze(0) - obstacle1_y).square())),
            torch.exp(-12.0 * ((xx.unsqueeze(0) - obstacle2_x).square() + (yy.unsqueeze(0) - obstacle2_y).square())),
            ((xx.unsqueeze(0) + 1.0) * 0.5).expand(batch_size, -1, -1),
        ),
        dim=1,
    )
    lidar = (lidar + randn((batch_size, 3, 224, 224), 0.01)).clamp(0.0, 1.0)

    measurements = torch.zeros(batch_size, 7, device=target_device)
    measurements[:, 3] = 1.0
    measurements[:, 6] = float(speed)
    target_point = torch.tensor([[10.0, 0.0]], device=target_device).repeat(batch_size, 1)

    return {
        "rgb": rgb,
        "rgb_left": rgb_left,
        "rgb_right": rgb_right,
        "rgb_center": rgb_center,
        "measurements": measurements,
        "target_point": target_point,
        "lidar": lidar,
    }


def outputs_to_dict(outputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    if len(outputs) != len(OUTPUT_NAMES):
        raise ValueError(f"expected {len(OUTPUT_NAMES)} outputs, got {len(outputs)}")
    return dict(zip(OUTPUT_NAMES, outputs, strict=True))


def output_shapes(outputs: tuple[torch.Tensor, ...]) -> dict[str, tuple[int, ...]]:
    return {name: tuple(tensor.shape) for name, tensor in outputs_to_dict(outputs).items()}


@torch.inference_mode()
def run_with_bev_perturbation(
    model: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    hook_module: str = BEV_HOOK_MODULE,
    capture_feature: bool = False,
) -> BevHookResult:
    """Forward InterFuser while replacing a LiDAR BEV feature with ``fn(feature)``.

    Supported hook modules:
      * ``lidar_patch_embed``: projected LiDAR token grid, normally ``B,256,7,7``.
      * ``lidar_backbone.layer2``: earlier LiDAR ResNet feature, normally ``B,128,28,28``.
    """

    hook_state: dict[str, object] = {"fired": False, "shape": None, "feature": None}

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: object) -> object:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise RuntimeError(f"{hook_module} output is not the expected (feature, global) tuple")
            feature, global_feature = output
        else:
            feature = output
            global_feature = None
        if not torch.is_tensor(feature) or feature.ndim != 4:
            raise RuntimeError(f"{hook_module} feature is not a B,C,H,W tensor")
        hook_state["fired"] = True
        hook_state["shape"] = tuple(feature.shape)
        if capture_feature:
            hook_state["feature"] = feature.detach().clone()

        replacement = fn(feature)
        if not torch.is_tensor(replacement):
            raise TypeError("BEV perturbation function must return a tensor")
        if replacement.shape != feature.shape:
            raise ValueError(
                f"BEV perturbation changed shape from {tuple(feature.shape)} to {tuple(replacement.shape)}"
            )
        if replacement is feature:
            return output
        if global_feature is None:
            return replacement
        replacement_global = replacement.mean(dim=(2, 3), keepdim=False)[:, :, None]
        return replacement, replacement_global

    module = _get_model_submodule(model, hook_module)
    if module is None:
        raise RuntimeError(f"InterFuser module {hook_module!r} was not found")

    handle = module.register_forward_hook(hook)
    try:
        outputs = tuple(model(inputs))
    finally:
        handle.remove()

    return BevHookResult(
        outputs=outputs,
        hook_fired=bool(hook_state["fired"]),
        feature_shape=hook_state["shape"],  # type: ignore[arg-type]
        clean_feature=hook_state["feature"],  # type: ignore[arg-type]
    )


def _get_model_submodule(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    if hasattr(model, "get_submodule"):
        try:
            return model.get_submodule(name)
        except AttributeError:
            pass
    return dict(model.named_modules()).get(name)


def _ensure_vendored_timm_namespace(root: Path) -> None:
    """Expose the vendored model modules without importing dataset-only extras."""

    vendor = vendored_interfuser_path(root)
    timm_dir = vendor / "timm"
    models_dir = timm_dir / "models"
    if not timm_dir.exists():
        raise FileNotFoundError(f"vendored timm path does not exist: {timm_dir}")
    vendor_str = str(vendor)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

    existing = sys.modules.get("timm")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "") or ".").resolve()
        if timm_dir.resolve() not in existing_file.parents and existing_file != timm_dir / "__init__.py":
            raise RuntimeError(f"non-vendored timm is already imported from {existing_file}")

    timm_pkg = sys.modules.get("timm") or types.ModuleType("timm")
    timm_pkg.__file__ = str(timm_dir / "__init__.py")
    timm_pkg.__path__ = [str(timm_dir)]  # type: ignore[attr-defined]
    timm_pkg.__version__ = _read_vendored_timm_version(timm_dir)  # type: ignore[attr-defined]
    sys.modules["timm"] = timm_pkg

    models_pkg = sys.modules.get("timm.models") or types.ModuleType("timm.models")
    models_pkg.__file__ = str(models_dir / "__init__.py")
    models_pkg.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules["timm.models"] = models_pkg

    data_pkg = sys.modules.get("timm.data") or types.ModuleType("timm.data")
    data_pkg.IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)  # type: ignore[attr-defined]
    data_pkg.IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)  # type: ignore[attr-defined]
    sys.modules["timm.data"] = data_pkg

    constants_pkg = sys.modules.get("timm.data.constants") or types.ModuleType("timm.data.constants")
    constants_pkg.IMAGENET_DEFAULT_MEAN = data_pkg.IMAGENET_DEFAULT_MEAN  # type: ignore[attr-defined]
    constants_pkg.IMAGENET_DEFAULT_STD = data_pkg.IMAGENET_DEFAULT_STD  # type: ignore[attr-defined]
    sys.modules["timm.data.constants"] = constants_pkg


def _read_vendored_timm_version(timm_dir: Path) -> str:
    version_file = timm_dir / "version.py"
    namespace: dict[str, str] = {}
    exec(version_file.read_text(encoding="utf-8"), namespace)
    return namespace.get("__version__", "vendored")


def _disable_vendored_pretrained_downloads() -> None:
    import timm.models.helpers as helpers

    def skip_pretrained(*_args: object, **_kwargs: object) -> None:
        return None

    helpers.load_pretrained = skip_pretrained
