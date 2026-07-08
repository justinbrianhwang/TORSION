"""BEV feature torsion operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bev_translate(
    Z: torch.Tensor,
    shift_xy: tuple[float, float],
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Shift a BEV feature grid in pixel coordinates with border padding.

    ``shift_xy`` is ``(dx, dy)`` in feature-grid pixels. Positive ``dx`` moves
    feature content to the right; positive ``dy`` moves feature content down.
    """

    tensor, squeeze_batch = _as_bchw(Z)
    _, _, height, width = tensor.shape
    dx = float(shift_xy[0])
    dy = float(shift_xy[1])
    if dx == 0.0 and dy == 0.0:
        out = tensor.clone()
        return out.squeeze(0) if squeeze_batch else out

    yy, xx = _pixel_grid(height, width, tensor)
    src_x = xx - dx
    src_y = yy - dy
    grid = _normalized_grid(src_x, src_y, height, width)
    grid = grid.unsqueeze(0).expand(tensor.shape[0], height, width, 2)

    out = F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return out.squeeze(0) if squeeze_batch else out


def gaussian_feature(Z: torch.Tensor, sigma: float, *, seed: int = 0) -> torch.Tensor:
    """Add deterministic seeded per-cell Gaussian feature noise."""

    if not torch.is_floating_point(Z):
        raise TypeError("Z must be a floating-point tensor")
    sigma_value = float(sigma)
    if sigma_value < 0.0:
        raise ValueError("sigma must be non-negative")
    if sigma_value == 0.0:
        return Z.clone()
    generator = torch.Generator(device=Z.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(Z.shape, device=Z.device, dtype=Z.dtype, generator=generator)
    return Z + sigma_value * noise


def random_warp_feature(
    Z: torch.Tensor,
    alpha: float,
    sigma: float,
    seed: int,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Apply a deterministic random-pivot/direction swirl to a BEV feature."""

    tensor, squeeze_batch = _as_bchw(Z)
    _, _, height, width = tensor.shape
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    pivot_x = torch.rand((), generator=generator, device=tensor.device, dtype=tensor.dtype) * (width - 1)
    pivot_y = torch.rand((), generator=generator, device=tensor.device, dtype=tensor.dtype) * (height - 1)
    direction = -1.0 if int(torch.randint(0, 2, (), generator=generator, device=tensor.device).item()) == 0 else 1.0
    out = bev_twist(
        tensor,
        pivot=(float(pivot_x.item()), float(pivot_y.item())),
        alpha=direction * float(alpha),
        sigma=float(sigma),
        mode=mode,
        padding_mode=padding_mode,
    )
    return out.squeeze(0) if squeeze_batch else out


def bev_twist(
    Z: torch.Tensor,
    *,
    pivot: tuple[float, float] | None = None,
    alpha: float,
    sigma: float,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Apply the canonical local swirl to a BEV feature tensor.

    ``Z`` may be ``(C, H, W)`` or ``(B, C, H, W)``.  The pivot is expressed in
    pixel coordinates as ``(x, y)``.  Warping uses the inverse twist field so
    each output pixel samples from the corresponding source location, and
    channels are sampled independently by ``grid_sample``.
    """

    tensor, squeeze_batch = _as_bchw(Z)
    _, _, height, width = tensor.shape

    alpha_value = float(alpha)
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError("sigma must be positive")
    if alpha_value == 0.0:
        out = tensor.clone()
        return out.squeeze(0) if squeeze_batch else out

    if pivot is None:
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
    else:
        center_x = float(pivot[0])
        center_y = float(pivot[1])

    yy, xx = _pixel_grid(height, width, tensor)
    rel_x = xx - center_x
    rel_y = yy - center_y
    radius = torch.sqrt(rel_x.square() + rel_y.square())
    scaled_radius = radius / sigma_value
    theta = -alpha_value * scaled_radius * torch.exp(-0.5 * scaled_radius.square())

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    src_x = center_x + cos_t * rel_x - sin_t * rel_y
    src_y = center_y + sin_t * rel_x + cos_t * rel_y

    grid = _normalized_grid(src_x, src_y, height, width)
    grid = grid.unsqueeze(0).expand(tensor.shape[0], height, width, 2)

    out = F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return out.squeeze(0) if squeeze_batch else out


def _as_bchw(Z: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if Z.ndim not in (3, 4):
        raise ValueError("Z must have shape (C, H, W) or (B, C, H, W)")
    if not torch.is_floating_point(Z):
        raise TypeError("Z must be a floating-point tensor")
    squeeze_batch = Z.ndim == 3
    tensor = Z.unsqueeze(0) if squeeze_batch else Z
    if tensor.shape[-2] < 2 or tensor.shape[-1] < 2:
        raise ValueError("Z spatial dimensions must both be at least 2")
    return tensor, squeeze_batch


def _pixel_grid(height: int, width: int, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.meshgrid(
        torch.arange(height, device=tensor.device, dtype=tensor.dtype),
        torch.arange(width, device=tensor.device, dtype=tensor.dtype),
        indexing="ij",
    )


def _normalized_grid(src_x: torch.Tensor, src_y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    grid_x = (2.0 * src_x / (width - 1)) - 1.0
    grid_y = (2.0 * src_y / (height - 1)) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)
