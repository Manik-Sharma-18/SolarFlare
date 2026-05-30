"""Tiled and multi-scale SSIM, built on the single-scale primitives."""
import torch
import torch.nn.functional as F
from typing import Optional

from .ssim_core import ssim, _compute_ssim_map


def _tiled_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    C1: float,
    C2: float,
    padding: int,
    on_mps: bool,
    tile_size: int,
    size_average: bool,
) -> torch.Tensor:
    """Compute SSIM over tiles to avoid OOM on large spatial tensors.

    Tiles overlap by 16 pixels. The overlap region is discarded from each
    tile's contribution to avoid double-counting.
    """
    overlap = 16
    H, W = pred.shape[2], pred.shape[3]
    step = tile_size - overlap

    weighted_sum = torch.tensor(0.0, device=pred.device)
    total_pixels = 0

    y = 0
    while y < H:
        x = 0
        y_end = min(y + tile_size, H)
        while x < W:
            x_end = min(x + tile_size, W)

            pred_tile = pred[:, :, y:y_end, x:x_end]
            tgt_tile = target[:, :, y:y_end, x:x_end]

            ssim_map = _compute_ssim_map(
                pred_tile, tgt_tile, window, C1, C2, padding, on_mps
            )

            # Determine valid (non-overlap) region within this tile's output
            valid_top = overlap // 2 if y > 0 else 0
            valid_left = overlap // 2 if x > 0 else 0
            valid_bottom = ssim_map.shape[2] - (overlap // 2 if y_end < H else 0)
            valid_right = ssim_map.shape[3] - (overlap // 2 if x_end < W else 0)

            valid_region = ssim_map[:, :, valid_top:valid_bottom, valid_left:valid_right]
            n_pixels = valid_region.numel()
            weighted_sum = weighted_sum + valid_region.sum()
            total_pixels += n_pixels

            x += step
        y += step

    if size_average:
        return weighted_sum / max(total_pixels, 1)
    # Non-averaged tiled SSIM isn't well-defined; return averaged anyway
    return weighted_sum / max(total_pixels, 1)


def ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    weights: Optional[torch.Tensor] = None,
    tiling_threshold: int = 256,
) -> torch.Tensor:
    """
    Multi-scale SSIM.

    Computes SSIM at multiple scales by progressively downsampling.

    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian
        data_range: Dynamic range of data
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        tiling_threshold: Spatial dimension above which tiling is used

    Returns:
        MS-SSIM value
    """
    if weights is None:
        # Default weights from original MS-SSIM paper
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=pred.device)

    levels = len(weights)
    ms_ssim_val = torch.ones(1, device=pred.device)

    for i in range(levels):
        ssim_val = ssim(
            pred, target, window_size, sigma, data_range,
            size_average=True, tiling_threshold=tiling_threshold,
        )

        if i < levels - 1:
            # Downsample for next scale
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)

            # Weight only contrast/structure for intermediate scales
            ms_ssim_val = ms_ssim_val * (ssim_val ** weights[i])
        else:
            # Final scale includes luminance
            ms_ssim_val = ms_ssim_val * (ssim_val ** weights[i])

    return ms_ssim_val
