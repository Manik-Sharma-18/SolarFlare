"""Single-scale SSIM with kernel caching and MPS-safe conv2d.

Holds the module-level kernel cache shared across the losses package.
"""
import torch
import torch.nn.functional as F
from typing import Dict

from utils.mps_ops import safe_outer, is_mps

# Module-level kernel cache: (size, sigma, device_str) -> kernel tensor
_KERNEL_CACHE: Dict[tuple, torch.Tensor] = {}


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel with caching.

    Kernels are cached by (size, sigma, device) so repeated calls with
    the same parameters reuse the cached tensor.
    """
    cache_key = (size, sigma, str(device))
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = safe_outer(g, g)

    _KERNEL_CACHE[cache_key] = kernel
    return kernel


def _ssim_conv2d(
    input_tensor: torch.Tensor,
    window: torch.Tensor,
    channels: int,
    padding: int,
    on_mps: bool,
) -> torch.Tensor:
    """Device-dispatched conv2d for SSIM computation.

    On MPS: loops over channels with groups=1 to avoid grouped conv2d bugs.
    On CUDA/CPU: uses native grouped conv2d (unchanged behaviour).
    """
    if on_mps:
        # Channel-loop fallback: run conv2d per channel to avoid MPS grouped conv bugs
        single_window = window[:1]  # (1, 1, K, K)
        slices = []
        for c in range(channels):
            x_c = input_tensor[:, c : c + 1, :, :]  # (B, 1, H, W)
            slices.append(F.conv2d(x_c, single_window, padding=padding, groups=1))
        return torch.cat(slices, dim=1)
    else:
        return F.conv2d(input_tensor, window, padding=padding, groups=channels)


def _compute_ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    C1: float,
    C2: float,
    padding: int,
    on_mps: bool,
) -> torch.Tensor:
    """Compute SSIM map for a (possibly tiled) region."""
    C = pred.size(1)

    mu1 = _ssim_conv2d(pred, window, C, padding, on_mps)
    mu2 = _ssim_conv2d(target, window, C, padding, on_mps)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _ssim_conv2d(pred ** 2, window, C, padding, on_mps) - mu1_sq
    sigma2_sq = _ssim_conv2d(target ** 2, window, C, padding, on_mps) - mu2_sq
    sigma12 = _ssim_conv2d(pred * target, window, C, padding, on_mps) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    size_average: bool = True,
    tiling_threshold: int = 256,
) -> torch.Tensor:
    """
    Compute SSIM between prediction and target.

    Supports MPS channel-loop fallback, kernel caching, and tiled
    computation for large spatial tensors to avoid OOM.

    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian
        data_range: Dynamic range of data
        size_average: If True, return mean SSIM
        tiling_threshold: Spatial dimension above which tiling is used (0 to disable)

    Returns:
        SSIM value(s)
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create / retrieve cached Gaussian window
    kernel = gaussian_kernel(window_size, sigma, pred.device)
    window = kernel.expand(pred.size(1), 1, window_size, window_size)

    on_mps = is_mps(pred.device)
    padding = window_size // 2
    H, W = pred.shape[2], pred.shape[3]

    # Tiled computation for large spatial tensors
    if tiling_threshold > 0 and (H > tiling_threshold or W > tiling_threshold):
        # Local import avoids a circular dependency: ssim_multiscale imports ssim.
        from .ssim_multiscale import _tiled_ssim
        return _tiled_ssim(
            pred, target, window, C1, C2, padding, on_mps,
            tiling_threshold, size_average,
        )

    # Standard (non-tiled) computation
    ssim_map = _compute_ssim_map(pred, target, window, C1, C2, padding, on_mps)

    if size_average:
        return ssim_map.mean()
    return ssim_map
