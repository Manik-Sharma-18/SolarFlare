"""
Composite loss functions for solar flare prediction.

Combines multiple loss components:
- L1 (MAE): Basic reconstruction loss
- MS-SSIM: Multi-scale structural similarity for sharper predictions
- Weighted MAE: Higher weight for extreme flux values (solar flares)
- Asymmetric extreme loss: Penalizes underestimation above threshold
- Temporal difference loss: L1 on frame-to-frame changes
- Temporal variation penalty: Rewards prediction variation up to target level
- Per-timestep weighting: Later timesteps contribute more to loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

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
        return _tiled_ssim(
            pred, target, window, C1, C2, padding, on_mps,
            tiling_threshold, size_average,
        )

    # Standard (non-tiled) computation
    ssim_map = _compute_ssim_map(pred, target, window, C1, C2, padding, on_mps)

    if size_average:
        return ssim_map.mean()
    return ssim_map


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


class WeightedMAELoss(nn.Module):
    """MAE loss with higher weight for extreme values using absolute threshold.

    Uses binary weighting based on an absolute threshold: pixels with
    |target| > threshold get extreme_weight, all others get base_weight.
    This ensures consistent penalty across samples regardless of batch content.

    Args:
        base_weight: Weight for normal regions (|target| <= threshold).
        extreme_weight: Weight for extreme regions (|target| > threshold).
        threshold: Absolute threshold for extreme region classification.
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        extreme_weight: float = 3.0,
        threshold: float = 0.3456,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.extreme_weight = extreme_weight
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MAE with absolute threshold binary weighting.

        Args:
            pred: Predicted tensor.
            target: Target tensor (same shape as pred).

        Returns:
            Scalar weighted MAE loss.
        """
        abs_error = torch.abs(pred - target)

        # Binary weighting: base_weight below threshold, extreme_weight above
        extreme_mask = (target.abs() > self.threshold).float()
        weights = self.base_weight + (self.extreme_weight - self.base_weight) * extreme_mask

        weighted_error = weights * abs_error
        return weighted_error.mean()


class AsymmetricExtremeLoss(nn.Module):
    """Asymmetric loss that penalizes underestimation more in extreme regions.

    Above the extreme threshold, underestimation (pred < target) is penalized
    alpha times more than overestimation. Below the threshold, the loss is
    symmetric (standard MAE).

    Args:
        alpha: Underestimation penalty multiplier for extreme regions.
        threshold: Absolute threshold for extreme region classification.
    """

    def __init__(self, alpha: float = 2.0, threshold: float = 0.3456):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric extreme loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor (same shape as pred).

        Returns:
            Scalar asymmetric loss.
        """
        above_threshold = (target.abs() > self.threshold).float()
        underestimation = (target - pred).clamp(min=0)  # positive where pred < target
        overestimation = (pred - target).clamp(min=0)    # positive where pred > target

        # Above threshold: asymmetric (alpha * under + over)
        # Below threshold: symmetric (under + over = |pred - target|)
        error = (
            above_threshold * (self.alpha * underestimation + overestimation)
            + (1 - above_threshold) * (underestimation + overestimation)
        )
        return error.mean()


def compute_temporal_diff_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """L1 loss on frame-to-frame changes (temporal dynamics matching).

    Computes the difference between consecutive frames for both prediction
    and target, then computes L1 loss between these difference sequences.
    This encourages the model to match the temporal dynamics, not just
    the absolute values.

    Args:
        pred: Predicted tensor (B, C, T, H, W).
        target: Target tensor (B, C, T, H, W).

    Returns:
        Scalar temporal difference loss. Returns 0.0 if T <= 1.
    """
    T = pred.shape[2]
    if T <= 1:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    return F.l1_loss(pred_diffs, target_diffs)


def compute_temporal_var_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """Negative loss that rewards prediction variation up to target level.

    Computes a penalty that is negative (subtracts from total loss) when
    the prediction has temporal variation. The penalty is capped at the
    target's variation level to prevent noisy/jittery predictions from
    being rewarded beyond what the data naturally exhibits.

    penalty = -lambda * min(pred_variation, target_variation)

    Args:
        pred: Predicted tensor (B, C, T, H, W).
        target: Target tensor (B, C, T, H, W).
        lambda_val: Penalty scaling factor.

    Returns:
        Scalar variation penalty (negative value). Returns 0.0 if T <= 1.
    """
    T = pred.shape[2]
    if T <= 1:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

    pred_var = pred_diffs.abs().mean()
    target_var = target_diffs.abs().mean()

    # Cap: no reward for exceeding target variation
    capped_var = torch.min(pred_var, target_var)
    return -lambda_val * capped_var


def apply_temporal_weights(
    loss_tensor: torch.Tensor, weights: List[float]
) -> torch.Tensor:
    """Apply per-timestep weights to a 5D loss tensor before reduction.

    Broadcasts weights along the temporal dimension (dim=2) and computes
    the weighted mean across all dimensions. Later timesteps can be given
    higher weights to prioritize future prediction accuracy.

    Args:
        loss_tensor: 5D tensor (B, C, T, H, W) of per-element losses.
        weights: List of per-timestep weights (truncated or padded to T).

    Returns:
        Scalar weighted mean loss.
    """
    T = loss_tensor.shape[2]
    tw = torch.tensor(
        weights[:T], device=loss_tensor.device, dtype=loss_tensor.dtype
    )
    tw = tw.view(1, 1, T, 1, 1)
    return (loss_tensor * tw).mean()


class CompositeLoss(nn.Module):
    """
    Composite loss combining L1, MS-SSIM, and weighted extreme loss.

    Total loss = l1_weight * L1 + ssim_weight * (1 - MS_SSIM) + extreme_weight * WeightedMAE
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        extreme_weight: float = 1.0,
        use_ms_ssim: bool = True,
        ssim_data_range: float = 2.0,
        ssim_tiling_threshold: int = 256,
    ):
        """
        Args:
            l1_weight: Weight for L1 (MAE) loss
            ssim_weight: Weight for SSIM loss (as 1 - SSIM)
            extreme_weight: Weight for extreme value loss
            use_ms_ssim: Use multi-scale SSIM (True) or single-scale (False)
            ssim_data_range: Data range for SSIM computation
            ssim_tiling_threshold: Spatial size above which SSIM tiles to avoid OOM
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.extreme_weight = extreme_weight
        self.use_ms_ssim = use_ms_ssim
        self.ssim_data_range = ssim_data_range
        self.ssim_tiling_threshold = ssim_tiling_threshold

        self.weighted_mae = WeightedMAELoss(base_weight=1.0, extreme_weight=2.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss.

        Args:
            pred: Predicted tensor (B, C, T, H, W) or (B, C, H, W)
            target: Target tensor (same shape as pred)
            return_components: If True, return dict with individual loss components

        Returns:
            Total loss value (or dict if return_components=True)
        """
        # Flatten temporal dimension if present
        if pred.dim() == 5:
            B, C, T, H, W = pred.shape
            pred = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # SSIM loss (1 - SSIM, so lower is better)
        if self.use_ms_ssim and min(pred.shape[-2:]) >= 32:
            ssim_val = ms_ssim(
                pred, target, data_range=self.ssim_data_range,
                tiling_threshold=self.ssim_tiling_threshold,
            )
        else:
            ssim_val = ssim(
                pred, target, data_range=self.ssim_data_range,
                tiling_threshold=self.ssim_tiling_threshold,
            )
        ssim_loss = 1.0 - ssim_val

        # Weighted extreme loss
        extreme_loss = self.weighted_mae(pred, target)

        # Combine
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.extreme_weight * extreme_loss
        )

        if return_components:
            return {
                'total': total_loss,
                'l1': l1_loss,
                'ssim': ssim_loss,
                'ssim_val': ssim_val,
                'extreme': extreme_loss
            }

        return total_loss


def get_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        config: Loss configuration dict with keys:
            - type: 'l1', 'composite', or 'weighted'
            - l1_weight, ssim_weight, extreme_weight (for composite)
            - ssim_tiling_threshold: tile size for large spatial SSIM (default 256)

    Returns:
        Loss function module
    """
    loss_type = config.get('type', 'l1')

    if loss_type == 'l1':
        return nn.L1Loss()

    elif loss_type == 'composite':
        return CompositeLoss(
            l1_weight=config.get('l1_weight', 1.0),
            ssim_weight=config.get('ssim_weight', 0.5),
            extreme_weight=config.get('extreme_weight', 1.0),
            use_ms_ssim=config.get('use_ms_ssim', True),
            ssim_data_range=config.get('ssim_data_range', 2.0),
            ssim_tiling_threshold=config.get('ssim_tiling_threshold', 256),
        )

    elif loss_type == 'weighted':
        return WeightedMAELoss(
            base_weight=config.get('base_weight', 1.0),
            extreme_weight=config.get('extreme_weight', 2.0)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
