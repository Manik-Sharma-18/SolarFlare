"""Composite loss functions for solar flare prediction.

Combines multiple loss components:
- L1 (MAE): Basic reconstruction loss
- MS-SSIM: Multi-scale structural similarity for sharper predictions
- Weighted MAE: Higher weight for extreme flux values (solar flares)
- Asymmetric extreme loss: Penalizes underestimation above threshold
- Temporal difference loss: L1 on frame-to-frame changes
- Temporal variation penalty: Rewards prediction variation up to target level
- Per-timestep weighting: Later timesteps contribute more to loss

See submodules: ssim_core.py, ssim_multiscale.py, components.py,
temporal.py, composite.py, factory.py
"""
from .ssim_core import gaussian_kernel, ssim, _KERNEL_CACHE
from .ssim_multiscale import ms_ssim
from .components import WeightedMAELoss, AsymmetricExtremeLoss
from .temporal import (
    compute_temporal_diff_loss,
    compute_temporal_var_penalty,
    apply_temporal_weights,
)
from .composite import CompositeLoss
from .dual_head import DualHeadLoss
from .quantile import PinballLoss
from .factory import get_loss_function

__all__ = [
    "gaussian_kernel",
    "ssim",
    "ms_ssim",
    "_KERNEL_CACHE",
    "WeightedMAELoss",
    "AsymmetricExtremeLoss",
    "compute_temporal_diff_loss",
    "compute_temporal_var_penalty",
    "apply_temporal_weights",
    "CompositeLoss",
    "DualHeadLoss",
    "PinballLoss",
    "get_loss_function",
]
