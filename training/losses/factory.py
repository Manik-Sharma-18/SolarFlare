"""Loss-function factory: build a loss module from a config dict."""
import torch.nn as nn
from typing import Dict

from .components import WeightedMAELoss
from .composite import CompositeLoss
from .dual_head import DualHeadLoss
from .quantile import PinballLoss


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
            temporal_diff_weight=config.get('temporal_diff_weight', 1.0),
            temporal_var_lambda=config.get('temporal_var_lambda', 0.1),
            asymmetric_weight=config.get('asymmetric_weight', 0.5),
            asymmetric_alpha=config.get('asymmetric_alpha', 2.0),
            extreme_threshold=config.get('extreme_threshold', 0.277),
            temporal_weights=config.get('temporal_weights', [1.0, 1.5, 2.0, 2.5]),
            extreme_pixel_weight=config.get('extreme_pixel_weight', 3.0),
        )

    elif loss_type == 'dual_head':
        return DualHeadLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 1.0),
            pos_weight=config.get('pos_weight', 60.0),
            extreme_threshold=config.get('extreme_threshold', 0.528),
            classification_loss=config.get('classification_loss', 'bce'),
            focal_gamma=config.get('focal_gamma', 2.0),
            focal_alpha=config.get('focal_alpha', 0.25),
            extreme_pixel_weight=config.get('extreme_pixel_weight', 1.0),
            base_pixel_weight=config.get('base_pixel_weight', 1.0),
            histogram_weight=config.get('histogram_weight', 0.0),
            histogram_n_bins=config.get('histogram_n_bins', 32),
            histogram_max=config.get('histogram_max', 3.0),
            temporal_grad_weight=config.get('temporal_grad_weight', 0.0),
            integrated_flux_weight=config.get('integrated_flux_weight', 0.0),
            sobel_weight=config.get('sobel_weight', 0.0),
            spectral_weight=config.get('spectral_weight', 0.0),
            lowpass_weight=config.get('lowpass_weight', 0.0),
            lowpass_pool=config.get('lowpass_pool', 32),
            spatial_mean_weight=config.get('spatial_mean_weight', 0.0),
        )

    elif loss_type == 'weighted':
        return WeightedMAELoss(
            base_weight=config.get('base_weight', 1.0),
            extreme_weight=config.get('extreme_weight', 2.0)
        )

    elif loss_type == 'quantile':
        return PinballLoss(tau=config.get('tau', 0.99))

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
