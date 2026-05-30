"""CompositeLoss: combines spatial + temporal terms for solar prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .ssim_core import ssim
from .ssim_multiscale import ms_ssim
from .components import WeightedMAELoss, AsymmetricExtremeLoss
from .temporal import compute_temporal_diff_loss, compute_temporal_var_penalty


class CompositeLoss(nn.Module):
    """
    Composite loss combining 6 components for temporal-aware solar prediction.

    Components:
    1. L1 (MAE): Basic reconstruction loss (per-timestep weighted when 5D)
    2. SSIM: Structural similarity loss (1 - SSIM)
    3. WeightedMAE: Higher weight for extreme flux values
    4. Temporal diff: L1 on frame-to-frame changes (temporal dynamics matching)
    5. Temporal var: Negative penalty rewarding prediction variation
    6. Asymmetric: Underestimation penalty for extreme regions

    Total loss = l1_weight * L1 + ssim_weight * (1-SSIM) + extreme_weight * WeightedMAE
                 + temporal_diff_weight * TemporalDiff + TemporalVar
                 + asymmetric_weight * AsymmetricExtreme
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        extreme_weight: float = 1.0,
        use_ms_ssim: bool = True,
        ssim_data_range: float = 2.0,
        ssim_tiling_threshold: int = 256,
        temporal_diff_weight: float = 1.0,
        temporal_var_lambda: float = 0.1,
        asymmetric_weight: float = 0.5,
        asymmetric_alpha: float = 2.0,
        extreme_threshold: float = 0.277,
        temporal_weights: Optional[List[float]] = None,
        extreme_pixel_weight: float = 3.0,
    ):
        """
        Args:
            l1_weight: Weight for L1 (MAE) loss
            ssim_weight: Weight for SSIM loss (as 1 - SSIM)
            extreme_weight: Component weight for extreme value loss in total sum
            use_ms_ssim: Use multi-scale SSIM (True) or single-scale (False)
            ssim_data_range: Data range for SSIM computation
            ssim_tiling_threshold: Spatial size above which SSIM tiles to avoid OOM
            temporal_diff_weight: Weight for temporal difference loss
            temporal_var_lambda: Lambda for temporal variation penalty
            asymmetric_weight: Weight for asymmetric extreme loss
            asymmetric_alpha: Underestimation penalty multiplier
            extreme_threshold: Absolute threshold for extreme region classification
            temporal_weights: Per-timestep weights (default [1.0, 1.5, 2.0, 2.5])
            extreme_pixel_weight: Per-pixel weight inside WeightedMAELoss for extreme regions
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.extreme_weight = extreme_weight
        self.use_ms_ssim = use_ms_ssim
        self.ssim_data_range = ssim_data_range
        self.ssim_tiling_threshold = ssim_tiling_threshold
        self.temporal_diff_weight = temporal_diff_weight
        self.temporal_var_lambda = temporal_var_lambda
        self.asymmetric_weight = asymmetric_weight
        self.temporal_weights = temporal_weights if temporal_weights is not None else [1.0, 1.5, 2.0, 2.5]

        self.weighted_mae = WeightedMAELoss(
            base_weight=1.0,
            extreme_weight=extreme_pixel_weight,
            threshold=extreme_threshold,
        )
        self.asymmetric_extreme = AsymmetricExtremeLoss(
            alpha=asymmetric_alpha,
            threshold=extreme_threshold,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss with temporal-aware two-phase computation.

        Phase 1: Compute temporal terms on 5D tensor (before flattening)
        Phase 2: Compute spatial terms on 4D tensor (after flattening)

        Args:
            pred: Predicted tensor (B, C, T, H, W) or (B, C, H, W)
            target: Target tensor (same shape as pred)
            return_components: If True, return dict with individual loss components

        Returns:
            Total loss value (or dict if return_components=True)
        """
        # Phase 1: Temporal terms on 5D tensor
        if pred.dim() == 5:
            B, C, T, H, W = pred.shape

            # Temporal difference loss (LOSS-01)
            temporal_diff_loss = compute_temporal_diff_loss(pred, target)

            # Temporal variation penalty (LOSS-03)
            temporal_var_loss = compute_temporal_var_penalty(
                pred, target, self.temporal_var_lambda
            )

            # Per-timestep weighting (LOSS-02): apply to element-wise L1 error
            elem_error = torch.abs(pred - target)
            tw = torch.tensor(
                self.temporal_weights[:T],
                device=pred.device, dtype=pred.dtype,
            )
            tw = tw.view(1, 1, T, 1, 1)
            weighted_l1_loss = (elem_error * tw).mean()

            # Flatten for spatial-only terms (SSIM, WeightedMAE, Asymmetric)
            pred_flat = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            target_flat = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            pred_flat = pred
            target_flat = target
            weighted_l1_loss = F.l1_loss(pred_flat, target_flat)
            temporal_diff_loss = torch.tensor(0.0, device=pred.device)
            temporal_var_loss = torch.tensor(0.0, device=pred.device)

        # Phase 2: Spatial terms on 4D tensor
        # SSIM loss (1 - SSIM, so lower is better)
        if self.use_ms_ssim and min(pred_flat.shape[-2:]) >= 32:
            ssim_val = ms_ssim(
                pred_flat, target_flat, data_range=self.ssim_data_range,
                tiling_threshold=self.ssim_tiling_threshold,
            )
        else:
            ssim_val = ssim(
                pred_flat, target_flat, data_range=self.ssim_data_range,
                tiling_threshold=self.ssim_tiling_threshold,
            )
        ssim_loss = 1.0 - ssim_val

        # Weighted extreme loss
        extreme_loss = self.weighted_mae(pred_flat, target_flat)

        # Asymmetric extreme loss
        asymmetric_loss = self.asymmetric_extreme(pred_flat, target_flat)

        # Combine all 6 components
        total_loss = (
            self.l1_weight * weighted_l1_loss
            + self.ssim_weight * ssim_loss
            + self.extreme_weight * extreme_loss
            + self.temporal_diff_weight * temporal_diff_loss
            + temporal_var_loss  # Already scaled by lambda, sign is negative
            + self.asymmetric_weight * asymmetric_loss
        )

        if return_components:
            return {
                'total': total_loss,
                'l1': weighted_l1_loss,
                'ssim': ssim_loss,
                'ssim_val': ssim_val,
                'extreme': extreme_loss,
                'temporal_diff': temporal_diff_loss,
                'temporal_var': temporal_var_loss,
                'asymmetric': asymmetric_loss,
            }

        return total_loss
