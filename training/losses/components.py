"""Pixel-wise loss components: weighted MAE and asymmetric extreme loss."""
import torch
import torch.nn as nn


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
        threshold: float = 0.277,
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

    def __init__(self, alpha: float = 2.0, threshold: float = 0.277):
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
