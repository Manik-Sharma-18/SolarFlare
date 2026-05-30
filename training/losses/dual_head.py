"""Dual-head loss: L1 regression + per-pixel BCE extreme classifier.

The CSI metric is a *thresholded binary* over ``|target|>extreme_threshold``,
but every S0..S11 arm trained a regression loss on amplitude and lost to
persistence. ``DualHeadLoss`` ends that mismatch — the classifier head emits
``logits = P(|x|>thr)`` directly, with class-imbalance corrected via either
``pos_weight`` BCE or focal loss. Used together with
``SimpleConvLSTM(enable_classifier_head=True)`` which returns
``(pred, ext_logits)``.
"""
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import WeightedMAELoss


def _binary_target(target: torch.Tensor, thr: float) -> torch.Tensor:
    """Same convention as ``accumulate_contingency``: ``abs(target) > thr``."""
    return (target.abs() > thr).float()


def _focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor,
                            gamma: float, alpha: float) -> torch.Tensor:
    """Focal BCE: ``-α (1-p)^γ log p`` on positives, ``-(1-α) p^γ log(1-p)`` on
    negatives. Reduces over all elements (mean). Numerically stable form."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


class DualHeadLoss(nn.Module):
    """``α · L1(pred, target) + β · BCE/Focal(logits, |target|>thr)``."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        pos_weight: float = 60.0,
        extreme_threshold: float = 0.528,
        classification_loss: str = "bce",      # 'bce' or 'focal'
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        extreme_pixel_weight: float = 1.0,     # >1 weights extreme pixels in L1 (S11 used 25)
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.thr = float(extreme_threshold)
        self.classification_loss = classification_loss
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.extreme_pixel_weight = float(extreme_pixel_weight)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.weighted_mae = (
            WeightedMAELoss(base_weight=1.0, extreme_weight=self.extreme_pixel_weight,
                            threshold=self.thr)
            if self.extreme_pixel_weight > 1.0 else None
        )

    def _classification_term(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        ext_target = _binary_target(target, self.thr)
        if self.classification_loss == "focal":
            return _focal_loss_with_logits(
                logits, ext_target, self.focal_gamma, self.focal_alpha
            )
        return F.binary_cross_entropy_with_logits(
            logits, ext_target, pos_weight=self.pos_weight
        )

    def forward(
        self,
        predictions: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        return_components: bool = False,
        ext_logits: Optional[torch.Tensor] = None,
    ):
        # Accept either tuple ``(pred, logits)`` from the dual-head model, or
        # tensor ``pred`` with logits passed via ``ext_logits`` kwarg.
        if isinstance(predictions, tuple):
            pred, logits = predictions
        else:
            pred, logits = predictions, ext_logits

        l1 = (self.weighted_mae(pred, target) if self.weighted_mae is not None
              else F.l1_loss(pred, target))
        if logits is None:
            # Regression-only fallback (e.g. classifier head disabled at eval).
            total = self.alpha * l1
            if return_components:
                return {"total": total, "l1": l1,
                        "bce": torch.zeros_like(l1)}
            return total

        bce = self._classification_term(logits, target)
        total = self.alpha * l1 + self.beta * bce
        if return_components:
            return {"total": total, "l1": l1, "bce": bce}
        return total
