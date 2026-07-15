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
from .histogram_match import HistogramMatchLoss


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
        base_pixel_weight: float = 1.0,        # <1 masks bg (S31: 0.01 = ~100% loss mass on extreme)
        histogram_weight: float = 0.0,         # S32: weight of histogram-match anti-collapse term
        histogram_n_bins: int = 32,
        histogram_max: float = 3.0,
        temporal_grad_weight: float = 0.0,     # S36: ||Δ_t pred - Δ_t gt||² physics smoothness term
        integrated_flux_weight: float = 0.0,   # S36b: |Σ pred - Σ gt| per timestep, bulk conservation
        sobel_weight: float = 0.0,             # S38: L1 on Sobel-filtered (boundary preservation)
        spectral_weight: float = 0.0,          # S39: MSE on |FFT2(·)| amplitude spectrum
        lowpass_weight: float = 0.0,           # S46: L1 on avg-pooled fields (EDA: only low-k predictable)
        lowpass_pool: int = 32,                # pool window px (32px = 11.5 Mm block means)
        spatial_mean_weight: float = 0.0,      # S51: L1 on per-frame spatial-mean |.| (the project objective curve)
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.thr = float(extreme_threshold)
        self.classification_loss = classification_loss
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.extreme_pixel_weight = float(extreme_pixel_weight)
        self.base_pixel_weight = float(base_pixel_weight)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        # Use the weighted MAE whenever either side of the weighting deviates
        # from a plain L1. S31's "masked loss" is base_pixel_weight=0.01,
        # extreme_pixel_weight=1.0.
        use_weighted = (self.extreme_pixel_weight != self.base_pixel_weight)
        self.weighted_mae = (
            WeightedMAELoss(base_weight=self.base_pixel_weight,
                            extreme_weight=self.extreme_pixel_weight,
                            threshold=self.thr)
            if use_weighted else None
        )
        self.histogram_weight = float(histogram_weight)
        self.hist_loss = (
            HistogramMatchLoss(n_bins=histogram_n_bins, bin_max=histogram_max,
                               on_abs=True)
            if self.histogram_weight > 0 else None
        )
        self.temporal_grad_weight = float(temporal_grad_weight)
        self.integrated_flux_weight = float(integrated_flux_weight)
        self.sobel_weight = float(sobel_weight)
        self.spectral_weight = float(spectral_weight)
        self.lowpass_weight = float(lowpass_weight)
        self.lowpass_pool = int(lowpass_pool)
        self.spatial_mean_weight = float(spatial_mean_weight)
        if self.sobel_weight > 0:
            sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            sy = sx.T
            self.register_buffer("_sx", sx.view(1, 1, 3, 3))
            self.register_buffer("_sy", sy.view(1, 1, 3, 3))

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
        hist = torch.zeros_like(l1)
        if self.hist_loss is not None:
            hist = self.hist_loss(pred, target)
            total = total + self.histogram_weight * hist
        tgrad = torch.zeros_like(l1)
        if self.temporal_grad_weight > 0 and pred.shape[2] >= 2:
            # T-dim assumed at index 2 for (B,C,T,H,W).
            dp = pred[:, :, 1:] - pred[:, :, :-1]
            dt = target[:, :, 1:] - target[:, :, :-1]
            tgrad = ((dp - dt) ** 2).mean()
            total = total + self.temporal_grad_weight * tgrad
        iflux = torch.zeros_like(l1)
        if self.integrated_flux_weight > 0:
            # Bulk-conservation: per-timestep sum over spatial dims should match.
            sp = pred.flatten(3).sum(dim=3)
            sg = target.flatten(3).sum(dim=3)
            iflux = (sp - sg).abs().mean()
            total = total + self.integrated_flux_weight * iflux
        sobel = torch.zeros_like(l1)
        if self.sobel_weight > 0:
            # Collapse (B,C,T,H,W) → (B·C·T, 1, H, W) for depthwise conv.
            B, C, T, H, W = pred.shape
            p2d = pred.reshape(B * C * T, 1, H, W)
            g2d = target.reshape(B * C * T, 1, H, W)
            sobel = (F.l1_loss(F.conv2d(p2d, self._sx, padding=1),
                               F.conv2d(g2d, self._sx, padding=1))
                   + F.l1_loss(F.conv2d(p2d, self._sy, padding=1),
                               F.conv2d(g2d, self._sy, padding=1)))
            total = total + self.sobel_weight * sobel
        lowpass = torch.zeros_like(l1)
        if self.lowpass_weight > 0:
            # Block-mean agreement only — high-k left unconstrained (EDA:
            # scales < ~14 Mm are stochastic; full-res MSE optimizes noise).
            B, C, T, H, W = pred.shape
            k = min(self.lowpass_pool, H, W)
            lowpass = F.l1_loss(
                F.avg_pool2d(pred.reshape(B * C * T, 1, H, W), k),
                F.avg_pool2d(target.reshape(B * C * T, 1, H, W), k))
            total = total + self.lowpass_weight * lowpass
        smean = torch.zeros_like(l1)
        if self.spatial_mean_weight > 0:
            # Per-frame spatial-mean magnitude curve — the staircase/spatial-
            # mean-flux objective, optimized directly (not just per-pixel L1).
            smean = F.l1_loss(pred.abs().mean(dim=(-1, -2)),
                              target.abs().mean(dim=(-1, -2)))
            total = total + self.spatial_mean_weight * smean
        spectral = torch.zeros_like(l1)
        if self.spectral_weight > 0:
            # FFT2 amplitude match — penalises wrong power-spectrum structure.
            fp = torch.fft.rfft2(pred).abs()
            fg = torch.fft.rfft2(target).abs()
            spectral = F.mse_loss(fp, fg)
            total = total + self.spectral_weight * spectral
        if return_components:
            return {"total": total, "l1": l1, "bce": bce, "hist": hist,
                    "tgrad": tgrad, "iflux": iflux,
                    "sobel": sobel, "spectral": spectral, "lowpass": lowpass,
                    "smean": smean}
        return total
