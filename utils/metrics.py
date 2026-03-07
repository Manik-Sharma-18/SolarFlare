"""Metrics computation for model evaluation."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
    """
    Compute evaluation metrics between predictions and targets.
    
    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)
    
    Returns:
        Dictionary with 'mae_total' and 'mae_per_timestep'
    """
    mae = F.l1_loss(pred, target, reduction='none')
    
    # Average over B, C, H, W to get per-timestep MAE
    mae_per_timestep = mae.mean(dim=(0, 1, 3, 4))
    mae_total = mae.mean()
    
    return {
        'mae_total': mae_total.item(),
        'mae_per_timestep': mae_per_timestep.cpu().numpy()
    }


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Squared Error."""
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse).item()


def compute_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
    denominator = torch.sqrt(
        ((pred_flat - pred_mean) ** 2).sum() * 
        ((target_flat - target_mean) ** 2).sum()
    )
    
    if denominator < 1e-8:
        return 0.0

    return (numerator / denominator).item()


# ---------------------------------------------------------------------------
# CSI (Critical Success Index) — EVAL-02
# ---------------------------------------------------------------------------
def compute_csi(tp: int, fp: int, fn: int) -> float:
    """Compute Critical Success Index from contingency counts.

    CSI = TP / (TP + FP + FN)

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        CSI value in [0, 1]. Returns 0.0 when denominator is zero.
    """
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return tp / denom


# ---------------------------------------------------------------------------
# HSS (Heidke Skill Score) — EVAL-03
# ---------------------------------------------------------------------------
def compute_hss(tp: int, fp: int, fn: int, tn: int) -> float:
    """Compute Heidke Skill Score from contingency counts.

    HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives

    Returns:
        HSS value in [-1, 1]. Returns 0.0 when denominator is zero.
    """
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Contingency Table — EVAL-02, EVAL-03
# ---------------------------------------------------------------------------
def accumulate_contingency(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> List[Tuple[int, int, int, int]]:
    """Accumulate contingency table (TP, FP, FN, TN) per timestep.

    Binary classification uses abs(value) > threshold.

    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)
        threshold: Absolute value threshold for binary classification

    Returns:
        List of (tp, fp, fn, tn) tuples, one per timestep.
    """
    T = pred.shape[2]
    binary_pred = (pred.abs() > threshold).long()
    binary_target = (target.abs() > threshold).long()

    results = []
    for t in range(T):
        bp = binary_pred[:, :, t, :, :]
        bt = binary_target[:, :, t, :, :]

        tp = int(((bp == 1) & (bt == 1)).sum().item())
        fp = int(((bp == 1) & (bt == 0)).sum().item())
        fn = int(((bp == 0) & (bt == 1)).sum().item())
        tn = int(((bp == 0) & (bt == 0)).sum().item())

        results.append((tp, fp, fn, tn))

    return results


# ---------------------------------------------------------------------------
# Persistence Baseline — EVAL-04
# ---------------------------------------------------------------------------
def compute_persistence_prediction(
    X_in: torch.Tensor,
    output_channels: int,
    T_out: int,
) -> torch.Tensor:
    """Create persistence prediction by repeating the last input frame.

    Extracts the last temporal frame from the first `output_channels` channels
    and expands it across T_out timesteps.

    Args:
        X_in: Input tensor (B, C_in, T_in, H, W)
        output_channels: Number of output channels to use from input
        T_out: Number of output timesteps to produce

    Returns:
        Persistence prediction tensor (B, output_channels, T_out, H, W)
    """
    # Extract last frame for output channels: (B, output_channels, 1, H, W)
    last_frame = X_in[:, :output_channels, -1:, :, :]
    # Expand to T_out timesteps
    return last_frame.expand(-1, -1, T_out, -1, -1).contiguous()


def compute_persistence_skill(
    model_mae: float,
    persistence_mae: float,
) -> float:
    """Compute skill-over-persistence as percentage improvement.

    skill = (1 - model_mae / persistence_mae) * 100

    Args:
        model_mae: Model's mean absolute error
        persistence_mae: Persistence baseline's mean absolute error

    Returns:
        Skill percentage. Positive means model is better. Returns 0.0 when
        persistence_mae is near zero (perfect persistence means no skill
        to measure).
    """
    if persistence_mae < 1e-8:
        return 0.0
    return (1.0 - model_mae / persistence_mae) * 100.0


# ---------------------------------------------------------------------------
# Standalone SSIM per-timestep — EVAL-05
# ---------------------------------------------------------------------------
def compute_ssim_per_timestep(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> List[float]:
    """Compute SSIM for each timestep separately.

    Imports and reuses the ssim() function from training.losses.

    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)
        data_range: Dynamic range of the data (default 2.0 for normalized asinh)

    Returns:
        List of T SSIM float values, one per timestep.
    """
    from training.losses import ssim

    T = pred.shape[2]
    results = []
    for t in range(T):
        # Extract (B, C, H, W) for this timestep
        pred_t = pred[:, :, t, :, :]
        target_t = target[:, :, t, :, :]
        ssim_val = ssim(
            pred_t, target_t, data_range=data_range, size_average=True
        )
        results.append(float(ssim_val.item()))
    return results


# ---------------------------------------------------------------------------
# Peak Flux Error — EVAL-06
# ---------------------------------------------------------------------------
def compute_peak_flux_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:
    """Compute per-timestep absolute peak flux error.

    For each timestep: abs(max(pred) - max(target)), averaged over batch.

    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)

    Returns:
        List of T float values representing mean absolute peak flux error.
    """
    T = pred.shape[2]
    results = []
    for t in range(T):
        # Get max over spatial dims (H, W) for each batch/channel: (B, C)
        pred_max = pred[:, :, t, :, :].amax(dim=(-2, -1))
        target_max = target[:, :, t, :, :].amax(dim=(-2, -1))
        # Absolute difference, then mean over batch and channels
        error = (pred_max - target_max).abs().mean()
        results.append(float(error.item()))
    return results


# ---------------------------------------------------------------------------
# Temporal Variation Ratio — EVAL-07
# ---------------------------------------------------------------------------
def compute_temporal_variation_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute ratio of temporal variation in predictions vs targets.

    ratio = mean(|pred[:,t+1] - pred[:,t]|) / mean(|target[:,t+1] - target[:,t]|)

    A ratio of 1.0 means predictions vary at the same rate as targets.
    A ratio near 0.0 means predictions are too static (near-persistence).

    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)

    Returns:
        Variation ratio (float). Returns 1.0 for single-frame input (T=1)
        and 0.0 when target has zero variation.
    """
    T = pred.shape[2]
    if T <= 1:
        return 1.0

    pred_diffs = (pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).abs().mean()
    target_diffs = (target[:, :, 1:, :, :] - target[:, :, :-1, :, :]).abs().mean()

    if target_diffs.item() < 1e-8:
        return 0.0

    return float((pred_diffs / target_diffs).item())


# ---------------------------------------------------------------------------
# Per-Timestep RMSE — EVAL-01
# ---------------------------------------------------------------------------
def compute_rmse_per_timestep(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:
    """Compute RMSE for each timestep separately.

    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)

    Returns:
        List of T RMSE float values, one per timestep.
    """
    T = pred.shape[2]
    results = []
    for t in range(T):
        diff = pred[:, :, t, :, :] - target[:, :, t, :, :]
        mse = (diff ** 2).mean()
        rmse = torch.sqrt(mse)
        results.append(float(rmse.item()))
    return results


# ---------------------------------------------------------------------------
# Per-Timestep Correlation — EVAL-01
# ---------------------------------------------------------------------------
def compute_correlation_per_timestep(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:
    """Compute Pearson correlation for each timestep separately.

    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)

    Returns:
        List of T correlation float values, one per timestep.
        Returns 0.0 for timesteps with zero variance.
    """
    T = pred.shape[2]
    results = []
    for t in range(T):
        p = pred[:, :, t, :, :].flatten()
        tgt = target[:, :, t, :, :].flatten()

        p_mean = p.mean()
        tgt_mean = tgt.mean()

        num = ((p - p_mean) * (tgt - tgt_mean)).sum()
        den = torch.sqrt(
            ((p - p_mean) ** 2).sum() * ((tgt - tgt_mean) ** 2).sum()
        )

        if den.item() < 1e-8:
            results.append(0.0)
        else:
            results.append(float((num / den).item()))
    return results

