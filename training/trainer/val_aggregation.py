"""Validation helpers: contingency accumulation, attention entropy, metric aggregation."""
from typing import List, Optional

import torch
import numpy as np

from utils.metrics import (
    compute_csi,
    compute_hss,
    compute_persistence_skill,
)


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Shannon entropy of attention distribution.

    Used as an overfitting diagnostic: collapsing attention (low entropy)
    indicates the model fixates on a single timestep, which suggests
    overfitting or failure to learn meaningful temporal patterns.

    Args:
        attn_weights: (B, T) attention probabilities (sum to 1 along last dim).

    Returns:
        Mean entropy across batch (scalar float).
    """
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)  # (B,)
    return entropy.mean().item()


def accumulate_contingency_totals(totals, batch_contingency):
    """Add a batch's per-timestep contingency counts into running totals.

    `totals` is a list of [tp, fp, fn, tn] per timestep, or None on the
    first call. Returns the updated totals list.
    """
    if totals is None:
        return [[tp, fp, fn, tn] for tp, fp, fn, tn in batch_contingency]
    for t, (tp, fp, fn, tn) in enumerate(batch_contingency):
        totals[t][0] += tp
        totals[t][1] += fp
        totals[t][2] += fn
        totals[t][3] += tn
    return totals


def aggregate_metrics(
    total_loss: float,
    valid_batches: int,
    all_mae_per_timestep: List,
    all_rmse_per_timestep: List,
    all_correlation_per_timestep: List,
    all_ssim_per_timestep: List,
    all_peak_flux_error_per_timestep: List,
    all_persistence_mae_per_timestep: List,
    all_temporal_variation_ratios: List,
    model_contingency_totals: Optional[list],
    persistence_contingency_totals: Optional[list],
    temporal_attn_entropies: List[float],
) -> dict:
    """Reduce per-batch accumulators into the final validation metrics dict."""
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')

    avg_mae_per_timestep = (
        np.mean(all_mae_per_timestep, axis=0).tolist()
        if all_mae_per_timestep else []
    )
    avg_rmse_per_timestep = (
        np.mean(all_rmse_per_timestep, axis=0).tolist()
        if all_rmse_per_timestep else []
    )
    avg_correlation_per_timestep = (
        np.mean(all_correlation_per_timestep, axis=0).tolist()
        if all_correlation_per_timestep else []
    )
    avg_ssim_per_timestep = (
        np.mean(all_ssim_per_timestep, axis=0).tolist()
        if all_ssim_per_timestep else []
    )
    avg_peak_flux_error_per_timestep = (
        np.mean(all_peak_flux_error_per_timestep, axis=0).tolist()
        if all_peak_flux_error_per_timestep else []
    )
    avg_persistence_mae_per_timestep = (
        np.mean(all_persistence_mae_per_timestep, axis=0).tolist()
        if all_persistence_mae_per_timestep else []
    )
    avg_temporal_variation_ratio = float(
        np.mean(all_temporal_variation_ratios)
        if all_temporal_variation_ratios else 0.0
    )

    # CSI/HSS from accumulated contingency tables (NOT batch-averaged)
    val_csi_per_timestep = []
    val_hss_per_timestep = []
    if model_contingency_totals:
        for tp, fp, fn, tn in model_contingency_totals:
            val_csi_per_timestep.append(compute_csi(tp, fp, fn))
            val_hss_per_timestep.append(compute_hss(tp, fp, fn, tn))

    # Pool model contingency across timesteps for overall CSI/HSS
    if model_contingency_totals:
        total_tp = sum(c[0] for c in model_contingency_totals)
        total_fp = sum(c[1] for c in model_contingency_totals)
        total_fn = sum(c[2] for c in model_contingency_totals)
        total_tn = sum(c[3] for c in model_contingency_totals)
        val_csi = compute_csi(total_tp, total_fp, total_fn)
        val_hss = compute_hss(total_tp, total_fp, total_fn, total_tn)
    else:
        val_csi = 0.0
        val_hss = 0.0

    # Persistence CSI/HSS from accumulated contingency
    if persistence_contingency_totals:
        p_total_tp = sum(c[0] for c in persistence_contingency_totals)
        p_total_fp = sum(c[1] for c in persistence_contingency_totals)
        p_total_fn = sum(c[2] for c in persistence_contingency_totals)
        p_total_tn = sum(c[3] for c in persistence_contingency_totals)
        persistence_csi = compute_csi(p_total_tp, p_total_fp, p_total_fn)
        persistence_hss = compute_hss(p_total_tp, p_total_fp, p_total_fn, p_total_tn)
    else:
        persistence_csi = 0.0
        persistence_hss = 0.0

    # SSIM average across timesteps
    val_ssim = float(np.mean(avg_ssim_per_timestep)) if avg_ssim_per_timestep else 0.0

    # Persistence skill per timestep
    persistence_skill_per_timestep = []
    if avg_mae_per_timestep and avg_persistence_mae_per_timestep:
        for model_mae, pers_mae in zip(avg_mae_per_timestep, avg_persistence_mae_per_timestep):
            persistence_skill_per_timestep.append(
                compute_persistence_skill(model_mae, pers_mae)
            )

    metrics = {
        'val_loss': float(avg_loss),
        'val_mae_per_timestep': avg_mae_per_timestep,
        'val_rmse_per_timestep': avg_rmse_per_timestep,
        'val_correlation_per_timestep': avg_correlation_per_timestep,
        'val_csi': val_csi,
        'val_csi_per_timestep': val_csi_per_timestep,
        'val_hss': val_hss,
        'val_hss_per_timestep': val_hss_per_timestep,
        'val_ssim': val_ssim,
        'val_ssim_per_timestep': avg_ssim_per_timestep,
        'persistence_mae_per_timestep': avg_persistence_mae_per_timestep,
        'persistence_skill_per_timestep': persistence_skill_per_timestep,
        'persistence_csi': persistence_csi,
        'persistence_hss': persistence_hss,
        'peak_flux_error_per_timestep': avg_peak_flux_error_per_timestep,
        'temporal_variation_ratio': avg_temporal_variation_ratio,
    }

    # Add temporal attention entropy if captured (ARCH-07 diagnostic)
    if temporal_attn_entropies:
        metrics['temporal_attn_entropy'] = sum(temporal_attn_entropies) / len(temporal_attn_entropies)

    return metrics
