"""Validation pass: per-timestep metrics, persistence baseline, attention entropy."""
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from utils.device import get_amp_context
from utils.metrics import (
    compute_metrics,
    accumulate_contingency,
    accumulate_contingency_from_logits,
    compute_persistence_prediction,
    compute_ssim_per_timestep,
    compute_peak_flux_error,
    compute_temporal_variation_ratio,
    compute_rmse_per_timestep,
    compute_correlation_per_timestep,
)
from .val_aggregation import (
    compute_attention_entropy,
    accumulate_contingency_totals,
    aggregate_metrics,
)

logger = logging.getLogger(__name__)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    show_progress: bool = True,
    output_channels: int = 1,
    extreme_threshold: float = 0.277,
    ssim_data_range: float = 2.0,
    verbose_metrics: bool = False,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> dict:
    """
    Validate model on a dataset.

    Returns a structured dict containing all evaluation metrics including
    per-timestep MAE, RMSE, correlation, CSI, HSS, SSIM, persistence
    baseline comparison, peak flux error, and temporal variation ratio.
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    nan_batches = 0

    # Running accumulators (will be averaged over batches)
    all_mae_per_timestep = []
    all_rmse_per_timestep = []
    all_correlation_per_timestep = []
    all_ssim_per_timestep = []
    all_peak_flux_error_per_timestep = []
    all_persistence_mae_per_timestep = []
    all_temporal_variation_ratios = []

    # Contingency table accumulators (summed across ALL batches, then CSI/HSS computed once)
    model_contingency_totals = None  # list of [tp, fp, fn, tn] per timestep
    persistence_contingency_totals = None
    # Dual-head classifier CSI: prediction = sigmoid(logits) > 0.5
    classifier_contingency_totals = None

    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()

    # Collect attention entropy if temporal attention is available (ARCH-07 diagnostic)
    temporal_attn_entropies = []
    _attn_hook_handle = None

    if hasattr(model, 'temporal_attn') and hasattr(model, 'use_temporal_attention') and model.use_temporal_attention:
        def _capture_attn_weights(module, input, output):
            context, attn_weights = output
            temporal_attn_entropies.append(compute_attention_entropy(attn_weights))

        _attn_hook_handle = model.temporal_attn.register_forward_hook(_capture_attn_weights)

    iterator = tqdm(dataloader, desc="Validating") if show_progress else dataloader

    with torch.no_grad():
        for batch_idx, (X_in, Y_out, _) in enumerate(iterator):
            X_in = X_in.to(device)
            Y_out = Y_out.to(device)

            with get_amp_context(use_amp, device):
                predictions = model(X_in, teacher_forcing_ratio=0.0)

                # Dual-head model returns (regression_pred, extreme_logits);
                # unpack so downstream metrics keep using regression tensor.
                ext_logits = None
                if isinstance(predictions, tuple):
                    predictions, ext_logits = predictions

                # For dual-channel: compare only flux channel
                Y_target = Y_out[:, :output_channels]
                loss_kwargs = ({"ext_logits": ext_logits}
                               if ext_logits is not None else {})
                loss = loss_fn(predictions, Y_target, **loss_kwargs)

            # Skip NaN/Inf losses in validation (log but don't abort)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                logger.warning(
                    "NaN/Inf loss in validation batch %d. Skipping.", batch_idx + 1
                )
                continue

            total_loss += loss.item()
            valid_batches += 1

            # --- Per-timestep MAE ---
            metrics = compute_metrics(predictions, Y_target)
            all_mae_per_timestep.append(metrics['mae_per_timestep'])

            # --- Per-timestep RMSE ---
            all_rmse_per_timestep.append(
                compute_rmse_per_timestep(predictions, Y_target)
            )

            # --- Per-timestep Correlation ---
            all_correlation_per_timestep.append(
                compute_correlation_per_timestep(predictions, Y_target)
            )

            # --- Contingency table for model predictions (accumulate across batches) ---
            model_contingency_totals = accumulate_contingency_totals(
                model_contingency_totals,
                accumulate_contingency(predictions, Y_target, extreme_threshold),
            )

            # --- Classifier-head contingency (dual-head only) ---
            if ext_logits is not None:
                classifier_contingency_totals = accumulate_contingency_totals(
                    classifier_contingency_totals,
                    accumulate_contingency_from_logits(
                        ext_logits, Y_target, extreme_threshold,
                    ),
                )

            # --- Persistence baseline ---
            T_out = predictions.shape[2]
            persistence_pred = compute_persistence_prediction(
                X_in, output_channels, T_out
            ).to(device)

            # Persistence MAE per timestep
            persistence_metrics = compute_metrics(persistence_pred, Y_target)
            all_persistence_mae_per_timestep.append(persistence_metrics['mae_per_timestep'])

            # Persistence contingency table
            persistence_contingency_totals = accumulate_contingency_totals(
                persistence_contingency_totals,
                accumulate_contingency(persistence_pred, Y_target, extreme_threshold),
            )

            # --- SSIM per timestep ---
            all_ssim_per_timestep.append(
                compute_ssim_per_timestep(predictions, Y_target, data_range=ssim_data_range)
            )

            # --- Peak flux error per timestep ---
            all_peak_flux_error_per_timestep.append(
                compute_peak_flux_error(predictions, Y_target)
            )

            # --- Temporal variation ratio ---
            all_temporal_variation_ratios.append(
                compute_temporal_variation_ratio(predictions, Y_target)
            )

    # Remove attention hook (must be outside torch.no_grad context)
    if _attn_hook_handle is not None:
        _attn_hook_handle.remove()

    if nan_batches > 0:
        logger.warning(
            "Validation had %d NaN/Inf batches out of %d total.",
            nan_batches, len(dataloader)
        )

    metrics = aggregate_metrics(
        total_loss, valid_batches,
        all_mae_per_timestep, all_rmse_per_timestep, all_correlation_per_timestep,
        all_ssim_per_timestep, all_peak_flux_error_per_timestep,
        all_persistence_mae_per_timestep, all_temporal_variation_ratios,
        model_contingency_totals, persistence_contingency_totals,
        temporal_attn_entropies,
    )
    # Dual-head: classifier-head CSI/HSS (sigmoid > 0.5 = positive). Reported
    # alongside the regression-derived val_csi so both signals are visible.
    if classifier_contingency_totals:
        from utils.metrics import compute_csi, compute_hss
        tp = sum(c[0] for c in classifier_contingency_totals)
        fp = sum(c[1] for c in classifier_contingency_totals)
        fn = sum(c[2] for c in classifier_contingency_totals)
        tn = sum(c[3] for c in classifier_contingency_totals)
        metrics["val_csi_classifier"] = compute_csi(tp, fp, fn)
        metrics["val_hss_classifier"] = compute_hss(tp, fp, fn, tn)
        metrics["val_csi_classifier_per_timestep"] = [
            compute_csi(c[0], c[1], c[2]) for c in classifier_contingency_totals
        ]
    return metrics
