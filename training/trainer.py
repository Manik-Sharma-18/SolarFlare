"""
Training loop and validation for the solar flux predictor.
"""
import logging
import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from utils.device import get_amp_context, get_grad_scaler, clear_device_cache
from utils.checkpoint import save_checkpoint, load_checkpoint_for_resume
from utils.metrics import (
    compute_metrics,
    accumulate_contingency,
    compute_csi,
    compute_hss,
    compute_persistence_prediction,
    compute_persistence_skill,
    compute_ssim_per_timestep,
    compute_peak_flux_error,
    compute_temporal_variation_ratio,
    compute_rmse_per_timestep,
    compute_correlation_per_timestep,
)
from .losses import get_loss_function, CompositeLoss

logger = logging.getLogger(__name__)


def _compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> float:
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


class NaNLossError(RuntimeError):
    """Raised when consecutive NaN/Inf losses exceed the configured threshold."""
    pass


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    teacher_forcing_ratio: float,
    epoch: int,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    show_progress: bool = True,
    output_channels: int = 1,
    max_consecutive_nan: int = 10,
    grad_norm_warning_threshold: float = 100.0
) -> Tuple[float, int, Optional[Dict[str, float]]]:
    """
    Train for one epoch with NaN detection and gradient monitoring.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for AMP
        device: Device to train on
        teacher_forcing_ratio: TF ratio for this epoch
        epoch: Current epoch number
        loss_fn: Loss function (default: L1)
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping norm
        show_progress: Show progress bar
        output_channels: Number of output channels (for dual-channel: only compare first N)
        max_consecutive_nan: Abort training after this many consecutive NaN/Inf losses
        grad_norm_warning_threshold: Log warning when gradient norm exceeds this value

    Returns:
        Tuple of (average_loss, consecutive_nan_count, avg_components) for the epoch.
        Average loss excludes NaN batches. consecutive_nan_count is the
        running count at epoch end (resets on any good batch).
        avg_components is a dict with per-component averages when loss_fn is
        CompositeLoss, or None otherwise.
    """
    model.train()
    total_loss = 0.0
    valid_batches = 0
    consecutive_nan_count = 0

    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()

    # Per-component tracking for CompositeLoss
    use_component_tracking = isinstance(loss_fn, CompositeLoss)
    component_keys = ['l1', 'ssim', 'extreme', 'temporal_diff', 'temporal_var', 'asymmetric']
    component_sums = {}
    if use_component_tracking:
        for key in component_keys:
            component_sums[key] = 0.0

    iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if show_progress else dataloader

    for batch_idx, (X_in, Y_out, _) in enumerate(iterator):
        X_in = X_in.to(device)
        Y_out = Y_out.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP
        with get_amp_context(use_amp, device):
            predictions = model(X_in, teacher_forcing_ratio, Y_out)

            # For dual-channel: compare only flux channel (first output_channels)
            # Model outputs: (B, output_channels, T, H, W)
            # Y_out may be: (B, 1 or 2, T, H, W)
            Y_target = Y_out[:, :output_channels]

            if use_component_tracking:
                components = loss_fn(predictions, Y_target, return_components=True)
                loss = components['total']
            else:
                loss = loss_fn(predictions, Y_target)

        # NaN/Inf detection — skip batch before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            consecutive_nan_count += 1
            logger.warning(
                "NaN/Inf loss detected at epoch %d, batch %d. "
                "Consecutive: %d/%d. Skipping batch.",
                epoch, batch_idx + 1, consecutive_nan_count, max_consecutive_nan
            )
            if consecutive_nan_count >= max_consecutive_nan:
                raise NaNLossError(
                    f"Training aborted: {consecutive_nan_count} consecutive NaN/Inf losses. "
                    f"Check learning rate, data normalization, or loss function."
                )
            continue  # Skip this batch entirely (no backward, no optimizer step)
        else:
            consecutive_nan_count = 0  # Reset on good batch

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient norm monitoring — clip_grad_norm_ returns the total norm before clipping
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm > grad_norm_warning_threshold:
            logger.warning(
                "High gradient norm: %.2f (threshold: %.2f). Epoch %d, batch %d.",
                total_norm, grad_norm_warning_threshold, epoch, batch_idx + 1
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        valid_batches += 1

        if use_component_tracking:
            for key in component_keys:
                component_sums[key] += components[key].item()

        if show_progress:
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')
    avg_components = None
    if use_component_tracking and valid_batches > 0:
        avg_components = {key: component_sums[key] / valid_batches for key in component_keys}
    return avg_loss, consecutive_nan_count, avg_components


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    show_progress: bool = True,
    output_channels: int = 1,
    extreme_threshold: float = 0.3456,
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

    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to use
        loss_fn: Loss function (default: L1)
        use_amp: Use automatic mixed precision
        show_progress: Show progress bar
        output_channels: Number of output channels
        extreme_threshold: Threshold for CSI/HSS binary classification
        ssim_data_range: Data range for SSIM computation
        verbose_metrics: Print per-timestep breakdown every epoch
        epoch: Current epoch number (for verbose/final-epoch logic)
        total_epochs: Total number of epochs (for final-epoch detection)

    Returns:
        Dict with keys: val_loss, val_mae_per_timestep, val_rmse_per_timestep,
        val_correlation_per_timestep, val_csi, val_csi_per_timestep, val_hss,
        val_hss_per_timestep, val_ssim, val_ssim_per_timestep,
        persistence_mae_per_timestep, persistence_skill_per_timestep,
        persistence_csi, persistence_hss, peak_flux_error_per_timestep,
        temporal_variation_ratio.
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

    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()

    # Collect attention entropy if temporal attention is available (ARCH-07 diagnostic)
    temporal_attn_entropies = []
    _attn_hook_handle = None

    if hasattr(model, 'temporal_attn') and hasattr(model, 'use_temporal_attention') and model.use_temporal_attention:
        def _capture_attn_weights(module, input, output):
            context, attn_weights = output
            temporal_attn_entropies.append(_compute_attention_entropy(attn_weights))

        _attn_hook_handle = model.temporal_attn.register_forward_hook(_capture_attn_weights)

    iterator = tqdm(dataloader, desc="Validating") if show_progress else dataloader

    with torch.no_grad():
        for batch_idx, (X_in, Y_out, _) in enumerate(iterator):
            X_in = X_in.to(device)
            Y_out = Y_out.to(device)

            with get_amp_context(use_amp, device):
                predictions = model(X_in, teacher_forcing_ratio=0.0)

                # For dual-channel: compare only flux channel
                Y_target = Y_out[:, :output_channels]
                loss = loss_fn(predictions, Y_target)

            # Skip NaN/Inf losses in validation (log but don't abort)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                logger.warning(
                    "NaN/Inf loss in validation batch %d. Skipping.", batch_idx + 1
                )
                continue

            total_loss += loss.item()
            valid_batches += 1

            # --- Per-timestep MAE (existing) ---
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
            batch_contingency = accumulate_contingency(
                predictions, Y_target, extreme_threshold
            )
            if model_contingency_totals is None:
                model_contingency_totals = [[tp, fp, fn, tn] for tp, fp, fn, tn in batch_contingency]
            else:
                for t, (tp, fp, fn, tn) in enumerate(batch_contingency):
                    model_contingency_totals[t][0] += tp
                    model_contingency_totals[t][1] += fp
                    model_contingency_totals[t][2] += fn
                    model_contingency_totals[t][3] += tn

            # --- Persistence baseline ---
            T_out = predictions.shape[2]
            persistence_pred = compute_persistence_prediction(
                X_in, output_channels, T_out
            ).to(device)

            # Persistence MAE per timestep
            persistence_metrics = compute_metrics(persistence_pred, Y_target)
            all_persistence_mae_per_timestep.append(persistence_metrics['mae_per_timestep'])

            # Persistence contingency table
            batch_persistence_contingency = accumulate_contingency(
                persistence_pred, Y_target, extreme_threshold
            )
            if persistence_contingency_totals is None:
                persistence_contingency_totals = [
                    [tp, fp, fn, tn] for tp, fp, fn, tn in batch_persistence_contingency
                ]
            else:
                for t, (tp, fp, fn, tn) in enumerate(batch_persistence_contingency):
                    persistence_contingency_totals[t][0] += tp
                    persistence_contingency_totals[t][1] += fp
                    persistence_contingency_totals[t][2] += fn
                    persistence_contingency_totals[t][3] += tn

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

    # --- Compute final metrics from accumulated values ---
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    normalization_params: dict = None
) -> Dict[str, List]:
    """
    Main training loop with early stopping and checkpointing.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Device to train on
    
    Returns:
        Training history dict with losses and metrics
    """
    # Extract config
    epochs = config.get('epochs', 25)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    tf_start = config.get('tf_start', 0.5)
    patience = config.get('patience', 8)
    use_amp = config.get('use_amp', True)
    grad_clip = config.get('grad_clip', 1.0)
    save_dir = Path(config.get('save_dir', './outputs'))
    checkpoint_name = config.get('checkpoint_name', 'best_model.pt')
    show_progress = config.get('show_progress', True)
    output_channels = config.get('output_channels', 1)

    # Error handling config
    error_handling = config.get('error_handling', {})
    max_consecutive_nan = error_handling.get('max_consecutive_nan', 10)
    grad_norm_warning_threshold = error_handling.get('grad_norm_warning_threshold', 100.0)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create loss function from config
    loss_config = config.get('loss', {'type': 'l1'})
    loss_fn = get_loss_function(loss_config)
    loss_fn = loss_fn.to(device)
    
    # Optimizer with parameter groups (exclude delta_scale from weight decay)
    if hasattr(model, 'delta_scale') and model.delta_scale is not None:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'delta_scale':
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    # Create scheduler based on config
    scheduler_config = config.get('scheduler', {'type': 'cosine'})
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        eta_min = scheduler_config.get('cosine_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
        scheduler_enabled = True
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('step_gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        scheduler_enabled = True
    elif scheduler_type == 'constant' or scheduler_type == 'none':
        scheduler = None
        scheduler_enabled = False
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Gradient scaler for mixed precision
    scaler = get_grad_scaler(use_amp, device)
    
    # Evaluation config
    eval_config = config.get('evaluation', {})
    extreme_threshold = eval_config.get('extreme_threshold', 0.3456)
    verbose_metrics = eval_config.get('verbose_metrics', False)
    ssim_data_range = config.get('loss', {}).get('ssim_data_range', 2.0)

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_per_timestep': [],
        'learning_rate': [],
        'val_csi': [],
        'val_hss': [],
        'val_ssim': [],
        'val_ssim_per_timestep': [],
        'persistence_skill_per_timestep': [],
        'persistence_csi': [],
        'persistence_hss': [],
        'peak_flux_error_per_timestep': [],
        'temporal_variation_ratio': [],
        'val_rmse_per_timestep': [],
        'val_correlation_per_timestep': [],
        'val_csi_per_timestep': [],
        'val_hss_per_timestep': [],
        'persistence_mae_per_timestep': [],
        'train_l1': [],
        'train_ssim': [],
        'train_extreme': [],
        'train_temporal_diff': [],
        'train_temporal_var': [],
        'train_asymmetric': [],
        'delta_scale': [],
        'temporal_attn_entropy': [],
    }

    # Resume from checkpoint if requested
    start_epoch = 1
    resume_from = config.get('resume_from')
    if resume_from:
        resume_path = Path(resume_from)
        logger.info("Resuming training from checkpoint: %s", resume_path)
        start_epoch, best_val_loss, patience_counter, history, ckpt_norm_params = (
            load_checkpoint_for_resume(
                filepath=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                current_config=config,
            )
        )
        # Use checkpoint normalization params if not explicitly provided
        if normalization_params is None:
            normalization_params = ckpt_norm_params
        print(f"Resumed from epoch {start_epoch - 1}. Continuing to epoch {epochs}.")

    # Checkpoint directory setup
    checkpoints_dir = save_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = None  # Track current latest for deletion

    print(f"\nStarting training for {epochs} epochs")
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    print(f"  Loss: {loss_config.get('type', 'l1')}")
    print(f"  LR Scheduler: {scheduler_type}")
    print(f"  Teacher forcing: {tf_start} -> 0.0")
    print(f"  NaN abort threshold: {max_consecutive_nan}")
    print(f"  Grad norm warning: {grad_norm_warning_threshold}")
    if resume_from:
        print(f"  Resuming from epoch: {start_epoch}")
    print()

    # --- Graceful shutdown via signal handling ---
    _shutdown_requested = False

    def _shutdown_handler(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Second signal = force quit immediately
            print("\nForce quit requested. Exiting immediately without saving.")
            sys.exit(1)
        _shutdown_requested = True
        sig_name = signal.Signals(signum).name
        print(f"\n{sig_name} received. Saving emergency checkpoint after current epoch...")
        print("(Press Ctrl+C again to force quit immediately)")

    old_sigint = signal.signal(signal.SIGINT, _shutdown_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _shutdown_handler)

    def _save_emergency_checkpoint(epoch_num, val_loss_val, reason):
        """Save an emergency checkpoint with metadata marking it as interrupted."""
        emergency_filename = f"EMERGENCY_checkpoint_epoch_{epoch_num:03d}.pt"
        emergency_path = checkpoints_dir / emergency_filename
        save_checkpoint(
            filepath=emergency_path, epoch=epoch_num, model=model,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_val_loss=val_loss_val, patience_counter=patience_counter,
            normalization_params=normalization_params, config=config,
            history=history, emergency=True, emergency_reason=reason
        )
        print(f"\nEmergency checkpoint saved to: {emergency_path}")

    try:
        for epoch in range(start_epoch, epochs + 1):
            # Teacher forcing schedule: linear decay
            tf_ratio = max(0.0, tf_start * (1 - epoch / epochs))
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.2e} | TF: {tf_ratio:.3f}")

            # Train (may raise NaNLossError)
            try:
                train_loss, _, avg_components = train_epoch(
                    model, train_loader, optimizer, scaler, device,
                    tf_ratio, epoch, loss_fn, use_amp, grad_clip, show_progress,
                    output_channels, max_consecutive_nan, grad_norm_warning_threshold
                )
            except NaNLossError:
                logger.error("NaN loss abort triggered. Saving emergency checkpoint.")
                _save_emergency_checkpoint(epoch, best_val_loss, 'nan_loss_abort')
                raise

            # Validate
            val_metrics = validate(
                model, val_loader, device, loss_fn, use_amp, show_progress,
                output_channels, extreme_threshold=extreme_threshold,
                ssim_data_range=ssim_data_range, verbose_metrics=verbose_metrics,
                epoch=epoch, total_epochs=epochs,
            )
            val_loss = val_metrics['val_loss']
            val_mae_per_timestep = val_metrics['val_mae_per_timestep']

            # Update scheduler
            if scheduler_enabled:
                scheduler.step()

            # Log epoch summary
            print(f"  Train Loss: {train_loss:.6f}")
            if avg_components:
                print(f"  Loss: {train_loss:.6f} | TDiff: {avg_components['temporal_diff']:.4f}"
                      f" | TVar: {avg_components['temporal_var']:.4f}"
                      f" | Extreme: {avg_components['extreme']:.4f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_mae_per_timestep}")
            print(f"  CSI: {val_metrics['val_csi']:.4f} | HSS: {val_metrics['val_hss']:.4f}"
                  f" | SSIM: {val_metrics['val_ssim']:.4f}")
            avg_persist_skill = (
                np.mean(val_metrics['persistence_skill_per_timestep'])
                if val_metrics['persistence_skill_per_timestep'] else 0.0
            )
            print(f"  Persistence Skill: {avg_persist_skill:.1f}%"
                  f" | Temporal Var Ratio: {val_metrics['temporal_variation_ratio']:.3f}")

            # Log delta_scale value if it exists (ARCH-02 diagnostic)
            if hasattr(model, 'delta_scale') and model.delta_scale is not None:
                print(f"  delta_scale: {model.delta_scale.item():.4f}")

            # Log temporal attention entropy (ARCH-07 overfitting diagnostic)
            if 'temporal_attn_entropy' in val_metrics:
                import math
                max_entropy = math.log(10)  # T_in=10, max entropy = ln(10) = 2.303
                print(f"  Temporal attn entropy: {val_metrics['temporal_attn_entropy']:.3f} "
                      f"(max: {max_entropy:.3f})")

            # Per-timestep breakdown at final epoch or if verbose_metrics
            is_final_epoch = (epoch == epochs)
            if verbose_metrics or is_final_epoch:
                T = len(val_mae_per_timestep)
                print("  --- Per-Timestep Breakdown ---")
                for t in range(T):
                    rmse_t = val_metrics['val_rmse_per_timestep'][t] if t < len(val_metrics['val_rmse_per_timestep']) else 0.0
                    corr_t = val_metrics['val_correlation_per_timestep'][t] if t < len(val_metrics['val_correlation_per_timestep']) else 0.0
                    csi_t = val_metrics['val_csi_per_timestep'][t] if t < len(val_metrics['val_csi_per_timestep']) else 0.0
                    hss_t = val_metrics['val_hss_per_timestep'][t] if t < len(val_metrics['val_hss_per_timestep']) else 0.0
                    ssim_t = val_metrics['val_ssim_per_timestep'][t] if t < len(val_metrics['val_ssim_per_timestep']) else 0.0
                    skill_t = val_metrics['persistence_skill_per_timestep'][t] if t < len(val_metrics['persistence_skill_per_timestep']) else 0.0
                    pfe_t = val_metrics['peak_flux_error_per_timestep'][t] if t < len(val_metrics['peak_flux_error_per_timestep']) else 0.0
                    print(f"    t+{t+1}: MAE={val_mae_per_timestep[t]:.4f} RMSE={rmse_t:.4f}"
                          f" Corr={corr_t:.4f} CSI={csi_t:.4f} HSS={hss_t:.4f}"
                          f" SSIM={ssim_t:.4f} Skill={skill_t:.1f}% PFE={pfe_t:.4f}")

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae_per_timestep'].append(
                val_mae_per_timestep if isinstance(val_mae_per_timestep, list)
                else val_mae_per_timestep.tolist() if hasattr(val_mae_per_timestep, 'tolist')
                else []
            )
            history['learning_rate'].append(current_lr)
            history['val_csi'].append(val_metrics['val_csi'])
            history['val_hss'].append(val_metrics['val_hss'])
            history['val_ssim'].append(val_metrics['val_ssim'])
            history['val_ssim_per_timestep'].append(val_metrics['val_ssim_per_timestep'])
            history['persistence_skill_per_timestep'].append(val_metrics['persistence_skill_per_timestep'])
            history['persistence_csi'].append(val_metrics['persistence_csi'])
            history['persistence_hss'].append(val_metrics['persistence_hss'])
            history['peak_flux_error_per_timestep'].append(val_metrics['peak_flux_error_per_timestep'])
            history['temporal_variation_ratio'].append(val_metrics['temporal_variation_ratio'])
            history['val_rmse_per_timestep'].append(val_metrics['val_rmse_per_timestep'])
            history['val_correlation_per_timestep'].append(val_metrics['val_correlation_per_timestep'])
            history['val_csi_per_timestep'].append(val_metrics['val_csi_per_timestep'])
            history['val_hss_per_timestep'].append(val_metrics['val_hss_per_timestep'])
            history['persistence_mae_per_timestep'].append(val_metrics['persistence_mae_per_timestep'])

            # Log per-component loss averages
            component_keys = ['l1', 'ssim', 'extreme', 'temporal_diff', 'temporal_var', 'asymmetric']
            if avg_components:
                for key in component_keys:
                    history[f'train_{key}'].append(avg_components[key])
            else:
                for key in component_keys:
                    history[f'train_{key}'].append(0.0)

            # Log delta_scale to history (ARCH-02)
            if hasattr(model, 'delta_scale') and model.delta_scale is not None:
                history['delta_scale'].append(model.delta_scale.item())

            # Log temporal attention entropy to history (ARCH-07)
            if 'temporal_attn_entropy' in val_metrics:
                history['temporal_attn_entropy'].append(val_metrics['temporal_attn_entropy'])

            # Free device caches between epochs (GPU/MPS memory)
            clear_device_cache(device)

            # Checkpointing — best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    filepath=checkpoints_dir / 'best_model.pt',
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    best_val_loss=val_loss, patience_counter=0,
                    normalization_params=normalization_params, config=config,
                    history=history
                )
                print(f"  Saved best model (val_loss: {val_loss:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")

            # Checkpointing — latest (rolling, delete old)
            new_filename = f"checkpoint_epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
            new_path = checkpoints_dir / new_filename
            save_checkpoint(
                filepath=new_path, epoch=epoch, model=model,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_val_loss=best_val_loss, patience_counter=patience_counter,
                normalization_params=normalization_params, config=config,
                history=history
            )
            if (latest_checkpoint_path is not None
                    and latest_checkpoint_path.exists()
                    and latest_checkpoint_path != checkpoints_dir / 'best_model.pt'):
                latest_checkpoint_path.unlink()
            latest_checkpoint_path = new_path

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Graceful shutdown check (after epoch completes)
            if _shutdown_requested:
                _save_emergency_checkpoint(epoch, val_loss, 'user_interrupt')
                # Restore original signal handlers before exiting
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
                sys.exit(0)

            print()

    finally:
        # Always restore original signal handlers (no leaked handlers)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    # Save training history
    if config.get('save_history', True):
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {history_path}")

    return history



