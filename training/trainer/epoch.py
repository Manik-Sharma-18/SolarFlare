"""Single-epoch training loop with NaN detection and gradient monitoring."""
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from utils.device import get_amp_context
from ..losses import CompositeLoss

logger = logging.getLogger(__name__)


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
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_steps = 0

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

            # Dual-head model returns (regression_pred, extreme_logits); single-
            # head returns regression_pred only. Downstream code uses
            # ``predictions`` as the regression tensor.
            ext_logits = None
            if isinstance(predictions, tuple):
                predictions, ext_logits = predictions

            # For dual-channel: compare only flux channel (first output_channels)
            # Model outputs: (B, output_channels, T, H, W)
            # Y_out may be: (B, 1 or 2, T, H, W)
            Y_target = Y_out[:, :output_channels]

            loss_kwargs = {"ext_logits": ext_logits} if ext_logits is not None else {}
            if use_component_tracking:
                components = loss_fn(predictions, Y_target, return_components=True,
                                     **loss_kwargs)
                loss = components['total']
            else:
                loss = loss_fn(predictions, Y_target, **loss_kwargs)

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
        gn_val = float(total_norm)
        if gn_val == gn_val:  # not NaN
            grad_norm_sum += gn_val
            grad_norm_max = max(grad_norm_max, gn_val)
            grad_steps += 1
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
    if grad_steps > 0:
        avg_grad_norm = grad_norm_sum / grad_steps
        print(
            f"  Grad norm  mean={avg_grad_norm:.4f}  max={grad_norm_max:.4f}  "
            f"clip={grad_clip}"
        )
    return avg_loss, consecutive_nan_count, avg_components
