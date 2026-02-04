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
from utils.metrics import compute_metrics
from .losses import get_loss_function, CompositeLoss

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
) -> Tuple[float, int]:
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
        Tuple of (average_loss, consecutive_nan_count) for the epoch.
        Average loss excludes NaN batches. consecutive_nan_count is the
        running count at epoch end (resets on any good batch).
    """
    model.train()
    total_loss = 0.0
    valid_batches = 0
    consecutive_nan_count = 0

    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()

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

        if show_progress:
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')
    return avg_loss, consecutive_nan_count


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    show_progress: bool = True,
    output_channels: int = 1
) -> tuple:
    """
    Validate model on a dataset.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to use
        loss_fn: Loss function (default: L1)
        use_amp: Use automatic mixed precision
        show_progress: Show progress bar
        output_channels: Number of output channels
    
    Returns:
        avg_loss: Average validation loss
        avg_mae_per_timestep: MAE for each output timestep
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    nan_batches = 0
    all_mae_per_timestep = []

    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()

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
            metrics = compute_metrics(predictions, Y_target)
            all_mae_per_timestep.append(metrics['mae_per_timestep'])

    if nan_batches > 0:
        logger.warning(
            "Validation had %d NaN/Inf batches out of %d total.",
            nan_batches, len(dataloader)
        )

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')
    avg_mae_per_timestep = np.mean(all_mae_per_timestep, axis=0) if all_mae_per_timestep else np.array([])

    return avg_loss, avg_mae_per_timestep


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
    
    # Optimizer and scheduler
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
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_per_timestep': [],
        'learning_rate': []
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
                train_loss, _ = train_epoch(
                    model, train_loader, optimizer, scaler, device,
                    tf_ratio, epoch, loss_fn, use_amp, grad_clip, show_progress,
                    output_channels, max_consecutive_nan, grad_norm_warning_threshold
                )
            except NaNLossError:
                logger.error("NaN loss abort triggered. Saving emergency checkpoint.")
                _save_emergency_checkpoint(epoch, best_val_loss, 'nan_loss_abort')
                raise

            # Validate
            val_loss, val_mae_per_timestep = validate(
                model, val_loader, device, loss_fn, use_amp, show_progress, output_channels
            )

            # Update scheduler
            if scheduler_enabled:
                scheduler.step()

            # Log
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_mae_per_timestep}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae_per_timestep'].append(val_mae_per_timestep.tolist()
                if hasattr(val_mae_per_timestep, 'tolist') else [])
            history['learning_rate'].append(current_lr)

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



