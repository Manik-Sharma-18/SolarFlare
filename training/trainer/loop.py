"""Main training loop: early stopping, checkpointing, LR scheduling, transfer learning."""
import logging
import sys
import json
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.device import clear_device_cache
from .epoch import train_epoch, NaNLossError
from .validation import validate
from .context import prepare_training
from .setup import unfreeze_and_rebuild
from .reporting import (
    log_epoch_summary, print_per_timestep_breakdown, update_history,
    save_epoch_checkpoints, save_emergency_checkpoint, install_shutdown_handlers,
)

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    normalization_params: dict = None
) -> Dict[str, List]:
    """Main training loop with early stopping and checkpointing.

    Returns the training-history dict with losses and metrics.
    """
    c = prepare_training(model, config, device, normalization_params)

    # Mutable working state pulled out of the context
    optimizer, scheduler, scaler = c.optimizer, c.scheduler, c.scaler
    best_val_loss, patience_counter = c.best_val_loss, c.patience_counter
    history, normalization_params = c.history, c.normalization_params
    _unfreeze_epoch = c.unfreeze_epoch
    latest_checkpoint_path = None

    is_shutdown_requested, restore_handlers = install_shutdown_handlers()

    try:
        for epoch in range(c.start_epoch, c.epochs + 1):
            # Unfreeze encoder at scheduled epoch (transfer learning)
            if _unfreeze_epoch is not None and epoch == _unfreeze_epoch:
                optimizer, scheduler, scaler, remaining = unfreeze_and_rebuild(
                    model, c.transfer_config, scheduler,
                    lr=c.lr, weight_decay=c.weight_decay, epochs=c.epochs, epoch=epoch,
                    use_amp=c.use_amp, scheduler_enabled=c.scheduler_enabled,
                    scheduler_type=c.scheduler_type, scheduler_config=c.scheduler_config,
                    device=device,
                )
                print(f"  ** Encoder unfrozen at epoch {epoch} (remaining: {remaining}) **")
                _unfreeze_epoch = None  # only unfreeze once

            # Teacher forcing schedule: linear decay over tf_decay_epochs
            # (defaults to total epochs). Shorter horizon → model reaches the
            # free-running regime before early-stop fires.
            tf_ratio = max(0.0, c.tf_start * (1 - epoch / c.tf_decay_epochs))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{c.epochs} | LR: {current_lr:.2e} | TF: {tf_ratio:.3f}")

            # Train (may raise NaNLossError)
            try:
                train_loss, _, avg_components = train_epoch(
                    model, train_loader, optimizer, scaler, device,
                    tf_ratio, epoch, c.loss_fn, c.use_amp, c.grad_clip, c.show_progress,
                    c.output_channels, c.max_consecutive_nan, c.grad_norm_warning_threshold
                )
            except NaNLossError:
                logger.error("NaN loss abort triggered. Saving emergency checkpoint.")
                save_emergency_checkpoint(
                    c.checkpoints_dir, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, patience_counter, normalization_params, config,
                    history, 'nan_loss_abort',
                )
                raise

            # Validate
            val_metrics = validate(
                model, val_loader, device, c.loss_fn, c.use_amp, c.show_progress,
                c.output_channels, extreme_threshold=c.extreme_threshold,
                ssim_data_range=c.ssim_data_range, verbose_metrics=c.verbose_metrics,
                epoch=epoch, total_epochs=c.epochs,
            )
            val_loss = val_metrics['val_loss']
            val_mae_per_timestep = val_metrics['val_mae_per_timestep']

            if c.scheduler_enabled:
                scheduler.step()

            log_epoch_summary(
                model, train_loss, avg_components, val_loss,
                val_mae_per_timestep, val_metrics,
            )
            if c.verbose_metrics or epoch == c.epochs:
                print_per_timestep_breakdown(val_mae_per_timestep, val_metrics)

            update_history(
                history, model, train_loss, val_loss, val_mae_per_timestep,
                current_lr, val_metrics, avg_components,
            )

            # Free device caches between epochs (GPU/MPS memory)
            clear_device_cache(device)

            # Checkpointing — best (on improvement) + rolling latest
            best_val_loss, patience_counter, latest_checkpoint_path = save_epoch_checkpoints(
                c.checkpoints_dir, latest_checkpoint_path,
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, val_loss=val_loss, best_val_loss=best_val_loss,
                patience_counter=patience_counter, patience=c.patience,
                normalization_params=normalization_params, config=config, history=history,
            )

            # Early stopping
            if patience_counter >= c.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Graceful shutdown check (after epoch completes)
            if is_shutdown_requested():
                save_emergency_checkpoint(
                    c.checkpoints_dir, epoch, model, optimizer, scheduler, scaler,
                    val_loss, patience_counter, normalization_params, config,
                    history, 'user_interrupt',
                )
                restore_handlers()
                sys.exit(0)

            print()

    finally:
        # Always restore original signal handlers (no leaked handlers)
        restore_handlers()

    # Save training history
    if config.get('save_history', True):
        history_path = c.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {history_path}")

    return history
