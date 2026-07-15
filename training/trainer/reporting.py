"""Console logging, history bookkeeping, banners, and shutdown/emergency-save helpers."""
import math
import signal
import sys

import numpy as np

from utils.checkpoint import save_checkpoint

# Legacy CompositeLoss keys retained so history rows in older runs stay
# populated. New components (bce, tgrad, iflux, sobel, spectral, …) are
# appended dynamically via setdefault in update_history.
_COMPONENT_KEYS = ['l1', 'ssim', 'extreme', 'temporal_diff', 'temporal_var', 'asymmetric']


def print_config_banner(epochs, device, use_amp, loss_config, scheduler_type,
                        tf_start, max_consecutive_nan, grad_norm_warning_threshold,
                        resume_from, start_epoch):
    """Print the training-configuration banner before the epoch loop."""
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


def log_epoch_summary(model, train_loss, avg_components, val_loss,
                      val_mae_per_timestep, val_metrics):
    """Print the per-epoch summary line block."""
    print(f"  Train Loss: {train_loss:.6f}")
    if avg_components:
        # Print whichever components the loss reported this epoch — keys vary
        # by loss class (CompositeLoss vs DualHeadLoss vs future variants).
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in avg_components.items())
        print(f"  Components: {parts}")
    print(f"  Val Loss:   {val_loss:.6f}")
    print(f"  Val MAE:    {val_mae_per_timestep}")
    cls_extra = f" | CLS-CSI: {val_metrics['val_csi_classifier']:.4f}" if 'val_csi_classifier' in val_metrics else ""
    print(f"  CSI: {val_metrics['val_csi']:.4f} | HSS: {val_metrics['val_hss']:.4f} | SSIM: {val_metrics['val_ssim']:.4f}{cls_extra}")
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
        max_entropy = math.log(10)  # T_in=10, max entropy = ln(10) = 2.303
        print(f"  Temporal attn entropy: {val_metrics['temporal_attn_entropy']:.3f} "
              f"(max: {max_entropy:.3f})")


def print_per_timestep_breakdown(val_mae_per_timestep, val_metrics):
    """Print the per-timestep metric table (final epoch / verbose mode)."""
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


def update_history(history, model, train_loss, val_loss, val_mae_per_timestep,
                   current_lr, val_metrics, avg_components):
    """Append this epoch's results to the training-history dict."""
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

    if 'val_csi_classifier' in val_metrics:
        history.setdefault('val_csi_classifier', []).append(val_metrics['val_csi_classifier'])
        history.setdefault('val_hss_classifier', []).append(val_metrics['val_hss_classifier'])
        history.setdefault('val_csi_classifier_per_timestep', []).append(
            val_metrics['val_csi_classifier_per_timestep'])

    # Per-component loss averages. Append zeros for legacy keys so older
    # downstream readers still find them; append actual values for whatever
    # keys this run's loss reported (auto-create the list via setdefault).
    reported = avg_components or {}
    for key in _COMPONENT_KEYS:
        history.setdefault(f'train_{key}', []).append(float(reported.get(key, 0.0)))
    for key, val in reported.items():
        if key in _COMPONENT_KEYS:
            continue
        history.setdefault(f'train_{key}', []).append(float(val))

    # delta_scale to history (ARCH-02)
    if hasattr(model, 'delta_scale') and model.delta_scale is not None:
        history['delta_scale'].append(model.delta_scale.item())

    # Temporal attention entropy to history (ARCH-07)
    if 'temporal_attn_entropy' in val_metrics:
        history['temporal_attn_entropy'].append(val_metrics['temporal_attn_entropy'])


def save_epoch_checkpoints(checkpoints_dir, latest_checkpoint_path, *, epoch, model,
                           optimizer, scheduler, scaler, val_loss, best_val_loss,
                           patience_counter, patience, normalization_params,
                           config, history, score_for_patience=None):
    """Save best (on improvement) and rolling-latest checkpoints for this epoch.

    Returns (best_val_loss, patience_counter, latest_checkpoint_path).
    """
    score = val_loss if score_for_patience is None else score_for_patience
    if score < best_val_loss:
        best_val_loss = score
        save_checkpoint(
            filepath=checkpoints_dir / 'best_model.pt',
            epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            best_val_loss=score, patience_counter=0,
            normalization_params=normalization_params, config=config,
            history=history
        )
        print(f"  Saved best model (score: {score:.6f}, val_loss: {val_loss:.6f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{patience})")

    new_path = checkpoints_dir / f"checkpoint_epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
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
    return best_val_loss, patience_counter, new_path


def save_emergency_checkpoint(checkpoints_dir, epoch_num, model, optimizer, scheduler,
                              scaler, val_loss_val, patience_counter,
                              normalization_params, config, history, reason):
    """Save a checkpoint flagged as interrupted/emergency."""
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


def install_shutdown_handlers():
    """Install SIGINT/SIGTERM handlers for graceful shutdown.

    Returns (is_requested, restore): a callable reporting whether shutdown
    was requested, and a callable restoring the previous handlers.
    """
    state = {'requested': False}

    def _handler(signum, frame):
        if state['requested']:
            # Second signal = force quit immediately
            print("\nForce quit requested. Exiting immediately without saving.")
            sys.exit(1)
        state['requested'] = True
        sig_name = signal.Signals(signum).name
        print(f"\n{sig_name} received. Saving emergency checkpoint after current epoch...")
        print("(Press Ctrl+C again to force quit immediately)")

    old_sigint = signal.signal(signal.SIGINT, _handler)
    old_sigterm = signal.signal(signal.SIGTERM, _handler)

    def restore():
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    return (lambda: state['requested']), restore
