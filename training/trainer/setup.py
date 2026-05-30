"""Pre-loop builders for train_model: optimizer, scheduler, history, transfer learning."""
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from utils.device import get_grad_scaler
from utils.checkpoint import load_checkpoint_for_resume

logger = logging.getLogger(__name__)


def maybe_resume(model, optimizer, scheduler, scaler, config, device,
                 best_val_loss, patience_counter, history, normalization_params):
    """Resume from a checkpoint if config['resume_from'] is set.

    Returns (start_epoch, best_val_loss, patience_counter, history, normalization_params).
    """
    start_epoch = 1
    resume_from = config.get('resume_from')
    if not resume_from:
        return start_epoch, best_val_loss, patience_counter, history, normalization_params

    resume_path = Path(resume_from)
    logger.info("Resuming training from checkpoint: %s", resume_path)
    start_epoch, best_val_loss, patience_counter, history, ckpt_norm_params = (
        load_checkpoint_for_resume(
            filepath=resume_path, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device,
            current_config=config,
        )
    )
    if normalization_params is None:
        normalization_params = ckpt_norm_params
    print(f"Resumed from epoch {start_epoch - 1}. Continuing to epoch {config.get('epochs', 25)}.")
    return start_epoch, best_val_loss, patience_counter, history, normalization_params


def build_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW optimizer; excludes delta_scale (ARCH-02) from weight decay."""
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
        return torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler_config: Dict[str, Any], epochs: int) -> Tuple:
    """Build LR scheduler from config. Returns (scheduler, enabled, type)."""
    scheduler_type = scheduler_config.get('type', 'cosine').lower()

    if scheduler_type == 'cosine':
        eta_min = scheduler_config.get('cosine_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
        return scheduler, True, scheduler_type
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('step_gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        return scheduler, True, scheduler_type
    elif scheduler_type == 'constant' or scheduler_type == 'none':
        return None, False, scheduler_type
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def init_history() -> Dict[str, list]:
    """Empty training-history dict with all tracked keys."""
    return {
        'train_loss': [], 'val_loss': [], 'val_mae_per_timestep': [],
        'learning_rate': [], 'val_csi': [], 'val_hss': [], 'val_ssim': [],
        'val_ssim_per_timestep': [], 'persistence_skill_per_timestep': [],
        'persistence_csi': [], 'persistence_hss': [],
        'peak_flux_error_per_timestep': [], 'temporal_variation_ratio': [],
        'val_rmse_per_timestep': [], 'val_correlation_per_timestep': [],
        'val_csi_per_timestep': [], 'val_hss_per_timestep': [],
        'persistence_mae_per_timestep': [],
        'train_l1': [], 'train_ssim': [], 'train_extreme': [],
        'train_temporal_diff': [], 'train_temporal_var': [], 'train_asymmetric': [],
        'delta_scale': [], 'temporal_attn_entropy': [],
    }


def apply_transfer_learning(
    model, transfer_config, optimizer, scheduler, scaler, *,
    lr, weight_decay, epochs, use_amp, start_epoch, device,
    scheduler_config, scheduler_type, scheduler_enabled,
):
    """Load pretrained weights, optionally freeze encoder, rebuild optim/sched/scaler.

    Returns (optimizer, scheduler, scaler, unfreeze_epoch).
    """
    from utils.transfer import (
        load_pretrained_weights, freeze_encoder, get_finetune_param_groups,
    )

    _unfreeze_epoch = None
    ptc_path = transfer_config['pretrained_checkpoint']
    print(f"\nTransfer learning from: {ptc_path}")

    loaded, skipped, reinited = load_pretrained_weights(
        model, ptc_path,
        reinit_mismatched=transfer_config.get('reinit_input_layers', True),
        device=device,
    )
    print(f"  Loaded {len(loaded)} parameter tensors from pretrained model")
    print(f"  Skipped {len(skipped)} (shape mismatch or missing)")
    print(f"  Reinitialized {len(reinited)} layers")

    # Freeze encoder if configured
    if transfer_config.get('freeze_encoder', False):
        frozen_names = freeze_encoder(model)
        print(f"  Frozen {len(frozen_names)} encoder parameters")

        uae = transfer_config.get('unfreeze_after_epochs', 0)
        if uae > 0:
            _unfreeze_epoch = start_epoch + uae
            print(f"  Will unfreeze encoder at epoch {_unfreeze_epoch}")

    # Rebuild optimizer with differential LR
    if transfer_config.get('reset_optimizer', True):
        lr_scale = transfer_config.get('lr_scale_pretrained', 0.1)
        param_groups = get_finetune_param_groups(model, lr, lr_scale)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        print(f"  Fresh optimizer: new layers lr={lr:.6f}, pretrained lr={lr * lr_scale:.6f}")

    # Rebuild scheduler with new optimizer
    if transfer_config.get('reset_scheduler', True) and scheduler_enabled:
        # Preserve original behaviour: only cosine/step are rebuilt here.
        if scheduler_type in ('cosine', 'step'):
            scheduler, _, _ = build_scheduler(optimizer, scheduler_config, epochs)
        scaler = get_grad_scaler(use_amp, device)
        print(f"  Fresh scheduler ({scheduler_type}) and grad scaler")

    return optimizer, scheduler, scaler, _unfreeze_epoch


def unfreeze_and_rebuild(
    model, transfer_config, scheduler, *,
    lr, weight_decay, epochs, epoch, use_amp,
    scheduler_enabled, scheduler_type, scheduler_config, device,
):
    """Unfreeze all params and rebuild optim/scaler for the remaining epochs.

    Rebuilds the (cosine) scheduler when applicable, otherwise returns the
    passed-in scheduler unchanged — matching the original loop behaviour.
    Returns (optimizer, scheduler, scaler, remaining_epochs).
    """
    from utils.transfer import unfreeze_all, get_finetune_param_groups

    unfreeze_all(model)
    lr_scale = transfer_config.get('lr_scale_pretrained', 0.1)
    param_groups = get_finetune_param_groups(model, lr, lr_scale)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    remaining = epochs - epoch + 1
    if scheduler_enabled and scheduler_type == 'cosine':
        eta_min = scheduler_config.get('cosine_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=eta_min
        )
    scaler = get_grad_scaler(use_amp, device)
    return optimizer, scheduler, scaler, remaining
