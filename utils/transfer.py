"""Transfer learning utilities for pretrain → fine-tune workflow.

Supports loading a pretrained checkpoint with partial key matching,
freezing/unfreezing encoder layers, and differential learning rates.
"""
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Layers whose weights depend on input_channels and may need reinitialization
INPUT_DEPENDENT_PREFIXES = (
    'preprocess.',
    'decoder_input_conv.',
)

# Encoder layers to freeze during fine-tuning
ENCODER_PREFIXES = (
    'encoder_conv1.',
    'encoder_conv2.',
    'encoder_conv3.',
    'downsample1.',
)


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    reinit_mismatched: bool = True,
    device: torch.device = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Load pretrained weights with partial key matching.

    Compares model and checkpoint state dicts key-by-key:
    - Matching shape: load from checkpoint
    - Mismatched shape: skip and optionally reinitialize
    - Missing from checkpoint: keep random init

    Args:
        model: Target model (may have different input_channels).
        checkpoint_path: Path to pretrained .pt checkpoint.
        reinit_mismatched: Reinitialize mismatched layers with Kaiming init.
        device: Target device (model is moved here after loading).

    Returns:
        (loaded_keys, skipped_keys, reinitialized_keys)
    """
    from utils.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path)
    pretrained_state = checkpoint['model_state_dict']
    model_state = model.state_dict()

    loaded_keys = []
    skipped_keys = []
    reinitialized_keys = []

    partial_state = {}
    for key in model_state:
        if key in pretrained_state:
            if model_state[key].shape == pretrained_state[key].shape:
                partial_state[key] = pretrained_state[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
                logger.info(
                    "Shape mismatch for '%s': model=%s, checkpoint=%s",
                    key, tuple(model_state[key].shape),
                    tuple(pretrained_state[key].shape),
                )
        else:
            skipped_keys.append(key)
            logger.info("Key '%s' not in checkpoint — keeping init", key)

    # Extra keys in checkpoint but not in model (e.g., removed features)
    extra = set(pretrained_state.keys()) - set(model_state.keys())
    if extra:
        logger.info("Ignoring %d extra checkpoint keys: %s", len(extra), sorted(extra))

    # Load compatible weights
    model.load_state_dict(partial_state, strict=False)

    # Reinitialize mismatched layers
    if reinit_mismatched:
        params_dict = dict(model.named_parameters())
        for key in skipped_keys:
            if key in params_dict:
                param = params_dict[key]
                if param.dim() >= 2:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.zeros_(param)
                reinitialized_keys.append(key)

    if device is not None:
        model.to(device)

    logger.info(
        "Transfer loading: %d loaded, %d skipped, %d reinitialized",
        len(loaded_keys), len(skipped_keys), len(reinitialized_keys),
    )
    return loaded_keys, skipped_keys, reinitialized_keys


def is_input_dependent(name: str) -> bool:
    """Check if a parameter name belongs to an input-channel-dependent layer."""
    return any(name.startswith(prefix) for prefix in INPUT_DEPENDENT_PREFIXES)


def freeze_encoder(model: nn.Module) -> List[str]:
    """Freeze encoder layers, leaving decoder and input layers trainable.

    Frozen: encoder_conv1/2/3, downsample1
    Unfrozen: preprocess, decoder_input_conv, decoder_*, upsample,
              refine_conv, attn_gate, temporal_attn, output_conv, delta_scale

    Returns:
        List of frozen parameter names.
    """
    frozen = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ENCODER_PREFIXES):
            param.requires_grad = False
            frozen.append(name)
    logger.info("Froze %d encoder parameters", len(frozen))
    return frozen


def unfreeze_all(model: nn.Module) -> None:
    """Restore requires_grad=True on all parameters."""
    count = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            count += 1
    logger.info("Unfroze %d parameters", count)


def get_finetune_param_groups(
    model: nn.Module,
    base_lr: float,
    lr_scale_pretrained: float = 0.1,
) -> list:
    """Create optimizer param groups with differential learning rates.

    Reinitialized/input-dependent layers get base_lr.
    Pretrained layers get base_lr * lr_scale_pretrained.
    Only includes parameters with requires_grad=True.

    Returns:
        List of param group dicts for optimizer construction.
    """
    new_params = []
    pretrained_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_input_dependent(name):
            new_params.append(param)
        else:
            pretrained_params.append(param)

    groups = []
    if new_params:
        groups.append({'params': new_params, 'lr': base_lr})
    if pretrained_params:
        groups.append({'params': pretrained_params, 'lr': base_lr * lr_scale_pretrained})

    logger.info(
        "Param groups: %d new (lr=%.6f), %d pretrained (lr=%.6f)",
        len(new_params), base_lr,
        len(pretrained_params), base_lr * lr_scale_pretrained,
    )
    return groups
