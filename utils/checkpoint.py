"""Centralized checkpoint I/O: atomic saves, versioned loads, cross-device portability."""
import copy
import logging
import os
import uuid
from pathlib import Path

import torch

from utils.device import _DummyGradScaler, resolve_device

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def _move_optimizer_state_to_cpu(optimizer_state_dict: dict) -> None:
    """Recursively walk optimizer state dict and move all tensors to CPU in-place."""
    if isinstance(optimizer_state_dict, dict):
        for key, value in optimizer_state_dict.items():
            if isinstance(value, torch.Tensor):
                optimizer_state_dict[key] = value.cpu()
            elif isinstance(value, dict):
                _move_optimizer_state_to_cpu(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        value[i] = item.cpu()
                    elif isinstance(item, dict):
                        _move_optimizer_state_to_cpu(item)
    elif isinstance(optimizer_state_dict, list):
        for i, item in enumerate(optimizer_state_dict):
            if isinstance(item, torch.Tensor):
                optimizer_state_dict[i] = item.cpu()
            elif isinstance(item, (dict, list)):
                _move_optimizer_state_to_cpu(item)


def _atomic_save(state_dict: dict, filepath: Path) -> None:
    """Save state dict atomically: write to temp file, fsync, rename."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = filepath.parent / f'.tmp_ckpt_{uuid.uuid4().hex}.pt'
    try:
        torch.save(state_dict, str(tmp_path))
        # Flush to disk before atomic rename
        with open(str(tmp_path), 'rb') as f:
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(filepath))
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def save_checkpoint(
    filepath,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_loss,
    patience_counter,
    normalization_params,
    config,
    history,
    emergency=False,
    emergency_reason=None,
):
    """
    Save a full training checkpoint atomically.

    All tensors are moved to CPU before saving so the checkpoint is
    device-neutral and can be loaded on any device.

    Args:
        filepath: Destination path for the checkpoint file.
        epoch: Current epoch number.
        model: The model (nn.Module).
        optimizer: The optimizer.
        scheduler: LR scheduler (or None).
        scaler: GradScaler or DummyGradScaler.
        best_val_loss: Best validation loss seen so far.
        patience_counter: Current early-stopping patience counter.
        normalization_params: Dict of normalization parameters from data loading.
        config: Full training config dict (snapshot for resume diffing).
        history: Training history dict (losses, LRs per epoch).
        emergency: Whether this is an emergency checkpoint.
        emergency_reason: Reason string if emergency is True.
    """
    filepath = Path(filepath)

    checkpoint = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'epoch': epoch,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': (
            scaler.state_dict()
            if hasattr(scaler, 'state_dict') and not isinstance(scaler, _DummyGradScaler)
            else None
        ),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'normalization_params': normalization_params,
        'config': config,
        'history': history,
        'emergency': emergency,
        'emergency_reason': emergency_reason,
    }

    # Move optimizer tensors to CPU for device-neutral checkpoint
    _move_optimizer_state_to_cpu(checkpoint['optimizer_state_dict'])

    _atomic_save(checkpoint, filepath)

    if emergency:
        logger.info("Emergency checkpoint saved: %s (reason: %s)", filepath, emergency_reason)
    else:
        logger.info(
            "Checkpoint saved: %s (epoch %d, val_loss %.6f)",
            filepath, epoch, best_val_loss,
        )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _diff_configs(saved_config, current_config, path=""):
    """
    Recursively compare two config dicts.

    Returns a list of human-readable diff strings such as:
        "  training.lr: 0.001 -> 0.0001"
    """
    diffs = []
    all_keys = set(list(saved_config.keys()) + list(current_config.keys()))
    for key in sorted(all_keys):
        full_key = f"{path}.{key}" if path else key
        old = saved_config.get(key)
        new = current_config.get(key)
        if key not in current_config:
            diffs.append(f"  {full_key}: {old!r} -> <removed>")
        elif key not in saved_config:
            diffs.append(f"  {full_key}: <absent> -> {new!r}")
        elif isinstance(old, dict) and isinstance(new, dict):
            diffs.extend(_diff_configs(old, new, full_key))
        elif old != new:
            diffs.append(f"  {full_key}: {old!r} -> {new!r}")
    return diffs


def _check_state_dict_compatibility(model, state_dict):
    """
    Compare model state dict keys with checkpoint state dict keys.

    Raises RuntimeError with a formatted diff of missing/unexpected keys
    if there is any mismatch.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing or unexpected:
        lines = ["Architecture mismatch between model and checkpoint:"]
        if missing:
            lines.append(f"\n  Missing keys ({len(missing)}) — in model but not in checkpoint:")
            for k in sorted(missing):
                lines.append(f"    - {k}")
        if unexpected:
            lines.append(f"\n  Unexpected keys ({len(unexpected)}) — in checkpoint but not in model:")
            for k in sorted(unexpected):
                lines.append(f"    + {k}")
        raise RuntimeError("\n".join(lines))


def _optimizer_to_device(optimizer, device):
    """Move all optimizer state tensors to *device* after loading."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# ---------------------------------------------------------------------------
# Public load functions
# ---------------------------------------------------------------------------


def load_checkpoint(filepath, device=None):
    """
    Load and validate a checkpoint file.

    Args:
        filepath: Path to the checkpoint ``.pt`` file.
        device: Unused (kept for API symmetry); loading always maps to CPU.

    Returns:
        The validated checkpoint dict.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        RuntimeError: If checkpoint version does not match CHECKPOINT_VERSION.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    version = checkpoint.get('checkpoint_version')
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Checkpoint version mismatch: file has v{version}, "
            f"expected v{CHECKPOINT_VERSION}. Cannot load."
        )

    return checkpoint


def load_checkpoint_for_resume(
    filepath,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    current_config=None,
):
    """
    High-level resume: load checkpoint and restore all training state.

    Args:
        filepath: Path to checkpoint file.
        model: The model instance (must match architecture).
        optimizer: The optimizer instance.
        scheduler: LR scheduler (or None).
        scaler: GradScaler or DummyGradScaler.
        device: Target device for training (e.g. ``torch.device('cuda')``).
        current_config: Current config dict; if provided, differences from the
            saved config are logged as warnings.

    Returns:
        Tuple of ``(start_epoch, best_val_loss, patience_counter, history, normalization_params)``.

    Raises:
        FileNotFoundError: If checkpoint file is missing.
        RuntimeError: On version mismatch or architecture mismatch.
    """
    checkpoint = load_checkpoint(filepath)

    # --- Architecture compatibility ---
    _check_state_dict_compatibility(model, checkpoint['model_state_dict'])

    # --- Restore model ---
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # --- Restore optimizer ---
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    _optimizer_to_device(optimizer, device)

    # --- Restore scheduler ---
    if scheduler and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # --- Restore scaler ---
    if (
        scaler is not None
        and not isinstance(scaler, _DummyGradScaler)
        and checkpoint.get('scaler_state_dict') is not None
    ):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # --- Config diff ---
    if current_config is not None:
        diffs = _diff_configs(checkpoint.get('config', {}), current_config)
        for diff in diffs:
            logger.warning("Config changed since checkpoint:%s", diff)

    # --- Device info ---
    saved_device = checkpoint.get('config', {}).get('device', 'unknown')
    if str(device) != saved_device and saved_device != 'unknown':
        logger.info(
            "Loading checkpoint trained on %s, resuming on %s",
            saved_device, device,
        )

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint['patience_counter']
    history = checkpoint.get('history', {})
    normalization_params = checkpoint.get('normalization_params')

    logger.info(
        "Resumed from checkpoint %s — starting at epoch %d (best_val_loss=%.6f)",
        filepath, start_epoch, best_val_loss,
    )

    return start_epoch, best_val_loss, patience_counter, history, normalization_params


def load_checkpoint_for_inference(filepath, device=None):
    """
    Load a checkpoint for inference use.

    Args:
        filepath: Path to checkpoint file.
        device: Target device. If ``None``, auto-resolved via
            :func:`utils.device.resolve_device`.

    Returns:
        Tuple of ``(checkpoint_dict, device)``.
    """
    if device is None:
        device = resolve_device('auto')

    checkpoint = load_checkpoint(filepath)

    saved_device = checkpoint.get('config', {}).get('device', 'unknown')
    if str(device) != saved_device and saved_device != 'unknown':
        logger.info(
            "Remapping checkpoint from %s to %s", saved_device, device,
        )

    return checkpoint, device
