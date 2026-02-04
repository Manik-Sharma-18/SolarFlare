"""Cross-platform device detection, AMP context, grad scaler, and cache cleanup."""
import gc
import logging
import platform

import torch
from contextlib import nullcontext

logger = logging.getLogger(__name__)


_VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(device_config: str) -> torch.device:
    """
    Resolve the compute device from a config string.

    Args:
        device_config: One of "auto", "cuda", "mps", "cpu".
            "auto" detects best available: CUDA > MPS > CPU.
            A specific device name forces that device and raises RuntimeError
            if it is unavailable.

    Returns:
        torch.device for the resolved device.

    Raises:
        ValueError: If device_config is not a recognized value.
        RuntimeError: If the requested device is not available.
    """
    device_config = device_config.strip().lower()

    if device_config not in _VALID_DEVICES:
        raise ValueError(
            f"Unknown device config '{device_config}'. "
            f"Expected one of: {', '.join(_VALID_DEVICES)}"
        )

    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # Forced device -- verify availability
        if device_config == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' requested but CUDA is not available on this system"
            )
        if device_config == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "Device 'mps' requested but MPS is not available on this system"
            )
        device = torch.device(device_config)

    # Startup log line
    _log_device(device)
    return device


def _log_device(device: torch.device) -> None:
    """Print a single startup log line describing the resolved device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using device: cuda ({name}, {vram_gb:.1f} GB)")
    elif device.type == "mps":
        chip = platform.processor() or "Apple Silicon"
        print(f"Using device: mps ({chip})")
    else:
        print("Using device: cpu")


def get_amp_context(use_amp: bool, device: torch.device):
    """
    Get the appropriate autocast context for mixed precision.

    Returns nullcontext if use_amp is False; otherwise returns
    torch.amp.autocast for the device type.
    """
    if not use_amp:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type)


def get_grad_scaler(use_amp: bool, device: torch.device):
    """
    Get gradient scaler for mixed precision training.

    Real GradScaler is only used for CUDA with AMP enabled.
    MPS and CPU always get DummyGradScaler (MPS does not support
    CUDA-style loss scaling).
    """
    if use_amp and device.type == "cuda":
        return torch.amp.GradScaler()
    return _DummyGradScaler()


def clear_device_cache(device: torch.device) -> None:
    """
    Clear device memory cache.

    For CUDA and MPS, gc.collect() is called first to release dangling
    Python tensor references before the device runtime reclaims memory.
    This is critical for MPS which has known memory leak issues with
    unreferenced tensors, and ensures flat memory usage over long
    training runs (50+ epochs).

    CUDA: gc.collect() + torch.cuda.empty_cache()
    MPS:  gc.collect() + torch.mps.empty_cache()
    CPU:  no-op
    """
    if device.type == "cuda":
        # gc.collect() releases dangling Python tensor references
        # before device cache clear
        gc.collect()
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # gc.collect() releases dangling Python tensor references
        # before device cache clear -- critical for MPS which has
        # known memory leak issues with unreferenced tensors
        gc.collect()
        torch.mps.empty_cache()
    # CPU: nothing to clear (OS/Python GC manages memory naturally)


class _DummyGradScaler:
    """A no-op gradient scaler for CPU or non-AMP training."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        # Check for NaN/Inf gradients before stepping
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    logger.warning(
                        "NaN/Inf gradients detected in DummyGradScaler.step(), "
                        "skipping optimizer step"
                    )
                    return
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass
