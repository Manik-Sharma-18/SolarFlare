"""
MPS-safe operation alternatives.

Provides device-dispatched wrappers for operations that have known
correctness bugs on Apple MPS (Metal Performance Shaders) backend.

- safe_outer: Replaces torch.outer which may produce wrong results on MPS
- safe_quantile: Replaces torch.quantile which is unsupported/buggy on MPS
- is_mps: Helper to detect MPS device from tensor or device object
"""
import logging
import torch

logger = logging.getLogger(__name__)

_mps_ops_logged = False


def is_mps(tensor_or_device) -> bool:
    """Check if a tensor or device is on MPS backend.

    Args:
        tensor_or_device: A torch.Tensor or torch.device instance.

    Returns:
        True if the device type is 'mps'.
    """
    if isinstance(tensor_or_device, torch.Tensor):
        return tensor_or_device.device.type == "mps"
    if isinstance(tensor_or_device, torch.device):
        return tensor_or_device.type == "mps"
    # Accept string device specs like "mps" or "mps:0"
    return str(tensor_or_device).startswith("mps")


def _log_mps_once():
    """Log MPS alternative ops usage exactly once."""
    global _mps_ops_logged
    if not _mps_ops_logged:
        logger.info(
            "MPS device detected: using alternative ops for correctness"
        )
        _mps_ops_logged = True


def safe_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute outer product, using broadcast multiply on MPS for correctness.

    On MPS, torch.outer may produce incorrect results due to Metal shader bugs.
    This function uses ``a.unsqueeze(-1) * b.unsqueeze(0)`` as a safe fallback.

    On CUDA and CPU, the native ``torch.outer`` is used directly.

    Args:
        a: 1-D tensor of shape (N,).
        b: 1-D tensor of shape (M,).

    Returns:
        2-D tensor of shape (N, M).
    """
    if is_mps(a):
        _log_mps_once()
        return a.unsqueeze(-1) * b.unsqueeze(0)
    return torch.outer(a, b)


def safe_quantile(
    tensor: torch.Tensor, q: float, dim: int = 0
) -> torch.Tensor:
    """Compute quantile, using sort-based interpolation on MPS for correctness.

    On MPS, torch.quantile is unsupported or produces wrong results.
    This function sorts along *dim*, computes the fractional index, and
    linearly interpolates between the floor/ceil neighbours to match
    ``torch.quantile`` behaviour.

    On CUDA and CPU, the native ``torch.quantile`` is used directly.

    Args:
        tensor: Input tensor.
        q: Quantile value in [0, 1].
        dim: Dimension along which to compute.

    Returns:
        Tensor with *dim* reduced (same behaviour as torch.quantile).
    """
    if is_mps(tensor):
        _log_mps_once()
        sorted_tensor, _ = torch.sort(tensor, dim=dim)
        n = tensor.size(dim)
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        lo_val = sorted_tensor.select(dim, lo)
        hi_val = sorted_tensor.select(dim, hi)
        return lo_val + frac * (hi_val - lo_val)
    return torch.quantile(tensor, q, dim=dim)
