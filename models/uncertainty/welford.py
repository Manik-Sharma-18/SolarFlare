"""Welford's online mean/variance update.

O(1) memory — avoids stacking all MC-Dropout samples for variance computation.

Reference: Welford, B.P. (1962). "Note on a method for calculating corrected
sums of squares and products." Technometrics, 4(3), 419-420.
"""
from typing import Tuple

import torch


def welford_update(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    new_value: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """One Welford step. Returns updated ``(count, mean, m2)``."""
    count += 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    m2 = m2 + delta * delta2
    return count, mean, m2
