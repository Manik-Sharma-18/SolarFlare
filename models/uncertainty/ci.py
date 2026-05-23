"""Gaussian-approximation confidence intervals from MC-Dropout samples."""
import math
from typing import Tuple

import torch
import torch.nn as nn

from .welford import welford_update


# Common two-tailed z-scores; fallback to erfinv for arbitrary levels.
_Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def _z_for(confidence_level: float) -> float:
    """Two-tailed normal quantile. ``z = sqrt(2) * erfinv(c)``.

    Avoids ``torch.quantile`` (incorrect on MPS) by sampling-then-
    moment-summarising; this returns the analytic quantile for the
    Gaussian approximation.
    """
    z = _Z_SCORES.get(confidence_level)
    if z is not None:
        return z
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1); got {confidence_level}"
        )
    # float64 CPU scalar; torch.erfinv supports CPU + CUDA.
    return float(
        torch.erfinv(torch.tensor(confidence_level, dtype=torch.float64)).item()
    ) * math.sqrt(2)


def predict_with_confidence_intervals(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    confidence_level: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict with two-tailed CI bands.

    Valid for ``n_samples >= 20`` by the CLT (Gaussian approximation).

    Returns:
        ``(mean, lower, upper)`` each ``(B, C, T_out, H, W)``.
    """
    z = _z_for(confidence_level)

    was_training = model.training
    model.train()

    try:
        with torch.no_grad():
            pred = model(x, teacher_forcing_ratio=0.0)
            count = 1
            mean = pred.clone()
            m2 = torch.zeros_like(pred)

            for _ in range(n_samples - 1):
                pred = model(x, teacher_forcing_ratio=0.0)
                count, mean, m2 = welford_update(count, mean, m2, pred)

        std = torch.sqrt(m2 / count + 1e-8)
        lower = mean - z * std
        upper = mean + z * std
        return mean, lower, upper
    finally:
        if not was_training:
            model.eval()
