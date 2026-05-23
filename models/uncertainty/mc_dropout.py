"""MC-Dropout uncertainty estimation via Welford running variance."""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .welford import welford_update


def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    teacher_forcing_ratio: float = 0.0,
    y_true: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC-Dropout mean + std via Welford's online algorithm.

    Forces ``model.train()`` to keep dropout active during inference, then
    restores the prior mode in ``finally``. Variance is computed in O(1)
    memory (no stacking).

    Args:
        model: Model with ``dropout_rate > 0``.
        x: ``(B, C, T_in, H, W)``.
        n_samples: Stochastic forward passes.
        teacher_forcing_ratio: Usually ``0.0`` at inference.
        y_true: Optional ground truth for teacher forcing.

    Returns:
        ``(mean, std)`` both ``(B, C, T_out, H, W)``.

    Raises:
        ValueError: If ``model.dropout_rate == 0.0``.
    """
    if hasattr(model, "dropout_rate") and model.dropout_rate == 0.0:
        raise ValueError(
            "Model dropout_rate must be > 0 for uncertainty estimation. "
            "Set dropout_rate=0.1 in model config and retrain."
        )

    was_training = model.training
    model.train()

    try:
        with torch.no_grad():
            pred = model(x, teacher_forcing_ratio=teacher_forcing_ratio, y_true=y_true)
            count = 1
            mean = pred.clone()
            m2 = torch.zeros_like(pred)

            for _ in range(n_samples - 1):
                pred = model(x, teacher_forcing_ratio=teacher_forcing_ratio, y_true=y_true)
                count, mean, m2 = welford_update(count, mean, m2, pred)

        std = torch.sqrt(m2 / count + 1e-8)
        return mean, std
    finally:
        if not was_training:
            model.eval()
