"""Inverse-uncertainty-weighted loss helper."""
import torch
import torch.nn as nn


def uncertainty_weighted_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    base_loss_fn: nn.Module = nn.L1Loss(reduction="none"),
) -> torch.Tensor:
    """Per-element loss reweighted by ``1 / (uncertainty + ε)``.

    Weights are mean-normalised before multiplication so the loss scale
    matches the un-weighted version on average. High-uncertainty regions
    contribute less, focusing the gradient on confident predictions.
    """
    base_loss = base_loss_fn(predictions, targets)
    weights = 1.0 / (uncertainty + 1e-6)
    weights = weights / weights.mean()
    return (base_loss * weights).mean()
