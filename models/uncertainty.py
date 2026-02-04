"""
Uncertainty Quantification via MC Dropout.

This module provides functions for estimating prediction uncertainty
using Monte Carlo Dropout. When dropout is enabled during inference,
running multiple forward passes produces a distribution of predictions
whose variance indicates model uncertainty.

Memory efficiency: Uses Welford's online algorithm for computing running
mean and variance in O(1) memory, avoiding the O(N) cost of stacking
all MC forward pass outputs.

Reference: Welford, B.P. (1962). "Note on a method for calculating
corrected sums of squares and products." Technometrics, 4(3), 419-420.

Usage:
    model = SolarFluxPredictor(dropout_rate=0.1, ...)
    mean_pred, uncertainty = predict_with_uncertainty(model, x, n_samples=20)
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


def _welford_update(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    new_value: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Single Welford update step. Returns (count, mean, m2)."""
    count += 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    m2 = m2 + delta * delta2
    return count, mean, m2


def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    teacher_forcing_ratio: float = 0.0,
    y_true: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC Dropout uncertainty estimation using Welford's online algorithm.

    Runs the model multiple times with dropout enabled to estimate
    prediction uncertainty. Uses Welford's algorithm to compute running
    mean and variance in O(1) memory (no stacking of all predictions).

    Args:
        model: Model with dropout layers (dropout_rate > 0)
        x: Input tensor (B, C, T_in, H, W)
        n_samples: Number of stochastic forward passes
        teacher_forcing_ratio: TF ratio for predictions (usually 0.0)
        y_true: Ground truth for teacher forcing (optional)

    Returns:
        mean: (B, C, T_out, H, W) - Mean prediction across samples
        std: (B, C, T_out, H, W) - Uncertainty (standard deviation)

    Raises:
        ValueError: If model has dropout_rate == 0.0
    """
    # Check that model has dropout enabled
    if hasattr(model, 'dropout_rate') and model.dropout_rate == 0.0:
        raise ValueError(
            "Model dropout_rate must be > 0 for uncertainty estimation. "
            "Set dropout_rate=0.1 in model config and retrain."
        )

    # Enable dropout during inference by setting model to train mode
    was_training = model.training
    model.train()

    try:
        with torch.no_grad():
            # First forward pass to initialize accumulators
            pred = model(x, teacher_forcing_ratio=teacher_forcing_ratio, y_true=y_true)
            count = 1
            mean = pred.clone()
            m2 = torch.zeros_like(pred)

            # Remaining passes with Welford updates
            for _ in range(n_samples - 1):
                pred = model(x, teacher_forcing_ratio=teacher_forcing_ratio, y_true=y_true)
                count, mean, m2 = _welford_update(count, mean, m2, pred)

        # Variance = M2 / count, add epsilon for numerical stability
        variance = m2 / count
        std = torch.sqrt(variance + 1e-8)

        return mean, std
    finally:
        # Restore original mode even if forward pass raises
        if not was_training:
            model.eval()


def predict_with_confidence_intervals(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    confidence_level: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict with confidence intervals using Gaussian approximation.

    Uses Welford's online algorithm for O(1) memory, then computes
    confidence intervals as mean +/- z * std (Gaussian approximation).
    This is valid for n_samples >= 20 by the Central Limit Theorem and
    avoids torch.quantile which produces incorrect results on MPS.

    Args:
        model: Model with dropout layers
        x: Input tensor (B, C, T_in, H, W)
        n_samples: Number of MC samples (>= 20 recommended)
        confidence_level: Confidence level for intervals (default 95%)

    Returns:
        mean: Mean prediction
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
    """
    # Z-scores for common confidence levels (Gaussian approximation)
    # For arbitrary levels: z = scipy.stats.norm.ppf(1 - alpha/2)
    # We use a lookup for common values to avoid scipy dependency.
    _z_scores = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    z = _z_scores.get(confidence_level)
    if z is None:
        # Approximate using inverse error function for arbitrary levels
        import math
        alpha = 1 - confidence_level
        # Rational approximation of inverse normal CDF
        # For the symmetric two-tailed case: z = sqrt(2) * erfinv(confidence_level)
        z = math.sqrt(2) * math.erfc(alpha) if hasattr(math, 'erfc') else 1.96
        # Use torch for erfinv if available
        z = float(torch.erfinv(torch.tensor(confidence_level)).item()) * math.sqrt(2)

    was_training = model.training
    model.train()

    try:
        with torch.no_grad():
            # First forward pass to initialize Welford accumulators
            pred = model(x, teacher_forcing_ratio=0.0)
            count = 1
            mean = pred.clone()
            m2 = torch.zeros_like(pred)

            # Remaining passes
            for _ in range(n_samples - 1):
                pred = model(x, teacher_forcing_ratio=0.0)
                count, mean, m2 = _welford_update(count, mean, m2, pred)

        variance = m2 / count
        std = torch.sqrt(variance + 1e-8)

        # Gaussian approximation: mean +/- z * std
        lower = mean - z * std
        upper = mean + z * std

        return mean, lower, upper
    finally:
        if not was_training:
            model.eval()


def uncertainty_weighted_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    base_loss_fn: nn.Module = nn.L1Loss(reduction='none')
) -> torch.Tensor:
    """
    Compute loss weighted by prediction uncertainty.

    Regions with high uncertainty contribute less to the loss,
    which can help training focus on learnable patterns.

    Args:
        predictions: Model predictions (B, C, T, H, W)
        targets: Ground truth (B, C, T, H, W)
        uncertainty: Uncertainty estimates (B, C, T, H, W)
        base_loss_fn: Base loss function (default L1)

    Returns:
        Uncertainty-weighted loss (scalar)
    """
    # Compute base loss
    base_loss = base_loss_fn(predictions, targets)  # (B, C, T, H, W)

    # Weight inversely by uncertainty (higher uncertainty = lower weight)
    # Add epsilon to avoid division by zero
    weights = 1.0 / (uncertainty + 1e-6)

    # Normalize weights
    weights = weights / weights.mean()

    # Weighted loss
    weighted_loss = (base_loss * weights).mean()

    return weighted_loss
