"""
Uncertainty Quantification via MC Dropout.

This module provides functions for estimating prediction uncertainty
using Monte Carlo Dropout. When dropout is enabled during inference,
running multiple forward passes produces a distribution of predictions
whose variance indicates model uncertainty.

Usage:
    model = SolarFluxPredictor(dropout_rate=0.1, ...)
    mean_pred, uncertainty = predict_with_uncertainty(model, x, n_samples=20)
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    teacher_forcing_ratio: float = 0.0,
    y_true: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC Dropout uncertainty estimation.
    
    Runs the model multiple times with dropout enabled to estimate
    prediction uncertainty. The standard deviation across samples
    indicates model uncertainty at each spatial location.
    
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
    
    Example:
        model = SolarFluxPredictor(dropout_rate=0.1)
        # ... train model ...
        
        mean, uncertainty = predict_with_uncertainty(model, test_input, n_samples=20)
        
        # High uncertainty regions indicate less confident predictions
        high_uncertainty_mask = uncertainty > uncertainty.mean() + uncertainty.std()
    """
    # Check that model has dropout enabled
    if hasattr(model, 'dropout_rate') and model.dropout_rate == 0.0:
        raise ValueError(
            "Model dropout_rate must be > 0 for uncertainty estimation. "
            "Set dropout_rate=0.1 in model config and retrain."
        )
    
    # Enable dropout during inference by setting model to train mode
    # This is the key trick of MC Dropout
    was_training = model.training
    model.train()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x, teacher_forcing_ratio=teacher_forcing_ratio, y_true=y_true)
            predictions.append(pred)
    
    # Restore original mode
    if not was_training:
        model.eval()
    
    # Stack predictions: (n_samples, B, C, T_out, H, W)
    predictions = torch.stack(predictions, dim=0)
    
    # Compute statistics across samples
    mean = predictions.mean(dim=0)  # (B, C, T_out, H, W)
    std = predictions.std(dim=0)    # (B, C, T_out, H, W)
    
    return mean, std


def predict_with_confidence_intervals(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    confidence_level: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict with confidence intervals.
    
    Args:
        model: Model with dropout layers
        x: Input tensor (B, C, T_in, H, W)
        n_samples: Number of MC samples
        confidence_level: Confidence level for intervals (default 95%)
    
    Returns:
        mean: Mean prediction
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
    """
    was_training = model.training
    model.train()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x, teacher_forcing_ratio=0.0)
            predictions.append(pred)
    
    if not was_training:
        model.eval()
    
    predictions = torch.stack(predictions, dim=0)
    
    # Compute percentiles for confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean = predictions.mean(dim=0)
    lower = torch.quantile(predictions, alpha / 2, dim=0)
    upper = torch.quantile(predictions, 1 - alpha / 2, dim=0)
    
    return mean, lower, upper


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

