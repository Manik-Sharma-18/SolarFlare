"""Metrics computation for model evaluation."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
    """
    Compute evaluation metrics between predictions and targets.
    
    Args:
        pred: Predicted tensor (B, C, T, H, W)
        target: Ground truth tensor (B, C, T, H, W)
    
    Returns:
        Dictionary with 'mae_total' and 'mae_per_timestep'
    """
    mae = F.l1_loss(pred, target, reduction='none')
    
    # Average over B, C, H, W to get per-timestep MAE
    mae_per_timestep = mae.mean(dim=(0, 1, 3, 4))
    mae_total = mae.mean()
    
    return {
        'mae_total': mae_total.item(),
        'mae_per_timestep': mae_per_timestep.cpu().numpy()
    }


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Squared Error."""
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse).item()


def compute_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
    denominator = torch.sqrt(
        ((pred_flat - pred_mean) ** 2).sum() * 
        ((target_flat - target_mean) ** 2).sum()
    )
    
    if denominator < 1e-8:
        return 0.0
    
    return (numerator / denominator).item()

