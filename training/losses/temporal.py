"""Temporal loss terms: frame-diff matching, variation reward, per-step weights."""
import torch
import torch.nn.functional as F
from typing import List


def compute_temporal_diff_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """L1 loss on frame-to-frame changes (temporal dynamics matching).

    Computes the difference between consecutive frames for both prediction
    and target, then computes L1 loss between these difference sequences.
    This encourages the model to match the temporal dynamics, not just
    the absolute values.

    Args:
        pred: Predicted tensor (B, C, T, H, W).
        target: Target tensor (B, C, T, H, W).

    Returns:
        Scalar temporal difference loss. Returns 0.0 if T <= 1.
    """
    T = pred.shape[2]
    if T <= 1:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    return F.l1_loss(pred_diffs, target_diffs)


def compute_temporal_var_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """Negative loss that rewards prediction variation up to target level.

    Computes a penalty that is negative (subtracts from total loss) when
    the prediction has temporal variation. The penalty is capped at the
    target's variation level to prevent noisy/jittery predictions from
    being rewarded beyond what the data naturally exhibits.

    penalty = -lambda * min(pred_variation, target_variation)

    Args:
        pred: Predicted tensor (B, C, T, H, W).
        target: Target tensor (B, C, T, H, W).
        lambda_val: Penalty scaling factor.

    Returns:
        Scalar variation penalty (negative value). Returns 0.0 if T <= 1.
    """
    T = pred.shape[2]
    if T <= 1:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

    pred_var = pred_diffs.abs().mean()
    target_var = target_diffs.abs().mean()

    # Cap: no reward for exceeding target variation
    capped_var = torch.min(pred_var, target_var)
    return -lambda_val * capped_var


def apply_temporal_weights(
    loss_tensor: torch.Tensor, weights: List[float]
) -> torch.Tensor:
    """Apply per-timestep weights to a 5D loss tensor before reduction.

    Broadcasts weights along the temporal dimension (dim=2) and computes
    the weighted mean across all dimensions. Later timesteps can be given
    higher weights to prioritize future prediction accuracy.

    Args:
        loss_tensor: 5D tensor (B, C, T, H, W) of per-element losses.
        weights: List of per-timestep weights (truncated or padded to T).

    Returns:
        Scalar weighted mean loss.
    """
    T = loss_tensor.shape[2]
    tw = torch.tensor(
        weights[:T], device=loss_tensor.device, dtype=loss_tensor.dtype
    )
    tw = tw.view(1, 1, T, 1, 1)
    return (loss_tensor * tw).mean()
