"""Teacher-forcing helper for the autoregressive decoder."""
from typing import Optional

import torch


def advance_input_frame(
    input_frame: torch.Tensor,
    pred_flux: torch.Tensor,
    y_true: Optional[torch.Tensor],
    t_idx: int,
    teacher_forcing_ratio: float,
    output_channels: int,
    input_channels: int,
) -> torch.Tensor:
    """Pick the next AR-input frame (teacher-forced or model-predicted).

    For multi-channel input (e.g. flux + extreme indicator), only the flux
    channels are replaced; non-flux channels carry forward from the current
    input frame unchanged.

    Args:
        input_frame: Current ``(B, C, H, W)`` AR input.
        pred_flux: Model's flux prediction ``(B, output_channels, H, W)``.
        y_true: Optional ground truth ``(B, C, T_out, H, W)``.
        t_idx: Current AR step index.
        teacher_forcing_ratio: Probability of using ground truth.
        output_channels: Flux channel count.
        input_channels: Total input channels (incl. non-flux).
    """
    use_teacher = (
        teacher_forcing_ratio > 0
        and y_true is not None
        and torch.rand(1).item() < teacher_forcing_ratio
    )
    next_flux = y_true[:, :output_channels, t_idx] if use_teacher else pred_flux

    if input_channels > output_channels:
        other_channels = input_frame[:, output_channels:]
        return torch.cat([next_flux, other_channels], dim=1)
    return next_flux
