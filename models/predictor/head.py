"""Output head: residual delta → flux frame."""
from typing import Tuple

import torch

from .spatial import match_spatial


def apply_head(
    model: "SolarFluxPredictor",  # noqa: F821
    refined: torch.Tensor,
    input_frame: torch.Tensor,
    target_hw: Tuple[int, int],
    flux_channel_idx: int = 0,
) -> torch.Tensor:
    """Project refined feature to a flux residual and add to the input flux.

    Reads ``model.{output_conv, delta_scale, output_channels}``.

    Args:
        refined: ``(B, c1, H, W)`` post-refinement features.
        input_frame: ``(B, C, H, W)`` current AR input frame (the model
            residual is added to this frame's flux channels).
        target_hw: Original spatial dims; residual is matched to these.
        flux_channel_idx: Starting channel index for flux in ``input_frame``.

    Returns:
        ``(B, output_channels, H, W)`` predicted flux frame.
    """
    raw_delta = model.output_conv(refined)
    delta = raw_delta * model.delta_scale if model.delta_scale is not None else raw_delta

    if delta.shape[2:] != target_hw:
        delta = match_spatial(delta, target_hw)

    input_flux = input_frame[:, flux_channel_idx:flux_channel_idx + model.output_channels]
    return input_flux + delta
