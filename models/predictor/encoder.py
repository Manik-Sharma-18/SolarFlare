"""Encoder forward pass — helper that operates on a SolarFluxPredictor instance."""
from typing import List, Optional, Tuple

import torch


def encoder_forward(
    model: "SolarFluxPredictor",  # noqa: F821 (forward-ref to avoid import cycle)
    x_prep: torch.Tensor,
    T_in: int,
) -> Tuple[
    torch.Tensor,                                # h1_skip
    List[Tuple[torch.Tensor, ...]],              # h2_states
    List[Tuple[torch.Tensor, ...]],              # h3_states
    Optional[torch.Tensor],                      # encoder_h3_packed
]:
    """Run the encoder over a preprocessed input sequence.

    Reads ``model.{encoder_conv1, downsample1, encoder_conv2, encoder_conv3,
    dropout_enc1, dropout_enc2, use_temporal_attention}``. Designed to be
    wrapped by :func:`torch.utils.checkpoint.checkpoint`.

    Args:
        model: :class:`SolarFluxPredictor` instance owning the encoder modules.
        x_prep: ``(B, c1, T_in, H, W)`` preprocessed input sequence.
        T_in: Number of input timesteps (used to build the temporal-attn
            tensor without re-deriving from shape).

    Returns:
        ``(h1_skip, h2_states, h3_states, encoder_h3_packed)`` where
        ``encoder_h3_packed`` is ``(B, T_in, c3, H, W)`` when temporal
        attention is enabled, else ``None``.
    """
    h1_seq, h1_states = model.encoder_conv1(x_prep)
    h1_skip = h1_states[0][0]  # last-timestep h of layer 0 (saved for skip)

    h1_seq = model.dropout_enc1(h1_seq)

    # Take last-timestep h1, downsample spatially, broadcast over T_in for the
    # downstream temporal ConvLSTMs.
    h1_down = model.downsample1(h1_seq[:, :, -1])
    h1_down_expanded = h1_down.unsqueeze(2).expand(-1, -1, T_in, -1, -1)

    h2_seq, h2_states = model.encoder_conv2(h1_down_expanded)
    h2_seq = model.dropout_enc2(h2_seq)

    h3_seq, h3_states = model.encoder_conv3(h2_seq)

    encoder_h3_packed: Optional[torch.Tensor] = None
    if model.use_temporal_attention:
        # Stack as tensor for checkpoint compatibility (lists aren't supported
        # as checkpointed outputs).
        encoder_h3_packed = torch.stack(
            [h3_seq[:, :, t] for t in range(T_in)], dim=1
        )

    return h1_skip, h2_states, h3_states, encoder_h3_packed
