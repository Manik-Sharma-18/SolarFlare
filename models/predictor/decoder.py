"""One autoregressive decoder step.

Pure helper operating on a :class:`SolarFluxPredictor` instance — keeps the
top-level state_dict flat (no nested ``decoder.*`` namespace) so existing
checkpoints continue to load.
"""
from typing import List, Optional, Tuple

import torch

from .spatial import match_spatial


def decoder_step(
    model: "SolarFluxPredictor",  # noqa: F821
    input_frame: torch.Tensor,
    decoder_state2: List[Tuple[torch.Tensor, ...]],
    decoder_state3: List[Tuple[torch.Tensor, ...]],
    refine_state: Optional[List[Tuple[torch.Tensor, ...]]],
    h1_skip: torch.Tensor,
    encoder_h3_states: Optional[List[torch.Tensor]],
) -> Tuple[
    torch.Tensor,                          # refined frame (B, c1, H, W)
    List[Tuple[torch.Tensor, ...]],        # decoder_state2
    List[Tuple[torch.Tensor, ...]],        # decoder_state3
    List[Tuple[torch.Tensor, ...]],        # refine_state
]:
    """One AR decode step ending at the refinement ConvLSTM output.

    The residual head + delta-scale + flux-merge are handled by
    :func:`models.predictor.head.apply_head`.

    Reads ``model.{decoder_input_conv, decoder_proj, decoder_conv2,
    decoder_conv3, dropout_dec, upsample, refine_conv, use_temporal_attention,
    temporal_attn, use_attention_gate, attn_gate}``.
    """
    # Map input frame to decoder-input channels.
    dec_input = model.decoder_input_conv(input_frame)

    # Project to latent resolution.
    dec_down = model.decoder_proj(dec_input).unsqueeze(2)  # (B, c2, 1, H_lat, W_lat)

    # Decoder ConvLSTMs.
    dec_h2, decoder_state2 = model.decoder_conv2(dec_down, decoder_state2)
    dec_h2 = model.dropout_dec(dec_h2)
    dec_h3, decoder_state3 = model.decoder_conv3(dec_h2, decoder_state3)

    # Optional temporal attention — additive context onto t=0 frame.
    if model.use_temporal_attention and encoder_h3_states is not None:
        # Query the decoder's current hidden state (richer than dec_h3 output).
        context, _attn = model.temporal_attn(decoder_state3[0][0], encoder_h3_states)
        dec_h3_frame = dec_h3[:, :, 0] + context
    else:
        dec_h3_frame = dec_h3[:, :, 0]

    # Upsample from latent to skip-connection resolution.
    dec_up = model.upsample(dec_h3_frame)
    if dec_up.shape[2:] != h1_skip.shape[2:]:
        dec_up = match_spatial(dec_up, h1_skip.shape[2:])

    # Optional attention gate on the encoder skip.
    if model.use_attention_gate:
        gated_skip = model.attn_gate(dec_up, h1_skip)
        dec_concat = torch.cat([dec_up, gated_skip], dim=1)
    else:
        dec_concat = torch.cat([dec_up, h1_skip], dim=1)

    refined, refine_state = model.refine_conv(dec_concat.unsqueeze(2), refine_state)
    return refined[:, :, 0], decoder_state2, decoder_state3, refine_state
