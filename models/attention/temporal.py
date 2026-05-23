"""Temporal attention over encoder hidden states."""
from typing import List, Tuple

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Decoder queries encoder states across time.

    Global average pooling collapses spatial dims for Q/K, so attention runs
    over the temporal axis only. Values keep spatial dims for context
    reconstruction. Additive injection: ``decoder + context`` preserves the
    baseline when learned weights are ~0.

    Args:
        channels: Hidden-state channels.
        proj_dim: Projection dim. Defaults to ``channels``.
        t_max: Hard cap on input timesteps; ``forward`` raises if exceeded.

    MPS-safe: manual ``bmm + softmax`` (no SDPA).
    """

    def __init__(self, channels: int, proj_dim: int = None, t_max: int = 20):
        super().__init__()
        proj_dim = proj_dim or channels
        self.proj_dim = proj_dim

        self.q_proj = nn.Conv2d(channels, proj_dim, 1)
        self.k_proj = nn.Conv2d(channels, proj_dim, 1)
        self.v_proj = nn.Conv2d(channels, proj_dim, 1)
        self.out_proj = nn.Conv2d(proj_dim, channels, 1)

        self.scale = proj_dim ** -0.5
        # Zero init keeps initial behavior identical to vanilla content attention.
        self.pos_embed = nn.Parameter(torch.zeros(t_max, proj_dim))

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute temporally-attended context.

        Args:
            decoder_state: ``(B, C, H, W)``.
            encoder_states: list of ``T`` encoder states each ``(B, C, H, W)``.

        Returns:
            ``(context, attn_weights)`` — context shape matches
            ``decoder_state``; weights ``(B, T)``.
        """
        B, C, H, W = decoder_state.shape
        T = len(encoder_states)
        if T > self.pos_embed.shape[0]:
            raise ValueError(
                f"TemporalAttention received {T} encoder states but "
                f"pos_embed was sized to t_max={self.pos_embed.shape[0]}. "
                f"Increase t_max in TemporalAttention(...)."
            )

        q = self.q_proj(decoder_state).mean(dim=(-2, -1))  # (B, proj_dim)

        keys = torch.stack(
            [self.k_proj(e).mean(dim=(-2, -1)) for e in encoder_states], dim=1
        )  # (B, T, proj_dim)
        keys = keys + self.pos_embed[:T]

        values = torch.stack(
            [self.v_proj(e) for e in encoder_states], dim=1
        )  # (B, T, proj_dim, H, W)

        logits = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)) * self.scale  # (B, 1, T)
        attn = torch.softmax(logits, dim=-1)  # (B, 1, T)

        values_flat = values.view(B, T, -1)
        context_flat = torch.bmm(attn, values_flat)  # (B, 1, proj_dim*H*W)
        context = context_flat.view(B, self.proj_dim, H, W)

        return self.out_proj(context), attn.squeeze(1)
