"""Attention modules for the SolarFluxPredictor encoder-decoder.

Implements temporal attention over encoder hidden states and spatial
attention gates on skip connections. Uses manual bmm+softmax for
MPS compatibility (no F.scaled_dot_product_attention).

Classes:
    TemporalAttention: Queries encoder states at each decoder step.
    AttentionGate: Attention U-Net gate for skip connections.
"""
import torch
import torch.nn as nn
from typing import List, Tuple


class TemporalAttention(nn.Module):
    """Temporal attention over encoder hidden states.

    Uses global average pooling for temporal-only attention (no spatial).
    Q/K/V projections via 1x1 Conv2d. The decoder queries all stored
    encoder hidden states at each decode step to produce a context vector.

    Additive injection: context is added to decoder output for graceful
    degradation (near-zero weights = baseline behavior preserved).

    Args:
        channels: Number of channels in encoder/decoder hidden states.
        proj_dim: Projection dimension for Q/K/V. Defaults to channels.
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

        # Learnable positional embedding (init zeros = starts as vanilla attention)
        self.pos_embed = nn.Parameter(torch.zeros(t_max, proj_dim))


    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute temporally-attended context from encoder states.

        Args:
            decoder_state: Current decoder hidden state (B, C, H, W).
            encoder_states: List of T encoder hidden states,
                each (B, C, H, W).

        Returns:
            context: Weighted combination of encoder states (B, C, H, W).
            attn_weights: Attention distribution over time (B, T).
        """
        B, C, H, W = decoder_state.shape
        T = len(encoder_states)

        # Query from decoder (pool spatial dims)
        q = self.q_proj(decoder_state).mean(dim=(-2, -1))  # (B, proj_dim)

        # Keys from all encoder states (pool spatial dims) + learnable PE
        keys = torch.stack([
            self.k_proj(e).mean(dim=(-2, -1)) for e in encoder_states
        ], dim=1)  # (B, T, proj_dim)
        keys = keys + self.pos_embed[:T]  # add learnable positional embedding

        # Values from encoder states (keep spatial dims)
        values = torch.stack([
            self.v_proj(e) for e in encoder_states
        ], dim=1)  # (B, T, proj_dim, H, W)

        # Attention logits
        logits = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)) * self.scale  # (B, 1, T)
        attn = torch.softmax(logits, dim=-1)  # (B, 1, T)

        # Weighted combination of values
        values_flat = values.view(B, T, -1)  # (B, T, proj_dim*H*W)
        context_flat = torch.bmm(attn, values_flat)  # (B, 1, proj_dim*H*W)
        context = context_flat.view(B, self.proj_dim, H, W)

        return self.out_proj(context), attn.squeeze(1)


class AttentionGate(nn.Module):
    """Attention U-Net gate for skip connections.

    Produces a spatial attention mask from encoder skip features and
    decoder upsampled features. The gate adapts per decoder timestep.
    No BatchNorm (batch_size=1 instability per CONTEXT.md).

    Architecture: Conv2d(decoder) + Conv2d(encoder) -> ReLU -> Conv2d
    -> Sigmoid -> multiply with encoder skip.

    Reference: Oktay et al., 2018 (Attention U-Net).

    Args:
        encoder_channels: Number of channels in encoder skip features.
        decoder_channels: Number of channels in decoder gating signal.
        f_int: Intermediate feature dimension. Defaults to
            max(encoder_channels // 2, 8).
    """

    def __init__(
        self,
        encoder_channels: int,
        decoder_channels: int,
        f_int: int = None,
    ):
        super().__init__()
        if f_int is None:
            f_int = max(encoder_channels // 2, 8)

        self.W_g = nn.Conv2d(decoder_channels, f_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(encoder_channels, f_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(f_int, 1, kernel_size=1, bias=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gate to skip connection features.

        Args:
            g: Gating signal from decoder (B, decoder_channels, H, W).
            x: Skip connection from encoder (B, encoder_channels, H, W).

        Returns:
            gated_x: Attention-weighted skip features
                (B, encoder_channels, H, W).
        """
        g1 = self.W_g(g)       # (B, f_int, H, W)
        x1 = self.W_x(x)       # (B, f_int, H, W)
        psi = torch.relu(g1 + x1)
        psi = torch.sigmoid(self.psi(psi))  # (B, 1, H, W)
        return x * psi
