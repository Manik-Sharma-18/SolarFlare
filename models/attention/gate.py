"""Attention U-Net gate for skip connections.

Produces a spatial gating mask from encoder skip features and the decoder's
upsampled gating signal. No BatchNorm — V4 trains with ``batch_size=1`` and
BN would be unstable.

Reference: Oktay et al., 2018 — Attention U-Net.
"""
import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """1x1 conv attention gate.

    Args:
        encoder_channels: Skip-connection channels.
        decoder_channels: Gating-signal channels.
        f_int: Intermediate dim. Defaults to ``max(encoder_channels // 2, 8)``.
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
        """Gate ``x`` by attention computed from ``g`` and ``x``.

        Args:
            g: gating signal ``(B, decoder_channels, H, W)``.
            x: encoder skip ``(B, encoder_channels, H, W)``.

        Returns:
            ``x`` masked by a per-pixel attention weight in ``[0, 1]``.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.relu(g1 + x1)
        psi = torch.sigmoid(self.psi(psi))
        return x * psi
