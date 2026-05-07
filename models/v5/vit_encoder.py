"""ViT-style per-frame encoder for V5 Path B (JEPA-from-scratch).

Replaces Surya: trainable from scratch, sized for AR-cutout cubes (~22M @ ViT-Small).
Per-frame patch embedding + bidirectional self-attention. Temporal modeling lives in
the predictor (block-causal). No fixed pos_embed buffer → handles variable HxW natively.

Used twice in V5JEPAModel:
- context_encoder (online, gradient-trained)
- target_encoder (EMA copy of context, no gradient — provides anti-collapse target)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .predictor import PredictorBlock
from .rope3d import build_token_coords


class ViTEncoder(nn.Module):
    """Per-frame ViT. Input [B, T, C_in, Hp, Wp] → tokens [B, T, hp, wp, D].

    Patch via Conv2d(C_in, D, k=patch, s=patch). Bidirectional self-attention
    within each frame (no causal mask). 2D positional info via RoPE3D with t=0
    (RoPE3D reused; t-axis contributes constant rotation across frames).
    """

    def __init__(
        self,
        in_ch: int = 13,
        embed_dim: int = 384,
        patch_size: int = 16,
        layers: int = 12,
        heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_grad_checkpoint = use_grad_checkpoint
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)

        dp_rates = [drop_path * i / max(1, layers - 1) for i in range(layers)]
        self.blocks = nn.ModuleList([
            PredictorBlock(
                hidden=embed_dim, heads=heads, mlp_ratio=mlp_ratio,
                dropout=dropout, drop_path=dp_rates[i],
            )
            for i in range(layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def encode_frames(
        self,
        x: torch.Tensor,
        cadence_min: float = 12.0,
        pixel_scale_mm: float = 0.364,
    ) -> torch.Tensor:
        """[B, T, C_in, Hp, Wp] → [B, T, hp, wp, D].

        Each frame encoded independently (no temporal mixing). Shared weights across T.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected [B,T,C,H,W]; got {tuple(x.shape)}")
        b, t, c, hp_in, wp_in = x.shape
        if c != self.in_ch:
            raise ValueError(f"Encoder in_ch={self.in_ch}, got {c}")
        if hp_in % self.patch_size or wp_in % self.patch_size:
            raise ValueError(
                f"Spatial dims must be multiples of patch_size={self.patch_size}; got {hp_in}x{wp_in}"
            )

        x_flat = x.reshape(b * t, c, hp_in, wp_in)
        tokens = self.proj(x_flat)                                # [B*T, D, hp, wp]
        hp = tokens.shape[-2]
        wp = tokens.shape[-1]
        tokens = tokens.permute(0, 2, 3, 1).reshape(b * t, hp * wp, self.embed_dim)

        # 2D RoPE: per-frame positional info (t=0 for all tokens, y/x vary)
        coords = build_token_coords(
            t_in=1, hp=hp, wp=wp, patch_size=self.patch_size,
            cadence_min=cadence_min, pixel_scale_mm=pixel_scale_mm, device=x.device,
        )

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training and tokens.requires_grad:
                tokens = ckpt.checkpoint(blk, tokens, coords, None, use_reentrant=False)
            else:
                tokens = blk(tokens, coords, None)                # mask=None → bidirectional
        tokens = self.norm(tokens)

        tokens = tokens.reshape(b, t, hp, wp, self.embed_dim)
        return tokens

    @torch.no_grad()
    def update_ema_from(self, source: "ViTEncoder", decay: float) -> None:
        """θ_target ← decay·θ_target + (1-decay)·θ_source. Buffers replicated exactly."""
        for p_t, p_s in zip(self.parameters(), source.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)
        for b_t, b_s in zip(self.buffers(), source.buffers()):
            b_t.data.copy_(b_s.data)
