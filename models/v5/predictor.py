"""Block-causal transformer predictor (V-JEPA 2-AC template).

Per docs/V5_JEPA/01_path_a.md §3.2:
    layers=12, hidden=768, heads=12, mlp_ratio=4, 3D RoPE physical units.
    Block-causal across time: frame t cannot attend to frame t+1..t+t_out.
    Within a frame: full bidirectional spatial attention (frames are spatial scenes).

Trainable. ~85 M params at default config.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .rope3d import RoPE3D, build_token_coords


def build_block_causal_mask(t_total: int, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    """Bool mask [N, N] where True = allow attention.

    Within-frame: full bidirectional. Across-frame: only past + present.
    Token order: frames stacked, each frame has `tokens_per_frame` tokens.
    """
    n = t_total * tokens_per_frame
    frame_idx = torch.arange(n, device=device) // tokens_per_frame
    return frame_idx[:, None] >= frame_idx[None, :]


class MultiheadAttentionRoPE(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads.")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = RoPE3D(self.head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # each [B, H, N, head_dim]
        q, k = self.rope(q, k, coords)

        if x.device.type == "mps" and attn_mask is not None:
            # MPS SDPA returns NaN with attn_mask under torch.no_grad. Manual fallback.
            attn = (q @ k.transpose(-2, -1)) * self.scale          # [B, H, N, N]
            keep = attn_mask.to(torch.bool)
            attn = attn.masked_fill(~keep[None, None], -1e4)
            attn = attn.softmax(dim=-1)
            out = attn @ v
        elif attn_mask is not None:
            keep = attn_mask[None, None, :, :].to(torch.bool)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, hidden: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        inner = hidden * mlp_ratio
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class PredictorBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = MultiheadAttentionRoPE(hidden, heads, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), coords, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BlockCausalPredictor(nn.Module):
    """Block-causal transformer over flattened (T, H, W) tokens.

    forward(z, t_in, hp, wp, patch, cadence_min, pixel_scale_mm, token_pad_mask=None)
        z: [B, T*hp*wp, hidden]  encoder embeddings (input frames)
    Returns same shape — predictor reuses encoder dim. Caller slices last `t_out` frames
    or feeds rollout extension tokens before forward.
    """

    def __init__(
        self,
        hidden: int = 768,
        layers: int = 12,
        heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        encoder_dim: int | None = None,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.use_grad_checkpoint = use_grad_checkpoint
        self.in_proj = nn.Linear(encoder_dim, hidden) if encoder_dim and encoder_dim != hidden else nn.Identity()
        self.out_proj = nn.Linear(hidden, encoder_dim) if encoder_dim and encoder_dim != hidden else nn.Identity()
        dp_rates = [drop_path * i / max(1, layers - 1) for i in range(layers)]
        self.blocks = nn.ModuleList([
            PredictorBlock(hidden, heads, mlp_ratio, dropout, dp_rates[i]) for i in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        z: torch.Tensor,
        t_total: int,
        hp: int,
        wp: int,
        patch: int,
        cadence_min: float,
        pixel_scale_mm: float,
        token_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = z.shape
        tokens_per_frame = hp * wp
        if n != t_total * tokens_per_frame:
            raise ValueError(f"Token count mismatch: {n} vs {t_total}*{tokens_per_frame}")

        x = self.in_proj(z)
        coords = build_token_coords(
            t_in=t_total, hp=hp, wp=wp, patch_size=patch,
            cadence_min=cadence_min, pixel_scale_mm=pixel_scale_mm, device=z.device,
        )
        mask = build_block_causal_mask(t_total, tokens_per_frame, device=z.device)
        if token_pad_mask is not None:
            keep = token_pad_mask.reshape(-1).repeat(t_total)  # tile spatial mask across frames
            mask = mask & keep[None, :] & keep[:, None]

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training and x.requires_grad:
                x = ckpt.checkpoint(blk, x, coords, mask, use_reentrant=False)
            else:
                x = blk(x, coords, mask)
        x = self.norm(x)
        return self.out_proj(x)
