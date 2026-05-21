"""Cross-attention predictor — toggle via predictor.cross_attn: true in config.

Q = all tokens (updated each layer), K/V = context-only tokens (fixed).
Causal cross-mask: token at frame t attends to context at frames <= t.
Attention cost: B * H * N * N_ctx vs B * H * N^2 for self-attn.
  With target_ratio=0.75 -> N_ctx ~ 0.25*N -> 0.25 N^2 (4x reduction).

Falls back to block-causal self-attention when ctx_mask=None (rollout mode).

NOTE: weights are incompatible with BlockCausalPredictor checkpoints.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .predictor import MLP, DropPath, build_block_causal_mask
from .rope3d import RoPE3D, build_token_coords


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape)


class MultiheadCrossAttnRoPE(nn.Module):
    """Cross-attention: Q from x_q, K/V from x_kv, each with own RoPE coords."""

    def __init__(self, hidden: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads.")
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.kv_proj = nn.Linear(hidden, 2 * hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = RoPE3D(self.head_dim)

    def forward(
        self,
        x_q: torch.Tensor,               # [B, N_q, D]
        x_kv: torch.Tensor,              # [B, N_kv, D]
        q_coords: torch.Tensor,          # [N_q, 3]
        kv_coords: torch.Tensor,         # [N_kv, 3]
        attn_mask: torch.Tensor | None,  # [N_q, N_kv] bool True=allow, or None
    ) -> torch.Tensor:
        b, n_q, c = x_q.shape
        n_kv = x_kv.shape[1]
        h, d = self.heads, self.head_dim

        q = self.q_proj(x_q).reshape(b, n_q, h, d).permute(0, 2, 1, 3)
        kv = self.kv_proj(x_kv).reshape(b, n_kv, 2, h, d).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # each [B, H, N_kv, d]

        # Apply RoPE with separate coords for Q and K.
        q_ang = self.rope._angles(q_coords).to(q.dtype)    # [N_q, D]
        k_ang = self.rope._angles(kv_coords).to(k.dtype)   # [N_kv, D]
        q = q * q_ang.cos()[None, None] + _rotate_half(q) * q_ang.sin()[None, None]
        k = k * k_ang.cos()[None, None] + _rotate_half(k) * k_ang.sin()[None, None]

        if x_q.device.type == "mps":
            # MPS SDPA returns NaN with attn_mask. Manual path.
            attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N_q, N_kv]
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask[None, None], -1e4)
            attn = attn.softmax(dim=-1)
            out = attn @ v
        else:
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask[None, None].to(torch.bool), dropout_p=0.0
                )
            else:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(b, n_q, c)
        return self.proj_drop(self.out_proj(out))


class CrossAttnPredictorBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.cross_attn = MultiheadCrossAttnRoPE(hidden, heads, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm_ff = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden, mlp_ratio, dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        q_coords: torch.Tensor,
        kv_coords: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x_q = x_q + self.drop_path(
            self.cross_attn(self.norm_q(x_q), self.norm_kv(x_kv), q_coords, kv_coords, attn_mask)
        )
        return x_q + self.drop_path(self.mlp(self.norm_ff(x_q)))


class CrossAttnPredictor(nn.Module):
    """Cross-attention drop-in for BlockCausalPredictor.

    Same __init__ and forward signature; ctx_mask is an optional extra arg.
    ctx_mask=None  -> block-causal self-attention (rollout / val-rollout mode).
    ctx_mask given -> cross-attention: K/V=context tokens, Q=all tokens.

    Rollout note: self-attn fallback still hits O(N^2) — cube allowlist required
    for MPS even with cross_attn=true (harp_86 and harp_245 remain excluded).
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
            CrossAttnPredictorBlock(hidden, heads, mlp_ratio, dropout, dp_rates[i])
            for i in range(layers)
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
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = z.shape
        tpf = hp * wp
        x = self.in_proj(z)
        all_coords = build_token_coords(t_total, hp, wp, patch, cadence_min, pixel_scale_mm, z.device)

        if ctx_mask is None:
            # Rollout fallback: full block-causal self-attention.
            sa_mask = build_block_causal_mask(t_total, tpf, z.device)
            if token_pad_mask is not None:
                keep = token_pad_mask.reshape(-1).repeat(t_total)
                sa_mask = sa_mask & keep[None, :] & keep[:, None]
            for blk in self.blocks:
                if self.use_grad_checkpoint and self.training and x.requires_grad:
                    x = ckpt.checkpoint(blk, x, x, all_coords, all_coords, sa_mask, use_reentrant=False)
                else:
                    x = blk(x, x, all_coords, all_coords, sa_mask)
            return self.out_proj(self.norm(x))

        # Cross-attention: K/V fixed to context tokens; Q = all tokens updated each layer.
        x_kv = x[:, ctx_mask]              # [B, N_ctx, D]  — copy, frozen across layers
        kv_coords = all_coords[ctx_mask]   # [N_ctx, 3]

        # JEPA masked pretext: predictor attends to all visible context (no causal).
        # Causality would empty rows when cross_time mask covers a full frame.
        n_ctx = int(ctx_mask.sum().item())
        ca_mask = torch.ones(n, n_ctx, dtype=torch.bool, device=z.device)

        if token_pad_mask is not None:
            q_pad = token_pad_mask.reshape(-1).repeat(t_total)  # [N]
            k_pad = q_pad[ctx_mask]
            ca_mask = ca_mask & q_pad[:, None] & k_pad[None, :]

        if not ca_mask.any(dim=-1).all():
            raise RuntimeError(
                "CrossAttnPredictor: query row with no valid context after pad mask"
            )

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training and x.requires_grad:
                x = ckpt.checkpoint(blk, x, x_kv, all_coords, kv_coords, ca_mask, use_reentrant=False)
            else:
                x = blk(x, x_kv, all_coords, kv_coords, ca_mask)

        return self.out_proj(self.norm(x))
