"""3D RoPE in physical units (min, Mm, Mm).

Per docs/V5_JEPA/01_path_a.md §3.1:
    token_coord = (t_idx · 12 min, y_idx · 0.364 Mm, x_idx · 0.364 Mm).
Frequency basis lives in physical space, NOT pixel index — variable cube dims with
constant 0.364 Mm/pixel scale transfer natively.

Head dim is split into 3 axes; even split when divisible, residual passthrough otherwise.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _split_head_dim(head_dim: int) -> tuple[int, int, int, int]:
    """Split head_dim into (d_t, d_y, d_x, d_passthrough). All d_* even (RoPE pairs)."""
    if head_dim % 2:
        raise ValueError("head_dim must be even (RoPE pairs).")
    third = head_dim // 3
    third -= third % 2
    d_t = d_y = d_x = third
    used = d_t + d_y + d_x
    return d_t, d_y, d_x, head_dim - used


def build_token_coords(
    t_in: int,
    hp: int,
    wp: int,
    patch_size: int,
    cadence_min: float = 12.0,
    pixel_scale_mm: float = 0.364,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-token (t_min, y_mm, x_mm) coordinates.

    Returns [t_in * hp * wp, 3] float32. Token order = (t, y, x) row-major,
    i.e. for fixed t and y, x varies fastest. Patch-center physical positions used.
    """
    device = device or torch.device("cpu")
    t = torch.arange(t_in, device=device, dtype=torch.float32) * cadence_min
    y = (torch.arange(hp, device=device, dtype=torch.float32) + 0.5) * patch_size * pixel_scale_mm
    x = (torch.arange(wp, device=device, dtype=torch.float32) + 0.5) * patch_size * pixel_scale_mm
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack([tt, yy, xx], dim=-1).reshape(-1, 3)


class RoPE3D(nn.Module):
    """Apply 3D RoPE to query/key tensors of shape [B, H, N, head_dim].

    coords: [N, 3] in physical units. Frequency bands per axis use the standard
    1 / base^(2i/d) schedule with three independent bases (one per axis).
    """

    def __init__(
        self,
        head_dim: int,
        base_t: float = 10000.0,
        base_y: float = 10000.0,
        base_x: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        d_t, d_y, d_x, d_pass = _split_head_dim(head_dim)
        self.d_t, self.d_y, self.d_x, self.d_pass = d_t, d_y, d_x, d_pass

        def _freqs(d: int, base: float) -> torch.Tensor:
            if d == 0:
                return torch.zeros(0)
            i = torch.arange(0, d, 2, dtype=torch.float32)
            return 1.0 / (base ** (i / d))

        self.register_buffer("freq_t", _freqs(d_t, base_t), persistent=False)
        self.register_buffer("freq_y", _freqs(d_y, base_y), persistent=False)
        self.register_buffer("freq_x", _freqs(d_x, base_x), persistent=False)

    def _angles(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, 3] -> per-axis angles concatenated, then duplicated for (cos, sin) pairs.
        ang_t = coords[:, 0:1] * self.freq_t  # [N, d_t/2]
        ang_y = coords[:, 1:2] * self.freq_y  # [N, d_y/2]
        ang_x = coords[:, 2:3] * self.freq_x  # [N, d_x/2]
        # Interleave each pair so dim-pair (2i, 2i+1) shares one angle.
        ang = torch.cat([ang_t, ang_y, ang_x], dim=-1)             # [N, (d_t+d_y+d_x)/2]
        ang = torch.repeat_interleave(ang, repeats=2, dim=-1)      # [N, d_t+d_y+d_x]
        if self.d_pass:
            pad = torch.zeros(ang.shape[0], self.d_pass, device=ang.device, dtype=ang.dtype)
            ang = torch.cat([ang, pad], dim=-1)
        return ang  # [N, head_dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: [B, H, N, head_dim]. coords: [N, 3]."""
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("q/k head_dim mismatch with RoPE3D head_dim.")
        ang = self._angles(coords).to(q.dtype)              # [N, D]
        cos = ang.cos()[None, None, :, :]                   # [1, 1, N, D]
        sin = ang.sin()[None, None, :, :]
        q_rot = _rotate_half(q) * sin + q * cos
        k_rot = _rotate_half(k) * sin + k * cos
        return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """For each (2i, 2i+1) pair, swap to (-x_{2i+1}, x_{2i})."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.reshape(*x.shape)
