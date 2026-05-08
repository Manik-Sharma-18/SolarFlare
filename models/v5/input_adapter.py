"""Input adapter: pad H,W to multiple of patch size + 1×1 conv 1→13.

Per docs/V5_JEPA/01_path_a.md §3.1: Surya native input is 13ch (8 AIA + 5 HMI).
Wind cube is 1ch signed scalar → learnable channel mixer projects to 13d embed space.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int, value: float = 0.0) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad last two dims (H, W) so each is multiple of `multiple`.

    Returns (x_padded, (pad_h, pad_w)). Padding applied on bottom/right only.
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=value)
    return x, (pad_h, pad_w)


def make_token_pad_mask(
    h: int, w: int, pad_h: int, pad_w: int, patch: int, device: torch.device
) -> torch.Tensor:
    """Per-token boolean mask. True = real, False = padding token.

    Shape: [(h+pad_h)//patch, (w+pad_w)//patch] = [Hp, Wp].
    """
    hp = (h + pad_h) // patch
    wp = (w + pad_w) // patch
    mask = torch.ones(hp, wp, dtype=torch.bool, device=device)
    # Only mark token rows/cols that are FULL padding. A partial pad (pad_h<patch)
    # fills an existing last token row — those tokens are still partly real.
    n_pad_rows = pad_h // patch
    n_pad_cols = pad_w // patch
    if n_pad_rows > 0:
        mask[-n_pad_rows:, :] = False
    if n_pad_cols > 0:
        mask[:, -n_pad_cols:] = False
    return mask


def valid_pixel_to_token_mask(
    valid_mask: torch.Tensor | None,
    hp: int, wp: int, Hp_pad: int, Wp_pad: int,
    patch: int, threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Pool a per-pixel valid mask [B,T,1,H,W] (bool) → per-token mask [B,T,hp,wp] (bool).

    Pad to (Hp_pad, Wp_pad) with False (pad is invalid), avg-pool the real-fraction
    over each `patch×patch` block, threshold to bool. Returns broadcast-friendly
    [1,1,hp,wp] all-True if `valid_mask` is None.
    """
    if valid_mask is None:
        return torch.ones(1, 1, hp, wp, dtype=torch.bool, device=device)
    B, T = valid_mask.shape[:2]
    H, W = valid_mask.shape[-2:]
    v = valid_mask.reshape(B * T, 1, H, W).float()
    pad_h = Hp_pad - H
    pad_w = Wp_pad - W
    if pad_h or pad_w:
        v = F.pad(v, (0, pad_w, 0, pad_h), value=0.0)
    v_pool = F.avg_pool2d(v, kernel_size=patch, stride=patch)
    return (v_pool > threshold).reshape(B, T, hp, wp)


class InputAdapter(nn.Module):
    """1ch wind → 13ch Surya input space, with shape-prep padding.

    forward(x) where x is [B, T, 1, H, W] → returns:
        x_padded:  [B, T, 13, Hp, Wp]
        token_mask: [Hp/patch, Wp/patch] bool
        pads:      (pad_h, pad_w)
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 13, patch_size: int = 16) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if x.dim() != 5 or x.shape[2] != self.in_ch:
            raise ValueError(f"Expected x of shape [B, T, {self.in_ch}, H, W]; got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        x_pad, (pad_h, pad_w) = pad_to_multiple(x_flat, self.patch_size, value=0.0)
        x_proj = self.proj(x_pad)
        x_out = x_proj.reshape(b, t, self.out_ch, x_proj.shape[-2], x_proj.shape[-1])
        token_mask = make_token_pad_mask(h, w, pad_h, pad_w, self.patch_size, x.device)
        return x_out, token_mask, (pad_h, pad_w)
