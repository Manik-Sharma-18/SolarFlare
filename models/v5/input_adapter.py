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
    if pad_h:
        mask[-(pad_h // patch):, :] = False
    if pad_w:
        mask[:, -(pad_w // patch):] = False
    return mask


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
