"""Spatial alignment helpers."""
from typing import Tuple

import torch
import torch.nn.functional as F

# Anything beyond this is a config bug (wrong input H/W vs encoder strides),
# not stride-2 round-off.
_MAX_DRIFT_PX = 2


def match_spatial(t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Center-crop or zero-pad ``t`` along its last two axes to ``target_hw``.

    Raises ``RuntimeError`` if the mismatch exceeds ``±2`` px per axis.
    Replaces the previous ``F.interpolate(mode='nearest')`` patch, which
    introduced sub-pixel shift on the residual head.
    """
    src_h, src_w = t.shape[-2], t.shape[-1]
    tgt_h, tgt_w = int(target_hw[0]), int(target_hw[1])

    if (abs(src_h - tgt_h) > _MAX_DRIFT_PX) or (abs(src_w - tgt_w) > _MAX_DRIFT_PX):
        raise RuntimeError(
            f"Spatial mismatch too large to resolve by center-crop/pad: "
            f"got ({src_h}, {src_w}), expected ({tgt_h}, {tgt_w})"
        )

    if src_h > tgt_h or src_w > tgt_w:
        h_off = max((src_h - tgt_h) // 2, 0)
        w_off = max((src_w - tgt_w) // 2, 0)
        t = t[..., h_off:h_off + tgt_h, w_off:w_off + tgt_w]
        src_h, src_w = t.shape[-2], t.shape[-1]

    if src_h < tgt_h or src_w < tgt_w:
        pad_h = tgt_h - src_h
        pad_w = tgt_w - src_w
        # F.pad order: (W_left, W_right, H_top, H_bot)
        t = F.pad(
            t,
            (pad_w // 2, pad_w - pad_w // 2,
             pad_h // 2, pad_h - pad_h // 2),
        )
    return t
