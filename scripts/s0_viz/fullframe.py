"""Full-frame prediction by tiling the 64x64 model across the cube.

The model only ever sees ``win x win`` patches, so a whole-frame forecast
is built by sliding a window over the frame (origins clamped to the edge so
every tile is full-size), predicting all tiles in one batched forward, and
averaging predictions where tiles overlap. Mirrors ``inference.infer_full_frame``
but operates on a pre-normalised cube and the SimpleConvLSTM API.
"""
from typing import List

import numpy as np
import torch


def _starts(extent: int, win: int, stride: int) -> List[int]:
    """Window origins covering ``[0, extent)``; last clamped to the edge."""
    if extent <= win:
        return [0]
    s = list(range(0, extent - win + 1, stride))
    if s[-1] != extent - win:
        s.append(extent - win)
    return s


@torch.no_grad()
def predict_full_frame(
    model,
    cube_norm: np.ndarray,
    t_start: int,
    t_in: int,
    t_out: int,
    win: int,
    stride: int,
    device: torch.device,
    max_batch: int = 64,
) -> np.ndarray:
    """Tiled whole-frame forecast → ``(t_out, H, W)`` normalised.

    Edge cubes smaller than ``win`` in a dim use a single full-extent tile.
    """
    T, H, W = cube_norm.shape
    wy, wx = min(win, H), min(win, W)
    ys, xs = _starts(H, wy, stride), _starts(W, wx, stride)

    tiles, coords = [], []
    for y in ys:
        for x in xs:
            tiles.append(cube_norm[t_start : t_start + t_in, y : y + wy, x : x + wx])
            coords.append((y, x))
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(tiles)))  # (N, t_in, wy, wx)
    batch = batch.unsqueeze(1).float()  # (N, 1, t_in, wy, wx)

    accum = np.zeros((t_out, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for i in range(0, batch.shape[0], max_batch):
        chunk = batch[i : i + max_batch].to(device)
        pred = model(chunk, teacher_forcing_ratio=0.0)  # (n, 1, t_out, wy, wx)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred[:, 0].cpu().numpy()
        for j in range(pred.shape[0]):
            y, x = coords[i + j]
            accum[:, y : y + wy, x : x + wx] += pred[j]
            count[y : y + wy, x : x + wx] += 1.0
    count = np.maximum(count, 1e-9)
    return (accum / count[None]).astype(np.float32)
