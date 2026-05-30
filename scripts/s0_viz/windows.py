"""Pick active 64x64 spatiotemporal windows from a cube.

"Active" = high winding-flux activity. We score each valid temporal
window by the spatial-mean ``|flux|`` over its output frames, take the
top ``n_sets`` starts (time-separated), and for each centre a 64x64 crop
on the peak-|flux| pixel within that window.
"""
from typing import List, Tuple

import numpy as np


def _crop_origin(center: int, extent: int, win: int) -> int:
    """Clamp a window origin so ``[o, o+win)`` stays inside ``[0, extent)``."""
    o = int(center) - win // 2
    return max(0, min(o, max(0, extent - win)))


def find_active_windows(
    cube: np.ndarray,
    t_in: int,
    t_out: int,
    win: int = 64,
    n_sets: int = 3,
) -> List[Tuple[int, int, int]]:
    """Return up to ``n_sets`` ``(t_start, y0, x0)`` active-window picks.

    Windows are spaced ≥ ``t_in + t_out`` apart in time so the three sets
    cover distinct phases of the cube's evolution.
    """
    T, H, W = cube.shape
    max_start = T - t_in - t_out
    if max_start < 0:
        return []

    absc = np.abs(cube)
    # Activity per start = mean |flux| over that window's output frames.
    scores = []
    for s in range(max_start + 1):
        out = absc[s + t_in : s + t_in + t_out]
        scores.append((float(out.mean()), s))
    scores.sort(reverse=True)

    picks: List[int] = []
    gap = t_in + t_out
    for _, s in scores:
        if all(abs(s - p) >= gap for p in picks):
            picks.append(s)
        if len(picks) == n_sets:
            break
    picks.sort()

    out_windows: List[Tuple[int, int, int]] = []
    for s in picks:
        # Peak-|flux| location across this window's frames → crop centre.
        frames = absc[s : s + t_in + t_out]
        peak = np.unravel_index(int(frames.sum(axis=0).argmax()), (H, W))
        y0 = _crop_origin(peak[0], H, win)
        x0 = _crop_origin(peak[1], W, win)
        out_windows.append((s, y0, x0))
    return out_windows
