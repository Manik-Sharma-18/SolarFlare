"""Spatially-averaged 1D winding flux target for probe head.

Per docs/V5_JEPA/11_winding_flux_probe_head.md:
    wind_target[t] = mean over valid pixels of cube.wind[:, :, t]
    Valid = finite AND |w| <= clip (defaults to WIND_FLUX_CLIP=1e8).
    Frames with Time==0 (invalid frame) → NaN sentinel (caller filters).
    Frames with zero valid pixels → NaN.

Two flavors:
- Signed mean (default): preserves chirality / sign convention.
- Abs mean: |wind| spatial average — robust when sign cancels across AR.

Clip override:
- Default 1e8 matches encoder data path (zarr_loader.WIND_FLUX_CLIP).
- Tighter clips (1e6, 1e5) useful for probe target to suppress pathological
  pixels (harp_8 has 14k px up to 1.68e10; even 1e8 clip leaves abs-mean ~5e6).

Cache layout (alongside zarr):
    data/<harp_id>_wind_1d.npy             fp32 [T]   signed,  clip 1e8
    data/<harp_id>_wind_1d_abs.npy         fp32 [T]   abs,     clip 1e8
    data/<harp_id>_wind_1d_clip1e6.npy     fp32 [T]   signed,  clip 1e6
    data/<harp_id>_wind_1d_abs_clip1e6.npy fp32 [T]   abs,     clip 1e6

Smoothing is plot-/probe-time post-processing, not cached.

Public:
    compute_wind_mean_1d(cube, abs_value=False, clip=WIND_FLUX_CLIP) -> [T] fp32
    smooth_1d(arr, window) -> [T] fp32   NaN-aware centered rolling mean
    load_or_compute_wind_1d(path, abs_value, clip, recompute) -> [T] fp32
    cache_path(path, abs_value, clip) -> Path
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .zarr_loader import WIND_FLUX_CLIP, CubeHandle, open_cube

# Probe-target clip. Tighter than encoder WIND_FLUX_CLIP=1e8.
# Per docs/V5_JEPA/OUTLIERS.md: per-pixel physical max ~1e7 (senior).
# clip=1e7 trims unphysical [1e7, 1e8] residue (harp_8 sentinel garbage at
# 1.68e10 already removed at 1e8, but tail residue persists). Physical
# pixels in [1e6, 1e7] preserved → harp_8 retains real high-flux signal.
# Sweep analysis (10 log clips × 21 cubes): healthy cubes lose <6% mean,
# harp_8 loses 12% — only the unphysical fraction. Below 1e7 cuts real
# signal differentially in harp_8 (high-flux AR) → not used.
WIND_PROBE_CLIP: float = 1.0e7


def _clip_tag(clip: float) -> str:
    if clip == WIND_FLUX_CLIP:
        return ""
    exp = int(round(np.log10(clip)))
    return f"_clip1e{exp}"


def cache_path(cube_path: str | Path, abs_value: bool = False, clip: float = WIND_FLUX_CLIP) -> Path:
    p = Path(cube_path)
    flavor = "_abs" if abs_value else ""
    return p.parent / f"{p.stem}_wind_1d{flavor}{_clip_tag(clip)}.npy"


def compute_wind_mean_1d(
    cube: CubeHandle,
    abs_value: bool = False,
    clip: float = WIND_FLUX_CLIP,
) -> np.ndarray:
    """Spatial mean of valid pixels per frame. Returns [T] fp32.

    Streams one frame at a time to bound peak memory at ~H*W*4 bytes.
    """
    _, _, t = cube.shape
    out = np.full((t,), np.nan, dtype=np.float32)
    for ti in range(t):
        if not cube.valid_frames[ti]:
            continue
        frame = np.asarray(cube.wind[:, :, ti], dtype=np.float32)
        valid = np.isfinite(frame) & (np.abs(frame) <= clip)
        if not valid.any():
            continue
        vals = frame[valid]
        if abs_value:
            vals = np.abs(vals)
        out[ti] = np.float32(vals.mean())
    return out


def smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centered rolling mean. Returns [T] fp32.

    window <= 1 returns arr unchanged.
    Positions with zero valid neighbors in window → NaN.
    Edges use available-only neighbors (no zero-padding bias).
    """
    if window is None or window <= 1:
        return arr.astype(np.float32, copy=False)
    a = arr.astype(np.float32, copy=False)
    finite = np.isfinite(a).astype(np.float32)
    vals = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32)
    num = np.convolve(vals, kernel, mode="same")
    den = np.convolve(finite, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan).astype(np.float32)
    return out


def load_or_compute_wind_1d(
    cube_path: str | Path,
    abs_value: bool = False,
    clip: float = WIND_FLUX_CLIP,
    recompute: bool = False,
) -> np.ndarray:
    """Return cached wind_1d [T] fp32; compute + cache if missing.

    Cache key includes clip via filename suffix (see cache_path).
    """
    out_path = cache_path(cube_path, abs_value=abs_value, clip=clip)
    if out_path.exists() and not recompute:
        return np.load(out_path).astype(np.float32, copy=False)
    cube = open_cube(cube_path)
    arr = compute_wind_mean_1d(cube, abs_value=abs_value, clip=clip)
    np.save(out_path, arr)
    return arr
