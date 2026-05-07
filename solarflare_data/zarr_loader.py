"""Lazy zarr-cube reader for V5 JEPA pipeline.

Per docs/V5_JEPA/06_data.md §11.1c–§11.5:
- wind[H, W, T] fp32 with NaN
- Time (T,) float64 epoch seconds; Time==0 = sentinel (skip)
- NaN→0 cast at loader boundary; emit valid_pixel_mask = ~isnan(orig) as second stream
- Loader returns fp32 (model casts to bf16 at compute boundary)
- Sparse-chunk fill is silent — never retry, never interpolate
- Sentinel outlier guard: |Bz| > BZ_CLIP_GAUSS treated as invalid (data quality —
  observed values up to 1e10 G in raw cubes, physical extreme sunspot ~5000 G).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


BZ_CLIP_GAUSS: float = 1.0e5  # |Bz| above this = sentinel/corruption, not physical


@dataclass
class CubeHandle:
    path: Path
    harp_id: str
    wind: zarr.Array       # [H, W, T] fp32 lazy
    time: np.ndarray       # [T] float64
    valid_frames: np.ndarray  # bool [T], time > 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.wind.shape)  # type: ignore[return-value]


def open_cube(path: str | Path) -> CubeHandle:
    p = Path(path)
    z = zarr.open(str(p), mode="r")
    wind = z["wind"]
    time = np.asarray(z["Time"][:], dtype=np.float64)
    valid = time > 0
    return CubeHandle(path=p, harp_id=p.stem, wind=wind, time=time, valid_frames=valid)


def read_window(cube: CubeHandle, t_start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Read [H, W, length] slice. Returns (wind_filled_fp32, valid_pixel_mask_bool).

    NaN positions → 0.0 in wind; valid_pixel_mask is True where original was finite.
    Caller must verify all `length` Time entries are > 0 (use `is_window_valid`).
    """
    if t_start < 0 or t_start + length > cube.shape[2]:
        raise IndexError(
            f"Window [{t_start}:{t_start + length}] out of bounds for cube T={cube.shape[2]}"
        )
    raw = np.asarray(cube.wind[:, :, t_start:t_start + length], dtype=np.float32)
    valid = np.isfinite(raw) & (np.abs(raw) <= BZ_CLIP_GAUSS)
    wind = np.where(valid, raw, 0.0).astype(np.float32, copy=False)
    return wind, valid


def is_window_valid(cube: CubeHandle, t_start: int, length: int) -> bool:
    """All Time[t_start:t_start+length] must be > 0."""
    if t_start < 0 or t_start + length > cube.shape[2]:
        return False
    return bool(cube.valid_frames[t_start:t_start + length].all())


def iter_valid_starts(cube: CubeHandle, length: int) -> Iterable[int]:
    """Yield every t_start such that the contiguous window of `length` frames is all valid."""
    valid = cube.valid_frames
    n = valid.shape[0]
    if length > n:
        return
    for t in range(n - length + 1):
        if valid[t:t + length].all():
            yield t


def cube_norm_stats(cube: CubeHandle, sample_frames: int = 32) -> tuple[float, float]:
    """NaN-aware (μ, σ) for per-cube z-score.

    Subsamples up to `sample_frames` valid frames to avoid full-cube read. Falls back to
    (0.0, 1.0) if no valid frames or σ collapses to zero.
    """
    idx = np.where(cube.valid_frames)[0]
    if idx.size == 0:
        return 0.0, 1.0
    if idx.size > sample_frames:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=sample_frames, replace=False)
        idx.sort()

    chunks: list[np.ndarray] = []
    for t in idx:
        frame = np.asarray(cube.wind[:, :, int(t)], dtype=np.float32)
        # Drop sentinel outliers before computing stats
        bounded = frame[np.isfinite(frame) & (np.abs(frame) <= BZ_CLIP_GAUSS)]
        if bounded.size:
            chunks.append(bounded)
    flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    mu = float(np.mean(flat)) if flat.size else 0.0
    sigma = float(np.std(flat)) if flat.size else 1.0
    if not np.isfinite(sigma) or sigma < 1e-8:
        sigma = 1.0
    if not np.isfinite(mu):
        mu = 0.0
    return mu, sigma
