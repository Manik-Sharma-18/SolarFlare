"""Cube loading + per-cube z-score normalisation for S0 visualisation.

Reuses the training loader's ``_densify_harp_cube`` so the (T, H, W) cube
and clip/sentinel handling exactly match what the model was trained on.
Per-cube z-score replicates ``_compute_per_cube_norm`` (mean/std over
non-zero finite pixels).
"""
from pathlib import Path
from typing import Tuple

import numpy as np

from solarflare_data.harp_loader import _densify_harp_cube, WIND_FLUX_CLIP


def load_cube(zarr_path: Path) -> Tuple[np.ndarray, dict]:
    """Densify one zarr cube to ``(T, H, W)`` float32 (sentinel/clip handled)."""
    return _densify_harp_cube(zarr_path, clip=WIND_FLUX_CLIP)


def cube_stats(cube: np.ndarray) -> Tuple[float, float]:
    """Per-cube ``(mu, sigma)`` over non-zero finite pixels (training logic)."""
    flat = cube.reshape(-1)
    nz = flat[(flat != 0) & np.isfinite(flat)]
    if nz.size == 0:
        return 0.0, 1.0
    mu = float(np.mean(nz))
    sigma = float(np.std(nz))
    if not np.isfinite(sigma) or sigma < 1e-9:
        sigma = 1.0
    return mu, sigma


def asinh_scale(cube: np.ndarray, softening: float = 1.0e6) -> Tuple[float, float]:
    """Per-cube ``(scale, softening)`` for ``signed_asinh`` (matches
    ``_compute_per_cube_norm``): ``scale = max(asinh(p99.99(|x|)/soft), 1.0)``."""
    flat = cube.reshape(-1)
    nz = flat[(flat != 0) & np.isfinite(flat)]
    if nz.size == 0:
        return 1.0, softening
    scale = float(np.arcsinh(np.nanpercentile(np.abs(nz), 99.99) / softening))
    return max(scale, 1.0), softening


def normalize(x: np.ndarray, p0: float, p1: float,
              method: str = "zscore_per_cube") -> np.ndarray:
    """zscore: (p0,p1)=(mu,sigma). signed_asinh: (p0,p1)=(scale,softening)."""
    if method == "signed_asinh":
        return np.sign(x) * np.arcsinh(np.abs(x) / p1) / p0
    return (x - p0) / p1


def denormalize(x: np.ndarray, p0: float, p1: float,
                method: str = "zscore_per_cube") -> np.ndarray:
    """Inverse of :func:`normalize` for the same ``method``."""
    if method == "signed_asinh":
        return np.sign(x) * p1 * np.sinh(np.abs(x) * p0)
    return x * p1 + p0
