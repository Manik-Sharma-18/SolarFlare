"""HARP zarr cube loader for V4 ConvLSTM.

Adapts the V5 HARP cube format (one zarr dir per active region; `wind[H,W,T]`
float32 + `Time[T]` float64 epoch seconds + sentinel `Time==0`) into the
mmap-backed `(T, H, W)` numpy cubes the V4 `SolarFluxDataset` expects.

Public entry point:
    load_harp_zarr_data(...) -> (train_ds, val_ds, test_ds, metadata)

Mirrors `loader.load_and_prepare_data` (whole-file split assignment,
training-only normalisation, on-the-fly normalisation inside the dataset)
but ingests `*.zarr` cubes instead of structured `*.npy` files.

Why a separate file: the parent `loader.py` is already 887 lines, well past
the 200-line repo convention. New HARP-specific code lives here.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

from .dataset import SolarFluxDataset, build_index, build_spatial_index
from .loader import (
    _center_crop,
    _compute_norm_params,
    assign_files_to_splits,
    assign_files_with_fixed_test,
)


def _compute_per_cube_norm(
    cube_paths: List[str],
    file_assignments: Dict[str, List[int]],
    method: str = "zscore_per_cube",
    softening: float = 1.0e6,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Compute per-cube normalisation statistics over valid pixels only.

    For each cube assigned to the training split, sample every 10th pixel
    via mmap and compute ``mu = nanmean(x)``, ``sigma = nanstd(x)`` over
    non-zero finite values (clipped-artefact pixels were zeroed in
    `_densify_harp_cube`; we keep zeros in the population but they don't
    distort the mean/std much when the cube is large).

    For ``signed_asinh`` also returns a per-cube ``scale`` so the
    transformed values are roughly unit-variance.

    Args:
        cube_paths: All cube `.npy` paths.
        file_assignments: Split assignment (we only fit on training).
        method: ``"zscore_per_cube"`` or ``"signed_asinh"``.
        softening: For ``signed_asinh``, the per-pixel softening constant
            (default 1e6, halfway between physical max and clip guard).

    Returns:
        ``(per_cube_norm, global_norm)``.
        ``per_cube_norm`` maps train-file_idx → ``{"mu", "sigma", ...}``;
        ``global_norm`` is the same dict computed over the concatenated
        training population — used as a fallback for val/test cubes.
    """
    per_cube_norm: Dict[int, Dict[str, float]] = {}
    pooled_nz: List[np.ndarray] = []
    for file_idx in file_assignments.get("train", []):
        mmap = np.load(cube_paths[file_idx], mmap_mode="r")
        flat = np.asarray(mmap).reshape(-1)
        sub = flat[::10]
        # Stats over non-zero, finite samples only — clipped artefact
        # pixels were zeroed in _densify_harp_cube and would deflate the
        # variance of real signal if pooled in.
        sub_nz = sub[(sub != 0) & np.isfinite(sub)]
        if sub_nz.size < 100:
            sub_nz = sub[np.isfinite(sub)]  # fallback for pathologically sparse cubes
        mu = float(np.nanmean(sub_nz)) if sub_nz.size else 0.0
        sigma = float(np.nanstd(sub_nz)) if sub_nz.size else 1.0
        if not np.isfinite(sigma) or sigma < 1e-9:
            sigma = 1.0
        entry: Dict[str, float] = {
            "mu": mu, "sigma": sigma,
            "n_nz": int(sub_nz.size), "n_sample": int(sub.size),
        }
        if method == "signed_asinh":
            scale_raw = float(np.arcsinh(
                np.nanpercentile(np.abs(sub_nz), 99.99) / softening
            )) if sub_nz.size else 1.0
            entry.update({"softening": softening,
                          "scale": max(scale_raw, 1.0)})
        per_cube_norm[file_idx] = entry
        pooled_nz.append(sub_nz)
    if not pooled_nz or sum(a.size for a in pooled_nz) == 0:
        raise ValueError("No training cubes (or all-zero) — cannot compute "
                         "normalisation")
    flat_nz = np.concatenate(pooled_nz)
    global_mu = float(np.nanmean(flat_nz))
    global_sigma = float(np.nanstd(flat_nz))
    if not np.isfinite(global_sigma) or global_sigma < 1e-9:
        global_sigma = 1.0
    global_norm: Dict[str, float] = {
        "mu": global_mu, "sigma": global_sigma,
        "n_nz": int(flat_nz.size),
    }
    if method == "signed_asinh":
        scale_raw = float(np.arcsinh(
            np.nanpercentile(np.abs(flat_nz), 99.99) / softening
        ))
        global_norm.update({"softening": softening,
                            "scale": max(scale_raw, 1.0)})
    return per_cube_norm, global_norm

logger = logging.getLogger(__name__)

# Per-pixel physical max ~1e7; guard at 1e8 (10x margin). Values above guard
# are treated as instrument artefacts (e.g. harp_8 sentinel garbage at 1.68e10)
# and zeroed before normalisation.
WIND_FLUX_CLIP: float = 1.0e8


def _densify_harp_cube(
    zarr_path: Path,
    clip: float = WIND_FLUX_CLIP,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Open a single HARP zarr cube and return `(cube, meta)`.

    Steps:
    - Open `zarr_path / wind` (layout `[H, W, T]`) and `Time[T]`.
    - Drop frames with `Time <= 0` (sentinel for missing frame).
    - Transpose to `(T, H, W)`.
    - NaN → 0, `|w| > clip` → 0 (per-pixel artefact mask).

    Returns `(T_keep, H, W) float32` cube and a metadata dict including
    `time_range`, `harp_id`, `n_invalid_frames`, `n_clipped_pixels`.
    """
    z = zarr.open(str(zarr_path), mode="r")
    time = np.asarray(z["Time"][:], dtype=np.float64)
    keep_t = time > 0
    n_invalid_frames = int((~keep_t).sum())
    time_kept = time[keep_t]
    keep_idx = np.where(keep_t)[0]
    H, W, _ = z["wind"].shape
    T_keep = len(keep_idx)

    # Chunked T-wise read/clean to bound peak memory at ~one chunk's worth
    # of temporaries. The naive path (`z["wind"][:]` then transpose + bool
    # masks) allocates 3-4× cube-size temporaries — fatal for the 14 GB
    # 5060ti host when harp_1028 (≈5.8 GB float32) is processed.
    CHUNK = 256
    wind_thw = np.empty((T_keep, H, W), dtype=np.float32)
    n_clipped_pixels = 0
    for s in range(0, T_keep, CHUNK):
        e = min(s + CHUNK, T_keep)
        idx = keep_idx[s:e]
        chunk_hwt = np.asarray(z["wind"][:, :, idx], dtype=np.float32)  # (H,W,k)
        # Transpose into the preallocated output, in place.
        wind_thw[s:e] = np.transpose(chunk_hwt, (2, 0, 1))
        del chunk_hwt
        # In-place clip + NaN→0
        block = wind_thw[s:e]
        bad = ~np.isfinite(block) | (np.abs(block) > clip)
        if bad.any():
            n_clipped_pixels += int(bad.sum())
            block[bad] = 0.0
        del bad

    meta = {
        "harp_id": zarr_path.stem,
        "shape": tuple(wind_thw.shape),
        "time_range": (float(time_kept.min()), float(time_kept.max()))
        if time_kept.size else (None, None),
        "n_invalid_frames": n_invalid_frames,
        "n_clipped_pixels": n_clipped_pixels,
        "T_raw": int(time.shape[0]),
        "T_kept": int(time_kept.size),
    }
    return wind_thw.astype(np.float32, copy=False), meta


def load_harp_zarr_data(
    data_dir: str,
    t_in: int = 8,
    t_out: int = 3,
    split_ratios: Optional[List[float]] = None,
    stride: int = 1,
    norm_method: str = "zscore_per_cube",
    norm_config: Optional[Dict] = None,
    augmentation: str = "none",
    dual_channel: bool = False,
    seed: int = 42,
    flare_extreme_threshold: Optional[float] = None,
    crop_size: Optional[Tuple[int, int]] = None,
    flare_density_threshold: float = 0.02,
    cube_allowlist: Optional[List[str]] = None,
    clip: float = WIND_FLUX_CLIP,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    test_cubes: Optional[List[str]] = None,
    val_cubes: Optional[List[str]] = None,
) -> Tuple[SolarFluxDataset, SolarFluxDataset, SolarFluxDataset, Dict[str, Any]]:
    """Load `*.zarr` HARP cubes and build train/val/test datasets.

    Each zarr cube becomes one densified `.npy` in a temp dir, then the
    standard whole-file split + on-the-fly normalisation flow runs (same
    as `load_and_prepare_data`).

    Args:
        data_dir: Directory containing `harp_*.zarr` cubes.
        t_in, t_out: Window lengths in frames (12-min cadence).
        split_ratios: ``[train, test, val]`` fractions; default `[0.7, 0.2, 0.1]`.
        stride: Step between sliding windows.
        norm_method: ``"asinh"`` (default — matches V4 production), ``"robust"``,
            or ``"fixed"``.
        norm_config: Extra params for the chosen normaliser.
        augmentation: ``"none"``, ``"balanced"``, ``"aggressive"``.
        dual_channel: Add a sigmoid extreme-indicator second channel.
        seed: Split + worker seed.
        flare_extreme_threshold: Pre-normalisation threshold for flare-flag
            sampling weights (passed to `build_index`).
        crop_size: Optional center crop ``(H, W)`` per cube.
        flare_density_threshold: Min flare-pixel fraction per window for
            flare-flag.
        cube_allowlist: If given, only HARP IDs in this list are loaded.
        clip: Per-pixel artefact guard; values above are zeroed.

    Returns:
        ``(train_ds, val_ds, test_ds, metadata)``.
    """
    if split_ratios is None:
        split_ratios = [0.7, 0.2, 0.1]

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    zarr_dirs = sorted(p for p in data_path.glob("*.zarr") if p.is_dir())
    if cube_allowlist is not None:
        allow = set(cube_allowlist)
        zarr_dirs = [p for p in zarr_dirs if p.stem in allow]
    if not zarr_dirs:
        raise FileNotFoundError(f"No *.zarr cubes found in {data_path}")

    print(f"Found {len(zarr_dirs)} HARP zarr cubes:")
    for p in zarr_dirs:
        print(f"  {p.name}")
    print()

    tmp_dir = Path(tempfile.mkdtemp(prefix="solarflare_harp_"))
    cube_paths: List[str] = []
    file_meta: Dict[str, Any] = {
        "files": [], "shapes": [], "time_ranges": [], "value_stats": [],
        "harp_meta": [],
    }
    import gc
    for i, zp in enumerate(zarr_dirs):
        cube, meta = _densify_harp_cube(zp, clip=clip)
        if crop_size is not None:
            ch, cw = crop_size
            if cube.shape[1] != ch or cube.shape[2] != cw:
                cube = _center_crop(cube, ch, cw)
        out_path = tmp_dir / f"{meta['harp_id']}.npy"
        np.save(out_path, cube)
        cube_paths.append(str(out_path))
        file_meta["files"].append(zp.name)
        file_meta["shapes"].append(meta["shape"])
        file_meta["time_ranges"].append(meta["time_range"])
        file_meta["value_stats"].append({
            "min": float(cube.min()), "max": float(cube.max()),
            "mean": float(cube.mean()), "std": float(cube.std()),
        })
        file_meta["harp_meta"].append(meta)
        h, w = cube.shape[1], cube.shape[2]
        t_kept = meta["T_kept"]
        n_clipped = meta["n_clipped_pixels"]
        harp_id = meta["harp_id"]
        # Release cube before next iteration — Python's allocator otherwise
        # retains arenas, so 28 sequential densifications would exceed RAM on
        # the 5060ti host (14 GB). gc.collect() forces malloc_trim.
        del cube
        gc.collect()
        print(f"  {harp_id:14s}  T={t_kept:4d}/{meta['T_raw']:4d}  "
              f"H×W={h}×{w}  n_clipped={n_clipped}")

    # Whole-file (= cube-level / AR-identity) split — eliminates AR-identity
    # leakage between train/val/test. ``test_cubes`` pins a fixed, informative
    # test set (cubes with extreme pixels) so flare CSI is comparable across
    # arms instead of split-luck; else fall back to seeded ratio split.
    if test_cubes:
        file_assignments = assign_files_with_fixed_test(
            [Path(p) for p in cube_paths], test_cubes, split_ratios, seed,
            val_ids=val_cubes,
        )
    else:
        file_assignments = assign_files_to_splits(
            [Path(p) for p in cube_paths], split_ratios, seed
        )

    # Normalisation: per-cube (default, V5-informed) or legacy global.
    per_cube_norm: Optional[Dict[int, Dict[str, float]]] = None
    norm_params: Optional[Dict[str, float]] = None
    extreme_threshold: Optional[float] = None
    if norm_method in ("zscore_per_cube", "signed_asinh"):
        nc = norm_config or {}
        # float() guards PyYAML's exponent quirk: "1.0e3" (no sign) is a str.
        softening = float(nc.get("softening",
                                 nc.get("signed_asinh_softening", 1.0e6)))
        per_cube_norm, global_norm = _compute_per_cube_norm(
            cube_paths, file_assignments, method=norm_method,
            softening=softening,
        )
        # Use global as fallback for val/test cubes
        norm_params = global_norm
        dataset_norm_method = norm_method
        print(f"\nNormalisation ({norm_method}, per-cube stats, "
              f"{len(per_cube_norm)} train cubes):")
        for fi in sorted(per_cube_norm):
            s = per_cube_norm[fi]
            stem = Path(cube_paths[fi]).stem
            print(f"  {stem:14s}  mu={s['mu']:+.3e}  sigma={s['sigma']:.3e}")
        print(f"  [global fallback]  mu={global_norm['mu']:+.3e}  "
              f"sigma={global_norm['sigma']:.3e}")
        # dual_channel extreme threshold approximated from training stats
        if dual_channel:
            extreme_threshold = 3.0  # roughly 3 sigma in normalised space
    else:
        # Legacy global asinh / robust / fixed path
        train_values: List[np.ndarray] = []
        for idx in file_assignments["train"]:
            mmap = np.load(cube_paths[idx], mmap_mode="r")
            flat = mmap.reshape(-1)
            train_values.append(flat[::10])
        if not train_values:
            raise ValueError("No training cubes — cannot compute normalisation")
        norm_params = _compute_norm_params(
            np.concatenate(train_values), norm_method, norm_config
        )
        dataset_norm_method = "asinh" if norm_method == "asinh" else "linear"
        print(f"\nNormalisation ({norm_method}, training cubes only):")
        if norm_method == "asinh":
            print(f"  softening: {norm_params['asinh_softening']:.2f}")
            print(f"  extreme threshold: {norm_params['extreme_threshold']:.2f}")
        else:
            print(f"  center: {norm_params['center']:.2f}")
        print(f"  scale: {norm_params['scale']:.2f}")
        if dual_channel:
            raw_et = norm_params.get("extreme_threshold")
            if raw_et is not None and norm_method == "asinh":
                import math
                extreme_threshold = math.asinh(
                    raw_et / norm_params.get("asinh_softening", 1000.0)
                ) / norm_params.get("scale", 1.0)
            else:
                extreme_threshold = norm_params.get("extreme_threshold")

    datasets_out: Dict[str, SolarFluxDataset] = {}
    split_flare_flags: Dict[str, List[bool]] = {}
    split_extreme_densities: Dict[str, List[float]] = {}
    for split_name in ("train", "val", "test"):
        aug = augmentation if split_name == "train" else "none"
        if window_size is not None:
            index, flare_flags, extreme_densities = build_spatial_index(
                file_paths=cube_paths,
                file_assignments=file_assignments,
                t_in=t_in, t_out=t_out,
                window_size=window_size,
                t_stride=stride,
                s_stride=window_stride,
                augmentation=aug, split=split_name,
                extreme_threshold=flare_extreme_threshold,
                flare_density_threshold=flare_density_threshold,
                per_cube_norm=per_cube_norm,
            )
        else:
            index, flare_flags, extreme_densities = build_index(
                file_paths=cube_paths,
                file_assignments=file_assignments,
                t_in=t_in, t_out=t_out, stride=stride,
                augmentation=aug, split=split_name,
                extreme_threshold=flare_extreme_threshold,
                flare_density_threshold=flare_density_threshold,
                per_cube_norm=per_cube_norm,
            )
        datasets_out[split_name] = SolarFluxDataset(
            file_paths=cube_paths,
            index=index, t_in=t_in, t_out=t_out,
            dual_channel=dual_channel,
            extreme_threshold=extreme_threshold,
            norm_params=norm_params,
            norm_method=dataset_norm_method,
            window_size=window_size,
            per_cube_norm=per_cube_norm,
            is_pseudoscalar=True,  # winding flux is a signed pseudoscalar
        )
        split_flare_flags[split_name] = flare_flags
        split_extreme_densities[split_name] = extreme_densities

    print("\nDataset splits (cube-level / AR-identity assignment):")
    for name in ("train", "val", "test"):
        n_flare = sum(split_flare_flags[name])
        print(f"  {name.capitalize()}: {len(datasets_out[name])} samples "
              f"({len(file_assignments[name])} cubes, {n_flare} flare windows)")

    metadata: Dict[str, Any] = {
        **file_meta,
        "normalization": norm_params,
        "n_datasets": len(cube_paths),
        "t_in": t_in, "t_out": t_out,
        "n_train": len(datasets_out["train"]),
        "n_val": len(datasets_out["val"]),
        "n_test": len(datasets_out["test"]),
        "dual_channel": dual_channel,
        "seed": seed,
        "split_ratios": split_ratios,
        "file_assignments": {k: v for k, v in file_assignments.items()},
        "train_flare_flags": split_flare_flags["train"],
        "train_extreme_densities": split_extreme_densities["train"],
        "clip": clip,
        "loader": "harp_zarr",
        "window_size": window_size,
        "window_stride": window_stride,
        "norm_method": norm_method,
        "per_cube_norm": per_cube_norm,
    }
    return (datasets_out["train"], datasets_out["val"],
            datasets_out["test"], metadata)
