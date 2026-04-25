#!/usr/bin/env python3
"""Preprocess downloaded magnetogram cubes for pretraining.

Normalizes (T, H, W) .npy cubes using asinh normalization and saves
as .npz files in the same format as the existing winding flux pipeline.

Usage:
    python scripts/preprocess_magnetograms.py
    python scripts/preprocess_magnetograms.py --input-dir ./data_magnetogram --output-dir ./data_magnetogram_processed

Reads:  data_magnetogram/magnetogram_HARP*.npy
Writes: data_magnetogram_processed/cube_*.npz + metadata.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT_DIR = Path("data_magnetogram")
DEFAULT_OUTPUT_DIR = Path("data_magnetogram_processed")

# Asinh softening for magnetogram Gauss values (different from winding flux 1000.0)
DEFAULT_ASINH_SOFTENING = 500.0
DEFAULT_EXTREME_PERCENTILE = 99


def compute_normalization(cubes: list, asinh_softening: float = DEFAULT_ASINH_SOFTENING,
                          extreme_percentile: float = DEFAULT_EXTREME_PERCENTILE) -> dict:
    """Compute asinh normalization parameters from training cubes.

    Same approach as preprocess_data.py but with magnetogram-appropriate defaults.
    """
    print("Computing normalization parameters...")
    all_values = []
    for cube in cubes:
        flat = cube.reshape(-1)
        all_values.append(flat[::10])  # Sample every 10th value
    all_values = np.concatenate(all_values)

    abs_max_p = np.percentile(np.abs(all_values), 99.99)
    scale = float(np.arcsinh(abs_max_p / asinh_softening))

    extreme_threshold = float(np.percentile(np.abs(all_values), extreme_percentile))
    extreme_normalized = float(np.arcsinh(extreme_threshold / asinh_softening) / scale)

    params = {
        'method': 'asinh',
        'center': 0.0,
        'scale': scale,
        'asinh_softening': asinh_softening,
        'extreme_threshold': extreme_threshold,
        'extreme_threshold_normalized': extreme_normalized,
        'extreme_threshold_percentile': extreme_percentile,
        'data_min': float(all_values.min()),
        'data_max': float(all_values.max()),
        'data_mean': float(all_values.mean()),
        'data_std': float(all_values.std()),
    }

    print(f"  Scale: {scale:.4f}")
    print(f"  Data range: [{params['data_min']:.1f}, {params['data_max']:.1f}] Gauss")
    print(f"  Extreme threshold: {extreme_threshold:.1f} Gauss ({extreme_normalized:.4f} normalized)")
    return params


def normalize_cube(cube: np.ndarray, norm_params: dict) -> np.ndarray:
    """Apply asinh normalization to a cube."""
    softening = norm_params['asinh_softening']
    scale = norm_params['scale']
    normalized = np.arcsinh(cube / softening) / scale
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess magnetogram cubes for pretraining"
    )
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--asinh-softening', type=float, default=DEFAULT_ASINH_SOFTENING)
    parser.add_argument('--extreme-percentile', type=float, default=DEFAULT_EXTREME_PERCENTILE)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover .npy cube files
    npy_files = sorted(input_dir.glob("magnetogram_HARP*.npy"))
    if not npy_files:
        print(f"No magnetogram_HARP*.npy files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(npy_files)} magnetogram cubes in {input_dir}")

    # Load all cubes (memory-mapped for efficiency)
    cubes = []
    cube_meta = []
    for f in npy_files:
        cube = np.load(str(f), mmap_mode='r')
        cubes.append(cube)
        cube_meta.append({
            'source_file': f.name,
            'shape': list(cube.shape),
        })
        print(f"  {f.name}: shape {cube.shape}")

    # Compute normalization from all cubes
    # Load fully into memory for stats computation
    cubes_full = [np.load(str(f)) for f in npy_files]
    norm_params = compute_normalization(
        cubes_full,
        asinh_softening=args.asinh_softening,
        extreme_percentile=args.extreme_percentile,
    )

    # Normalize and save as .npz
    print(f"\nSaving normalized cubes to {output_dir}/")
    for i, (cube, meta) in enumerate(zip(cubes_full, cube_meta)):
        cube_norm = normalize_cube(cube, norm_params)
        output_file = output_dir / f"cube_{i:03d}.npz"
        np.savez_compressed(
            str(output_file),
            data=cube_norm,
            source_file=meta['source_file'],
            shape=meta['shape'],
        )
        print(f"  cube_{i:03d}.npz ← {meta['source_file']} ({cube_norm.shape})")

    # Save metadata
    metadata = {
        'normalization': norm_params,
        'num_cubes': len(cubes_full),
        'cubes': cube_meta,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved: {metadata_path}")
    print(f"\nDone! {len(cubes_full)} cubes preprocessed.")
    print(f"\nNext step: python main.py --config configs/pretrain_magnetogram.yaml")


if __name__ == '__main__':
    main()
