"""
Data Preprocessing Script

Converts raw structured .npy files to dense cubes for fast loading during training.
Run this once before training to speed up data loading.

Usage:
    python preprocess_data.py
    python preprocess_data.py --input ./data --output ./data_processed
"""
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def structured_to_cube(data: np.ndarray) -> tuple:
    """Convert structured array to dense (T, H, W) cube."""
    x_coords = np.unique(data['X'])
    y_coords = np.unique(data['Y'])
    times = np.unique(data['time'])
    
    H, W = len(y_coords), len(x_coords)
    T = len(times)
    
    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    flux_cube = np.zeros((T, H, W), dtype=np.float32)
    
    print(f"    Converting {len(data):,} records to cube ({T}, {H}, {W})...")
    for i in tqdm(range(len(data)), desc="    Building cube", leave=False):
        t_idx = time_to_idx[data['time'][i]]
        h_idx = y_to_idx[data['Y'][i]]
        w_idx = x_to_idx[data['X'][i]]
        flux_cube[t_idx, h_idx, w_idx] = data['windTotal'][i]
    
    metadata = {
        'time_range': (str(times[0]), str(times[-1])),
        'shape': (T, H, W),
        'x_range': (float(x_coords.min()), float(x_coords.max())),
        'y_range': (float(y_coords.min()), float(y_coords.max())),
    }
    
    return flux_cube, metadata


def compute_normalization(cubes: list, method: str = 'robust', 
                          asinh_softening: float = 1000.0,
                          extreme_threshold_percentile: float = 99.5) -> dict:
    """Compute normalization parameters from all cubes.
    
    Args:
        cubes: List of flux cubes
        method: 'robust', 'fixed', or 'asinh'
        asinh_softening: Softening parameter for asinh transform
        extreme_threshold_percentile: Percentile for extreme event threshold
    """
    # Sample values from all cubes
    all_values = []
    for cube in cubes:
        all_values.append(cube.flatten()[::10])
    all_values = np.concatenate(all_values)
    
    # Compute data statistics (always useful)
    data_stats = {
        'data_min': float(all_values.min()),
        'data_max': float(all_values.max()),
        'data_mean': float(all_values.mean()),
        'data_std': float(all_values.std()),
    }
    
    if method == 'asinh':
        # Asinh transformation: preserves sign, compresses dynamic range
        # No clipping - extreme values are compressed but preserved
        center = 0.0  # Asinh is symmetric, no centering needed
        abs_max_percentile = float(np.percentile(np.abs(all_values), 99.99))
        scale = float(np.arcsinh(abs_max_percentile / asinh_softening))
        
        # Compute threshold for extreme events (for dual-channel)
        extreme_threshold = float(np.percentile(np.abs(all_values), extreme_threshold_percentile))

        # Also compute normalized-space extreme threshold for dual-channel mode
        extreme_threshold_normalized = float(np.arcsinh(extreme_threshold / asinh_softening) / max(scale, 1.0))

        return {
            'method': 'asinh',
            'center': center,
            'scale': scale,
            'asinh_softening': asinh_softening,
            'extreme_threshold': extreme_threshold,
            'extreme_threshold_normalized': extreme_threshold_normalized,
            'extreme_threshold_percentile': extreme_threshold_percentile,
            'scale_percentile': 99.99,
            **data_stats,
        }
    
    elif method == 'robust':
        low = np.percentile(all_values, 1)
        high = np.percentile(all_values, 99)
        center = float(np.median(all_values))
        scale = float(max((high - low) / 2, 1.0))
        
        return {
            'method': 'robust',
            'center': center,
            'scale': scale,
            **data_stats,
        }
    
    else:  # fixed
        center = 0.0
        scale = 40000.0
        
        return {
            'method': 'fixed',
            'center': center,
            'scale': scale,
            **data_stats,
        }


def preprocess(input_dir: str = './data', output_dir: str = './data_processed',
               norm_method: str = 'asinh', asinh_softening: float = 1000.0,
               extreme_threshold_percentile: float = 99.5):
    """
    Preprocess raw .npy files and save as compressed cubes.
    
    Args:
        input_dir: Directory containing raw windTotal*.npy files
        output_dir: Directory to save processed cubes
        norm_method: 'robust', 'fixed', or 'asinh' (recommended for solar flare data)
        asinh_softening: Softening parameter for asinh transform (default 1000.0)
        extreme_threshold_percentile: Percentile for extreme event detection
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find raw files
    npy_files = sorted(input_path.glob('windTotal*.npy'))
    if len(npy_files) == 0:
        npy_files = sorted(input_path.glob('*.npy'))
    
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {input_path}")
    
    print(f"Found {len(npy_files)} raw data files")
    print(f"Output directory: {output_path}\n")
    
    # First pass: convert to cubes
    cubes = []
    cube_metadata = []
    
    for file_path in npy_files:
        print(f"Processing {file_path.name}...")
        
        data = np.load(file_path)
        cube, meta = structured_to_cube(data)
        
        print(f"    Shape: {cube.shape}")
        print(f"    Value range: [{cube.min():.2f}, {cube.max():.2f}]")
        
        cubes.append(cube)
        cube_metadata.append({
            'source_file': file_path.name,
            **meta
        })
    
    # Compute normalization
    print(f"\nComputing normalization ({norm_method})...")
    norm_params = compute_normalization(cubes, norm_method, asinh_softening, 
                                        extreme_threshold_percentile)
    print(f"  Method: {norm_params['method']}")
    if norm_method == 'asinh':
        print(f"  Softening: {norm_params['asinh_softening']:.2f}")
        print(f"  Extreme threshold: {norm_params['extreme_threshold']:.2f}")
    else:
        print(f"  Center: {norm_params['center']:.2f}")
    print(f"  Scale: {norm_params['scale']:.2f}")
    
    # Normalize and save cubes
    print(f"\nSaving preprocessed cubes...")
    
    for i, (cube, meta) in enumerate(zip(cubes, cube_metadata)):
        # Normalize based on method
        if norm_method == 'asinh':
            # Asinh transform: preserves extreme values
            cube_norm = np.arcsinh(cube / norm_params['asinh_softening']) / norm_params['scale']
            cube_norm = np.clip(cube_norm, -1.0, 1.0)
        else:
            # Linear normalization (robust or fixed)
            cube_norm = (cube - norm_params['center']) / norm_params['scale']
        
        # Save as compressed npz
        output_name = f"cube_{i:03d}.npz"
        output_file = output_path / output_name
        
        np.savez_compressed(
            output_file,
            data=cube_norm,
            **{k: np.array(v) if isinstance(v, tuple) else v for k, v in meta.items()}
        )
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  Saved {output_name} ({size_mb:.1f} MB)")
    
    # Save metadata
    metadata = {
        'normalization': norm_params,
        'cubes': cube_metadata,
        'n_cubes': len(cubes),
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nPreprocessing complete!")
    print(f"  {len(cubes)} cubes saved to {output_path}")
    print(f"  Metadata saved to {output_path / 'metadata.json'}")


if __name__ == '__main__':
    import sys
    
    # Simple argument parsing (avoiding argparse per user preference)
    input_dir = './data'
    output_dir = './data_processed'
    norm_method = 'asinh'  # Default to asinh for solar flare data
    asinh_softening = 1000.0
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--input' and i + 1 < len(args):
            input_dir = args[i + 1]
            i += 2
        elif args[i] == '--output' and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif args[i] == '--method' and i + 1 < len(args):
            norm_method = args[i + 1]
            i += 2
        elif args[i] == '--softening' and i + 1 < len(args):
            asinh_softening = float(args[i + 1])
            i += 2
        else:
            i += 1
    
    print("=" * 60)
    print("SOLAR FLARE DATA PREPROCESSING")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Method: {norm_method}")
    if norm_method == 'asinh':
        print(f"Asinh softening: {asinh_softening}")
    print("=" * 60 + "\n")
    
    preprocess(input_dir, output_dir, norm_method, asinh_softening)

