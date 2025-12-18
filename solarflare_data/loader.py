"""
Data loading and preprocessing utilities.

Handles loading .npy files, normalization, and train/val/test splitting.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader

from .dataset import SolarFluxDataset


def load_and_prepare_data(
    data_dir: str,
    t_in: int = 8,
    t_out: int = 3,
    train_split: float = 0.7,
    val_split: float = 0.15,
    norm_method: str = "robust",
    norm_config: Dict = None,
    augment_train: bool = True,
    dual_channel: bool = False
) -> Tuple[SolarFluxDataset, SolarFluxDataset, SolarFluxDataset, Dict[str, Any]]:
    """
    Load .npy files, normalize, and create train/val/test datasets.
    
    Args:
        data_dir: Path to directory containing .npy files
        t_in: Number of input timesteps
        t_out: Number of output timesteps
        train_split: Fraction for training
        val_split: Fraction for validation (remainder is test)
        norm_method: "robust", "fixed", or "asinh" (recommended for solar flare)
        norm_config: Dict with normalization parameters
        augment_train: Whether to augment training data
        dual_channel: If True, output 2 channels (flux + extreme indicator)
    
    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find .npy files (prefer windTotal pattern)
    npy_files = sorted(data_path.glob('windTotal*.npy'))
    if len(npy_files) == 0:
        npy_files = sorted(data_path.glob('*.npy'))
    
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {data_path}")
    
    print(f"Found {len(npy_files)} data files:")
    for f in npy_files:
        print(f"  {f.name}")
    print()
    
    # Load and convert each file to dense cube
    datasets = []
    metadata = {
        'files': [],
        'shapes': [],
        'time_ranges': [],
        'value_stats': []
    }
    
    # First pass: collect statistics for normalization
    all_values = []
    
    for file_path in npy_files:
        print(f"Loading {file_path.name}...")
        try:
            data = np.load(file_path)
            flux_cube, cube_meta = _structured_to_cube(data)
            
            print(f"  Shape: T={flux_cube.shape[0]}, H={flux_cube.shape[1]}, W={flux_cube.shape[2]}")
            print(f"  Value range: [{flux_cube.min():.2f}, {flux_cube.max():.2f}]")
            
            datasets.append(flux_cube)
            metadata['files'].append(file_path.name)
            metadata['shapes'].append(flux_cube.shape)
            metadata['time_ranges'].append(cube_meta['time_range'])
            metadata['value_stats'].append({
                'min': float(flux_cube.min()),
                'max': float(flux_cube.max()),
                'mean': float(flux_cube.mean()),
                'std': float(flux_cube.std())
            })
            
            # Sample values for normalization stats
            all_values.append(flux_cube.flatten()[::100])  # Sample every 100th value
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if len(datasets) == 0:
        raise ValueError("No datasets loaded successfully")
    
    # Compute normalization parameters
    all_values = np.concatenate(all_values)
    norm_params = _compute_norm_params(all_values, norm_method, norm_config)
    metadata['normalization'] = norm_params
    
    print(f"\nNormalization ({norm_method}):")
    if norm_method == 'asinh':
        print(f"  Softening: {norm_params['asinh_softening']:.2f}")
        print(f"  Extreme threshold: {norm_params['extreme_threshold']:.2f}")
    else:
        print(f"  Center: {norm_params['center']:.2f}")
    print(f"  Scale: {norm_params['scale']:.2f}")
    
    # Normalize all datasets
    for i in range(len(datasets)):
        if norm_method == 'asinh':
            # Asinh transform: preserves extreme values
            datasets[i] = np.arcsinh(datasets[i] / norm_params['asinh_softening']) / norm_params['scale']
        else:
            # Linear normalization (robust or fixed)
            datasets[i] = (datasets[i] - norm_params['center']) / norm_params['scale']
    
    # Create sliding window samples
    all_samples_train = []
    all_samples_val = []
    all_samples_test = []
    
    for dataset_id, flux_cube in enumerate(datasets):
        T = flux_cube.shape[0]
        n_samples = T - t_in - t_out + 1
        
        if n_samples <= 0:
            print(f"Warning: Dataset {dataset_id} too short ({T} frames), skipping")
            continue
        
        # Time-based split within each dataset
        train_end = int(train_split * n_samples)
        val_end = int((train_split + val_split) * n_samples)
        
        for start_idx in range(n_samples):
            sample = (dataset_id, start_idx)
            if start_idx < train_end:
                all_samples_train.append(sample)
            elif start_idx < val_end:
                all_samples_val.append(sample)
            else:
                all_samples_test.append(sample)
    
    print("\nDataset splits:")
    print(f"  Train: {len(all_samples_train)} samples")
    print(f"  Val:   {len(all_samples_val)} samples")
    print(f"  Test:  {len(all_samples_test)} samples")
    
    # Get extreme threshold for dual-channel mode
    extreme_threshold = norm_params.get('extreme_threshold', None) if dual_channel else None
    if dual_channel:
        print("\nDual-channel mode enabled:")
        print("  Channel 1: Normalized flux")
        print(f"  Channel 2: Extreme event indicator (threshold: {extreme_threshold:.4f})")
    
    # Create datasets
    train_dataset = SolarFluxDataset(
        all_samples_train, datasets, t_in, t_out, 
        augment=augment_train, dual_channel=dual_channel, extreme_threshold=extreme_threshold
    )
    val_dataset = SolarFluxDataset(
        all_samples_val, datasets, t_in, t_out, 
        augment=False, dual_channel=dual_channel, extreme_threshold=extreme_threshold
    )
    test_dataset = SolarFluxDataset(
        all_samples_test, datasets, t_in, t_out, 
        augment=False, dual_channel=dual_channel, extreme_threshold=extreme_threshold
    )
    
    metadata['n_datasets'] = len(datasets)
    metadata['t_in'] = t_in
    metadata['t_out'] = t_out
    metadata['n_train'] = len(all_samples_train)
    metadata['n_val'] = len(all_samples_val)
    metadata['n_test'] = len(all_samples_test)
    metadata['dual_channel'] = dual_channel
    
    return train_dataset, val_dataset, test_dataset, metadata


def load_preprocessed_data(
    preprocessed_dir: str,
    t_in: int = 8,
    t_out: int = 3,
    train_split: float = 0.7,
    val_split: float = 0.15,
    augment_train: bool = True,
    dual_channel: bool = False
) -> Tuple[SolarFluxDataset, SolarFluxDataset, SolarFluxDataset, Dict[str, Any]]:
    """
    Load preprocessed cube files for fast training.
    
    Use this after running preprocess_data.py to skip the slow
    structured-to-cube conversion.
    
    Args:
        preprocessed_dir: Path to directory with cube_*.npz files
        t_in: Number of input timesteps
        t_out: Number of output timesteps
        train_split: Fraction for training
        val_split: Fraction for validation
        augment_train: Whether to augment training data
        dual_channel: If True, output 2 channels (flux + extreme indicator)
    
    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    import json
    
    data_path = Path(preprocessed_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {data_path}")
    
    # Load metadata
    metadata_file = data_path / 'metadata.json'
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {data_path}. "
            f"Run preprocess_data.py first."
        )
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loading preprocessed data from {data_path}")
    print(f"  Normalization: center={metadata['normalization']['center']:.2f}, "
          f"scale={metadata['normalization']['scale']:.2f}")
    
    # Load cube files
    cube_files = sorted(data_path.glob('cube_*.npz'))
    
    if len(cube_files) == 0:
        raise FileNotFoundError(f"No cube_*.npz files found in {data_path}")
    
    datasets = []
    for cube_file in cube_files:
        print(f"  Loading {cube_file.name}...")
        npz = np.load(cube_file)
        cube = npz['data']
        datasets.append(cube)
        print(f"    Shape: {cube.shape}")
    
    # Create sliding window samples (same logic as load_and_prepare_data)
    all_samples_train = []
    all_samples_val = []
    all_samples_test = []
    
    for dataset_id, flux_cube in enumerate(datasets):
        T = flux_cube.shape[0]
        n_samples = T - t_in - t_out + 1
        
        if n_samples <= 0:
            print(f"Warning: Dataset {dataset_id} too short ({T} frames), skipping")
            continue
        
        train_end = int(train_split * n_samples)
        val_end = int((train_split + val_split) * n_samples)
        
        for start_idx in range(n_samples):
            sample = (dataset_id, start_idx)
            if start_idx < train_end:
                all_samples_train.append(sample)
            elif start_idx < val_end:
                all_samples_val.append(sample)
            else:
                all_samples_test.append(sample)
    
    print("\nDataset splits:")
    print(f"  Train: {len(all_samples_train)} samples")
    print(f"  Val:   {len(all_samples_val)} samples")
    print(f"  Test:  {len(all_samples_test)} samples")
    
    # Get extreme threshold for dual-channel mode
    extreme_threshold = metadata['normalization'].get('extreme_threshold', None) if dual_channel else None
    if dual_channel:
        print("\nDual-channel mode enabled:")
        print("  Channel 1: Normalized flux")
        if extreme_threshold:
            print(f"  Channel 2: Extreme event indicator (threshold: {extreme_threshold:.4f})")
        else:
            print("  Warning: No extreme_threshold in metadata, using default")
            extreme_threshold = 0.8  # Default threshold in normalized space
    
    # Create datasets
    train_dataset = SolarFluxDataset(
        all_samples_train, datasets, t_in, t_out, 
        augment=augment_train, dual_channel=dual_channel, extreme_threshold=extreme_threshold
    )
    val_dataset = SolarFluxDataset(
        all_samples_val, datasets, t_in, t_out, 
        augment=False, dual_channel=dual_channel, extreme_threshold=extreme_threshold
    )
    test_dataset = SolarFluxDataset(
        all_samples_test, datasets, t_in, t_out, 
        augment=False, dual_channel=dual_channel, extreme_threshold=extreme_threshold
    )
    
    # Update metadata with runtime info
    metadata['n_datasets'] = len(datasets)
    metadata['t_in'] = t_in
    metadata['t_out'] = t_out
    metadata['n_train'] = len(all_samples_train)
    metadata['n_val'] = len(all_samples_val)
    metadata['n_test'] = len(all_samples_test)
    metadata['dual_channel'] = dual_channel
    
    return train_dataset, val_dataset, test_dataset, metadata


def _structured_to_cube(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Convert structured array to dense (T, H, W) cube.
    
    Args:
        data: Structured numpy array with X, Y, time, windTotal fields
    
    Returns:
        flux_cube: Dense array (T, H, W)
        metadata: Dict with coordinate info
    """
    # Extract unique coordinates
    x_coords = np.unique(data['X'])
    y_coords = np.unique(data['Y'])
    times = np.unique(data['time'])
    
    H, W = len(y_coords), len(x_coords)
    T = len(times)
    
    # Create mappings
    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    # Build dense cube
    flux_cube = np.zeros((T, H, W), dtype=np.float32)
    
    for i in range(len(data)):
        t_idx = time_to_idx[data['time'][i]]
        h_idx = y_to_idx[data['Y'][i]]
        w_idx = x_to_idx[data['X'][i]]
        flux_cube[t_idx, h_idx, w_idx] = data['windTotal'][i]
    
    metadata = {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'times': times,
        'time_range': (str(times[0]), str(times[-1]))
    }
    
    return flux_cube, metadata


def _compute_norm_params(
    values: np.ndarray, 
    method: str, 
    config: Dict = None
) -> Dict:
    """
    Compute normalization parameters.
    
    Args:
        values: Flattened sample of all values
        method: "robust", "fixed", or "asinh"
        config: Additional configuration
    
    Returns:
        Dict with normalization parameters
    """
    config = config or {}
    
    # Compute data statistics (always useful)
    data_stats = {
        'data_min': float(values.min()),
        'data_max': float(values.max()),
        'data_mean': float(values.mean()),
        'data_std': float(values.std()),
    }
    
    if method == "asinh":
        # Asinh transformation: preserves sign, compresses dynamic range
        softening = config.get('asinh_softening', 1000.0)
        extreme_pct = config.get('extreme_threshold_percentile', 99.5)
        
        center = 0.0  # Asinh is symmetric
        scale = float(np.arcsinh(max(abs(values.min()), abs(values.max())) / softening))
        extreme_threshold = float(np.percentile(np.abs(values), extreme_pct))
        
        return {
            'method': 'asinh',
            'center': center,
            'scale': float(max(scale, 1.0)),
            'asinh_softening': softening,
            'extreme_threshold': extreme_threshold,
            'extreme_threshold_percentile': extreme_pct,
            **data_stats
        }
    
    elif method == "robust":
        # Percentile-based normalization (robust to outliers)
        p_low = config.get('percentile_low', 1)
        p_high = config.get('percentile_high', 99)
        
        low = np.percentile(values, p_low)
        high = np.percentile(values, p_high)
        center = np.median(values)
        scale = (high - low) / 2  # Scale so most values are in [-1, 1]
        
        return {
            'method': 'robust',
            'center': float(center),
            'scale': float(max(scale, 1.0)),  # Avoid division by zero
            'percentile_low': p_low,
            'percentile_high': p_high,
            **data_stats
        }
    
    elif method == "fixed":
        # Fixed normalization factor
        factor = config.get('fixed_factor', 40000.0)
        return {
            'method': 'fixed',
            'center': 0.0,
            'scale': float(factor),
            **data_stats
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_dataloaders(
    train_dataset: SolarFluxDataset,
    val_dataset: SolarFluxDataset,
    test_dataset: SolarFluxDataset,
    batch_size: int = 1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

