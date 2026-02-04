"""
Data loading and preprocessing utilities.

Handles loading .npy files, normalization, and train/val/test splitting.
Includes pre-flight data validation that scans all files before loading,
aborting early if too many files are corrupted or unreadable.

Key design decisions (04-02):
- Whole-file split assignment: entire .npy files assigned to train/test/val
- File assignment is seeded and reproducible
- Normalization computed from training files only
- DataLoader uses spawn on macOS, fork on Linux
- pin_memory True only on CUDA
- seed_worker seeds numpy and random per worker for reproducibility
- Collate function skips None samples from error handling
"""
import json
import logging
import platform
import random as stdlib_random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .dataset import SolarFluxDataset, build_index

logger = logging.getLogger(__name__)

# Required fields for raw structured .npy files
_REQUIRED_STRUCTURED_FIELDS = ('X', 'Y', 'time', 'windTotal')


class DataValidationError(Exception):
    """Raised when data pre-flight validation fails (too many bad files)."""
    pass


def _preflight_scan_npy(npy_files: List[Path], failure_threshold: float) -> List[Path]:
    """
    Pre-flight scan of raw .npy structured array files.

    Memory-maps each file and checks it is a readable structured array with
    required fields.  Returns the list of valid file paths.

    Raises:
        DataValidationError: If the fraction of failed files exceeds
            *failure_threshold*.
    """
    valid_files: List[Path] = []
    failed_files: Dict[str, str] = {}

    for file_path in npy_files:
        try:
            data = np.load(file_path, mmap_mode='r')
            # Verify it is a structured array with required fields
            if data.dtype.names is None:
                failed_files[str(file_path)] = "Not a structured array (no named fields)"
                continue
            missing = [f for f in _REQUIRED_STRUCTURED_FIELDS if f not in data.dtype.names]
            if missing:
                failed_files[str(file_path)] = f"Missing required fields: {missing}"
                continue
            valid_files.append(file_path)
        except Exception as e:
            failed_files[str(file_path)] = str(e)

    total = len(npy_files)
    n_failed = len(failed_files)
    n_valid = len(valid_files)

    print(f"Pre-flight data scan: {n_valid}/{total} files OK")

    if n_failed > 0:
        failure_pct = n_failed / total
        # Always log each failed file
        for fpath, reason in failed_files.items():
            logger.warning("Pre-flight: %s -- %s", Path(fpath).name, reason)

        if failure_pct > failure_threshold:
            detail_lines = [f"  {Path(fp).name}: {reason}" for fp, reason in failed_files.items()]
            detail = "\n".join(detail_lines)
            raise DataValidationError(
                f"{n_failed} of {total} files failed validation "
                f"({failure_pct:.0%}), exceeding {failure_threshold:.0%} threshold.\n"
                f"Failed files:\n{detail}"
            )
        else:
            print(f"  ({n_failed} file(s) failed but within {failure_threshold:.0%} threshold -- skipping them)")

    if n_valid == 0:
        raise DataValidationError(f"All {total} data files failed validation -- cannot proceed")

    return valid_files


def _preflight_scan_npz(npz_files: List[Path], failure_threshold: float) -> List[Path]:
    """
    Pre-flight scan of preprocessed .npz cube files.

    Checks each file is loadable and contains a 'data' key.

    Raises:
        DataValidationError: If the fraction of failed files exceeds
            *failure_threshold*.
    """
    valid_files: List[Path] = []
    failed_files: Dict[str, str] = {}

    for file_path in npz_files:
        try:
            npz = np.load(file_path)
            if 'data' not in npz.files:
                failed_files[str(file_path)] = "Missing 'data' key in .npz archive"
                continue
            valid_files.append(file_path)
        except Exception as e:
            failed_files[str(file_path)] = str(e)

    total = len(npz_files)
    n_failed = len(failed_files)
    n_valid = len(valid_files)

    print(f"Pre-flight data scan: {n_valid}/{total} files OK")

    if n_failed > 0:
        failure_pct = n_failed / total
        for fpath, reason in failed_files.items():
            logger.warning("Pre-flight: %s -- %s", Path(fpath).name, reason)

        if failure_pct > failure_threshold:
            detail_lines = [f"  {Path(fp).name}: {reason}" for fp, reason in failed_files.items()]
            detail = "\n".join(detail_lines)
            raise DataValidationError(
                f"{n_failed} of {total} files failed validation "
                f"({failure_pct:.0%}), exceeding {failure_threshold:.0%} threshold.\n"
                f"Failed files:\n{detail}"
            )
        else:
            print(f"  ({n_failed} file(s) failed but within {failure_threshold:.0%} threshold -- skipping them)")

    if n_valid == 0:
        raise DataValidationError(f"All {total} data files failed validation -- cannot proceed")

    return valid_files


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


# ---------------------------------------------------------------------------
# Whole-file split assignment
# ---------------------------------------------------------------------------


def assign_files_to_splits(
    file_paths: List[Path],
    split_ratios: List[float],
    seed: int,
) -> Dict[str, List[int]]:
    """Assign entire files to train/test/val splits by shuffled index.

    Each file goes to exactly one split -- no file is ever split across
    train/test/val.  The assignment is seeded for reproducibility.

    Args:
        file_paths: Ordered list of file paths (used only for count + logging).
        split_ratios: ``[train, test, val]`` fractions summing to ~1.0.
        seed: Random seed for reproducible shuffling.

    Returns:
        ``{"train": [...], "test": [...], "val": [...]}`` with file indices.

    Raises:
        ValueError: If ratios do not sum to ~1.0 or fewer than 3 values given.
    """
    if len(split_ratios) != 3:
        raise ValueError(f"split_ratios must have 3 elements [train, test, val], got {len(split_ratios)}")

    ratio_sum = sum(split_ratios)
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(f"split_ratios must sum to ~1.0, got {ratio_sum:.4f}")

    n = len(file_paths)
    indices = list(range(n))
    rng = stdlib_random.Random(seed)
    rng.shuffle(indices)

    n_train = round(n * split_ratios[0])
    n_test = round(n * split_ratios[1])
    n_val = n - n_train - n_test

    # Guard against negative val from rounding
    if n_val < 0:
        n_train += n_val  # reduce train to compensate
        n_val = 0

    assignments = {
        "train": indices[:n_train],
        "test": indices[n_train:n_train + n_test],
        "val": indices[n_train + n_test:],
    }

    logger.info(
        "File split assignment (seed=%d): train=%d, test=%d, val=%d (total=%d)",
        seed, len(assignments["train"]), len(assignments["test"]),
        len(assignments["val"]), n,
    )
    for split_name, idxs in assignments.items():
        for idx in idxs:
            logger.debug("  %s -> %s", split_name, file_paths[idx].name if hasattr(file_paths[idx], 'name') else file_paths[idx])

    return assignments


# ---------------------------------------------------------------------------
# Public data loading functions
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    data_dir: str,
    t_in: int = 8,
    t_out: int = 3,
    split_ratios: Optional[List[float]] = None,
    stride: int = 1,
    norm_method: str = "robust",
    norm_config: Optional[Dict] = None,
    augmentation: str = "none",
    dual_channel: bool = False,
    failure_threshold: float = 0.1,
    seed: int = 42,
) -> Tuple[SolarFluxDataset, SolarFluxDataset, SolarFluxDataset, Dict[str, Any]]:
    """
    Load raw .npy files and create train/val/test datasets with whole-file splitting.

    Runs a pre-flight validation scan before loading.  Normalization is
    computed from *training files only* (via mmap sampling) and applied
    on-the-fly in each dataset's ``__getitem__``.

    Args:
        data_dir: Path to directory containing .npy files.
        t_in: Number of input timesteps.
        t_out: Number of output timesteps.
        split_ratios: ``[train, test, val]`` fractions.  Defaults to ``[0.7, 0.2, 0.1]``.
        stride: Step between consecutive sliding-window starts.
        norm_method: ``"robust"``, ``"fixed"``, or ``"asinh"``.
        norm_config: Extra parameters for the chosen method.
        augmentation: ``"none"``, ``"balanced"``, ``"aggressive"``.
        dual_channel: If True, output 2 channels (flux + extreme indicator).
        failure_threshold: Max fraction of files that can fail before aborting.
        seed: Random seed for split assignment and DataLoader workers.

    Returns:
        ``(train_dataset, val_dataset, test_dataset, metadata)``
    """
    if split_ratios is None:
        split_ratios = [0.7, 0.2, 0.1]

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Discover .npy files (prefer windTotal pattern)
    npy_files = sorted(data_path.glob('windTotal*.npy'))
    if len(npy_files) == 0:
        npy_files = sorted(data_path.glob('*.npy'))
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {data_path}")

    print(f"Found {len(npy_files)} data files:")
    for f in npy_files:
        print(f"  {f.name}")
    print()

    # Pre-flight validation scan (memory-mapped, minimal I/O)
    valid_files = _preflight_scan_npy(npy_files, failure_threshold)

    # ------------------------------------------------------------------
    # Detect whether files are structured arrays or cubes
    # ------------------------------------------------------------------
    sample = np.load(valid_files[0], mmap_mode='r')
    is_structured = sample.dtype.names is not None

    if is_structured:
        # Convert structured arrays to .npy cubes in a temp directory
        print("\nConverting structured arrays to dense cubes...")
        tmp_dir = Path(tempfile.mkdtemp(prefix="solarflare_cubes_"))
        cube_paths: List[str] = []
        file_meta: Dict[str, Any] = {
            'files': [], 'shapes': [], 'time_ranges': [], 'value_stats': []
        }

        for i, file_path in enumerate(valid_files):
            print(f"  Converting {file_path.name}...")
            data = np.load(file_path)
            flux_cube, cube_meta = _structured_to_cube(data)
            out_path = tmp_dir / f"cube_{i:04d}.npy"
            np.save(out_path, flux_cube)
            cube_paths.append(str(out_path))
            file_meta['files'].append(file_path.name)
            file_meta['shapes'].append(flux_cube.shape)
            file_meta['time_ranges'].append(cube_meta['time_range'])
            file_meta['value_stats'].append({
                'min': float(flux_cube.min()),
                'max': float(flux_cube.max()),
                'mean': float(flux_cube.mean()),
                'std': float(flux_cube.std()),
            })
    else:
        # Already 3-D cubes -- use directly
        cube_paths = [str(f) for f in valid_files]
        file_meta = {'files': [], 'shapes': [], 'time_ranges': [], 'value_stats': []}
        for file_path in valid_files:
            mmap = np.load(file_path, mmap_mode='r')
            file_meta['files'].append(file_path.name)
            file_meta['shapes'].append(mmap.shape)
            # time_ranges not available for raw cubes
            file_meta['time_ranges'].append(None)
            file_meta['value_stats'].append({
                'min': float(mmap.min()),
                'max': float(mmap.max()),
                'mean': float(mmap.mean()),
                'std': float(mmap.std()),
            })

    # ------------------------------------------------------------------
    # Whole-file split assignment
    # ------------------------------------------------------------------
    file_assignments = assign_files_to_splits(
        [Path(p) for p in cube_paths], split_ratios, seed
    )

    # ------------------------------------------------------------------
    # Normalization from TRAINING files only (mmap + subsample)
    # ------------------------------------------------------------------
    train_values: List[np.ndarray] = []
    for idx in file_assignments["train"]:
        mmap = np.load(cube_paths[idx], mmap_mode='r')
        flat = mmap.reshape(-1)
        train_values.append(flat[::100])  # sample every 100th value

    if len(train_values) == 0:
        raise ValueError("No training files assigned -- cannot compute normalization")

    all_train_values = np.concatenate(train_values)
    norm_params = _compute_norm_params(all_train_values, norm_method, norm_config)

    # Map norm_method to the method string SolarFluxDataset expects
    if norm_method == "asinh":
        dataset_norm_method: Optional[str] = "asinh"
    else:
        dataset_norm_method = "linear"

    print(f"\nNormalization ({norm_method}, training files only):")
    if norm_method == 'asinh':
        print(f"  Softening: {norm_params['asinh_softening']:.2f}")
        print(f"  Extreme threshold: {norm_params['extreme_threshold']:.2f}")
    else:
        print(f"  Center: {norm_params['center']:.2f}")
    print(f"  Scale: {norm_params['scale']:.2f}")

    # ------------------------------------------------------------------
    # Build index and datasets for each split
    # ------------------------------------------------------------------
    extreme_threshold = norm_params.get('extreme_threshold', None) if dual_channel else None

    datasets_out = {}
    for split_name in ("train", "val", "test"):
        aug = augmentation if split_name == "train" else "none"
        index = build_index(
            file_paths=cube_paths,
            file_assignments=file_assignments,
            t_in=t_in,
            t_out=t_out,
            stride=stride,
            augmentation=aug,
            split=split_name,
        )
        ds = SolarFluxDataset(
            file_paths=cube_paths,
            index=index,
            t_in=t_in,
            t_out=t_out,
            dual_channel=dual_channel,
            extreme_threshold=extreme_threshold,
            norm_params=norm_params,
            norm_method=dataset_norm_method,
        )
        datasets_out[split_name] = ds

    print("\nDataset splits (whole-file assignment):")
    for name in ("train", "val", "test"):
        print(f"  {name.capitalize()}: {len(datasets_out[name])} samples "
              f"({len(file_assignments[name])} files)")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata: Dict[str, Any] = {
        **file_meta,
        'normalization': norm_params,
        'n_datasets': len(cube_paths),
        't_in': t_in,
        't_out': t_out,
        'n_train': len(datasets_out["train"]),
        'n_val': len(datasets_out["val"]),
        'n_test': len(datasets_out["test"]),
        'dual_channel': dual_channel,
        'seed': seed,
        'split_ratios': split_ratios,
        'file_assignments': {k: v for k, v in file_assignments.items()},
    }

    return datasets_out["train"], datasets_out["val"], datasets_out["test"], metadata


def load_preprocessed_data(
    preprocessed_dir: str,
    t_in: int = 8,
    t_out: int = 3,
    split_ratios: Optional[List[float]] = None,
    stride: int = 1,
    augmentation: str = "none",
    dual_channel: bool = False,
    failure_threshold: float = 0.1,
    seed: int = 42,
) -> Tuple[SolarFluxDataset, SolarFluxDataset, SolarFluxDataset, Dict[str, Any]]:
    """
    Load preprocessed cube files for fast training with whole-file splitting.

    Preprocessed cubes are already normalized, so ``norm_params=None`` is
    passed to the dataset (no on-the-fly normalization).  Uses whole-file
    split assignment identical to :func:`load_and_prepare_data`.

    Args:
        preprocessed_dir: Path to directory with ``cube_*.npz`` files.
        t_in: Number of input timesteps.
        t_out: Number of output timesteps.
        split_ratios: ``[train, test, val]`` fractions.  Defaults to ``[0.7, 0.2, 0.1]``.
        stride: Step between consecutive sliding-window starts.
        augmentation: ``"none"``, ``"balanced"``, ``"aggressive"``.
        dual_channel: If True, output 2 channels (flux + extreme indicator).
        failure_threshold: Max fraction of files that can fail before aborting.
        seed: Random seed for split assignment.

    Returns:
        ``(train_dataset, val_dataset, test_dataset, metadata)``
    """
    if split_ratios is None:
        split_ratios = [0.7, 0.2, 0.1]

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
    norm_info = metadata.get('normalization', {})
    print(f"  Normalization: center={norm_info.get('center', 'N/A')}, "
          f"scale={norm_info.get('scale', 'N/A')}")

    # Discover and validate cube files
    cube_files = sorted(data_path.glob('cube_*.npz'))
    if len(cube_files) == 0:
        raise FileNotFoundError(f"No cube_*.npz files found in {data_path}")

    valid_cube_files = _preflight_scan_npz(cube_files, failure_threshold)

    # Extract cubes and save as .npy for mmap-based SolarFluxDataset
    tmp_dir = Path(tempfile.mkdtemp(prefix="solarflare_preproc_"))
    cube_paths: List[str] = []

    for i, cube_file in enumerate(valid_cube_files):
        print(f"  Loading {cube_file.name}...")
        npz = np.load(cube_file)
        cube = npz['data']
        out_path = tmp_dir / f"cube_{i:04d}.npy"
        np.save(out_path, cube)
        cube_paths.append(str(out_path))
        print(f"    Shape: {cube.shape}")

    # Whole-file split assignment
    file_assignments = assign_files_to_splits(
        [Path(p) for p in cube_paths], split_ratios, seed
    )

    # Extreme threshold for dual-channel mode
    extreme_threshold = norm_info.get('extreme_threshold', None) if dual_channel else None
    if dual_channel:
        print("\nDual-channel mode enabled:")
        print("  Channel 1: Normalized flux")
        if extreme_threshold:
            print(f"  Channel 2: Extreme event indicator (threshold: {extreme_threshold:.4f})")
        else:
            print("  Warning: No extreme_threshold in metadata, using default")
            extreme_threshold = 0.8

    # Build index and datasets -- no on-the-fly normalization (already done)
    datasets_out = {}
    for split_name in ("train", "val", "test"):
        aug = augmentation if split_name == "train" else "none"
        index = build_index(
            file_paths=cube_paths,
            file_assignments=file_assignments,
            t_in=t_in,
            t_out=t_out,
            stride=stride,
            augmentation=aug,
            split=split_name,
        )
        ds = SolarFluxDataset(
            file_paths=cube_paths,
            index=index,
            t_in=t_in,
            t_out=t_out,
            dual_channel=dual_channel,
            extreme_threshold=extreme_threshold,
            norm_params=None,   # Already normalized
            norm_method=None,
        )
        datasets_out[split_name] = ds

    print("\nDataset splits (whole-file assignment):")
    for name in ("train", "val", "test"):
        print(f"  {name.capitalize()}: {len(datasets_out[name])} samples "
              f"({len(file_assignments[name])} files)")

    # Update metadata with runtime info
    metadata['n_datasets'] = len(cube_paths)
    metadata['t_in'] = t_in
    metadata['t_out'] = t_out
    metadata['n_train'] = len(datasets_out["train"])
    metadata['n_val'] = len(datasets_out["val"])
    metadata['n_test'] = len(datasets_out["test"])
    metadata['dual_channel'] = dual_channel
    metadata['seed'] = seed
    metadata['split_ratios'] = split_ratios
    metadata['file_assignments'] = {k: v for k, v in file_assignments.items()}

    return datasets_out["train"], datasets_out["val"], datasets_out["test"], metadata


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def _seed_worker(worker_id: int) -> None:
    """Seed numpy and stdlib random per DataLoader worker for reproducibility.

    Follows the PyTorch recommended pattern using ``torch.initial_seed()``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    stdlib_random.seed(worker_seed)


def _skip_none_collate(batch):
    """Custom collate that filters out None samples from error-handling.

    When :class:`SolarFluxDataset.__getitem__` encounters a read error it
    returns ``None``.  This collate function silently drops those entries
    and collates the remaining valid samples.

    Returns:
        Collated batch, or ``None`` if all samples were ``None``.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def create_dataloaders(
    train_dataset: SolarFluxDataset,
    val_dataset: SolarFluxDataset,
    test_dataset: SolarFluxDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create platform-aware DataLoaders from datasets.

    Key behaviours:
    - ``pin_memory`` is ``True`` only when *device* is CUDA.
    - On macOS (Darwin), ``multiprocessing_context="spawn"`` is used to
      avoid fork-safety issues with mmap file descriptors.
    - Each worker is seeded via :func:`_seed_worker` for reproducibility.
    - A custom collate function filters out ``None`` samples.
    - ``persistent_workers=True`` when ``num_workers > 0``.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size for all loaders.
        num_workers: Number of data-loading workers.
        device: Target device (used only for pin_memory decision).
        seed: Random seed for the generator.

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    pin_memory = device is not None and device.type == "cuda"

    # Platform-aware multiprocessing context (only relevant with workers)
    mp_context = None
    if num_workers > 0:
        mp_context = "spawn" if platform.system() == "Darwin" else None

    # Reproducible generator
    g = torch.Generator()
    g.manual_seed(seed)

    common_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "generator": g,
        "collate_fn": _skip_none_collate,
    }

    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        if mp_context is not None:
            common_kwargs["multiprocessing_context"] = mp_context

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader
