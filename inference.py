"""
Inference Script - Load trained model and make predictions

Usage:
    python inference.py

This script demonstrates how to:
1. Load a trained model from checkpoint
2. Load and normalize new data
3. Make predictions
4. Unnormalize predictions to get actual flux values
"""
import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SolarFluxPredictor
from utils.device import resolve_device
from utils.checkpoint import load_checkpoint_for_inference


def load_model(checkpoint_path: str, device: torch.device = None):
    """
    Load trained model from checkpoint.

    Returns model AND normalization_params (self-contained, no metadata.json needed).

    Args:
        checkpoint_path: Path to best_model.pt
        device: torch.device to load onto (auto-detected if None)

    Returns:
        Tuple of (model, normalization_params). normalization_params may be None
        for legacy checkpoints that don't contain them.
    """
    checkpoint, device = load_checkpoint_for_inference(Path(checkpoint_path), device)

    config = checkpoint.get('config', {})

    # Create model with same architecture
    # Handle both new nested format (config.model.X) and legacy flat format (config.X)
    model = SolarFluxPredictor(
        input_channels=config.get('input_channels',
                        config.get('model', {}).get('input_channels', 1)),
        output_channels=config.get('output_channels',
                        config.get('model', {}).get('output_channels', 1)),
        t_out=config.get('t_out',
              config.get('data', {}).get('t_out', 3)),
        channels=config.get('channels',
                 config.get('model', {}).get('channels', [16, 32, 64])),
        kernel_size=config.get('kernel_size',
                    config.get('model', {}).get('kernel_size', 3)),
        downsample_input=config.get('downsample_input',
                         config.get('model', {}).get('downsample_input', True)),
        use_checkpointing=config.get('use_checkpointing',
                          config.get('model', {}).get('use_checkpointing', False)),
        dropout_rate=config.get('dropout_rate',
                     config.get('model', {}).get('dropout_rate', 0.0))
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    normalization_params = checkpoint.get('normalization_params')

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if checkpoint.get('best_val_loss') is not None:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")

    return model, normalization_params


def load_normalization(metadata_path: str) -> dict:
    """Load normalization parameters from metadata.

    Note: Prefer using normalization_params from checkpoint directly
    (returned by load_model). This function is for legacy checkpoints
    that don't contain normalization_params.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['normalization']


def normalize(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """Normalize data using saved parameters."""
    return (data - norm_params['center']) / norm_params['scale']


def unnormalize(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """Convert normalized values back to original scale."""
    return data * norm_params['scale'] + norm_params['center']


def predict(
    model: SolarFluxPredictor,
    input_data: np.ndarray,
    device: torch.device = None,
) -> np.ndarray:
    """
    Make predictions on input data.

    Args:
        model: Loaded model
        input_data: numpy array of shape (T_in, H, W) - already normalized
        device: torch.device (inferred from model parameters if None)

    Returns:
        predictions: numpy array of shape (T_out, H, W)
    """
    if device is None:
        device = next(model.parameters()).device
    # Ensure correct shape: (B, C, T, H, W)
    if input_data.ndim == 3:
        input_tensor = input_data[np.newaxis, np.newaxis, ...]  # (1, 1, T, H, W)
    elif input_data.ndim == 4:
        input_tensor = input_data[np.newaxis, ...]  # (1, C, T, H, W)
    else:
        input_tensor = input_data
    
    # Convert to tensor
    x = torch.from_numpy(input_tensor).float().to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(x, teacher_forcing_ratio=0.0)
    
    # Return as (T_out, H, W)
    return predictions[0, 0].cpu().numpy()


def load_raw_data(npy_path: str, t_in: int = 8, start_idx: int = 0) -> np.ndarray:
    """
    Load raw .npy file and extract a sequence for prediction.
    
    Args:
        npy_path: Path to windTotal*.npy file
        t_in: Number of input frames
        start_idx: Starting time index
    
    Returns:
        flux_cube: (T_in, H, W) array of raw flux values
    """
    data = np.load(npy_path)
    
    # Get unique coordinates
    x_coords = np.unique(data['X'])
    y_coords = np.unique(data['Y'])
    times = np.unique(data['time'])
    
    H, W = len(y_coords), len(x_coords)
    T = len(times)
    
    print(f"Loaded data: T={T}, H={H}, W={W}")
    
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
    
    # Extract sequence
    sequence = flux_cube[start_idx:start_idx + t_in]
    print(f"Extracted sequence from t={start_idx} to t={start_idx + t_in - 1}")
    
    return sequence


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Configuration - UPDATE THESE PATHS
    CHECKPOINT_PATH = './outputs/best_model.pt'
    METADATA_PATH = './outputs/metadata.json'
    DATA_PATH = './data/windTotal_MM_2024-05-09_0000_2348.npy'  # Example
    
    T_IN = 8  # Must match training config
    START_IDX = 50  # Which time index to start from
    
    # Device selection
    device = resolve_device("auto")
    print("=" * 60)
    
    # Load model (includes normalization from checkpoint if available)
    print("\n1. Loading model...")
    model, norm_params_from_ckpt = load_model(CHECKPOINT_PATH, device)

    # Use normalization from checkpoint if available, fall back to metadata.json
    print("\n2. Loading normalization parameters...")
    if norm_params_from_ckpt:
        norm_params = norm_params_from_ckpt
        print("   Loaded normalization from checkpoint (self-contained)")
    else:
        print("   Loading normalization from metadata.json (legacy checkpoint)")
        norm_params = load_normalization(METADATA_PATH)
    print(f"   Center: {norm_params['center']:.2f}")
    print(f"   Scale: {norm_params['scale']:.2f}")
    
    # Load and prepare input data
    print("\n3. Loading input data...")
    raw_input = load_raw_data(DATA_PATH, T_IN, START_IDX)
    print(f"   Raw flux range: [{raw_input.min():.2f}, {raw_input.max():.2f}]")
    
    # Normalize input
    normalized_input = normalize(raw_input, norm_params)
    print(f"   Normalized range: [{normalized_input.min():.2f}, {normalized_input.max():.2f}]")
    
    # Make prediction
    print("\n4. Making predictions...")
    predictions_normalized = predict(model, normalized_input, device)
    print(f"   Output shape: {predictions_normalized.shape}")  # (T_out, H, W)
    
    # Unnormalize to get actual flux values
    predictions_actual = unnormalize(predictions_normalized, norm_params)
    print(f"   Predicted flux range: [{predictions_actual.min():.2f}, {predictions_actual.max():.2f}]")
    
    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Input: {T_IN} frames starting at index {START_IDX}")
    print(f"Output: {predictions_actual.shape[0]} predicted frames")
    print(f"  Frame t+1: mean={predictions_actual[0].mean():.2f}, std={predictions_actual[0].std():.2f}")
    print(f"  Frame t+2: mean={predictions_actual[1].mean():.2f}, std={predictions_actual[1].std():.2f}")
    print(f"  Frame t+3: mean={predictions_actual[2].mean():.2f}, std={predictions_actual[2].std():.2f}")
    
    # Optional: Save predictions
    # np.save('predictions.npy', predictions_actual)
    # print("\nSaved predictions to predictions.npy")
