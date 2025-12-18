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


def load_model(checkpoint_path: str, device: str = 'cuda') -> SolarFluxPredictor:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pt
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model ready for inference
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint (or use defaults)
    config = checkpoint.get('config', {})
    
    # Create model with same architecture
    model = SolarFluxPredictor(
        input_channels=config.get('input_channels', 1),
        t_out=config.get('t_out', 3),
        channels=config.get('channels', [16, 32, 64]),
        kernel_size=config.get('kernel_size', 3),
        downsample_input=config.get('downsample_input', True)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model


def load_normalization(metadata_path: str) -> dict:
    """Load normalization parameters from metadata."""
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
    device: str = 'cuda'
) -> np.ndarray:
    """
    Make predictions on input data.
    
    Args:
        model: Loaded model
        input_data: numpy array of shape (T_in, H, W) - already normalized
        device: 'cuda' or 'cpu'
    
    Returns:
        predictions: numpy array of shape (T_out, H, W)
    """
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model = load_model(CHECKPOINT_PATH, device)
    
    # Load normalization parameters
    print("\n2. Loading normalization parameters...")
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
