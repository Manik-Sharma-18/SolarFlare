"""
Inference Script - Load trained model and make predictions

Usage:
    python inference.py
    python inference.py --checkpoint ./outputs/checkpoints/best_model.pt
    python inference.py --config config.yaml --checkpoint best_model.pt --data ./data/example.npy

This script demonstrates how to:
1. Load a trained model from checkpoint + config
2. Load and normalize new data
3. Make predictions
4. Unnormalize predictions to get actual flux values
"""
import argparse
import torch
import numpy as np
import json
from pathlib import Path
import sys

import yaml

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SolarFluxPredictor
from utils.device import resolve_device
from utils.checkpoint import load_checkpoint_for_inference


def load_model(checkpoint_path: str, config: dict, device: torch.device = None):
    """
    Load trained model from checkpoint using architecture params from config.

    The checkpoint stores training state and normalization params.
    The config (config.yaml) provides model architecture and data params.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        config: Full config dict (from config.yaml).
        device: torch.device to load onto (auto-detected if None).

    Returns:
        Tuple of (model, normalization_params, device).
    """
    checkpoint, device = load_checkpoint_for_inference(Path(checkpoint_path), device)

    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    model = SolarFluxPredictor(
        input_channels=model_cfg.get('input_channels', 2),
        output_channels=model_cfg.get('output_channels', 1),
        t_out=data_cfg.get('t_out', 4),
        channels=model_cfg.get('channels', [32, 64, 128]),
        kernel_size=model_cfg.get('kernel_size', 5),
        downsample_input=model_cfg.get('downsample_input', True),
        use_checkpointing=False,  # Not needed for inference
        dropout_rate=model_cfg.get('dropout_rate', 0.0),
        use_sa_convlstm=model_cfg.get('use_sa_convlstm', False),
        temporal_attention=model_cfg.get('temporal_attention', False),
        attention_gate=model_cfg.get('attention_gate', False),
        delta_scale_init=model_cfg.get('delta_scale_init', 0.0),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    normalization_params = checkpoint.get('normalization_params', {})

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if checkpoint.get('best_val_loss') is not None:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    print(f"  Architecture: SA-ConvLSTM={model_cfg.get('use_sa_convlstm', False)}, "
          f"channels={model_cfg.get('channels')}, "
          f"input_channels={model_cfg.get('input_channels', 2)}")

    return model, normalization_params, device


def normalize_asinh(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """Normalize raw flux data using asinh transformation."""
    softening = norm_params['asinh_softening']
    scale = norm_params['scale']
    return np.arcsinh(data / softening) / scale


def unnormalize_asinh(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """Convert asinh-normalized values back to original flux scale."""
    softening = norm_params['asinh_softening']
    scale = norm_params['scale']
    return np.sinh(data * scale) * softening


def compute_extreme_channel(flux: np.ndarray, threshold: float) -> np.ndarray:
    """Compute soft extreme event indicator channel.

    Args:
        flux: Normalized flux values (T, H, W).
        threshold: Extreme threshold in normalized space.

    Returns:
        Indicator array (T, H, W) in [0, 1].
    """
    abs_flux = np.abs(flux)
    scaled = (abs_flux - threshold) / (threshold * 0.5 + 1e-6)
    return (1.0 / (1.0 + np.exp(-scaled * 2))).astype(np.float32)


def center_crop(data: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Center-crop spatial dimensions (..., H, W)."""
    h, w = data.shape[-2], data.shape[-1]
    if h == crop_h and w == crop_w:
        return data
    if h < crop_h or w < crop_w:
        raise ValueError(
            f"Data spatial dims ({h}, {w}) smaller than crop size ({crop_h}, {crop_w})"
        )
    sh = (h - crop_h) // 2
    sw = (w - crop_w) // 2
    return data[..., sh:sh + crop_h, sw:sw + crop_w]


def prepare_input(
    flux_frames: np.ndarray,
    norm_params: dict,
    config: dict,
    already_normalized: bool = False,
) -> torch.Tensor:
    """Prepare flux frames for model input.

    Handles normalization, center-cropping, and dual-channel construction.

    Args:
        flux_frames: Raw or normalized flux array (T_in, H, W).
        norm_params: Normalization parameters from checkpoint.
        config: Full config dict.
        already_normalized: If True, skip normalization step.

    Returns:
        Input tensor (1, C, T_in, H, W) ready for the model.
    """
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})

    # Normalize
    if not already_normalized:
        method = config.get('normalization', {}).get('method', 'asinh')
        if method == 'asinh':
            flux_frames = normalize_asinh(flux_frames, norm_params)
        else:
            center = norm_params.get('center', 0.0)
            scale = norm_params['scale']
            flux_frames = (flux_frames - center) / scale

    flux_frames = flux_frames.astype(np.float32)

    # Center-crop to training dimensions
    crop_size = data_cfg.get('crop_size')
    if crop_size:
        flux_frames = center_crop(flux_frames, crop_size[0], crop_size[1])

    # Build channels
    dual_channel = data_cfg.get('dual_channel', False)
    input_channels = model_cfg.get('input_channels', 1)

    if dual_channel and input_channels == 2:
        extreme_thresh = norm_params.get(
            'extreme_threshold_normalized',
            config.get('evaluation', {}).get('extreme_threshold', 0.528),
        )
        extreme_ch = compute_extreme_channel(flux_frames, extreme_thresh)
        x = np.stack([flux_frames, extreme_ch], axis=0)  # (2, T, H, W)
    else:
        x = flux_frames[np.newaxis, ...]  # (1, T, H, W)

    return torch.from_numpy(x.copy()).float().unsqueeze(0)  # (1, C, T, H, W)


def predict(
    model: SolarFluxPredictor,
    input_tensor: torch.Tensor,
    device: torch.device = None,
) -> np.ndarray:
    """
    Run model inference on a prepared input tensor.

    Args:
        model: Loaded model in eval mode.
        input_tensor: Tensor of shape (1, C, T_in, H, W).
        device: Device (inferred from model if None).

    Returns:
        Predicted flux frames (T_out, H, W) in normalized space.
    """
    if device is None:
        device = next(model.parameters()).device

    x = input_tensor.to(device)

    with torch.no_grad():
        predictions = model(x, teacher_forcing_ratio=0.0)

    # Return flux channel only: (T_out, H, W)
    return predictions[0, 0].cpu().numpy()


def load_raw_data(npy_path: str, t_in: int = 10, start_idx: int = 0) -> np.ndarray:
    """
    Load raw structured .npy file and extract a sequence.

    Args:
        npy_path: Path to windTotal*.npy structured array file.
        t_in: Number of input frames.
        start_idx: Starting time index.

    Returns:
        flux_cube: (T_in, H, W) array of raw flux values.
    """
    data = np.load(npy_path)

    x_coords = np.unique(data['X'])
    y_coords = np.unique(data['Y'])
    times = np.unique(data['time'])

    H, W = len(y_coords), len(x_coords)
    T = len(times)

    print(f"Loaded data: T={T}, H={H}, W={W}")

    if start_idx + t_in > T:
        raise ValueError(
            f"start_idx={start_idx} + t_in={t_in} = {start_idx + t_in} exceeds T={T}"
        )

    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    time_to_idx = {t: i for i, t in enumerate(times)}

    flux_cube = np.zeros((T, H, W), dtype=np.float32)
    for i in range(len(data)):
        t_idx = time_to_idx[data['time'][i]]
        h_idx = y_to_idx[data['Y'][i]]
        w_idx = x_to_idx[data['X'][i]]
        flux_cube[t_idx, h_idx, w_idx] = data['windTotal'][i]

    sequence = flux_cube[start_idx:start_idx + t_in]
    print(f"Extracted sequence from t={start_idx} to t={start_idx + t_in - 1}")

    return sequence


def load_preprocessed_data(
    npz_path: str, t_in: int = 10, start_idx: int = 0
) -> np.ndarray:
    """
    Load preprocessed .npz cube and extract a sequence.

    Preprocessed cubes store raw (unnormalized) flux in the 'data' key.

    Args:
        npz_path: Path to cube_XXX.npz file.
        t_in: Number of input frames.
        start_idx: Starting time index.

    Returns:
        flux_frames: (T_in, H, W) array of flux values.
    """
    cube = np.load(npz_path, allow_pickle=True)['data']
    T = cube.shape[0]
    print(f"Loaded preprocessed cube: shape={cube.shape}")

    if start_idx + t_in > T:
        raise ValueError(
            f"start_idx={start_idx} + t_in={t_in} = {start_idx + t_in} exceeds T={T}"
        )

    return cube[start_idx:start_idx + t_in].astype(np.float32)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solar Flare Prediction Inference')
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to config.yaml (for model architecture params)',
    )
    parser.add_argument(
        '--checkpoint', default='./outputs/checkpoints/best_model.pt',
        help='Path to checkpoint file',
    )
    parser.add_argument(
        '--data', default=None,
        help='Path to input data (.npy raw or .npz preprocessed)',
    )
    parser.add_argument('--start', type=int, default=50, help='Starting time index')
    parser.add_argument('--device', default='auto', help='Device: auto, cuda, mps, cpu')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = resolve_device(args.device)
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    model, norm_params, device = load_model(args.checkpoint, config, device)

    t_in = config.get('data', {}).get('t_in', 10)
    t_out = config.get('data', {}).get('t_out', 4)
    print(f"  t_in={t_in}, t_out={t_out}")
    print(f"  Normalization: {norm_params.get('method', 'asinh')}, "
          f"scale={norm_params.get('scale', '?')}")

    # 2. Load data
    data_path = args.data
    if data_path is None:
        data_path = './data_processed/cube_000.npz'

    print(f"\n2. Loading data from {data_path}...")

    if data_path.endswith('.npz'):
        raw_input = load_preprocessed_data(data_path, t_in, args.start)
        # Check if data looks already normalized (range ~[-1, 1])
        already_normalized = abs(raw_input.max()) <= 1.5 and abs(raw_input.min()) <= 1.5
        if already_normalized:
            print("  Data appears pre-normalized (range within [-1.5, 1.5])")
    else:
        raw_input = load_raw_data(data_path, t_in, args.start)
        already_normalized = False

    print(f"  Input range: [{raw_input.min():.4f}, {raw_input.max():.4f}]")

    # 3. Prepare input (normalize, crop, build dual channel)
    print("\n3. Preparing input...")
    input_tensor = prepare_input(raw_input, norm_params, config, already_normalized)
    print(f"  Input tensor shape: {input_tensor.shape}")
    print(f"  Flux channel range: [{input_tensor[0, 0].min():.4f}, {input_tensor[0, 0].max():.4f}]")

    # 4. Predict
    print("\n4. Making predictions...")
    predictions_normalized = predict(model, input_tensor, device)
    print(f"  Output shape: {predictions_normalized.shape}")

    # 5. Unnormalize
    method = config.get('normalization', {}).get('method', 'asinh')
    if method == 'asinh' and not already_normalized:
        predictions_raw = unnormalize_asinh(predictions_normalized, norm_params)
    else:
        predictions_raw = predictions_normalized

    # 6. Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Input: {t_in} frames starting at index {args.start}")
    print(f"Output: {t_out} predicted frames")
    extreme_thresh = norm_params.get('extreme_threshold_normalized', 0.528)
    for t in range(t_out):
        frame = predictions_normalized[t]
        extreme_frac = (np.abs(frame) > extreme_thresh).mean() * 100
        print(f"  t+{t+1}: range=[{frame.min():.4f}, {frame.max():.4f}] "
              f"std={frame.std():.4f} extreme={extreme_frac:.3f}%")
