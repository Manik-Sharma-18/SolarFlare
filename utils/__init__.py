"""Utility modules for Solar Flare Prediction."""
from .device import resolve_device, get_amp_context, get_grad_scaler, clear_device_cache
from .config_validator import validate_config, ConfigValidationError
from .checkpoint import (
    save_checkpoint, load_checkpoint, load_checkpoint_for_resume,
    load_checkpoint_for_inference, CHECKPOINT_VERSION,
)
from .metrics import compute_metrics
from .visualization import (
    visualize_predictions,
    visualize_with_uncertainty,
    visualize_uncertainty_statistics
)
from .animation import (
    load_flux_data,
    animate_flare_sequence,
    interactive_flare_viewer,
    animate_prediction_vs_truth,
    animate_with_uncertainty,
    create_difference_animation
)
from .mps_ops import safe_outer, safe_quantile, is_mps
