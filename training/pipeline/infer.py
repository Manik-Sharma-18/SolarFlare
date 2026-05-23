"""Inference-only helper: load checkpoint into a fresh model."""
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models import build_forecaster
from utils import resolve_device
from utils.checkpoint import load_checkpoint


def run_inference(
    config: Dict[str, Any],
    checkpoint_path: str,
    data_path: Optional[str] = None,
) -> nn.Module:
    """Build the model, load weights, return it ready for ``.eval()`` forward.

    Honors ``config['model']['kind']`` (``solar_flux`` or ``simple_convlstm``).
    ``data_path`` is accepted for API parity with the prior ``main.py`` shim;
    actual single-sequence inference is handled by ``inference.py`` at repo
    root.
    """
    device = resolve_device(config["device"])

    model = build_forecaster(config["model"], config["data"]["t_out"]).to(device)

    ckpt = load_checkpoint(Path(checkpoint_path))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Model loaded and ready for inference")
    return model
