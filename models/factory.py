"""Forecaster model factory — ``kind``-aware construction.

Single source of truth for turning a ``config['model']`` block into a
model instance, shared by training (``training/pipeline/model_setup.py``)
and inference (``inference.py``, ``training/pipeline/infer.py``) so the
``model.kind`` selector can't drift between code paths.

Does NOT move to device, compile, or log — callers own that.
"""
from typing import Any, Dict

import torch.nn as nn

from .predictor import SolarFluxPredictor
from .simple_convlstm import SimpleConvLSTM


def build_forecaster(model_cfg: Dict[str, Any], t_out: int) -> nn.Module:
    """Construct the forecaster named by ``model_cfg['kind']``.

    Args:
        model_cfg: The ``config['model']`` mapping.
        t_out: Output horizon (from ``config['data']['t_out']``).

    Returns:
        An un-moved, un-compiled ``nn.Module``.

    Raises:
        ValueError: on an unrecognised ``kind``.
    """
    kind = model_cfg.get("kind", "solar_flux")

    if kind == "simple_convlstm":
        return SimpleConvLSTM(
            input_channels=model_cfg["input_channels"],
            output_channels=model_cfg.get("output_channels", 1),
            t_out=t_out,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            kernel_size=model_cfg.get("kernel_size", 3),
            num_layers=model_cfg.get("num_layers", 2),
        )

    if kind == "solar_flux":
        return SolarFluxPredictor(
            input_channels=model_cfg["input_channels"],
            output_channels=model_cfg.get("output_channels", 1),
            t_out=t_out,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            use_checkpointing=model_cfg.get("use_checkpointing", False),
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            use_sa_convlstm=model_cfg.get("use_sa_convlstm", False),
            temporal_attention=model_cfg.get("temporal_attention", False),
            attention_gate=model_cfg.get("attention_gate", False),
            delta_scale_init=model_cfg.get("delta_scale_init", 0.0),
        )

    raise ValueError(
        f"Unknown model.kind: {kind!r} (expected 'solar_flux' or 'simple_convlstm')"
    )
