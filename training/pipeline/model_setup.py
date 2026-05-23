"""Build and log the forecaster model from config.

``model.kind`` selects the architecture: ``solar_flux`` (default, the
6-layer attention-capable :class:`SolarFluxPredictor`) or
``simple_convlstm`` (minimal 2-layer :class:`SimpleConvLSTM` baseline).
"""
from typing import Any, Dict

import torch

from models import build_forecaster


def build_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Instantiate the model named by ``config['model']['kind']`` and log it."""
    model_cfg = config["model"]
    kind = model_cfg.get("kind", "solar_flux")

    model = build_forecaster(model_cfg, config["data"]["t_out"]).to(device)

    print(f"Model kind: {kind}")
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print(f"Input channels: {model_cfg['input_channels']}")
    print(f"Output channels: {model_cfg.get('output_channels', 1)}")
    if kind == "simple_convlstm":
        print(f"Hidden dim: {model_cfg.get('hidden_dim', 64)}  "
              f"layers: {model_cfg.get('num_layers', 2)}  "
              f"kernel: {model_cfg.get('kernel_size', 3)}")
    else:
        print(f"Channel progression: {model_cfg['channels']}")
        print(f"Gradient checkpointing: {model_cfg.get('use_checkpointing', False)}")
        print(f"Dropout rate: {model_cfg.get('dropout_rate', 0.0)}")

    _log_transfer_block(config)
    return _maybe_compile(model, config, device)


def _maybe_compile(model, config: Dict[str, Any], device: torch.device):
    """torch.compile on CUDA when ``model.compile`` truthy (default True on CUDA).

    MPS support is fragile (PROJECT.md out-of-scope note) so we skip there.
    Set ``model.compile: false`` in YAML to opt out on CUDA.
    """
    if device.type != "cuda":
        return model
    cfg = config.get("model", {})
    if cfg.get("compile", True) is False:
        return model
    mode = cfg.get("compile_mode", "default")
    print(f"torch.compile(mode={mode!r}) on CUDA model")
    return torch.compile(model, mode=mode)


def _log_transfer_block(config: Dict[str, Any]) -> None:
    transfer = config.get("transfer_learning") or {}
    if not transfer.get("pretrained_checkpoint"):
        return
    print(f"\nTransfer learning mode: {transfer.get('mode', 'finetune')}")
    print(f"  Pretrained checkpoint: {transfer['pretrained_checkpoint']}")
    print(f"  Freeze encoder: {transfer.get('freeze_encoder', False)}")
    unfreeze = transfer.get("unfreeze_after_epochs", 0)
    if unfreeze > 0:
        print(f"  Unfreeze after: {unfreeze} epochs")
