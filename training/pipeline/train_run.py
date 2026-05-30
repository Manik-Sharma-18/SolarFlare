"""Top-level ``run_training`` orchestration: data → model → train → test."""
import json
from pathlib import Path
from typing import Any, Dict

import torch

from training import train_model
from utils import resolve_device, validate_config, visualize_predictions
from utils.checkpoint import load_checkpoint
from utils.mps_ops import is_mps, _log_mps_once
from utils.visualization import plot_training_history

from .data_setup import build_loaders
from .model_setup import build_model
from .seeding import seed_everything
from .test_eval import evaluate_on_test
from .uncertainty_eval import run_uncertainty


def run_training(config: Dict[str, Any]) -> None:
    """End-to-end training: validate config → build → train → test → uncertainty."""
    validate_config(config)
    seed_everything(config.get("seed", 42))

    device = resolve_device(config["device"])
    if is_mps(device):
        _log_mps_once()

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_loader, val_loader, test_loader, test_dataset, metadata = build_loaders(config, device)

    output_dir = Path(config["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    model = build_model(config, device)

    train_cfg = {
        **config["training"],
        "save_dir": config["output"]["save_dir"],
        "checkpoint_name": config["output"]["checkpoint_name"],
        "save_history": config["output"]["save_history"],
        "show_progress": config["logging"]["progress_bar"],
        "loss": config.get("loss", {"type": "l1"}),
        "output_channels": config["model"].get("output_channels", 1),
        "error_handling": config.get("error_handling", {}),
        "resume_from": config.get("resume_from"),
        "evaluation": config.get("evaluation", {}),
        "transfer_learning": config.get("transfer_learning"),
    }

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = train_model(
        model, train_loader, val_loader, train_cfg, device,
        normalization_params=metadata.get("normalization", {}),
    )

    if config["output"]["save_visualizations"]:
        plot_training_history(history, str(output_dir / "training_history.png"))

    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    best_ckpt = output_dir / "checkpoints" / "best_model.pt"
    ckpt = load_checkpoint(best_ckpt)
    # Checkpoints store bare keys; torch.compile wraps the model as
    # OptimizedModule (keys gain a "_orig_mod." prefix). Load into the
    # underlying module so reload works whether or not the model is compiled.
    load_target = getattr(model, "_orig_mod", model)
    load_target.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    output_channels = config["model"].get("output_channels", 1)
    evaluate_on_test(model, test_loader, config, device, output_dir, output_channels)

    if config["output"]["save_visualizations"]:
        print("\n" + "=" * 60)
        print("VISUALIZING")
        print("=" * 60)
        visualize_predictions(
            model, test_dataset, device,
            n_samples=3,
            save_path=str(output_dir / "predictions.png"),
            use_amp=config["training"]["use_amp"],
        )

    run_uncertainty(model, test_dataset, device, config, output_dir, output_channels)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Outputs saved to: {output_dir}")
