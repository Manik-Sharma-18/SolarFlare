"""Post-training test-set evaluation + ``test_results.json`` emission."""
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from training import validate
from training.losses import get_loss_function


def evaluate_on_test(
    model: torch.nn.Module,
    test_loader,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    output_channels: int,
) -> Dict[str, Any]:
    """Run validation on the test loader and write ``test_results.json``."""
    loss_fn = get_loss_function(config.get("loss", {"type": "l1"})).to(device)
    eval_cfg = config.get("evaluation", {})

    metrics = validate(
        model, test_loader, device,
        loss_fn=loss_fn,
        use_amp=config["training"]["use_amp"],
        show_progress=config["logging"]["progress_bar"],
        output_channels=output_channels,
        extreme_threshold=eval_cfg.get("extreme_threshold", 0.277),
        ssim_data_range=config.get("loss", {}).get("ssim_data_range", 2.0),
    )
    _print_summary(metrics)
    _write_results_json(metrics, output_dir)
    return metrics


def _print_summary(m: Dict[str, Any]) -> None:
    print(f"Test Loss: {m['val_loss']:.6f}")
    print(f"Test MAE per timestep: {m['val_mae_per_timestep']}")
    print(f"Test CSI: {m['val_csi']:.4f} | HSS: {m['val_hss']:.4f}")
    print(f"Test SSIM: {m['val_ssim']:.4f}")
    persist = m["persistence_skill_per_timestep"]
    avg_persist = float(np.mean(persist)) if persist else 0.0
    print(f"Test Persistence Skill: {avg_persist:.1f}%")
    print(f"Test Temporal Var Ratio: {m['temporal_variation_ratio']:.3f}")


def _write_results_json(m: Dict[str, Any], output_dir: Path) -> None:
    mae_per_t = m["val_mae_per_timestep"]
    if isinstance(mae_per_t, list):
        mae_list = mae_per_t
    elif hasattr(mae_per_t, "tolist"):
        mae_list = mae_per_t.tolist()
    else:
        mae_list = []
    results = {
        "test_loss": float(m["val_loss"]),
        "test_mae_per_timestep": mae_list,
        "test_rmse_per_timestep": m["val_rmse_per_timestep"],
        "test_correlation_per_timestep": m["val_correlation_per_timestep"],
        "test_csi": m["val_csi"],
        "test_csi_per_timestep": m["val_csi_per_timestep"],
        "test_hss": m["val_hss"],
        "test_hss_per_timestep": m["val_hss_per_timestep"],
        "test_ssim": m["val_ssim"],
        "test_ssim_per_timestep": m["val_ssim_per_timestep"],
        "persistence_mae_per_timestep": m["persistence_mae_per_timestep"],
        "persistence_skill_per_timestep": m["persistence_skill_per_timestep"],
        "persistence_csi": m["persistence_csi"],
        "persistence_hss": m["persistence_hss"],
        "peak_flux_error_per_timestep": m["peak_flux_error_per_timestep"],
        "temporal_variation_ratio": m["temporal_variation_ratio"],
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
