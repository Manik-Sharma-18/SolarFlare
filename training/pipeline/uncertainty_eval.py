"""MC-Dropout uncertainty pass over the first few test samples."""
from pathlib import Path
from typing import Any, Dict

import torch

from models.uncertainty import predict_with_uncertainty
from utils.visualization import visualize_with_uncertainty


def run_uncertainty(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: Path,
    output_channels: int,
) -> None:
    """Run MC-Dropout on up to 3 samples; save maps + print summary stats."""
    uq_cfg = config.get("uncertainty", {})
    if not uq_cfg.get("enabled", False):
        return

    print("\n" + "=" * 60)
    print("UNCERTAINTY QUANTIFICATION")
    print("=" * 60)

    if getattr(model, "dropout_rate", 0.0) == 0.0:
        print("Warning: Dropout rate is 0. Uncertainty estimation requires dropout_rate > 0.")
        print("Set model.dropout_rate in config and retrain for meaningful uncertainty.")
        return

    n_samples = uq_cfg.get("n_samples", 20)
    save_maps = uq_cfg.get("save_uncertainty_maps", True)
    print(f"Running MC Dropout with {n_samples} samples...")

    for i in range(min(3, len(test_dataset))):
        X_in, Y_out, _ = test_dataset[i]
        X_in = X_in.unsqueeze(0).to(device)
        Y_out = Y_out.unsqueeze(0).to(device)

        mean_pred, uncertainty = predict_with_uncertainty(model, X_in, n_samples=n_samples)

        if save_maps:
            visualize_with_uncertainty(
                mean_pred,
                uncertainty,
                Y_out[:, :output_channels],
                save_path=str(output_dir / f"uncertainty_sample_{i}.png"),
            )

        unc_mean = uncertainty.mean().item()
        unc_max = uncertainty.max().item()
        print(f"  Sample {i}: Mean uncertainty = {unc_mean:.6f}, Max = {unc_max:.6f}")
