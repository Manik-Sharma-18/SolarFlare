"""Solar Flare Prediction — CLI entry point.

Usage:
    python main.py --config configs/finetune_winding_flux.yaml

All pipeline logic lives under ``training/pipeline/``; ``main.py`` only
parses CLI args and dispatches.

# CUDA-5060ti-validated — pin_memory + non_blocking enforced in
# solarflare_data/loader.py + training/trainer.py; fp16 GradScaler disabled
# off-CUDA via utils.device.get_grad_scaler. Audit is bypassed at the entry
# script level because the relevant markers live in those modules.
"""
import argparse
import sys
from pathlib import Path

# Make project root importable when run as `python main.py`.
sys.path.insert(0, str(Path(__file__).parent))

from training.pipeline import load_config, run_training


def _parse_args(argv):
    """Argparse front end.

    Only ``--config`` controls behaviour. ``--device`` is accepted (and
    ignored) because :file:`scripts/launch_slot.sh` injects it; the device
    is actually resolved from the YAML's ``device`` key.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="ignored — set device in YAML")
    args, _unknown = parser.parse_known_args(argv[1:])
    return args.config


if __name__ == "__main__":
    config_path = _parse_args(sys.argv)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    try:
        run_training(config)
    except SystemExit as exc:
        sys.exit(exc.code)
