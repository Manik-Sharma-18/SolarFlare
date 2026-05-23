"""Seeding + config loading helpers used by the pipeline entry points."""
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed torch (CPU + CUDA), numpy, and Python ``random``."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
