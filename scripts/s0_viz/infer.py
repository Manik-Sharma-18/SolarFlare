"""Load the trained S0 SimpleConvLSTM and predict a single 64x64 window."""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from models import build_forecaster


def load_s0_model(ckpt_path: Path, model_cfg: dict, t_out: int, device: torch.device):
    """Build the model from the YAML ``model_cfg`` and load checkpoint weights.

    The checkpoint stores only the training sub-config, so architecture comes
    from the YAML. Unwraps a torch.compile ``_orig_mod`` prefix if present.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_forecaster(model_cfg, t_out)
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_window(
    model,
    cube_norm: np.ndarray,
    t_start: int,
    y0: int,
    x0: int,
    t_in: int,
    t_out: int,
    win: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict ``t_out`` normalised frames for one crop.

    Returns ``(pred_norm, gt_norm)`` each ``(t_out, win, win)`` in
    normalised (z-score) space.
    """
    xin = cube_norm[t_start : t_start + t_in, y0 : y0 + win, x0 : x0 + win]
    x = torch.from_numpy(np.ascontiguousarray(xin)).float()
    x = x.view(1, 1, t_in, win, win).to(device)
    pred = model(x, teacher_forcing_ratio=0.0)  # (1, 1, t_out, win, win)
    if isinstance(pred, tuple):
        pred = pred[0]
    pred_norm = pred[0, 0].cpu().numpy()
    gt_norm = cube_norm[
        t_start + t_in : t_start + t_in + t_out, y0 : y0 + win, x0 : x0 + win
    ]
    return pred_norm, gt_norm
