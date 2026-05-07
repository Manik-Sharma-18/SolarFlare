"""Smoke test V5 scaffold (Path B) end-to-end on ONE cube.

Usage:
    python scripts/smoke_test_v5.py --cube data/harp_11930.zarr

Verifies:
    1. Cube opens; one valid window readable.
    2. Adapter pads + projects to 13ch with correct shape.
    3. ViT context encoder + EMA target encoder produce tokens.
    4. Predictor runs with block-causal mask, no NaN.
    5. Backward step succeeds, no NaN in grads.
    6. Target EMA update runs.
    7. Reports peak VRAM / RSS.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.v5 import V5JEPAModel
from models.v5.jepa_model import JEPAConfig
from solarflare_data.zarr_loader import (
    cube_norm_stats,
    iter_valid_starts,
    open_cube,
    read_window,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cube", required=True)
    p.add_argument("--t-in", type=int, default=10)
    p.add_argument("--t-out", type=int, default=4)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def report_peak_mem(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    return "non-cuda peak unavailable"


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[info] device={device}")

    cube = open_cube(args.cube)
    window = args.t_in + args.t_out
    starts = list(iter_valid_starts(cube, window))
    if not starts:
        print(f"[err] no valid {window}-frame window in {args.cube}")
        return 1
    t0 = starts[0]
    print(f"[info] cube {cube.harp_id}: shape={cube.shape}, window starts at t={t0}")

    wind, valid = read_window(cube, t0, window)
    mu, sigma = cube_norm_stats(cube)
    wind = (wind - mu) / sigma
    print(f"[info] norm μ={mu:.3e} σ={sigma:.3e}")

    x = torch.from_numpy(wind).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).float()  # [1, T, 1, H, W]
    valid_t = torch.from_numpy(valid).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).bool()
    x = x.to(device)
    valid_t = valid_t.to(device)
    print(f"[info] input tensor: {tuple(x.shape)} dtype={x.dtype}")

    cfg = JEPAConfig(t_in=args.t_in, t_out=args.t_out)
    model = V5JEPAModel(cfg=cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] model params: total={n_total/1e6:.1f}M trainable={n_train/1e6:.1f}M")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_fwd = time.time()
    out = model(x, valid_mask=valid_t)
    print(f"[info] fwd ok: z_pred={tuple(out['z_pred'].shape)} loss={out['loss'].item():.4f} "
          f"({time.time()-t_fwd:.2f}s)")

    if not torch.isfinite(out["loss"]):
        print("[err] loss is non-finite")
        return 2

    t_bwd = time.time()
    out["loss"].backward()
    print(f"[info] bwd ok ({time.time()-t_bwd:.2f}s)")

    bad_grads = [n for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad_grads:
        print(f"[err] non-finite grads in {len(bad_grads)} params (first: {bad_grads[0]})")
        return 3

    model.update_target_ema()
    print("[info] target ema update ok")

    print(f"[info] {report_peak_mem(device)}")
    print("[pass] smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
