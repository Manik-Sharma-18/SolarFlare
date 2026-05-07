"""Diagnose val_loss=NaN: run val cube through full-rollout, log per-step ranges.

Sanity val cube (seed=0, val_frac=0.34, 4 cubes) → 1 random val. We try each cube
to find which path produces NaN.
"""
from __future__ import annotations

import sys
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


def run_one(path: str, t_in: int, t_out: int, device: torch.device) -> None:
    cube = open_cube(path)
    starts = list(iter_valid_starts(cube, t_in + t_out))
    if not starts:
        print(f"[skip] {path}: no valid window")
        return
    t0 = starts[0]
    wind, _ = read_window(cube, t0, t_in + t_out)
    mu, sigma = cube_norm_stats(cube)
    wind = (wind - mu) / sigma
    finite = torch.isfinite(torch.from_numpy(wind)).all().item()
    print(f"\n=== {cube.harp_id}: shape={cube.shape}, mu={mu:.2e} sigma={sigma:.2e} "
          f"max|z|={float(abs(wind).max()):.2f} finite={finite}")
    x = torch.from_numpy(wind).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).float().to(device)
    cfg = JEPAConfig(t_in=t_in, t_out=t_out, encoder_dim=192, encoder_layers=4,
                     encoder_heads=4, predictor_hidden=192, predictor_layers=3,
                     predictor_heads=4, dropout=0.0, drop_path=0.0)
    model = V5JEPAModel(cfg=cfg).to(device).eval()
    mode = "EVAL+NO_GRAD"
    print(f"[mode] {mode}")

    with torch.no_grad():
        x_adapt, token_pad_mask, _ = model.adapter(x)
        print(f"x_adapt: range=[{x_adapt.min():.2f}, {x_adapt.max():.2f}] "
              f"finite={torch.isfinite(x_adapt).all().item()}")

        z_t = model.target_encoder.encode_frames(x_adapt)
        print(f"z_target: shape={tuple(z_t.shape)} range=[{z_t.min():.2f}, {z_t.max():.2f}] "
              f"finite={torch.isfinite(z_t).all().item()}")

        z_c = model.encoder.encode_frames(x_adapt[:, :t_in])
        print(f"z_ctx: range=[{z_c.min():.2f}, {z_c.max():.2f}] "
              f"finite={torch.isfinite(z_c).all().item()}")

        b, _, hp, wp, d = z_t.shape
        tpf = hp * wp
        z_seq = z_c.reshape(b, t_in * tpf, d)

        for step in range(t_out):
            cur_t = t_in + step
            z_out = model.predictor(
                z_seq, t_total=cur_t, hp=hp, wp=wp, patch=cfg.patch_size,
                cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
                token_pad_mask=token_pad_mask,
            )
            ok = torch.isfinite(z_out).all().item()
            print(f"  step={step} cur_t={cur_t} z_out: range=[{z_out.min():.2f}, "
                  f"{z_out.max():.2f}] finite={ok}")
            if not ok:
                bad = (~torch.isfinite(z_out)).sum().item()
                print(f"    ! NaN/Inf count: {bad}/{z_out.numel()}")
                return
            z_last = z_out.reshape(b, cur_t, tpf, d)[:, -1:]
            z_seq = torch.cat([z_seq, z_last.reshape(b, tpf, d)], dim=1)


def main() -> int:
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device={dev}")
    cubes = ["data/harp_17.zarr", "data/harp_83.zarr", "data/harp_45.zarr", "data/harp_51.zarr"]
    for c in cubes:
        run_one(c, t_in=4, t_out=2, device=dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
