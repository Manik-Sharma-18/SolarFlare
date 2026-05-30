"""Driver: S0 predictions + spatial-mean viz for every HARP cube.

Phase A: densify each cube, compute per-cube z-score stats + the full-cube
spatial-mean time series; accumulate train-cube stats into a global
fallback (the normalisation val/test cubes saw in training).
Phase B: per cube, normalise (train→own stats, val/test→global), pick 3
active 64x64 windows, predict, and render one combined figure.

Run: python3 -m scripts.s0_viz.run
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solarflare_data.loader import assign_files_to_splits
from scripts.s0_viz.cube_io import load_cube, cube_stats, asinh_scale, normalize, denormalize
from scripts.s0_viz.windows import find_active_windows
from scripts.s0_viz.infer import load_s0_model, predict_window
from scripts.s0_viz.fullframe import predict_full_frame
from scripts.s0_viz.figure import render_cube_figure

CKPT = Path("outputs/ablations/S0_simple_convlstm/checkpoints/best_model.pt")
CONFIG = Path("configs/ablations/S0_simple_convlstm.yaml")
OUT = Path("outputs/s0_viz")
WIN = 64


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _args():
    import argparse
    p = argparse.ArgumentParser(description="S0-style prediction + spatial-mean viz")
    p.add_argument("--ckpt", default=str(CKPT), help="checkpoint .pt (best_model.pt, mid-train OK)")
    p.add_argument("--config", default=str(CONFIG), help="YAML giving model/data arch")
    p.add_argument("--out", default=str(OUT), help="output dir for figures + spatial_mean")
    p.add_argument("--full-frame", action="store_true",
                   help="predict the whole frame (tiled) instead of one 64x64 crop")
    p.add_argument("--stride", type=int, default=32, help="tile stride for --full-frame")
    return p.parse_args()


def main() -> None:
    args = _args()
    out_dir = Path(args.out)
    device = _device()
    cfg = yaml.safe_load(Path(args.config).read_text())
    norm_method = cfg.get("normalization", {}).get("method", "zscore_per_cube")
    t_in, t_out = cfg["data"]["t_in"], cfg["data"]["t_out"]
    ratios = cfg["data"].get("split_ratios", [0.7, 0.2, 0.1])
    seed = cfg.get("seed", 42)
    data_dir = Path(cfg["data"]["data_dir"])
    model = load_s0_model(Path(args.ckpt), cfg["model"], t_out, device)

    # Title label derived from config + checkpoint (not hardcoded).
    arm = Path(cfg.get("output", {}).get("save_dir", args.config)).name
    mc = cfg["model"]
    flags = [f"hidden{mc.get('hidden_dim', 64)}", f"L{mc.get('num_layers', 2)}"]
    if mc.get("residual"): flags.append("residual")
    if cfg.get("training", {}).get("tf_start", 0): flags.append("TF")
    flags.append(f"norm={mc.get('norm_type', 'batch')}")
    ck_ep = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("epoch", "?")
    model_label = f"{arm}  [{mc.get('kind', 'solar_flux')}: {', '.join(flags)}; best ep{ck_ep}]"

    zarr_dirs = sorted(p for p in data_dir.glob("*.zarr") if p.is_dir())
    assign = assign_files_to_splits(zarr_dirs, ratios, seed)
    split_of = {}
    for name, idxs in assign.items():
        for i in idxs:
            split_of[i] = name
    print(f"{len(zarr_dirs)} cubes | device={device} | t_in={t_in} t_out={t_out}")

    # Phase A — stats + global fallback + cached densified cubes.
    cache = Path(out_dir, "_cache"); cache.mkdir(parents=True, exist_ok=True)
    per_cube = {}              # idx -> (mu, sigma)
    ts_means = {}              # idx -> full-cube spatial-mean time series
    g_sum = g_sq = g_cnt = 0.0
    for i, zp in enumerate(zarr_dirs):
        cube, _ = load_cube(zp)
        np.save(cache / f"{zp.stem}.npy", cube)
        # signed_asinh: per-cube (scale, softening); else zscore (mu, sigma).
        # asinh viz uses per-cube scale for all cubes (incl test) — minor vs the
        # training global fallback, acceptable for qualitative viz.
        mu, sigma = (asinh_scale(cube) if norm_method == "signed_asinh"
                     else cube_stats(cube))
        per_cube[i] = (mu, sigma)
        ts_means[i] = cube.mean(axis=(1, 2)).astype(np.float32)
        if split_of[i] == "train":
            nz = cube[(cube != 0) & np.isfinite(cube)]
            g_sum += float(nz.sum()); g_sq += float((nz.astype(np.float64) ** 2).sum())
            g_cnt += nz.size
        print(f"  [A] {zp.stem:14s} {split_of[i]:5s} T={cube.shape[0]} "
              f"HxW={cube.shape[1]}x{cube.shape[2]} mu={mu:+.2e} sig={sigma:.2e}")
    g_mu = g_sum / g_cnt if g_cnt else 0.0
    g_sigma = float(np.sqrt(max(g_sq / g_cnt - g_mu ** 2, 1e-18))) if g_cnt else 1.0
    print(f"  global fallback mu={g_mu:+.3e} sigma={g_sigma:.3e}")

    # Phase B — normalise, predict, render.
    sm_dir = Path(out_dir, "spatial_mean"); sm_dir.mkdir(parents=True, exist_ok=True)
    for i, zp in enumerate(zarr_dirs):
        name, split = zp.stem, split_of[i]
        cube = np.load(cache / f"{name}.npy")
        np.savez(sm_dir / f"{name}.npz", frame=np.arange(ts_means[i].size),
                 spatial_mean=ts_means[i], split=split)
        if norm_method == "signed_asinh":
            mu, sigma = per_cube[i]          # per-cube scale for all cubes
        else:
            mu, sigma = per_cube[i] if split == "train" else (g_mu, g_sigma)
        cube_norm = normalize(cube, mu, sigma, norm_method)
        wins = find_active_windows(cube, t_in, t_out, WIN, n_sets=3)
        if not wins:
            print(f"  [B] {name:14s} SKIP (T={cube.shape[0]} < {t_in+t_out})")
            continue
        sets = []
        for (ts, y0, x0) in wins:
            if args.full_frame:
                pn = predict_full_frame(model, cube_norm, ts, t_in, t_out,
                                        WIN, args.stride, device)
                gn = cube_norm[ts + t_in : ts + t_in + t_out]
            else:
                pn, gn = predict_window(model, cube_norm, ts, y0, x0,
                                        t_in, t_out, WIN, device)
            sets.append({"pred": denormalize(pn, mu, sigma, norm_method),
                         "gt": denormalize(gn, mu, sigma, norm_method), "t_start": ts})
        out_png = out_dir / f"{name}.png"
        note = (f"full frame, tiled {WIN}x{WIN} stride {args.stride} + averaged"
                if args.full_frame else "crops are 64x64 around peak-|flux|")
        render_cube_figure(name, split, t_out, sets, out_png,
                            region_note=note, model_label=model_label)
        print(f"  [B] {name:14s} {split:5s} {len(sets)} sets -> {out_png}")

    print(f"\nDone. Figures + spatial_mean/*.npz under {out_dir}/")


if __name__ == "__main__":
    main()
