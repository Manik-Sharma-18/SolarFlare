"""Staircase autoregressive prediction, full-frame tiled at the training window.

Per step: predict t_out frames (tiled), commit first `stride` to the output
series, slide the input buffer by `stride`. Pure autoregressive — model never
sees real frames past t0+T_in. Reports per-frame integrated winding flux MAE +
variance ratio + drift correlation; plots integrated-flux series + grid.
Cube (--cube), tile (config window_size) and softening come from the arm config
so pooled-domain / signed_asinh arms work unchanged.

Usage:
  python3 -u -m scripts.s0_viz._staircase_harp11930 --arm S47 --cube harp_245
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.s0_viz.cube_io import load_cube, cube_stats, asinh_scale, normalize, denormalize
from scripts.s0_viz.windows import find_active_windows
from scripts.s0_viz.infer import load_s0_model

ARMS = {
    "S3":  "S3_simple_convlstm_composite",
    "S11": "S11_simple_convlstm_fasttf_extreme",
    "S13": "S13_simple_convlstm_dual_posweight100",
    "S16": "S16_simple_convlstm_dual_extreme_weighted",
    "S18": "S18_simple_convlstm_dual_extreme_pw100",
    "S20p5": "S20p5_dilated_gates",
    "S30": "S30_stratified_sampler",
    "S30b": "S30b_sampler_plus_fixes",
    "S31": "S31_masked_loss",
    "S32": "S32_histogram_match",
    "S19": "S19_quantile_head_tau099",
    "S22": "S22_orthogonal_init",
    "S20": "S20_depthwise_widen192",
    "S33": "S33_S16_val_fix",
    "S34": "S34_S33_tin20",
    "S36": "S36_physics_smoothness",
    "S37": "S37_event_detection",
    "S42b": "S42b_S16_new_data_xval",
    "S43": "S43_S20_arch_new_data",
    "S44": "S44_S43_k5_hidden256",
    "S46": "S46_lowpass_asinh1e3",
    "S47": "S47_pool4_tin25",
    "S48": "S48_pool4_tin25_tfdecay12",
    "S49": "S49_pool4_ss_perstep",
    "S50": "S50_pool4_s16arch",
    "S51": "S51_s50_trajloss",
    "S52": "S52_longroll_tout16",
}
# Defaults; --cube overrides, TILE/softening come from the arm config.
# harp_11930: clean high-ceiling cube. harp_245/274/49 have LIMB EFFECT
# (foreshortening + edge data loss) → unreliable curve metrics; focus on 11930.
DEFAULT_CUBE = "harp_11930"


def device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict_one_step(model, state, t_out, dev, max_batch=8):
    """state: (T_in, H, W) normalized. Returns (t_out, H, W) normalized prediction.

    Tiles full frame at TILE/TILE_STRIDE; runs model on each tile; averages overlaps."""
    T_in, H, W = state.shape
    ys = list(range(0, H - TILE + 1, TILE_STRIDE))
    if ys[-1] != H - TILE: ys.append(H - TILE)
    xs = list(range(0, W - TILE + 1, TILE_STRIDE))
    if xs[-1] != W - TILE: xs.append(W - TILE)
    tiles, coords = [], []
    for y in ys:
        for x in xs:
            tiles.append(state[:, y : y + TILE, x : x + TILE])
            coords.append((y, x))
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(tiles))).unsqueeze(1).float()
    accum = np.zeros((t_out, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for i in range(0, batch.shape[0], max_batch):
        chunk = batch[i : i + max_batch].to(dev)
        out = model(chunk, teacher_forcing_ratio=0.0)
        if isinstance(out, tuple): out = out[0]
        out = out[:, 0].cpu().numpy()
        for j in range(out.shape[0]):
            y, x = coords[i + j]
            accum[:, y : y + TILE, x : x + TILE] += out[j]
            count[y : y + TILE, x : x + TILE] += 1.0
    count = np.maximum(count, 1e-9)
    return (accum / count[None]).astype(np.float32)


def staircase(model, cube_norm, t0, t_in, t_out, n_steps, stride, dev):
    """Returns (committed_concat (n_steps*stride,H,W), full_preds [list of (t_out,H,W)])."""
    state = cube_norm[t0 : t0 + t_in].astype(np.float32).copy()
    committed_list, full_list = [], []
    for k in range(n_steps):
        out = predict_one_step(model, state, t_out, dev)
        committed = out[:stride]
        committed_list.append(committed); full_list.append(out.copy())
        state = np.concatenate([state[stride:], committed], axis=0)
        print(f"  step {k+1:2d}/{n_steps} committed {stride}f @ abs {t0+t_in+k*stride}..{t0+t_in+(k+1)*stride-1}")
    return np.concatenate(committed_list, axis=0), full_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--cube", default=DEFAULT_CUBE)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2,
                    help="frames committed per step (also slide amount)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid-steps", type=int, default=4,
                    help="show only first N steps in the grid viz (rest still predicted)")
    args = ap.parse_args()
    cube_name = args.cube

    arm_full = ARMS[args.arm]
    cfg = yaml.safe_load(Path(f"configs/ablations/{arm_full}.yaml").read_text())
    t_in, t_out = cfg["data"]["t_in"], cfg["data"]["t_out"]
    if args.stride > t_out:
        raise ValueError(f"stride={args.stride} must be <= t_out={t_out}")
    nm = cfg.get("normalization", {}).get("method", "zscore_per_cube")
    data_dir = Path(cfg["data"]["data_dir"])
    # Tile = training window; softening from config (s1e3 arms differ from 1e6).
    global TILE, TILE_STRIDE
    TILE = int(cfg["data"].get("window_size") or 64)
    TILE_STRIDE = TILE
    soft = float(cfg.get("normalization", {}).get("signed_asinh_softening", 1.0e6))

    zarr_dirs = sorted(p for p in data_dir.glob("*.zarr") if p.is_dir())
    zp = next(p for p in zarr_dirs if p.stem == cube_name)
    cube_arr, _ = load_cube(zp)
    mu, sigma = (asinh_scale(cube_arr, soft) if nm == "signed_asinh" else cube_stats(cube_arr))
    cube_norm = normalize(cube_arr, mu, sigma, nm)
    print(f"cube {cube_name}: T={cube_arr.shape[0]} HxW={cube_arr.shape[1]}x{cube_arr.shape[2]}")

    sets_meta = find_active_windows(cube_arr, t_in, t_out, TILE, n_sets=3)
    horizon = args.n_steps * args.stride
    grid_cols = (args.n_steps - 1) * args.stride + t_out
    needed_frames = t_in + max(horizon, grid_cols)
    sets_meta = [(ts, y, x) for (ts, y, x) in sets_meta if ts + needed_frames <= cube_arr.shape[0]]
    if not sets_meta:
        raise SystemExit(f"No active window has {needed_frames} frames available.")
    t0 = sets_meta[0][0]
    print(f"t0={t0}, t_in={t_in}, t_out={t_out}, stride={args.stride}, "
          f"n_steps={args.n_steps}, total committed = {horizon} frames")

    dev = device()
    ckpt = Path(f"outputs/ablations/{arm_full}/checkpoints/best_model.pt")
    model = load_s0_model(ckpt, cfg["model"], t_out, dev)

    print(f"\n=== STAIRCASE {args.arm} ({arm_full}) ===")
    pred_n, full_preds_n = staircase(model, cube_norm, t0, t_in, t_out,
                                     args.n_steps, args.stride, dev)
    gt_n = cube_norm[t0 + t_in : t0 + t_in + horizon]
    pred = denormalize(pred_n, mu, sigma, nm)
    gt = denormalize(gt_n, mu, sigma, nm)

    pred_int = pred.sum(axis=(1, 2))
    gt_int = gt.sum(axis=(1, 2))
    abs_err = np.abs(pred_int - gt_int)
    mae = float(np.mean(abs_err))
    pred_var = float(np.var(pred_int))
    gt_var = float(np.var(gt_int))
    drift_corr = float(np.corrcoef(np.arange(horizon), abs_err)[0, 1])
    print(f"\nMAE={mae:.3e} | var ratio(pred/gt)={pred_var/gt_var:.3f} (1.0=match) | "
          f"drift corr={drift_corr:+.3f} (>0=error grows with horizon)")
    out_dir = Path(args.out or f"outputs/staircase_{args.arm.lower()}_{cube_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "series.npz",
             frame_abs=np.arange(t0 + t_in, t0 + t_in + horizon),
             pred_int=pred_int, gt_int=gt_int, abs_err=abs_err,
             pred=pred, gt=gt,
             meta=np.array([t0, t_in, t_out, args.stride, args.n_steps]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.s0_viz._staircase_viz import render_staircase_grid
        x = np.arange(t0 + t_in, t0 + t_in + horizon)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(x, gt_int, "s-", color="tab:blue", label="GT")
        axes[0].plot(x, pred_int, "o-", color="tab:red", label=f"{args.arm} pred (AR)")
        axes[0].set_ylabel("integrated winding flux (Σ pixels)")
        axes[0].set_title(f"{cube_name} staircase, {args.arm} ({arm_full}), t0={t0}, "
                          f"stride={args.stride}\nMAE={mae:.2e} | var ratio={pred_var/gt_var:.3f}")
        axes[1].plot(x, abs_err, "o-", color="tab:purple", label="|pred - gt|")
        axes[1].set_xlabel("absolute frame index"); axes[1].set_ylabel("|abs err|")
        for a in axes: a.legend(); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "integrated_flux.png", dpi=110)
        plt.close(fig)
        gs = min(args.grid_steps, args.n_steps)
        gt_full = denormalize(cube_norm[t0 + t_in : t0 + t_in + (gs - 1) * args.stride + t_out],
                              mu, sigma, nm)
        full_preds = [denormalize(p, mu, sigma, nm) for p in full_preds_n[:gs]]
        render_staircase_grid(gt_full, full_preds, t0 + t_in, args.stride, t_out,
                              f"{args.arm} ({arm_full}) — first {gs} of {args.n_steps} steps",
                              out_dir / f"staircase_grid_{gs}steps.png")
        print(f"\nWrote {out_dir}/ (integrated_flux.png, staircase_grid, series.npz)")
    except Exception as e:
        print(f"plotting skipped: {e}")


if __name__ == "__main__":
    main()
