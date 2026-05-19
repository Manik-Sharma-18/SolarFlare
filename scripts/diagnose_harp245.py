"""Diagnose why harp_245 calibration explodes (a=3.7, b=-6.6e3 → 220% medAPE).

Hypotheses:
  H1: target scale / range very different from other novel cubes
  H2: encoder features degenerate (low variance / collapse)
  H3: raw pred range far from y range → affine has nothing to grab
  H4: target slow-varying (persist MAPE 5.5% confirms) so cal/eval split
      gives a,b fit on early part that doesn't generalize
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from models.v5.wind_probe_head import build_probe
from solarflare_data.probe_dataset import ProbeFrameDataset, TargetStats
from scripts.probe_eval_metrics import predict_cube


def load_probe(path):
    p = torch.load(str(path), map_location="cpu", weights_only=False)
    head = build_probe(p["kind"], dim=int(p["dim"]),
                       hidden=int(p["config"]["probe"].get("hidden", 256)),
                       dropout=float(p["config"]["probe"].get("dropout", 0.1)))
    head.load_state_dict(p["state_dict"]); head.eval()
    return head, TargetStats.from_dict(p["stats"])


def stats(name, x):
    x = x[np.isfinite(x)]
    return {
        "name": name, "n": int(x.size),
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "std": float(x.std()),
        "med": float(np.median(x)),
    }


def main():
    cfg = yaml.safe_load(Path("configs/probe_e30.yaml").read_text())
    head, tstats = load_probe(Path("outputs_probe/E30_linear/best.pt"))
    data_dir = Path(cfg["data"]["data_dir"])
    feat_tag = cfg["encoder"]["feat_tag"]
    abs_value = bool(cfg["data"]["target"]["abs_value"])
    clip = float(cfg["data"]["target"]["clip"])

    novel = ["harp_17", "harp_51", "harp_245", "harp_may2024", "harp_nov2025"]
    ds = ProbeFrameDataset(novel, data_dir, feat_tag, abs_value, clip, tstats)
    arrs_all = ds.per_cube_arrays()

    print(f"{'cube':<14} {'n_valid':>7} {'y_med':>10} {'y_mean':>10} {'y_std':>10} "
          f"{'pred_med':>10} {'pred_mean':>10} {'pred_std':>10} "
          f"{'feat_std':>10} {'persist_acf1':>12}")
    for h in novel:
        a = arrs_all[h]
        feat, y, valid = a["feat"], a["y"], a["valid"]
        pred = predict_cube(head, feat, valid, tstats, torch.device("cpu"))
        ye = y.astype(float); ye[~np.isfinite(ye)] = np.nan
        ye[ye <= 0] = np.nan
        pe = np.where(valid, pred, np.nan)
        # feat std (avg per-dim std, indicates encoder collapse if low)
        f_v = feat[valid]
        feat_std = float(f_v.std(axis=0).mean())
        # lag-1 autocorrelation of y (high → trivial persistence)
        yv = ye[valid]; yv = yv[np.isfinite(yv)]
        acf1 = float(np.corrcoef(yv[:-1], yv[1:])[0, 1]) if yv.size > 5 else float("nan")
        print(f"{h:<14} {int(valid.sum()):>7} "
              f"{np.nanmedian(ye):>10.3e} {np.nanmean(ye):>10.3e} {np.nanstd(ye):>10.3e} "
              f"{np.nanmedian(pe):>10.3e} {np.nanmean(pe):>10.3e} {np.nanstd(pe):>10.3e} "
              f"{feat_std:>10.4f} {acf1:>12.4f}")

    # Inspect calibration cal/eval split on harp_245 specifically.
    print("\n--- harp_245 detail ---")
    a = arrs_all["harp_245"]
    feat, y, valid = a["feat"], a["y"], a["valid"]
    pred = predict_cube(head, feat, valid, tstats, torch.device("cpu"))
    ye = y.copy().astype(float); ye[~np.isfinite(ye)] = np.nan; ye[ye <= 0] = np.nan
    valid_idx = np.where(valid)[0]
    n_cal = max(5, int(round(valid_idx.size * 0.30)))
    cal_idx = valid_idx[:n_cal]
    eval_idx = valid_idx[n_cal:]
    print(f"n_valid={valid_idx.size}  n_cal={n_cal}  n_eval={eval_idx.size}")
    yc, pc = ye[cal_idx], pred[cal_idx]
    ye_, pe_ = ye[eval_idx], pred[eval_idx]
    m = np.isfinite(yc) & np.isfinite(pc) & (yc > 0)
    a_, b_ = np.polyfit(pc[m], yc[m], 1)
    print(f"cal a={a_:.4f} b={b_:.4e}")
    print(f"cal y: med={np.nanmedian(yc):.3e} mean={np.nanmean(yc):.3e} std={np.nanstd(yc):.3e}")
    print(f"eval y: med={np.nanmedian(ye_):.3e} mean={np.nanmean(ye_):.3e} std={np.nanstd(ye_):.3e}")
    print(f"cal pred: med={np.nanmedian(pc):.3e} mean={np.nanmean(pc):.3e} std={np.nanstd(pc):.3e}")
    print(f"eval pred: med={np.nanmedian(pe_):.3e} mean={np.nanmean(pe_):.3e} std={np.nanstd(pe_):.3e}")
    # Distribution shift cal→eval?
    print(f"cal y range: [{np.nanmin(yc):.3e}, {np.nanmax(yc):.3e}]")
    print(f"eval y range: [{np.nanmin(ye_):.3e}, {np.nanmax(ye_):.3e}]")


if __name__ == "__main__":
    main()
