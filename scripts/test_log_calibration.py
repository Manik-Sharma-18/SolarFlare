"""Test log-space affine calibration vs linear-space.

Hypothesis: heavy-tail targets (harp_245) break linear polyfit. Fitting
log(y) = a·log(pred) + b yields multiplicative+offset cal that is robust
to extreme spikes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import yaml

from models.v5.wind_probe_head import build_probe
from solarflare_data.probe_dataset import ProbeFrameDataset, TargetStats
from scripts.probe_eval_metrics import predict_cube, _r2, _pearson, _mape


def load_probe(path):
    p = torch.load(str(path), map_location="cpu", weights_only=False)
    head = build_probe(p["kind"], dim=int(p["dim"]),
                       hidden=int(p["config"]["probe"].get("hidden", 256)),
                       dropout=float(p["config"]["probe"].get("dropout", 0.1)))
    head.load_state_dict(p["state_dict"]); head.eval()
    return head, TargetStats.from_dict(p["stats"])


def fit_log_affine(p, y):
    m = np.isfinite(p) & np.isfinite(y) & (y > 0) & (p > 0)
    if m.sum() < 5: return 1.0, 0.0
    a, b = np.polyfit(np.log(p[m]), np.log(y[m]), 1)
    return float(a), float(b)


def apply_log(pred, a, b):
    out = np.full_like(pred, np.nan)
    m = pred > 0
    out[m] = np.exp(a * np.log(pred[m]) + b)
    return out


def main():
    cfg = yaml.safe_load(Path("configs/probe_e30.yaml").read_text())
    head, tstats = load_probe(Path("outputs_probe/E30_linear/best.pt"))
    data_dir = Path(cfg["data"]["data_dir"])
    feat_tag = cfg["encoder"]["feat_tag"]
    abs_value = bool(cfg["data"]["target"]["abs_value"])
    clip = float(cfg["data"]["target"]["clip"])

    all_eval = cfg["data"]["val_cubes"] + ["harp_17", "harp_51", "harp_245",
                                            "harp_may2024", "harp_nov2025"]
    ds = ProbeFrameDataset(all_eval, data_dir, feat_tag, abs_value, clip, tstats)
    arrs_all = ds.per_cube_arrays()

    print(f"{'cube':<14} {'a_log':>7} {'b_log':>10} {'R²_lin':>10} {'R²_log':>10} "
          f"{'MAPE_lin':>10} {'MAPE_log':>10}")
    for h in all_eval:
        a = arrs_all[h]
        feat, y, valid = a["feat"], a["y"], a["valid"]
        pred = predict_cube(head, feat, valid, tstats, torch.device("cpu"))
        ye = y.astype(float); ye[~np.isfinite(ye)] = np.nan; ye[ye <= 0] = np.nan
        vi = np.where(valid)[0]
        n_cal = max(5, int(round(vi.size * 0.30)))
        cal_idx = vi[:n_cal]; eval_idx = vi[n_cal:]
        # linear cal
        m = np.isfinite(pred[cal_idx]) & np.isfinite(ye[cal_idx]) & (ye[cal_idx] > 0)
        a_lin, b_lin = np.polyfit(pred[cal_idx][m], ye[cal_idx][m], 1)
        # log cal
        a_log, b_log = fit_log_affine(pred[cal_idx], ye[cal_idx])
        # apply on eval
        yt = ye[eval_idx]; pe = pred[eval_idx]
        yp_lin = pe * a_lin + b_lin
        yp_log = apply_log(pe, a_log, b_log)
        print(f"{h:<14} {a_log:>7.3f} {b_log:>+10.3f} "
              f"{_r2(yt, yp_lin):>+10.3f} {_r2(yt, yp_log):>+10.3f} "
              f"{_mape(yt, yp_lin):>9.1f}% {_mape(yt, yp_log):>9.1f}%")


if __name__ == "__main__":
    main()
