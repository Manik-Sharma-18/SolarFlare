"""Three-panel calibration recovery for harp_245 (Ch 6 §probe-logcal).

Shows the same cube under: (left) raw probe, (mid) linear per-cube affine
recalibration y = a*yhat + b, (right) log-affine recalibration
log y = a*log(yhat) + b. Demonstrates that the heavy-tailed cube is
defeated by the linear fit but stabilised by the log-affine fit.

Output: thesis/assets/figures/probe_harp245_recovery.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.v5.wind_probe_head import build_probe  # noqa: E402
from solarflare_data.probe_dataset import TargetStats  # noqa: E402
from solarflare_data.wind_target import cache_path as wind_cache_path  # noqa: E402

DATA = REPO / "data"
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = REPO / "outputs_probe" / "E30_mlp" / "best.pt"
CAL_PATH = REPO / "outputs_probe" / "E30_eval" / "probe_calibration.json"
HARP = "harp_245"


def load_setup():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    kind = payload["kind"]
    dim = int(payload["dim"])
    stats = TargetStats.from_dict(payload["stats"])
    hidden = int(cfg.get("probe", {}).get("hidden", 256))
    head = build_probe(kind, dim=dim, hidden=hidden, dropout=0.0)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    abs_value = bool(cfg.get("target", {}).get("abs", True))
    clip = float(cfg.get("target", {}).get("clip", 1.0e7))
    feat_tag = cfg.get("encoder", {}).get("feat_tag", "E30")
    cal = json.loads(CAL_PATH.read_text())
    row = next(r for r in cal["linear"] if r["harp"] == HARP)
    return head, stats, abs_value, clip, feat_tag, row


def load_cube(abs_value: bool, clip: float, feat_tag: str):
    feat = np.load(DATA / f"{HARP}_feat_{feat_tag}.npy").astype(np.float32)
    valid_feat = np.load(DATA / f"{HARP}_feat_{feat_tag}_valid.npy").astype(bool)
    y = np.load(wind_cache_path(DATA / f"{HARP}.zarr",
                                abs_value=abs_value, clip=clip)).astype(np.float32)
    T = min(feat.shape[0], valid_feat.shape[0], y.shape[0])
    feat, valid_feat, y = feat[:T], valid_feat[:T], y[:T]
    keep = valid_feat & np.isfinite(y) & np.all(np.isfinite(feat), axis=1)
    return feat[keep], y[keep]


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    head, stats, abs_value, clip, feat_tag, row = load_setup()
    feat, y = load_cube(abs_value, clip, feat_tag)
    with torch.no_grad():
        yhat_norm = head(torch.from_numpy(feat).float()).cpu().numpy()
    yhat_raw = stats.invert(yhat_norm)
    a, b = float(row["a"]), float(row["b"])
    yhat_lin = a * yhat_raw + b
    a_log, b_log = float(row["a_log"]), float(row["b_log"])
    # log calibration: log10(y) = a_log * log10(yhat) + b_log
    safe = np.maximum(yhat_raw, 1.0)
    yhat_log = np.power(10.0, a_log * np.log10(safe) + b_log)

    panels = [
        ("raw", yhat_raw, "#dd8452"),
        ("linear affine $a\\hat y+b$", yhat_lin, "#4c72b0"),
        ("log-affine $\\hat y^a 10^b$", yhat_log, "#55a868"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, (label, yhat, col) in zip(axes, panels):
        ymin = min(y.min(), yhat.min())
        ymax = max(y.max(), yhat.max())
        lo, hi = ymin - 0.05 * abs(ymin), ymax + 0.05 * abs(ymax)
        ax.scatter(y, yhat, s=6, alpha=0.45, color=col, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6,
                linestyle="--", label="$y=\\hat y$")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        rr = r2(y, yhat)
        ax.set_title(f"{label}\n$R^2={rr:+.2f}$", fontsize=10)
        ax.set_xlabel("actual $\\langle W \\rangle$ (harp\\_245)", fontsize=9)
        ax.set_ylabel("predicted $\\hat{\\langle W \\rangle}$", fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7)
    fig.suptitle("harp\\_245 calibration recovery: linear affine fits one "
                 "rescale + offset; log-affine fits a power-law structure "
                 "that absorbs the heavy-tailed target", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUT / "probe_harp245_recovery.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
