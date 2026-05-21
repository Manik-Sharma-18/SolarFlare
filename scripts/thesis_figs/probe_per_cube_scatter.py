"""Per-cube predicted-vs-actual scatter grid for the wind-flux probe (Ch 6).

Loads the E30 MLP probe head + cached features, computes per-frame
predictions for the eight evaluation cubes (encoder-val + novel), and
renders an 8-panel scatter. Each panel shows the predicted-vs-actual
spatial-mean winding flux, with y=x diagonal and per-cube Pearson r
annotation.

Output: thesis/assets/figures/probe_per_cube_scatter.pdf
"""
from __future__ import annotations
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
EVAL_CUBES = [
    ("harp_11930", "val"),
    ("harp_221",   "val"),
    ("harp_86",    "val"),
    ("harp_17",    "novel"),
    ("harp_51",    "novel"),
    ("harp_may2024", "novel"),
    ("harp_nov2025", "novel"),
    ("harp_245",   "novel"),
]


def load_head_and_stats():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    kind = payload["kind"]
    dim = int(payload["dim"])
    stats = TargetStats.from_dict(payload["stats"])
    hidden = int(cfg.get("probe", {}).get("hidden", 256))
    dropout = float(cfg.get("probe", {}).get("dropout", 0.0))
    head = build_probe(kind, dim=dim, hidden=hidden, dropout=dropout)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    abs_value = bool(cfg.get("target", {}).get("abs", True))
    clip = float(cfg.get("target", {}).get("clip", 1.0e7))
    feat_tag = cfg.get("encoder", {}).get("feat_tag", "E30")
    return head, stats, abs_value, clip, feat_tag


def load_cube(harp: str, abs_value: bool, clip: float, feat_tag: str):
    feat = np.load(DATA / f"{harp}_feat_{feat_tag}.npy").astype(np.float32)
    valid_feat = np.load(DATA / f"{harp}_feat_{feat_tag}_valid.npy").astype(bool)
    target_path = wind_cache_path(DATA / f"{harp}.zarr",
                                  abs_value=abs_value, clip=clip)
    y = np.load(target_path).astype(np.float32)
    T = min(feat.shape[0], valid_feat.shape[0], y.shape[0])
    feat, valid_feat, y = feat[:T], valid_feat[:T], y[:T]
    keep = valid_feat & np.isfinite(y) & np.all(np.isfinite(feat), axis=1)
    return feat[keep], y[keep]


def pearson(a, b):
    if a.size < 2:
        return float("nan")
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    head, stats, abs_value, clip, feat_tag = load_head_and_stats()
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5))
    for ax, (harp, split) in zip(axes.flat, EVAL_CUBES):
        try:
            feat, y = load_cube(harp, abs_value, clip, feat_tag)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f"missing\n{harp}\n{e.filename if hasattr(e,'filename') else ''}",
                    ha="center", va="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); continue
        if feat.size == 0:
            ax.text(0.5, 0.5, f"{harp}\n(no valid frames)",
                    ha="center", va="center", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); continue
        with torch.no_grad():
            yhat_norm = head(torch.from_numpy(feat).float()).cpu().numpy()
        yhat = stats.invert(yhat_norm)
        # both axes in linear; clip x/y for visibility
        ymin = min(y.min(), yhat.min())
        ymax = max(y.max(), yhat.max())
        lo, hi = ymin - 0.05 * abs(ymin), ymax + 0.05 * abs(ymax)
        ax.scatter(y, yhat, s=4, alpha=0.35, color="#1f77b4", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6,
                linestyle="--", label="$y=\\hat y$")
        r = pearson(y, yhat)
        ax.set_title(f"{harp} ({split})\nn={y.size}, r={r:.2f}", fontsize=9)
        ax.set_xlabel("actual $\\langle W \\rangle$", fontsize=8)
        ax.set_ylabel("predicted $\\hat{\\langle W \\rangle}$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.suptitle("Per-cube predicted vs actual spatial-mean winding flux "
                 "(E30 v2 MLP probe, raw — no per-cube affine)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT / "probe_per_cube_scatter.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
