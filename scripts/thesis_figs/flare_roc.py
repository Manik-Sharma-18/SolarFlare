"""ROC curves for the C+/12h MLP flare classifier (thesis Ch 6).

Loads the cached E30 features + the trained head from
outputs_flare/E30_C_12h_temporal_mlp/best.pt, regenerates per-cube logits on
the trailing 30%% eval slice (within-cube temporal split), and renders a
4-panel ROC figure with the persistence baseline overlaid.

Output: thesis/assets/figures/flare_roc.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FEAT_TAG = "E30"
LABEL_TAG = "C_12h"
CKPT = REPO / "outputs_flare" / "E30_C_12h_temporal_mlp" / "best.pt"
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_CUBES = ["harp_49", "harp_54", "harp_8", "harp_318"]
SPLIT_FRAC = 0.7


def load_per_cube(harp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat = np.load(DATA / f"{harp}_feat_{FEAT_TAG}.npy").astype(np.float32)
    valid = np.load(DATA / f"{harp}_feat_{FEAT_TAG}_valid.npy").astype(bool)
    lab = np.load(DATA / f"{harp}_labels_{LABEL_TAG}.npy").astype(np.int64)
    T = min(feat.shape[0], valid.shape[0], lab.shape[0])
    feat, valid, lab = feat[:T], valid[:T], lab[:T]
    keep = valid & np.all(np.isfinite(feat), axis=1)
    return feat, lab, keep


def load_head() -> torch.nn.Module:
    import sys
    sys.path.insert(0, str(REPO))
    from models.v5.wind_probe_head import build_probe
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    kind = ck.get("kind", "mlp")
    dim = int(ck.get("dim", 384))
    head = build_probe(kind, dim=dim, hidden=256, dropout=0.0)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head


def roc_curve(logits: np.ndarray, y: np.ndarray, n_thr: int = 200):
    thrs = np.linspace(logits.min() - 1e-6, logits.max() + 1e-6, n_thr)
    tprs, fprs = [], []
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    for thr in thrs[::-1]:
        p = (logits >= thr).astype(np.int64)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return np.array(fprs), np.array(tprs)


def persistence_point(lab: np.ndarray, keep: np.ndarray):
    pred = np.zeros_like(lab); pred[1:] = lab[:-1]
    k = keep.copy(); k[0] = False
    y = lab[k]; p = pred[k]
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return None
    tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return fp / max(fp + tn, 1), tp / max(tp + fn, 1)


def auc_trapz(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def main() -> None:
    head = load_head()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    for ax, harp in zip(axes.flat, PANEL_CUBES):
        feat, lab, keep = load_per_cube(harp)
        T = lab.shape[0]
        train_cut = int(SPLIT_FRAC * T)
        train_keep = np.zeros(T, dtype=bool); train_keep[:train_cut] = True
        em = keep & ~train_keep
        with torch.no_grad():
            z = torch.from_numpy(feat[em]).float()
            logits = head(z).cpu().numpy()
        y = lab[em]
        n_pos = int(y.sum()); n_neg = int(y.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            ax.text(0.5, 0.5, f"{harp}\n(degenerate: pos={n_pos}/{y.size})",
                    ha="center", va="center", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        fpr, tpr = roc_curve(logits, y)
        auc = auc_trapz(fpr, tpr)
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=1.6,
                label=f"MLP head (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.6, linestyle="--",
                label="random")
        pp = persistence_point(lab, em)
        if pp is not None:
            ax.scatter([pp[0]], [pp[1]], marker="*", s=110, color="#d62728",
                       zorder=5, label="lag-1 persistence")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title(f"{harp}  (n={y.size}, pos={n_pos})", fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("C+ / 12 h MLP flare-classifier ROC vs persistence "
                 "(within-cube temporal split, trailing 30%)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT / "flare_roc.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
