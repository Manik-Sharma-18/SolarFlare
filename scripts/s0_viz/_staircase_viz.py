"""Staircase grid viz: top row = GT, below rows = per-step preds at their
absolute frame positions so overlaps are visible. Each cell is a small
heatmap thumbnail with shared symmetric color scale.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def _symlog_norm(arr):
    a = np.abs(arr.ravel())
    vmax = float(np.percentile(a, 99.5)) or 1.0
    nz = a[a > 0]
    lt = float(np.percentile(nz, 50)) if nz.size else vmax / 100.0
    lt = min(max(lt, vmax / 1e4), vmax / 2)
    return SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)


def render_staircase_grid(gt_seq, step_preds_full, abs_start, stride, t_out,
                          arm_label, out_path, downsample=4):
    """gt_seq: (horizon+t_out-stride, H, W) full GT covering all pred positions.
       step_preds_full: list of (t_out, H, W) preds, one per staircase step.
       abs_start: absolute frame index of first GT frame in gt_seq.
       Each pred row offset by k*stride from leftmost column.
    """
    n_steps = len(step_preds_full)
    n_cols = (n_steps - 1) * stride + t_out
    gt_seq = gt_seq[:n_cols]                         # clip to grid width
    ds = downsample
    gt_thumb = gt_seq[:, ::ds, ::ds]
    preds_thumb = [p[:, ::ds, ::ds] for p in step_preds_full]

    norm = _symlog_norm(np.concatenate([gt_thumb.ravel(),
                                        np.concatenate([p.ravel() for p in preds_thumb])]))

    fig_w = max(10.0, 1.6 * n_cols)
    fig_h = 1.6 * (n_steps + 1) + 1.2
    fig, axes = plt.subplots(n_steps + 1, n_cols,
                             figsize=(fig_w, fig_h), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_linewidth(0.3)

    for c in range(n_cols):
        axes[0, c].imshow(gt_thumb[c], cmap="seismic", norm=norm)
        axes[0, c].set_title(f"f{abs_start+c}", fontsize=8)
    axes[0, 0].set_ylabel("GT", fontsize=10, rotation=0, ha="right", va="center")

    for k, pred in enumerate(preds_thumb):
        row = k + 1
        col0 = k * stride
        for j in range(t_out):
            c = col0 + j
            if c < n_cols:
                axes[row, c].imshow(pred[j], cmap="seismic", norm=norm)
                if j < stride:
                    # committed (non-overlap) frames — highlight border
                    for spine in axes[row, c].spines.values():
                        spine.set_color("limegreen"); spine.set_linewidth(1.0)
                else:
                    for spine in axes[row, c].spines.values():
                        spine.set_color("gold"); spine.set_linewidth(1.0)
        # blank out columns outside this step's span
        for c in range(n_cols):
            if c < col0 or c >= col0 + t_out:
                axes[row, c].set_visible(False)
        axes[row, 0].set_ylabel(f"step{k}", fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle(f"Staircase predictions — {arm_label}\n"
                 f"green border = committed (stride={stride}), gold = overlap (look-ahead)",
                 fontsize=10)
    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.04, right=0.99,
                         wspace=0.05, hspace=0.15)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
