"""Render one combined figure per cube.

Layout (n_sets active-region rows + 1 time-series row)::

    set row:  [pred t+1..t+K] [GT t+1..t+K] [crop spatial-mean: pred vs actual]
    last row: full-cube spatial-mean winding flux vs time (spans width)

Image pairs share a colour scale (from GT) so blur/under-prediction is
visible; the mean graph shares one y-axis for pred and actual.
"""
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def _symlog_norm(gt: np.ndarray, pred: np.ndarray) -> SymLogNorm:
    """Robust signed-log scale for heavy-tailed flux.

    Bounds from the 99.5th percentile of |values| (clips rare extremes that
    otherwise wash everything to grey); linear region near 0 set to the
    median non-zero magnitude so bulk structure stays visible.
    """
    a = np.abs(np.concatenate([gt.ravel(), pred.ravel()]))
    vmax = float(np.percentile(a, 99.5)) or 1.0
    nz = a[a > 0]
    lt = float(np.percentile(nz, 50)) if nz.size else vmax / 100.0
    lt = min(max(lt, vmax / 1e4), vmax / 2)  # keep linthresh sane vs vmax
    return SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)


def render_cube_figure(
    name: str,
    split: str,
    t_out: int,
    sets: List[Dict],
    out_path: Path,
    region_note: str = "crops are 64x64 around peak-|flux|",
    model_label: str = "SimpleConvLSTM",
) -> None:
    """Write the combined PNG for one cube. ``sets`` items carry
    ``pred``/``gt`` ``(t_out, h, w)`` physical arrays + ``t_start``.

    Layout: one image row per set (pred frames + GT frames), then a bottom
    row of ``len(sets)`` small graphs — predicted vs actual crop spatial-mean
    flux over the predicted frames, one per set.
    """
    ncol = 2 * t_out
    nrow = len(sets) + 1
    fig = plt.figure(figsize=(2.0 * ncol, 2.2 * nrow))
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0.45, wspace=0.3)

    # Norm computed from GT only — keeps GT spectrum identical across arms
    # (pred clips/saturates visibly when it overshoots, which is informative).
    all_gt = np.concatenate([st["gt"].ravel() for st in sets])
    norm = _symlog_norm(all_gt, all_gt)
    all_axes, im = [], None

    for r, st in enumerate(sets):
        pred, gt = st["pred"], st["gt"]
        row_axes = []
        for k in range(t_out):
            ax = fig.add_subplot(gs[r, k])
            im = ax.imshow(pred[k], cmap="seismic", norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            row_axes.append(ax)
            if r == 0:
                ax.set_title(f"pred t+{k+1}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"set {r+1}\nt0={st['t_start']}", fontsize=8)
        for k in range(t_out):
            ax = fig.add_subplot(gs[r, t_out + k])
            im = ax.imshow(gt[k], cmap="seismic", norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            row_axes.append(ax)
            if r == 0:
                ax.set_title(f"GT t+{k+1}", fontsize=8)
        all_axes.extend(row_axes)

    cb = fig.colorbar(im, ax=all_axes, fraction=0.012, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("winding flux (symlog)", fontsize=7)

    # bottom row: one small pred-vs-actual spatial-mean graph per set
    bottom = GridSpecFromSubplotSpec(1, len(sets), subplot_spec=gs[nrow - 1, :], wspace=0.35)
    steps = np.arange(1, t_out + 1)
    for j, st in enumerate(sets):
        axg = fig.add_subplot(bottom[0, j])
        axg.plot(steps, st["pred"].sum(axis=(1, 2)), "o-", label="predicted", color="tab:red")
        axg.plot(steps, st["gt"].sum(axis=(1, 2)), "s-", label="actual", color="tab:blue")
        axg.set_title(f"set {j+1} (t0={st['t_start']}) — integrated winding flux", fontsize=9)
        axg.set_xlabel("predicted frame (t+)", fontsize=8)
        if j == 0:
            axg.set_ylabel("integrated winding flux (Σ pixels)", fontsize=8)
            axg.legend(fontsize=8)
        axg.set_xticks(steps)
        axg.tick_params(labelsize=7)

    fig.suptitle(
        f"{name}  [{split.upper()}]   {model_label}\n"
        f"{region_note}; symlog colour scale shared across all sets (matches staircase)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
