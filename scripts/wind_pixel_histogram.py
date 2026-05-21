"""Per-cube log10(|wind|) pixel histogram — validates clip=1e7 choice.

For each cube: histogram of log10(|wind|) over all finite, nonzero pixels.
Vertical lines mark candidate clips at 1e6, 1e7, 1e8.

Special focus on harp_8: its [1e6,1e7] tail (real high-flux) vs [1e7,1e8]
residue (unphysical sentinel tail) should be visible.

Outputs:
    outputs_probe/wind_pixel_hist_grid.png   — 21-panel grid
    outputs_probe/wind_pixel_hist_harp_8.png — harp_8 detail + clip ratios

Usage:
    python3 scripts/wind_pixel_histogram.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solarflare_data.zarr_loader import open_cube  # noqa: E402

CLIP_LINES = [1e6, 1e7, 1e8]
CLIP_COLORS = ["C2", "C1", "C3"]
SAMPLE_FRAMES = 50  # subsample for speed
N_BINS = 80
LOG_MIN, LOG_MAX = 0.0, 11.0


def cube_log_histogram(cube_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stream cube. Return (bin_centers, counts, stats). log10(|wind|) > 0."""
    cube = open_cube(cube_path)
    _, _, t = cube.shape
    bins = np.linspace(LOG_MIN, LOG_MAX, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts = np.zeros(N_BINS, dtype=np.int64)
    valid_t = np.where(cube.valid_frames)[0]
    if valid_t.size > SAMPLE_FRAMES:
        idx = np.linspace(0, valid_t.size - 1, SAMPLE_FRAMES).astype(int)
        valid_t = valid_t[idx]
    px_total = 0
    px_above_1e7 = 0
    px_above_1e8 = 0
    for ti in valid_t:
        frame = np.asarray(cube.wind[:, :, ti], dtype=np.float32)
        absf = np.abs(frame)
        m = np.isfinite(absf) & (absf > 0)
        if not m.any():
            continue
        vals = absf[m]
        log_vals = np.log10(vals)
        c, _ = np.histogram(log_vals, bins=bins)
        counts += c
        px_total += vals.size
        px_above_1e7 += int((vals > 1e7).sum())
        px_above_1e8 += int((vals > 1e8).sum())
    stats = {
        "px_total": px_total,
        "frac_above_1e7": (px_above_1e7 / px_total) if px_total else 0.0,
        "frac_above_1e8": (px_above_1e8 / px_total) if px_total else 0.0,
    }
    return centers, counts, stats


def plot_grid(hists: dict, out_path: Path) -> None:
    rows, cols = 7, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 18), constrained_layout=True)
    axes_flat = axes.flatten()
    fig.suptitle("log10(|wind|) pixel histograms — clip candidates 1e6/1e7/1e8 (green/orange/red)", fontsize=12)
    for i, (harp_id, (centers, counts, stats)) in enumerate(hists.items()):
        ax = axes_flat[i]
        ax.bar(centers, counts, width=(centers[1] - centers[0]) * 0.95,
               color="C0", alpha=0.7, edgecolor="none")
        for c, col in zip(CLIP_LINES, CLIP_COLORS):
            ax.axvline(np.log10(c), color=col, lw=1.0, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlim(LOG_MIN, LOG_MAX)
        title = (f"{harp_id}  >1e7={stats['frac_above_1e7']*100:.3f}%  "
                 f">1e8={stats['frac_above_1e8']*100:.5f}%")
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, lw=0.3)
        if i % cols == 0:
            ax.set_ylabel("count (log)", fontsize=7)
        if i // cols == rows - 1:
            ax.set_xlabel("log10(|wind|)", fontsize=7)
    for j in range(len(hists), len(axes_flat)):
        axes_flat[j].set_axis_off()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_harp_8_detail(hists: dict, out_path: Path) -> None:
    if "harp_8" not in hists:
        return
    centers, counts, stats = hists["harp_8"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(centers, counts, width=(centers[1] - centers[0]) * 0.95,
                color="C3", alpha=0.75)
    axes[0].set_yscale("log")
    for c, col in zip(CLIP_LINES, CLIP_COLORS):
        axes[0].axvline(np.log10(c), color=col, lw=1.2, label=f"clip={c:.0e}")
    axes[0].set_xlabel("log10(|wind|)")
    axes[0].set_ylabel("pixel count (log)")
    axes[0].set_title(f"harp_8 full distribution — "
                      f"{stats['frac_above_1e7']*100:.3f}% > 1e7, "
                      f"{stats['frac_above_1e8']*100:.5f}% > 1e8")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    median_cube = max((h for h in hists if h != "harp_8"),
                      key=lambda h: hists[h][2]["px_total"])
    h_c, h_n, h_s = hists[median_cube]
    norm_8 = counts / max(counts.sum(), 1)
    norm_h = h_n / max(h_n.sum(), 1)
    axes[1].plot(centers, norm_8, color="C3", lw=1.6, label="harp_8")
    axes[1].plot(h_c, norm_h, color="C0", lw=1.2, label=median_cube)
    for c, col in zip(CLIP_LINES, CLIP_COLORS):
        axes[1].axvline(np.log10(c), color=col, lw=1.0, alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("log10(|wind|)")
    axes[1].set_ylabel("density (log)")
    axes[1].set_title(f"harp_8 vs {median_cube} (largest healthy cube)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    with Path("data/manifest.json").open() as fh:
        cubes = sorted(json.load(fh)["cubes"], key=lambda c: c["harp_id"])
    outdir = Path("outputs_probe")
    outdir.mkdir(parents=True, exist_ok=True)
    hists: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    print(f"Histogramming {len(cubes)} cubes ({SAMPLE_FRAMES} frames each)…")
    for c in cubes:
        if not Path(c["path"]).exists():
            continue
        print(f"  {c['harp_id']:<14}", end="", flush=True)
        centers, counts, stats = cube_log_histogram(c["path"])
        hists[c["harp_id"]] = (centers, counts, stats)
        print(f"  total_px={stats['px_total']:>10}  >1e7={stats['frac_above_1e7']*100:.4f}%")
    plot_grid(hists, outdir / "wind_pixel_hist_grid.png")
    plot_harp_8_detail(hists, outdir / "wind_pixel_hist_harp_8.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
