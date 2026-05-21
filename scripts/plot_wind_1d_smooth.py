"""Clip + smoothing comparison plots for ⟨wind⟩(t).

Emits:
    outputs_probe/wind_1d_clip_sweep_harp_8.png       — harp_8 across clips (pathology)
    outputs_probe/wind_1d_clip_sweep_overlay.png      — 21-cube abs mean per clip
    outputs_probe/wind_1d_smooth_grid_abs.png         — 21 panels, abs clip=1e6, smoothing overlay
    outputs_probe/wind_1d_smooth_showcase.png         — 4 representative cubes, smoothing detail

Usage:
    python3 scripts/plot_wind_1d_smooth.py
    python3 scripts/plot_wind_1d_smooth.py --clip 1e6 --smooth-windows 1 5 15 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solarflare_data.wind_target import load_or_compute_wind_1d, smooth_1d  # noqa: E402

CADENCE_MIN = 12.0
SHOWCASE_CUBES = ["harp_43", "harp_274", "harp_11930", "harp_may2024"]


def hours(arr: np.ndarray) -> np.ndarray:
    return np.arange(arr.size, dtype=np.float32) * (CADENCE_MIN / 60.0)


def plot_clip_sweep_one_cube(cube_path: str, clips: list[float], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, abs_value, label in zip(axes, [False, True], ["signed", "abs"]):
        for clip in clips:
            arr = load_or_compute_wind_1d(cube_path, abs_value=abs_value, clip=clip)
            ax.plot(hours(arr), arr, lw=0.6, label=f"clip={clip:.0e}", alpha=0.85)
        if not abs_value:
            ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.set_ylabel(f"⟨{label} wind⟩")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("hours")
    name = Path(cube_path).stem
    fig.suptitle(f"{name} — clip sweep")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_clip_overlay_all(cubes: list[dict], clips: list[float], abs_value: bool, out_path: Path) -> None:
    """One panel per clip; all 21 cubes overlaid (max-normalized) for visual compare."""
    fig, axes = plt.subplots(1, len(clips), figsize=(6 * len(clips), 5), sharey=True)
    if len(clips) == 1:
        axes = [axes]
    for ax, clip in zip(axes, clips):
        for c in cubes:
            if not Path(c["path"]).exists():
                continue
            arr = load_or_compute_wind_1d(c["path"], abs_value=abs_value, clip=clip)
            scale = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
            if scale <= 0:
                continue
            ax.plot(hours(arr), arr / scale, lw=0.4, alpha=0.5)
        ax.set_title(f"clip={clip:.0e}  ({'abs' if abs_value else 'signed'})")
        ax.set_xlabel("hours")
        ax.axhline(0, color="k", lw=0.3, alpha=0.5)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("per-cube max-normalized")
    fig.suptitle(f"⟨wind⟩(t) overlay across clips — {len(cubes)} cubes")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_smooth_grid(cubes: list[dict], clip: float, windows: list[int], out_path: Path) -> None:
    """21-panel grid: abs ⟨wind⟩ raw + smoothed overlays at clip."""
    rows, cols = 7, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 18), constrained_layout=True)
    axes_flat = axes.flatten()
    fig.suptitle(f"⟨|wind|⟩(t) smoothing — clip={clip:.0e}, windows={windows}", fontsize=13)

    for i, c in enumerate(cubes):
        ax = axes_flat[i]
        if not Path(c["path"]).exists():
            ax.set_axis_off()
            continue
        arr = load_or_compute_wind_1d(c["path"], abs_value=True, clip=clip)
        xh = hours(arr)
        ax.plot(xh, arr, lw=0.4, alpha=0.4, color="0.5", label="raw")
        for w in windows:
            if w <= 1:
                continue
            sm = smooth_1d(arr, w)
            hr = w * CADENCE_MIN / 60.0
            ax.plot(xh, sm, lw=0.9, label=f"w={w} ({hr:.1f}h)")
        ax.set_title(c["harp_id"], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, lw=0.3)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")
        if i % cols == 0:
            ax.set_ylabel("⟨|w|⟩", fontsize=8)
        if i // cols == rows - 1:
            ax.set_xlabel("hours", fontsize=8)

    for j in range(len(cubes), len(axes_flat)):
        axes_flat[j].set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_smooth_showcase(cubes_by_id: dict, clip: float, windows: list[int], out_path: Path) -> None:
    pick = [cubes_by_id[k] for k in SHOWCASE_CUBES if k in cubes_by_id]
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 3.0 * len(pick)), sharex=False)
    if len(pick) == 1:
        axes = [axes]
    fig.suptitle(f"Smoothing showcase — abs, clip={clip:.0e}, windows={windows}")
    for ax, c in zip(axes, pick):
        arr = load_or_compute_wind_1d(c["path"], abs_value=True, clip=clip)
        xh = hours(arr)
        ax.plot(xh, arr, lw=0.4, color="0.6", alpha=0.7, label="raw")
        for w in windows:
            if w <= 1:
                continue
            ax.plot(xh, smooth_1d(arr, w), lw=1.0, label=f"w={w} ({w*CADENCE_MIN/60:.1f}h)")
        ax.set_title(c["harp_id"])
        ax.set_ylabel("⟨|wind|⟩")
        ax.set_xlabel("hours")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--outdir", default="outputs_probe")
    p.add_argument("--clip", type=float, default=1e6)
    p.add_argument("--clip-sweep", type=float, nargs="+", default=[1e8, 1e7, 1e6])
    p.add_argument("--smooth-windows", type=int, nargs="+", default=[1, 5, 15, 30])
    args = p.parse_args()

    with Path(args.manifest).open() as fh:
        cubes = sorted(json.load(fh)["cubes"], key=lambda c: c["harp_id"])
    by_id = {c["harp_id"]: c for c in cubes}
    outdir = Path(args.outdir)
    print(f"Plotting {len(cubes)} cubes  clip={args.clip:.0e}  windows={args.smooth_windows}")

    if "harp_8" in by_id:
        plot_clip_sweep_one_cube(by_id["harp_8"]["path"], args.clip_sweep, outdir / "wind_1d_clip_sweep_harp_8.png")
    plot_clip_overlay_all(cubes, args.clip_sweep, abs_value=True, out_path=outdir / "wind_1d_clip_sweep_overlay.png")
    plot_smooth_grid(cubes, args.clip, args.smooth_windows, outdir / "wind_1d_smooth_grid_abs.png")
    plot_smooth_showcase(by_id, args.clip, args.smooth_windows, outdir / "wind_1d_smooth_showcase.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
