"""Plot ⟨wind⟩(t) curves for every cube in manifest.

Two figures, 7×3 grid each (21 panels):
    outputs_probe/wind_1d_signed.png   — signed mean over valid pixels
    outputs_probe/wind_1d_abs.png      — |wind| mean over valid pixels

X-axis: hours since first valid frame (12-min cadence).
NaN frames render as gaps (matplotlib breaks line on NaN).

Usage:
    python3 scripts/plot_wind_1d.py
    python3 scripts/plot_wind_1d.py --outdir custom/dir
    python3 scripts/plot_wind_1d.py --manifest data/manifest.json
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

from solarflare_data.wind_target import load_or_compute_wind_1d  # noqa: E402

CADENCE_MIN = 12.0


def load_curve(cube_path: str, abs_value: bool) -> tuple[np.ndarray, np.ndarray]:
    arr = load_or_compute_wind_1d(cube_path, abs_value=abs_value)
    hours = np.arange(arr.size, dtype=np.float32) * (CADENCE_MIN / 60.0)
    return hours, arr


def panel_stats(arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    if not finite.any():
        return "empty"
    v = arr[finite]
    return f"n={finite.sum()}/{arr.size}  μ={v.mean():.2e}  σ={v.std():.2e}"


def plot_grid(cubes: list[dict], abs_value: bool, out_path: Path) -> None:
    rows, cols = 7, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 18), constrained_layout=True)
    axes_flat = axes.flatten()

    title_flavor = "|wind| (abs)" if abs_value else "wind (signed)"
    fig.suptitle(f"⟨{title_flavor}⟩(t) — 21 cubes, {CADENCE_MIN:.0f}-min cadence", fontsize=14)

    for i, c in enumerate(cubes):
        ax = axes_flat[i]
        harp_id = c["harp_id"]
        cube_path = c["path"]
        if not Path(cube_path).exists():
            ax.set_title(f"{harp_id} (missing)", fontsize=9)
            ax.set_axis_off()
            continue
        hours, arr = load_curve(cube_path, abs_value=abs_value)

        ax.plot(hours, arr, lw=0.7, color="C0" if not abs_value else "C3")
        if not abs_value:
            ax.axhline(0, color="k", lw=0.3, alpha=0.4)

        ax.set_title(f"{harp_id}  {panel_stats(arr)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, lw=0.3)
        if i % cols == 0:
            ax.set_ylabel("⟨wind⟩", fontsize=8)
        if i // cols == rows - 1:
            ax.set_xlabel("hours", fontsize=8)

    for j in range(len(cubes), len(axes_flat)):
        axes_flat[j].set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_overview(cubes: list[dict], out_path: Path) -> None:
    """Single-figure overlay: all 21 signed curves rescaled to [-1, 1] per-cube.

    Helps eyeball common dynamics / sharp events across cubes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    n_drawn = 0
    for c in cubes:
        cube_path = c["path"]
        if not Path(cube_path).exists():
            continue
        hours, arr = load_curve(cube_path, abs_value=False)
        finite = np.isfinite(arr)
        if finite.sum() < 8:
            continue
        v = arr.copy()
        scale = float(np.nanmax(np.abs(v)))
        if scale == 0 or not np.isfinite(scale):
            continue
        v = v / scale
        ax.plot(hours, v, lw=0.5, alpha=0.55, label=c["harp_id"])
        n_drawn += 1

    ax.axhline(0, color="k", lw=0.3, alpha=0.5)
    ax.set_title(f"⟨wind⟩(t) per-cube max-normalized — {n_drawn} cubes", fontsize=11)
    ax.set_xlabel("hours since first frame")
    ax.set_ylabel("signed mean / max(|signed mean|)")
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.legend(loc="upper right", fontsize=6, ncol=3, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--outdir", default="outputs_probe")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    cubes = sorted(manifest.get("cubes", []), key=lambda c: c["harp_id"])
    if not cubes:
        print("ERROR: no cubes in manifest", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    print(f"Plotting {len(cubes)} cubes to {outdir}/")
    plot_grid(cubes, abs_value=False, out_path=outdir / "wind_1d_signed.png")
    plot_grid(cubes, abs_value=True, out_path=outdir / "wind_1d_abs.png")
    plot_overview(cubes, out_path=outdir / "wind_1d_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
