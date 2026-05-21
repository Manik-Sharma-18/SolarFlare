"""Render harp_8 outlier histogram + per-cube outlier-count bar for thesis Ch 3.

Scans all data/*.zarr cubes once. Counts |w| > 1e8 per cube. Plots
two panels: (left) log-y histogram of |w| in harp_8 with the 1e8
guard marked; (right) per-cube outlier-count bar chart.

Run: python3 scripts/thesis_figs/clip_audit.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CLIP = 1.0e8


def cube_outliers(path: Path) -> tuple[int, int, float]:
    """Return (n_outlier_pixels, total_finite_pixels, peak_abs)."""
    root = zarr.open(str(path), mode="r")
    wind = np.asarray(root["wind"])
    finite = np.isfinite(wind)
    abs_w = np.abs(wind, where=finite)
    extreme = finite & (abs_w > CLIP)
    return int(extreme.sum()), int(finite.sum()), float(abs_w.max(initial=0.0))


def harp8_abs(path: Path) -> np.ndarray:
    root = zarr.open(str(path), mode="r")
    wind = np.asarray(root["wind"])
    finite = np.isfinite(wind)
    return np.abs(wind[finite])


def main() -> None:
    cubes = sorted((REPO / "data").glob("harp_*.zarr"))
    if not cubes:
        print("no cubes found"); return

    counts: dict[str, int] = {}
    for c in cubes:
        n_out, _, peak = cube_outliers(c)
        counts[c.stem] = n_out
        print(f"  {c.stem:12s}  outliers={n_out:6d}  peak={peak:.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))

    # left: harp_8 histogram
    h8 = REPO / "data" / "harp_8.zarr"
    if h8.exists():
        arr = harp8_abs(h8)
        ax1.hist(arr, bins=np.logspace(0, 11, 80),
                 color="#c44e52", edgecolor="black", linewidth=0.3)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.axvline(CLIP, color="black", linestyle="--", linewidth=1.0,
                    label=r"guard $|w|=10^{8}$")
        ax1.set_xlabel(r"$|w|$ (winding-flux units)")
        ax1.set_ylabel("pixel count")
        ax1.set_title("harp\\_8 absolute winding-flux distribution")
        ax1.legend(fontsize=8, frameon=False)
        ax1.grid(alpha=0.3, linewidth=0.4)
    else:
        ax1.text(0.5, 0.5, "harp_8.zarr missing", ha="center")

    # right: per-cube outlier count
    labels = list(counts.keys())
    vals   = [counts[k] for k in labels]
    ypos   = list(range(len(labels)))
    ax2.barh(ypos, vals, color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([s.replace("_", r"\_") for s in labels], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xscale("symlog")
    ax2.set_xlabel(r"pixels with $|w| > 10^{8}$")
    ax2.set_title("Per-cube outlier counts above the guard")
    ax2.grid(axis="x", alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    out_path = OUT / "clip_audit.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
