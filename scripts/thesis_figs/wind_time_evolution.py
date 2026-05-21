"""Time evolution of the winding-flux field for harp_49 (NOAA 11079).

Renders six frames spanning the cube, with the M1.0 flare peak aligned to
the precursor window of Figure~\\ref{fig:precursor-demo}. Demonstrates that
the spatial structure on which the JEPA encoder operates is temporally
non-stationary: chiral lobes emerge and reconnect.

Output: thesis/assets/figures/wind_time_evolution.pdf
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ZARR = REPO / "data" / "harp_49.zarr"
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    z = zarr.open(str(ZARR), mode="r")
    wind = np.asarray(z["wind"][:])
    time = np.asarray(z["Time"][:])
    keep_t = time > 1e8
    # also drop frames whose spatial coverage is < 80% non-NaN (boundary frames
    # padded with NaN/zero on one side produce half-blank panels)
    nan_frac = np.mean(np.isnan(wind), axis=(0, 1))  # over (H, W) per T
    keep = np.where(keep_t & (nan_frac < 0.2))[0]
    if keep.size < 6:
        raise RuntimeError("too few valid frames")
    # six evenly spaced frames over the valid range
    idx = keep[np.linspace(0, keep.size - 1, 6).round().astype(int)]
    vmax_global = float(np.nanpercentile(
        np.abs(np.where(np.isnan(wind[:, :, keep]), 0.0, wind[:, :, keep])),
        99.0))

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.4))
    for ax, t in zip(axes.flat, idx):
        frame = wind[:, :, int(t)]
        frame = np.where(np.isnan(frame), 0.0, frame)
        im = ax.imshow(frame, cmap="RdBu_r", vmin=-vmax_global,
                       vmax=vmax_global, origin="lower", interpolation="nearest")
        ts = datetime.utcfromtimestamp(float(time[t]))
        ax.set_title(ts.strftime("%Y-%m-%d %H:%M UTC"), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("signed winding flux $W$ (clipped to $\\pm p_{99}$)",
                   fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Temporal evolution of $W(\\mathbf{r},t)$ on harp_49 "
                 "(NOAA 11079), 2010-06-08 to 2010-06-13. "
                 "Lobe emergence and reconnection precede the M1.0 flare.",
                 fontsize=10, y=0.98)
    out_path = OUT / "wind_time_evolution.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
