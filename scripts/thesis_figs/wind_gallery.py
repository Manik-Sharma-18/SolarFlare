"""Gallery of winding-flux frames across the corpus.

Renders one mid-frame per cube for nine selected cubes spanning the
spatial-extent bins (S/M/L/XL), so the reader can see the diversity of
spatial structure on which the JEPA encoder pretrains.

Output: thesis/assets/figures/wind_gallery.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Selected by spatial-extent bin and split membership so the gallery covers
# small/medium/large/XL AR cubes from train + val + novel partitions.
GALLERY = [
    ("harp_316", "harp_316 (train, S)"),
    ("harp_49",  "harp_49 (train, M)"),
    ("harp_43",  "harp_43 (train, L)"),
    ("harp_116", "harp_116 (train, XL)"),
    ("harp_11930", "harp_11930 (val, M)"),
    ("harp_86",  "harp_86 (val, L)"),
    ("harp_17",  "harp_17 (novel)"),
    ("harp_may2024", "harp_may2024 (novel)"),
    ("harp_245", "harp_245 (novel, long)"),
]


def mid_frame(harp: str) -> tuple[np.ndarray, int, int]:
    z = zarr.open(str(DATA / f"{harp}.zarr"), mode="r")
    wind = z["wind"]
    time = np.asarray(z["Time"][:])
    valid = np.where(time > 1e8)[0]
    t = int(valid[len(valid) // 2]) if valid.size else wind.shape[2] // 2
    arr = np.asarray(wind[:, :, t])
    return np.where(np.isnan(arr), 0.0, arr), arr.shape[0], arr.shape[1]


def main() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10.0, 9.0))
    for ax, (harp, label) in zip(axes.flat, GALLERY):
        try:
            frame, h, w = mid_frame(harp)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"missing\n{harp}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([]); continue
        vmax = float(np.nanpercentile(np.abs(frame), 98.0)) or 1.0
        im = ax.imshow(frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower", interpolation="nearest")
        ax.set_title(f"{label}\n{h}$\\times${w} px, $p_{{98}}{{=}}${vmax:.0f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.75)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Winding flux $W(\\mathbf{r},t)$ gallery: mid-frame per cube, "
                 "diverging cmap $\\pm p_{98}(|W|)$ (blue = positive chirality, "
                 "red = negative)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT / "wind_gallery.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
