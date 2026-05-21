"""Render two example winding-flux frames (diverging cmap) for thesis Ch 3.

Reads two zarr cubes, picks a middle frame from each, displays with
symmetric vmax = p98(|x|). Output: single 2-panel PDF.

Run: python3 scripts/thesis_figs/wind_examples.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CUBES = [
    ("data/harp_11930.zarr", "harp\\_11930"),
    ("data/harp_43.zarr",    "harp\\_43"),
]


def load_frame(path: Path, frame_frac: float = 0.5) -> np.ndarray:
    root = zarr.open(str(path), mode="r")
    wind = root["wind"]
    T = wind.shape[2]
    t = int(round(frame_frac * (T - 1)))
    arr = np.asarray(wind[:, :, t])
    return np.where(np.isnan(arr), 0.0, arr)


def render_panel(ax, frame: np.ndarray, title: str) -> None:
    vmax = float(np.nanpercentile(np.abs(frame), 98.0))
    if vmax <= 0.0:
        vmax = float(np.abs(frame).max() + 1.0)
    im = ax.imshow(frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main() -> None:
    fig, axes = plt.subplots(1, len(CUBES), figsize=(8.4, 3.6))
    if len(CUBES) == 1:
        axes = [axes]
    for ax, (rel, label) in zip(axes, CUBES):
        path = REPO / rel
        if not path.exists():
            ax.text(0.5, 0.5, f"missing\n{rel}", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        frame = load_frame(path)
        im = render_panel(ax, frame, label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    fig.suptitle("Winding flux $W(\\mathbf{r},t)$: example frames "
                 "(diverging colourmap, $\\pm p_{98}(|x|)$)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUT / "wind_examples.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
