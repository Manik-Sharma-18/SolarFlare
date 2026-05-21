"""Render one realisation of each V5 mask family for thesis Appendix B.

Calls solarflare_data.mask_catalog directly. Output: a single 5-panel PDF
showing each family as a time-vs-spatial slab so families that mask whole
frames (future, cross-time, tail) are not solid-black.

Run: python3 scripts/thesis_figs/render_masks.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from solarflare_data import mask_catalog as mc  # noqa: E402

OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def slab(mask: torch.Tensor) -> torch.Tensor:
    """Flatten (T,Hp,Wp) → (Hp*Wp, T) so x=time, y=spatial-token."""
    T, Hp, Wp = mask.shape
    return mask.reshape(T, Hp * Wp).t()


def render_panel(ax, mask: torch.Tensor, title: str) -> None:
    s = slab(mask).numpy()
    ax.imshow(s, cmap="Greys", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time index", fontsize=8)
    ax.set_ylabel("spatial token (flat)", fontsize=8)
    ax.tick_params(labelsize=7)


def main() -> None:
    T, Hp, Wp = 14, 16, 16
    gen = torch.Generator().manual_seed(0)

    fig, axes = plt.subplots(1, 5, figsize=(12.0, 2.8))

    m_short = mc._sample_tube_one(T, Hp, Wp, area=0.15, gen=gen)
    render_panel(axes[0], m_short, "short tube (15\\%)")

    m_long = mc._sample_tube_one(T, Hp, Wp, area=0.40, gen=gen)
    render_panel(axes[1], m_long, "long tube (40\\%)")

    m_fut = mc._sample_future_one(T, Hp, Wp, frac=0.30, gen=gen)
    render_panel(axes[2], m_fut, "future block (last $K$)")

    m_cross = mc._sample_cross_time_one(T, Hp, Wp, frac=0.30, gen=gen)
    render_panel(axes[3], m_cross, "cross-time")

    m_tail = mc._sample_tail_one(T, Hp, Wp, t_out=2)
    render_panel(axes[4], m_tail, "tail (deployment)")

    fig.suptitle("V5 mask catalogue: space--time slab per family "
                 "(black = masked, white = visible)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = OUT / "mask_catalog.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
