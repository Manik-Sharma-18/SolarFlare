"""Per-config × per-cube TSS heatmap for the C+ flare classifier (Ch 6).

Reads temporal_eval.json from each of the four C+ configurations
(6h/12h × linear/MLP), extracts per-cube best-threshold TSS and the
lag-one persistence TSS, and renders a heatmap with cubes on the x-axis
and configurations on the y-axis. The bottom row is the persistence
baseline. Cells annotated with the TSS value; missing entries
(degenerate cube) are left blank.

Output: thesis/assets/figures/flare_tss_heatmap.pdf
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("C+ / 6 h linear",  REPO / "outputs_flare" / "E30_C_6h_temporal_linear"),
    ("C+ / 6 h MLP",     REPO / "outputs_flare" / "E30_C_6h_temporal_mlp"),
    ("C+ / 12 h linear", REPO / "outputs_flare" / "E30_C_12h_temporal_linear"),
    ("C+ / 12 h MLP",    REPO / "outputs_flare" / "E30_C_12h_temporal_mlp"),
]
# Order cubes by total positive mass across configs for readability
CUBE_ORDER = ["harp_49", "harp_54", "harp_8", "harp_318", "harp_11930",
              "harp_51", "harp_156", "harp_245"]


def load_per_cube(eval_dir: Path) -> dict:
    j = json.loads((eval_dir / "temporal_eval.json").read_text())
    return j.get("per_cube", {})


def main():
    head_grid = np.full((len(CONFIGS), len(CUBE_ORDER)), np.nan)
    persist_row = np.full(len(CUBE_ORDER), np.nan)
    n_pos_row = np.full(len(CUBE_ORDER), np.nan)
    for ci, (_, d) in enumerate(CONFIGS):
        pc = load_per_cube(d)
        for cj, h in enumerate(CUBE_ORDER):
            m = pc.get(h, {})
            tss = m.get("tss_best", m.get("tss", float("nan")))
            if isinstance(tss, (int, float)) and not math.isnan(tss):
                head_grid[ci, cj] = tss
            if math.isnan(persist_row[cj]):
                pers = m.get("tss_persist", float("nan"))
                if isinstance(pers, (int, float)) and not math.isnan(pers):
                    persist_row[cj] = pers
            npos = m.get("n_pos", float("nan"))
            if isinstance(npos, (int, float)) and not math.isnan(npos):
                n_pos_row[cj] = float(npos)

    full = np.vstack([head_grid, persist_row[None, :]])
    row_labels = [c[0] for c in CONFIGS] + ["lag-1 persistence"]
    col_labels = [f"{h}\n(n_pos={int(n_pos_row[i]) if not math.isnan(n_pos_row[i]) else '?'})"
                  for i, h in enumerate(CUBE_ORDER)]

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    im = ax.imshow(full, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CUBE_ORDER)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    # separator above persistence row
    ax.axhline(len(CONFIGS) - 0.5, color="black", linewidth=0.7)
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            v = full[i, j]
            if math.isnan(v):
                txt = "—"
                col = "black"
            else:
                txt = f"{v:+.2f}"
                col = "black" if abs(v) < 0.6 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("true skill statistic (TSS)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("C+ flare classifier: per-cube TSS across four "
                 "head--window configurations, with lag-1 persistence "
                 "baseline (bottom row)", fontsize=10)
    fig.tight_layout()
    out_path = OUT / "flare_tss_heatmap.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
