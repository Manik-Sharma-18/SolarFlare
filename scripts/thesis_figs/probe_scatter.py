"""Render probe diagnostics for thesis Ch 6.

Inputs:
  outputs_probe/E30_eval/probe_calibration.json -- per-cube linear + log calibration

Outputs two PDF panels (three-series grouped bars: raw / linear-cal / log-cal):
  probe_r2_bars.pdf
  probe_mape_bars.pdf

Run: python3 scripts/thesis_figs/probe_scatter.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CAL_PATH = REPO / "outputs_probe" / "E30_eval" / "probe_calibration.json"


def load_cal() -> list[dict]:
    if not CAL_PATH.exists():
        return []
    return json.loads(CAL_PATH.read_text()).get("linear", [])


def grouped_bars(rows: list[dict], metric: str, ylabel: str,
                 fname: str, clip: tuple[float, float] | None = None,
                 sort_key: str = "raw") -> None:
    if not rows:
        print(f"  skip {fname}: no data")
        return
    rows = sorted(rows, key=lambda r: r[sort_key][metric])
    labels = [r["harp"] for r in rows]
    raw = [r["raw"][metric] for r in rows]
    cal = [r["cal"][metric] for r in rows]
    log = [r.get("log", r["cal"])[metric] for r in rows]

    raw_clipped, cal_clipped, log_clipped = raw[:], cal[:], log[:]
    if clip:
        lo, hi = clip
        raw_clipped = [max(lo, min(hi, v)) for v in raw]
        cal_clipped = [max(lo, min(hi, v)) for v in cal]
        log_clipped = [max(lo, min(hi, v)) for v in log]

    n = len(labels)
    xpos = list(range(n))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * n + 3.0), 3.8))
    b1 = ax.bar([x - width for x in xpos], raw_clipped, width=width,
                color="#dd8452", edgecolor="black", linewidth=0.5,
                label="raw probe")
    b2 = ax.bar(xpos, cal_clipped, width=width,
                color="#4c72b0", edgecolor="black", linewidth=0.5,
                hatch="///", label="+ linear affine")
    b3 = ax.bar([x + width for x in xpos], log_clipped, width=width,
                color="#55a868", edgecolor="black", linewidth=0.5,
                hatch="xxx", label="+ log-affine")

    if clip:
        lo, hi = clip
        for x, v in zip(xpos, raw):
            if v < lo or v > hi:
                ax.text(x - width, lo if v < lo else hi, f"{v:.1f}",
                        ha="center", va="bottom" if v < lo else "top",
                        fontsize=6, rotation=0)

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    out_path = OUT / fname
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO)}")


def main() -> None:
    rows = load_cal()
    print(f"loaded {len(rows)} per-cube calibration entries")
    grouped_bars(rows, "r2", r"$R^{2}$ (clipped to $[-2, 1]$, raw value annotated)",
                 "probe_r2_bars.pdf", clip=(-2.0, 1.0))
    grouped_bars(rows, "mape", "MAPE (%, clipped to [0, 60])",
                 "probe_mape_bars.pdf", clip=(0.0, 60.0))


if __name__ == "__main__":
    main()
