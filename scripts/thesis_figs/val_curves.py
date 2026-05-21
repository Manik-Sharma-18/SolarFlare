"""Render val-loss convergence curves + ablation summary bars for thesis Ch 5.

Reads `outputs_v5_e*/run.jsonl`. Writes PDFs to thesis/assets/figures/.

Run from repo root:  python3 scripts/thesis_figs/val_curves.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# (run_dir, label) tuples grouped by ablation
MASK_POLICY = [
    ("outputs_v5_e12_tube_only_cuda",   "tube only (E12)"),
    ("outputs_v5_e13_tube_future_cuda", "tube + future (E13)"),
    ("outputs_v5_e14_tube_cross_cuda",  "tube + cross (E14)"),
    ("outputs_v5_e15_uniform_cuda",     "uniform mix (E15)"),
]
EMA_SWEEP = [
    ("outputs_v5_e17_ema_0990", r"$\tau=0.990$ (E17)"),
    ("outputs_v5_e18_ema_0994", r"$\tau=0.994$ (E18)"),
    ("outputs_v5_e19_ema_0998", r"$\tau=0.998$ (E19)"),
    ("outputs_v5_e20_ema_09995", r"$\tau=0.9995$ (E20)"),
]
THESIS_RUN = [
    ("outputs_v5_e30_lowdim_long", "E30 thesis run"),
    ("outputs_v5_e31_lowdim_long", "E31 thesis run"),
]

# Best-validation losses for runs whose raw run.jsonl is not available
# locally (executed on a remote CUDA slot; only summary numbers synced back).
# Sourced from thesis Tables 5.2 and 5.3 / Appendix D summary table.
EXTRA_BARS = [
    (r"$\tau=0.9995$ (E20)", 0.03010),
    ("ratio $0.60$ (E25)",   0.00960),
    ("ratio $0.75$ (E26)",   0.00876),
    ("ratio $0.85$ (E27)",   0.01597),
    ("ratio $0.90$ (E28, term.)", 0.05798),
]


def load_val(run_dir: Path) -> list[float]:
    """Return val loss per epoch, in order, from a run.jsonl file."""
    path = run_dir / "run.jsonl"
    if not path.exists():
        return []
    losses: list[float] = []
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "val" and "loss" in rec:
                losses.append(float(rec["loss"]))
    return losses


def plot_group(runs, title: str, fname: str, ylog: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for sub, label in runs:
        losses = load_val(REPO / sub)
        if not losses:
            print(f"  skip empty: {sub}")
            continue
        epochs = list(range(1, len(losses) + 1))
        ax.plot(epochs, losses, label=label, linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (smooth-$L_1$, embedding)")
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    out_path = OUT / fname
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO)}")


def summary_table() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for group in (MASK_POLICY, EMA_SWEEP, THESIS_RUN):
        for sub, label in group:
            losses = load_val(REPO / sub)
            if not losses:
                continue
            rows[label] = {
                "best": min(losses),
                "best_ep": int(losses.index(min(losses))) + 1,
                "last": losses[-1],
                "n_epoch": len(losses),
            }
    return rows


def plot_ablation_bars() -> None:
    table = summary_table()
    rows: list[tuple[str, float, str]] = []
    for k, v in table.items():
        rows.append((k, v["best"], _group_of(k)))
    have_labels = {k for k, _, _ in rows}
    for label, best in EXTRA_BARS:
        if label not in have_labels:
            rows.append((label, best, _group_of(label)))
    if not rows:
        print("  no rows for ablation bars")
        return
    group_order = {"mask": 0, "ema": 1, "ratio": 2, "thesis": 3}
    rows.sort(key=lambda r: (group_order.get(r[2], 9), r[1]))
    labels = [r[0] for r in rows]
    bests = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    palette = {"mask": "#4c72b0", "ema": "#dd8452",
               "ratio": "#55a868", "thesis": "#c44e52"}
    colors = [palette.get(g, "#888") for g in groups]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(labels) + 1.6))
    ypos = list(range(len(labels)))
    ax.barh(ypos, bests, color=colors, edgecolor="black", linewidth=0.5)
    for y, b in zip(ypos, bests):
        ax.text(b * 1.05, y, f"{b:.4f}", va="center", fontsize=7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("best validation loss (smooth-$L_1$, embedding) — lower is better")
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[g],
                             edgecolor="black", linewidth=0.5)
               for g in ("mask", "ema", "ratio", "thesis")]
    ax.legend(handles, ["mask policy", "EMA decay", "mask ratio", "thesis run"],
              fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    out_path = OUT / "ablation_summary.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO)}")


def _group_of(label: str) -> str:
    s = label.lower()
    if "e12" in s or "e13" in s or "e14" in s or "e15" in s or "tube" in s or "uniform" in s:
        return "mask"
    if "tau" in s or "ema" in s or any(x in s for x in ("e17", "e18", "e19", "e20", "0.99")):
        return "ema"
    if "ratio" in s or any(x in s for x in ("e25", "e26", "e27", "e28")):
        return "ratio"
    if "thesis" in s or "e30" in s or "e31" in s:
        return "thesis"
    return "other"


def write_summary_text() -> None:
    table = summary_table()
    out_path = OUT / "ablation_summary.txt"
    lines = [f"{k:40s}  best={v['best']:.5f} @ep{v['best_ep']:3d}"
             f"  last={v['last']:.5f}  ({v['n_epoch']} ep)"
             for k, v in table.items()]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    print("rendering convergence curves...")
    plot_group(MASK_POLICY, "Mask-policy ablation (E12--E15)",  "curves_mask_policy.pdf")
    plot_group(EMA_SWEEP,   r"EMA $\tau$ sweep (E17--E19)",     "curves_ema_sweep.pdf")
    plot_group(THESIS_RUN,  "Thesis curated run (E30/E31)",      "curves_thesis_run.pdf")
    print("rendering ablation summary bars...")
    plot_ablation_bars()
    write_summary_text()
    print("done.")
