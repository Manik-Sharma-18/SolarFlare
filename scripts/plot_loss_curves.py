"""Plot val/train loss curves for V5 JEPA experiments.

Reads each run.jsonl, emits per-experiment PNG + a composite comparison.
Output: docs/V5_JEPA/figures/<tag>_loss.png and figures/all_val_compare.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Experiments to plot — (tag, run.jsonl path, label, color)
EXPERIMENTS = [
    ("E05_mask_on_50ep", "outputs_v5_mini_mask_on/run.jsonl",
     "E05 mask-ON 50ep (fast curriculum)", "#d62728"),
    ("E06_mask_off_50ep", "outputs_v5_mini_mask_off/run.jsonl",
     "E06 mask-OFF 50ep", "#2ca02c"),
    ("E09_mask_on_100ep_slow", "outputs_v5_mini_mask_on_slow/run.jsonl",
     "E09 mask-ON 100ep (slow curriculum) — anchor 0.00831", "#1f77b4"),
    ("E12_tube_only", "outputs_v5_e12_tube_only_cuda/run.jsonl",
     "E12 tube-only mask (diverges ep65+)", "#e377c2"),
    ("E13_tube_future", "outputs_v5_e13_tube_future_cuda/run.jsonl",
     "E13 tube+future", "#bcbd22"),
    ("E14_tube_cross", "outputs_v5_e14_tube_cross_cuda/run.jsonl",
     "E14 tube+cross", "#17becf"),
    ("E15_uniform_mix", "outputs_v5_e15_uniform_cuda/run.jsonl",
     "E15 uniform mix — SOTA 0.00530", "#ff7f0e"),
    ("E17_ema_0990", "outputs_v5_e17_ema_0990/run.jsonl",
     "E17 τ=0.990 — 0.00761", "#9467bd"),
    ("E30_v2_thesis_curated", "outputs_v5_thesis_curated/run.jsonl",
     "E30 v2 thesis curated — SOTA 0.00268", "#000000"),
]

# Mask-policy ablation group for dedicated comparison (E09, E12–E15)
MASK_ABLATION = ["E09_mask_on_100ep_slow", "E12_tube_only",
                 "E13_tube_future", "E14_tube_cross", "E15_uniform_mix"]

# EMA sweep group (E09 anchor + E17)
EMA_SWEEP = ["E09_mask_on_100ep_slow", "E17_ema_0990"]

# Thesis vs sanity-scale anchors
THESIS_VS_SANITY = ["E09_mask_on_100ep_slow", "E15_uniform_mix",
                    "E17_ema_0990", "E30_v2_thesis_curated"]


def load_curves(path: Path) -> dict:
    """Read run.jsonl. Return {train_ep: [...], train_loss: [...], val_ep: [...], val_loss: [...]}."""
    train_loss_by_ep, val_loss_by_ep = {}, {}
    t_idx = v_idx = 0
    with path.open() as f:
        for line in f:
            try: rec = json.loads(line)
            except Exception: continue
            kind = rec.get("kind")
            if kind == "train" and "loss" in rec:
                ep = rec.get("epoch", t_idx)
                train_loss_by_ep.setdefault(ep, []).append(rec["loss"])
                t_idx += 1
            elif kind == "val" and "loss" in rec:
                ep = rec.get("epoch", v_idx)
                val_loss_by_ep[ep] = rec["loss"]
                v_idx += 1
    # Bin train per epoch (mean).
    train_eps = sorted(train_loss_by_ep)
    train_means = [float(np.mean(train_loss_by_ep[e])) for e in train_eps]
    val_eps = sorted(val_loss_by_ep)
    val_losses = [val_loss_by_ep[e] for e in val_eps]
    return {"train_ep": train_eps, "train_loss": train_means,
            "val_ep": val_eps, "val_loss": val_losses}


def plot_single(tag: str, label: str, color: str, curves: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    if curves["train_ep"]:
        ax.plot(curves["train_ep"], curves["train_loss"],
                color=color, alpha=0.35, lw=1.2, label="train (epoch mean)")
    if curves["val_ep"]:
        ax.plot(curves["val_ep"], curves["val_loss"],
                color=color, lw=2.0, marker="o", ms=3, label="val")
        v = curves["val_loss"]
        best_i = int(np.argmin(v))
        best_ep, best_v = curves["val_ep"][best_i], v[best_i]
        ax.axhline(best_v, ls="--", color=color, alpha=0.4, lw=0.8)
        ax.annotate(f"best ep{best_ep}: {best_v:.5f}",
                    xy=(best_ep, best_v),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=9, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("smooth-L1 embedding loss")
    ax.set_yscale("log")
    ax.set_title(label)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_compare(experiments, results: dict[str, dict], out_path: Path,
                 title: str = "V5 JEPA mini — val-loss comparison across experiments  (★ = best)"
                 ) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    for tag, _path, label, color in experiments:
        c = results.get(tag)
        if not c or not c["val_ep"]: continue
        ax.plot(c["val_ep"], c["val_loss"], color=color, lw=2.0,
                marker="o", ms=2.5, label=label)
        best_i = int(np.argmin(c["val_loss"]))
        ax.scatter([c["val_ep"][best_i]], [c["val_loss"][best_i]],
                   color=color, s=70, marker="*", zorder=5,
                   edgecolor="black", lw=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val smooth-L1 (log scale)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_group(group_tags: list[str], experiments: list, results: dict,
               title: str, out_path: Path) -> None:
    subset = [e for e in experiments if e[0] in group_tags]
    plot_compare(subset, results, out_path, title=title)


def main() -> int:
    out_dir = Path("docs/V5_JEPA/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag, jsonl_path, label, color in EXPERIMENTS:
        p = Path(jsonl_path)
        if not p.exists():
            print(f"[skip] {tag}: missing {p}"); continue
        c = load_curves(p)
        results[tag] = c
        plot_single(tag, label, color, c, out_dir / f"{tag}_loss.png")
        if c["val_loss"]:
            best_v = min(c["val_loss"])
            best_i = c["val_loss"].index(best_v)
            print(f"  {tag}: best val={best_v:.5f} @ ep{c['val_ep'][best_i]} "
                  f"(n_val={len(c['val_loss'])})")
    plot_compare(EXPERIMENTS, results, out_dir / "all_val_compare.png")
    plot_group(MASK_ABLATION, EXPERIMENTS, results,
               "V5 JEPA mini — mask-policy ablation (E09 anchor + E12–E15)  (★ = best)",
               out_dir / "mask_ablation_compare.png")
    plot_group(EMA_SWEEP, EXPERIMENTS, results,
               "V5 JEPA mini — EMA τ sweep (E09 anchor + E17)  (★ = best)",
               out_dir / "ema_sweep_compare.png")
    plot_group(THESIS_VS_SANITY, EXPERIMENTS, results,
               "V5 JEPA — thesis E30 v2 vs sanity anchors (E09/E15/E17)  (★ = best)",
               out_dir / "E30_v2_vs_sanity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
