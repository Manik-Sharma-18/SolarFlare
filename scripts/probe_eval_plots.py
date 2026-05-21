"""Plotting helpers for probe eval. Split from probe_eval.py for 200-line cap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solarflare_data.probe_dataset import ProbeFrameDataset, TargetStats
from scripts.probe_eval_metrics import predict_cube, _persistence_pred


def plot_overlays(out_dir: Path, harps: list[str], data_dir: Path, feat_tag: str,
                  abs_value: bool, clip: float,
                  heads: dict[str, tuple[torch.nn.Module, TargetStats]],
                  device: torch.device, cadence_min: float = 12.0) -> list[Path]:
    any_stats = next(iter(heads.values()))[1]
    ds = ProbeFrameDataset(harps, data_dir, feat_tag, abs_value, clip, any_stats)
    cubes = ds.per_cube_arrays()
    paths = []
    for h in harps:
        d = cubes[h]
        y, valid = d["y"], d["valid"]
        y_eval = y.copy(); y_eval[~valid] = np.nan; y_eval[y_eval <= 0] = np.nan
        t = np.arange(y_eval.size) * (cadence_min / 60.0)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, y_eval, label="true", color="black", lw=1.5)
        for kind, (head, stats) in heads.items():
            pred = predict_cube(head, d["feat"], valid, stats, device)
            ax.plot(t, pred, label=f"probe ({kind})", lw=1.2, alpha=0.85)
        persist = _persistence_pred(y_eval)
        ax.plot(t, persist, label="persistence", lw=0.8, ls="--", alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("⟨|wind|⟩ (clipped 1e7)")
        ax.set_title(f"{h} — wind-flux probe overlay")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"overlay_{h}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        paths.append(p)
    return paths


def plot_jepa_curve(out_dir: Path, run_jsonl: Path) -> Path | None:
    if not run_jsonl.exists(): return None
    train_e, train_l, val_e, val_l = [], [], [], []
    t_idx, v_idx = 0, 0
    for line in run_jsonl.read_text().splitlines():
        try: rec = json.loads(line)
        except Exception: continue
        if rec.get("kind") == "train" and "loss" in rec:
            train_e.append(rec.get("epoch", t_idx)); train_l.append(rec["loss"]); t_idx += 1
        elif rec.get("kind") == "val" and "loss" in rec:
            val_e.append(rec.get("epoch", v_idx)); val_l.append(rec["loss"]); v_idx += 1
    if not val_e: return None
    fig, ax = plt.subplots(figsize=(8, 4))
    if train_e: ax.plot(train_e, train_l, label="train", alpha=0.6, lw=1.0)
    ax.plot(val_e, val_l, label="val", color="red", lw=1.4)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("smooth-L1 (embedding)")
    ax.set_title("E09 JEPA backbone — slow curriculum (val 0.00831 ep98)")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    p = out_dir / "e09_jepa_curve.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p
