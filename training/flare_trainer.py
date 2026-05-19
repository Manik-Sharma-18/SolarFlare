"""Trainer for frozen-encoder binary flare classifier (M+ or C+ in next N hours).

Single-loop trainer over cached features [T,D] + binary labels [T]. BCE-w-logits
with pos_weight for class imbalance. Eval emits TSS (True Skill Statistic) at the
threshold that maximises it on the val set + ROC-AUC.

TSS = TPR − FPR = recall − false_alarm_rate. Operational space-weather metric;
matches Prithish's XGBoost baseline reporting (TSS 0.5–0.6 on holdout).

Saves:
- best.pt by val TSS (NOT loss — TSS is what we care about)
- last.pt latest epoch
- run.jsonl append-only
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class FlareState:
    epoch: int = 0
    global_step: int = 0
    best_tss: float = -float("inf")


def build_flare_optimizer(head: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt = cfg["training"]["optimizer"].lower()
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    if opt == "adamw":
        return torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    if opt == "adam":
        return torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt}")


def _binary_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """Compute TSS at every threshold, return best + ROC-AUC.
    logits, y: 1D np arrays. y in {0,1}."""
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        # Degenerate: all positive or all negative → TSS undefined; emit pos_rate marker.
        return {"tss_best": float("nan"), "thr_best": 0.0, "auc": float("nan"),
                 "pos_rate": float(y.mean()) if y.size else float("nan"),
                 "tpr_at_best": float("nan"), "fpr_at_best": float("nan")}
    # Sort by logit descending for ROC sweep
    order = np.argsort(-logits)
    y_sorted = y[order].astype(np.float64)
    n_pos = float(y.sum())
    n_neg = float(y.size - y.sum())
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tss = tpr - fpr
    k = int(np.argmax(tss))
    thr_best = float(logits[order][k])
    # ROC-AUC via trapezoid (prepend (0,0), append (1,1))
    fpr_aug = np.concatenate([[0.0], fpr, [1.0]])
    tpr_aug = np.concatenate([[0.0], tpr, [1.0]])
    auc = float(np.trapezoid(tpr_aug, fpr_aug))
    return {"tss_best": float(tss[k]), "thr_best": thr_best, "auc": auc,
             "pos_rate": float(y.mean()), "tpr_at_best": float(tpr[k]),
             "fpr_at_best": float(fpr[k])}


def train_one_epoch(head: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, state: FlareState, pos_weight: torch.Tensor) -> dict:
    head.train()
    loss_sum, n_seen = 0.0, 0
    all_logits, all_y = [], []
    for batch in loader:
        z = batch["z"].to(device)
        y = batch["y"].to(device)
        logits = head(z)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * z.shape[0]
        n_seen += int(z.shape[0])
        state.global_step += 1
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits) if all_logits else np.zeros(0)
    y_np = np.concatenate(all_y) if all_y else np.zeros(0)
    m = _binary_metrics(logits_np, y_np)
    return {"loss": loss_sum / max(1, n_seen), "n": n_seen, **m}


@torch.no_grad()
def validate(head: nn.Module, loader: DataLoader, device: torch.device,
             state: FlareState, pos_weight: torch.Tensor) -> dict:
    head.eval()
    loss_sum, n_seen = 0.0, 0
    all_logits, all_y = [], []
    for batch in loader:
        z = batch["z"].to(device)
        y = batch["y"].to(device)
        logits = head(z)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight)
        loss_sum += float(loss.item()) * z.shape[0]
        n_seen += int(z.shape[0])
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.cpu().numpy())
    logits_np = np.concatenate(all_logits) if all_logits else np.zeros(0)
    y_np = np.concatenate(all_y) if all_y else np.zeros(0)
    m = _binary_metrics(logits_np, y_np)
    return {"loss": loss_sum / max(1, n_seen), "n": n_seen, **m}


def save_flare_ckpt(path: Path, head: nn.Module, kind: str, dim: int, cls: str,
                    window_hr: int, pos_weight: float, thr_best: float,
                    cfg: dict, state: FlareState, val_metrics: dict) -> None:
    payload = {
        "state_dict": head.state_dict(),
        "kind": kind, "dim": dim,
        "cls": cls, "window_hr": int(window_hr),
        "pos_weight": float(pos_weight),
        "thr_best": float(thr_best),
        "config": cfg,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "val_metrics": val_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
