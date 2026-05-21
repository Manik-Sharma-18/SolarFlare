"""Trainer for frozen-encoder wind-flux probe heads.

Single-loop trainer (no AMP, no compile, no grad accum) — features are cached
to disk and the head is tiny, so simplicity wins. Logs per-epoch train/val
loss + MSE→R² in normalized space; eval script does original-space R²/r.

Saves:
- best.pt: {'state_dict', 'kind', 'dim', 'stats', 'config', 'epoch', 'val_loss'}
- last.pt: same keys, latest epoch
- run.jsonl: append-only per-epoch records
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from solarflare_data.probe_dataset import TargetStats


@dataclass
class ProbeState:
    epoch: int = 0
    global_step: int = 0
    best_val: float = float("inf")


def build_probe_optimizer(probe: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt = cfg["training"]["optimizer"].lower()
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    if opt == "adamw":
        return torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    if opt == "adam":
        return torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(probe.parameters(), lr=lr,
                               momentum=float(cfg["training"].get("momentum", 0.9)),
                               weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt}")


def _r2_norm(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R² in the (already normalized) target space — quick health metric."""
    ss_res = ((pred - target) ** 2).sum().item()
    mean = target.mean()
    ss_tot = ((target - mean) ** 2).sum().item()
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def train_one_epoch(probe: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, state: ProbeState) -> dict:
    probe.train()
    loss_sum, n_seen = 0.0, 0
    preds, targets = [], []
    for batch in loader:
        z = batch["z"].to(device)
        y = batch["y_norm"].to(device)
        pred = probe(z)
        loss = nn.functional.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * z.shape[0]
        n_seen += int(z.shape[0])
        state.global_step += 1
        preds.append(pred.detach()); targets.append(y.detach())
    preds_t = torch.cat(preds); targets_t = torch.cat(targets)
    return {"loss": loss_sum / max(1, n_seen), "r2_norm": _r2_norm(preds_t, targets_t),
            "n": n_seen}


@torch.no_grad()
def validate(probe: nn.Module, loader: DataLoader, device: torch.device, state: ProbeState) -> dict:
    probe.eval()
    loss_sum, n_seen = 0.0, 0
    preds, targets = [], []
    for batch in loader:
        z = batch["z"].to(device)
        y = batch["y_norm"].to(device)
        pred = probe(z)
        loss = nn.functional.mse_loss(pred, y)
        loss_sum += float(loss.item()) * z.shape[0]
        n_seen += int(z.shape[0])
        preds.append(pred); targets.append(y)
    preds_t = torch.cat(preds); targets_t = torch.cat(targets)
    return {"loss": loss_sum / max(1, n_seen), "r2_norm": _r2_norm(preds_t, targets_t),
            "n": n_seen}


def save_probe_ckpt(path: Path, probe: nn.Module, kind: str, dim: int,
                    stats: TargetStats, cfg: dict, state: ProbeState, val_loss: float) -> None:
    payload = {
        "state_dict": probe.state_dict(),
        "kind": kind,
        "dim": dim,
        "stats": stats.to_dict(),
        "config": cfg,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "val_loss": float(val_loss),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def write_split_record(out_dir: Path, train_harps: list[str], val_harps: list[str],
                       novel_harps: list[str], stats: TargetStats) -> None:
    rec = {
        "train": train_harps, "val": val_harps, "novel_eval": novel_harps,
        "target_stats": stats.to_dict(),
    }
    (out_dir / "splits.json").write_text(json.dumps(rec, indent=2))
