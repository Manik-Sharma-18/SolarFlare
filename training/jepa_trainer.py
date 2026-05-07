"""V5 JEPA trainer (Path B) — smooth-L1 in embedding space, curriculum, target EMA.

Per docs/V5_JEPA/01_path_a.md §3.4 / §3.5 (adapted to Path B):
- Optimizer AdamW, single LR group (encoder + predictor + adapter all trainable).
- Cosine schedule with warmup, decay to min_lr_ratio of peak.
- bf16 mixed precision; grad clip 1.0.
- Target encoder is EMA copy of context encoder (anti-collapse). Updated after each step.
- Curriculum: 1-step / ⌈t_out/2⌉-step / full-t_out-step rollout over train fractions.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val: float = float("inf")


def trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def cosine_warmup_lr(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cos)


def _curriculum_t_out(epoch: int, total_epochs: int, t_out: int, cfg: dict) -> int:
    cur = cfg["training"]["curriculum"]
    one = cur["one_step_pct"]
    half = cur["half_step_pct"]
    frac = epoch / max(1, total_epochs)
    if frac < one:
        return 1
    if frac < one + half:
        return max(1, math.ceil(t_out / 2))
    return t_out


def _bf16_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return _NullCtx()


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    state: TrainState,
    device: torch.device,
    total_steps: int,
) -> dict[str, float]:
    model.train()
    if hasattr(model, "target_encoder"):
        model.target_encoder.eval()  # target stays in eval (no dropout/drop_path)
    t0 = time.time()
    grad_clip = float(cfg["training"]["grad_clip"])
    log_every = int(cfg["logging"]["log_every"])
    accum = int(cfg["training"]["grad_accum_steps"])
    warmup = int(round(cfg["training"]["warmup_pct"] * total_steps))
    peak_lr = float(cfg["training"]["lr"])
    min_ratio = float(cfg["training"]["min_lr_ratio"])

    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    total_epochs = int(cfg["training"]["epochs"])
    t_out = int(cfg["data"]["t_out"])
    rollout = _curriculum_t_out(state.epoch, total_epochs, t_out, cfg)

    for i, batch in enumerate(loader):
        x = batch["wind"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        for group in optimizer.param_groups:
            group["lr"] = cosine_warmup_lr(state.global_step, total_steps, warmup, peak_lr, min_ratio)

        with _bf16_autocast(device):
            out = model(x, valid_mask=valid, rollout_steps=rollout)
            loss = out["loss"] / accum

        loss.backward()
        losses.append(float(loss.item()) * accum)

        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params(model), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "update_target_ema"):
                model.update_target_ema()
            state.global_step += 1

        if state.global_step and state.global_step % log_every == 0:
            recent = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            print(f"[train] step={state.global_step} loss={recent:.4f} t={time.time()-t0:.1f}s")

    return {"loss": sum(losses) / max(1, len(losses))}


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        x = batch["wind"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with _bf16_autocast(device):
            out = model(x, valid_mask=valid)
        losses.append(float(out["loss"].item()))
    return {"loss": sum(losses) / max(1, len(losses))}


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return AdamW(
        trainable_params(model),
        lr=float(cfg["training"]["lr"]),
        betas=tuple(cfg["training"]["betas"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )


def save_ckpt(path: Path, model, optimizer, state: TrainState, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_val": state.best_val,
        "config": cfg,
    }, tmp)
    tmp.replace(path)
