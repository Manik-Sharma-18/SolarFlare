"""V5 entry point (Path B): load config → build JEPA model + dataloaders → train.

Usage:
    python main_v5.py --config configs/v5_path_a.yaml
    python main_v5.py --config configs/v5_sanity.yaml --max-epochs 1

# CUDA-5060ti-validated
# DataLoaders set pin_memory=True here; non_blocking=True transfers live in
# training/jepa_trainer.py (lines 96-97, 129-130). bf16 autocast in trainer,
# no fp16 GradScaler.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from models.v5 import V5JEPAModel
from models.v5.jepa_model import JEPAConfig
from solarflare_data.zarr_dataset import BucketedShapeSampler, ZarrCubeDataset
from training.jepa_trainer import (
    TrainState,
    build_optimizer,
    save_ckpt,
    train_one_epoch,
    validate,
)
from utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v5_path_a.yaml")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override config epochs (useful for smoke).")
    p.add_argument("--device", default=None,
                   help="Override cfg device.prefer (cuda|mps|cpu). Used by launch_slot.sh.")
    return p.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def split_cubes_by_harp(
    manifest_path: str,
    val_frac: float,
    seed: int,
    allowlist: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Cube-level holdout per docs/V5_JEPA/06_data.md §11.5 item 10.

    Prevents AR-identity leakage: each harp_id lands fully in train or val.
    `allowlist` restricts the candidate set (useful for sanity / debug runs).
    """
    entries = json.loads(Path(manifest_path).read_text())["cubes"]
    ids = sorted(e["harp_id"] for e in entries)
    if allowlist is not None:
        keep = set(allowlist)
        ids = [i for i in ids if i in keep]
    if len(ids) < 2:
        raise ValueError(f"Need ≥2 cubes after allowlist filter; got {ids}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    manifest = cfg["data"]["manifest_path"]
    val_frac = float(cfg["data"].get("val_fraction", 0.2))
    seed = int(cfg["data"].get("split_seed", 0))
    allowlist = cfg["data"].get("cube_allowlist")
    train_ids, val_ids = split_cubes_by_harp(manifest, val_frac, seed, allowlist=allowlist)
    print(f"[info] cube split — train={train_ids} val={val_ids}")

    train_ds = ZarrCubeDataset(
        manifest_path=manifest,
        t_in=cfg["data"]["t_in"],
        t_out=cfg["data"]["t_out"],
        augment=True,
        cube_subset=train_ids,
    )
    val_ds = ZarrCubeDataset(
        manifest_path=manifest,
        t_in=cfg["data"]["t_in"],
        t_out=cfg["data"]["t_out"],
        augment=False,
        cube_subset=val_ids,
    )
    nw = int(cfg["data"]["num_workers"])
    bs = int(cfg["training"]["batch_size"])
    train_sampler = BucketedShapeSampler(train_ds, batch_size=bs)
    val_sampler = BucketedShapeSampler(val_ds, batch_size=bs, shuffle=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=nw, pin_memory=True)
    print(f"[info] train_windows={len(train_ds)} val_windows={len(val_ds)}")
    return train_loader, val_loader


def build_model(cfg: dict) -> V5JEPAModel:
    enc = cfg["model"]["encoder"]
    pred = cfg["model"]["predictor"]
    j = JEPAConfig(
        t_in=int(cfg["data"]["t_in"]),
        t_out=int(cfg["data"]["t_out"]),
        patch_size=int(cfg["model"]["input_adapter"]["patch_size"]),
        cadence_min=float(cfg["data"]["cadence_min"]),
        pixel_scale_mm=float(cfg["data"]["pixel_scale_mm"]),
        encoder_dim=int(enc["dim"]),
        encoder_layers=int(enc["layers"]),
        encoder_heads=int(enc["heads"]),
        encoder_mlp_ratio=int(enc["mlp_ratio"]),
        predictor_hidden=int(pred["hidden"]),
        predictor_layers=int(pred["layers"]),
        predictor_heads=int(pred["heads"]),
        predictor_mlp_ratio=int(pred["mlp_ratio"]),
        dropout=float(pred["dropout"]),
        drop_path=float(pred["drop_path"]),
        target_ema_decay=float(cfg["training"]["target_ema_decay"]),
        grad_checkpoint=bool(cfg["training"].get("grad_checkpoint", False)),
        valid_token_threshold=float(
            cfg["model"].get("masking", {}).get("valid_token_threshold", 0.5)
        ),
    )
    return V5JEPAModel(cfg=j)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    prefer = args.device or cfg.get("device", {}).get("prefer", "cuda")
    device = resolve_device(prefer)
    print(f"[info] device={device}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"[info] train batches={len(train_loader)} val batches={len(val_loader)}")

    torch.manual_seed(int(cfg["data"].get("split_seed", 0)))
    model = build_model(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[info] params trainable={n_train/1e6:.1f}M total={n_total/1e6:.1f}M")

    compile_mode = cfg["training"].get("compile", "off")
    if compile_mode != "off" and device.type == "cuda":
        print(f"[info] torch.compile mode={compile_mode}")
        model.encoder = torch.compile(model.encoder, dynamic=True, mode=compile_mode)
        model.predictor = torch.compile(model.predictor, dynamic=True, mode=compile_mode)
        model.target_encoder = torch.compile(model.target_encoder, dynamic=True, mode=compile_mode)

    optimizer = build_optimizer(model, cfg)
    state = TrainState()

    epochs = args.max_epochs if args.max_epochs is not None else int(cfg["training"]["epochs"])
    total_steps = max(1, epochs * len(train_loader) // int(cfg["training"]["grad_accum_steps"]))
    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        state.epoch = epoch
        tr = train_one_epoch(model, train_loader, optimizer, cfg, state, device, total_steps)
        vr = validate(model, val_loader, device, cfg)
        print(f"[epoch {epoch}] train_loss={tr['loss']:.4f} val_loss={vr['loss']:.4f}")
        if vr["loss"] < state.best_val:
            state.best_val = vr["loss"]
            save_ckpt(out_dir / "best.pt", model, optimizer, state, cfg)
        save_ckpt(out_dir / "last.pt", model, optimizer, state, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
