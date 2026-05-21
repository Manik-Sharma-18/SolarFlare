"""Probe entry: load cached encoder features → train LinearProbe / MLPProbe.

Splits respect the encoder training cube allowlist. Probe-train cubes overlap
encoder-train cubes (cached features only — encoder is frozen). Probe-val cube
was the encoder's held-out val cube. Off-distribution cubes (the other 17 in
the manifest) are evaluated by `scripts/probe_eval.py`, not here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from models.v5.wind_probe_head import build_probe
from solarflare_data.probe_dataset import (
    ProbeFrameDataset,
    fit_target_stats,
    split_cubes_for_probe,
)
from training.probe_trainer import (
    ProbeState,
    build_probe_optimizer,
    save_probe_ckpt,
    train_one_epoch,
    validate,
    write_split_record,
)
from utils.device import resolve_device
from utils.run_logger import log_jsonl, log_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/probe_e09.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--kind", default=None, help="Override probe.kind ('linear' or 'mlp').")
    p.add_argument("--out-dir", default=None, help="Override logging.out_dir (e.g. outputs_probe/linear).")
    return p.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.kind: cfg["probe"]["kind"] = args.kind
    if args.out_dir: cfg["logging"]["out_dir"] = args.out_dir
    if args.max_epochs is not None: cfg["training"]["epochs"] = int(args.max_epochs)

    prefer = args.device or cfg.get("device", {}).get("prefer", "cpu")
    device = resolve_device(prefer)
    print(f"[info] device={device}")

    manifest_path = Path(cfg["data"]["manifest_path"])
    data_dir = Path(cfg["data"]["data_dir"])
    feat_tag = cfg["encoder"]["feat_tag"]
    abs_value = bool(cfg["data"]["target"]["abs_value"])
    clip = float(cfg["data"]["target"]["clip"])

    if cfg["data"].get("train_cubes") and cfg["data"].get("val_cubes"):
        train_harps = sorted(cfg["data"]["train_cubes"])
        val_harps = sorted(cfg["data"]["val_cubes"])
    else:
        train_harps, val_harps = split_cubes_for_probe(
            manifest_path,
            val_frac=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["data"]["split_seed"]),
            allowlist=cfg["data"].get("cube_allowlist"),
        )
    all_ids = sorted(json.loads(manifest_path.read_text())["cubes"], key=lambda e: e["harp_id"])
    novel_harps = [e["harp_id"] for e in all_ids
                   if e["harp_id"] not in set(train_harps) | set(val_harps)]
    print(f"[info] split — train={train_harps} val={val_harps}")
    print(f"[info] novel cubes (encoder never saw): n={len(novel_harps)}")

    # Fit target stats on TRAIN cubes only (no val/novel leakage).
    stats = fit_target_stats(train_harps, data_dir, abs_value=abs_value, clip=clip)
    print(f"[info] target log-z stats — μ={stats.mu:.4f} σ={stats.sigma:.4f}")

    train_ds = ProbeFrameDataset(train_harps, data_dir, feat_tag, abs_value, clip, stats)
    val_ds = ProbeFrameDataset(val_harps, data_dir, feat_tag, abs_value, clip, stats)
    print(f"[info] train_frames={len(train_ds)} val_frames={len(val_ds)} dim={train_ds.dim}")

    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    kind = cfg["probe"]["kind"]
    probe = build_probe(kind, dim=int(cfg["encoder"]["dim"]),
                        hidden=int(cfg["probe"].get("hidden", 256)),
                        dropout=float(cfg["probe"].get("dropout", 0.1))).to(device)
    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[info] probe={kind} trainable_params={n_params}")

    optimizer = build_probe_optimizer(probe, cfg)

    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split_record(out_dir, train_harps, val_harps, novel_harps, stats)

    run_log = out_dir / "run.jsonl"
    log_meta(run_log, event="run_start", config=cfg, device=str(device),
             train_frames=len(train_ds), val_frames=len(val_ds), n_params=n_params,
             train_harps=train_harps, val_harps=val_harps, novel_harps=novel_harps)

    state = ProbeState()
    train_fn = log_jsonl(run_log, "train")(train_one_epoch)
    val_fn = log_jsonl(run_log, "val")(validate)
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(epochs):
        state.epoch = epoch
        tr = train_fn(probe, train_loader, optimizer, device, state)
        vr = val_fn(probe, val_loader, device, state)
        msg = (f"[epoch {epoch}] train_mse={tr['loss']:.4f} val_mse={vr['loss']:.4f}"
               f"  r2_norm tr={tr['r2_norm']:.3f} val={vr['r2_norm']:.3f}")
        print(msg)
        if vr["loss"] < state.best_val:
            state.best_val = vr["loss"]
            save_probe_ckpt(out_dir / "best.pt", probe, kind, train_ds.dim,
                            stats, cfg, state, vr["loss"])
            log_meta(run_log, event="best_ckpt", epoch=epoch, val_loss=float(vr["loss"]))
        save_probe_ckpt(out_dir / "last.pt", probe, kind, train_ds.dim,
                        stats, cfg, state, vr["loss"])
    log_meta(run_log, event="run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
