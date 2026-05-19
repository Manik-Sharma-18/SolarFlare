"""Flare-classifier entry: load cached encoder features → train binary head.

Splits respect the encoder cube allowlist (E30 thesis_curated: 13/3/5).
Date-named cubes (harp_may2024, harp_nov2025) auto-dropped — no NOAA mapping
yet, so no label files. Holdout cubes (encoder-novel) evaluated by
`scripts/flare_eval.py` (TBD), not here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from models.v5.wind_probe_head import build_probe
from solarflare_data.flare_dataset import (
    FlareFrameDataset,
    compute_pos_weight,
    filter_labeled_harps,
    write_split_record,
)
from training.flare_trainer import (
    FlareState,
    build_flare_optimizer,
    save_flare_ckpt,
    train_one_epoch,
    validate,
)
from utils.device import resolve_device
from utils.run_logger import log_jsonl, log_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/flare_e30.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--kind", default=None, help="Override head.kind ('linear' or 'mlp').")
    p.add_argument("--cls", default=None, help="Override label.cls ('C', 'M', or 'X').")
    p.add_argument("--out-dir", default=None, help="Override logging.out_dir.")
    return p.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.kind: cfg["head"]["kind"] = args.kind
    if args.cls: cfg["label"]["cls"] = args.cls
    if args.out_dir: cfg["logging"]["out_dir"] = args.out_dir
    if args.max_epochs is not None: cfg["training"]["epochs"] = int(args.max_epochs)

    prefer = args.device or cfg.get("device", {}).get("prefer", "cpu")
    device = resolve_device(prefer)
    print(f"[info] device={device}")

    data_dir = Path(cfg["data"]["data_dir"])
    feat_tag = cfg["encoder"]["feat_tag"]
    cls = cfg["label"]["cls"]
    window_hr = int(cfg["label"]["window_hr"])

    train_harps = sorted(cfg["data"]["train_cubes"])
    val_harps = sorted(cfg["data"]["val_cubes"])
    train_harps = filter_labeled_harps(train_harps, data_dir, cls, window_hr)
    val_harps = filter_labeled_harps(val_harps, data_dir, cls, window_hr)

    manifest_path = Path(cfg["data"]["manifest_path"])
    all_ids = sorted(json.loads(manifest_path.read_text())["cubes"], key=lambda e: e["harp_id"])
    novel_harps = [e["harp_id"] for e in all_ids
                   if e["harp_id"] not in set(train_harps) | set(val_harps)]
    novel_harps = filter_labeled_harps(novel_harps, data_dir, cls, window_hr)
    print(f"[info] split — train={train_harps}")
    print(f"[info]         val={val_harps}")
    print(f"[info]         novel_labeled={novel_harps}")

    train_ds = FlareFrameDataset(train_harps, data_dir, feat_tag, cls, window_hr)
    val_ds = FlareFrameDataset(val_harps, data_dir, feat_tag, cls, window_hr)
    pn, tn = train_ds.pos_count()
    pv, vn = val_ds.pos_count()
    print(f"[info] train_frames={tn} pos={pn} ({100*pn/max(tn,1):.1f}%)"
          f"   val_frames={vn} pos={pv} ({100*pv/max(vn,1):.1f}%)   dim={train_ds.dim}")

    pos_weight = compute_pos_weight(train_harps, data_dir, cls, window_hr)
    print(f"[info] pos_weight={pos_weight:.2f}")
    pw_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    kind = cfg["head"]["kind"]
    head = build_probe(kind, dim=int(cfg["encoder"]["dim"]),
                       hidden=int(cfg["head"].get("hidden", 256)),
                       dropout=float(cfg["head"].get("dropout", 0.1))).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[info] head={kind} trainable_params={n_params}")

    optimizer = build_flare_optimizer(head, cfg)
    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split_record(out_dir, train_harps, val_harps, novel_harps, cls, window_hr, pos_weight)

    run_log = out_dir / "run.jsonl"
    log_meta(run_log, event="run_start", config=cfg, device=str(device),
             train_frames=tn, val_frames=vn, train_pos=pn, val_pos=pv,
             pos_weight=pos_weight, n_params=n_params,
             train_harps=train_harps, val_harps=val_harps, novel_harps=novel_harps)

    state = FlareState()
    train_fn = log_jsonl(run_log, "train")(train_one_epoch)
    val_fn = log_jsonl(run_log, "val")(validate)
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(epochs):
        state.epoch = epoch
        tr = train_fn(head, train_loader, optimizer, device, state, pw_t)
        vr = val_fn(head, val_loader, device, state, pw_t)
        msg = (f"[ep{epoch:>3}] loss tr={tr['loss']:.4f} val={vr['loss']:.4f}"
               f"  TSS tr={tr['tss_best']:.3f} val={vr['tss_best']:.3f}"
               f"  AUC tr={tr['auc']:.3f} val={vr['auc']:.3f}")
        print(msg)
        if vr["tss_best"] > state.best_tss:
            state.best_tss = vr["tss_best"]
            save_flare_ckpt(out_dir / "best.pt", head, kind, int(cfg["encoder"]["dim"]),
                            cls, window_hr, pos_weight, vr["thr_best"], cfg, state, vr)
            log_meta(run_log, event="best_ckpt", epoch=epoch,
                     val_tss=float(vr["tss_best"]), thr=float(vr["thr_best"]))
        save_flare_ckpt(out_dir / "last.pt", head, kind, int(cfg["encoder"]["dim"]),
                        cls, window_hr, pos_weight, vr["thr_best"], cfg, state, vr)
    log_meta(run_log, event="run_end", best_val_tss=float(state.best_tss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
