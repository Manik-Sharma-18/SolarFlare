"""Flare classifier with within-cube temporal split.

Each cube split per-frame: first `split_frac` = train, trailing (1-split_frac) = eval.
Sidesteps cross-cube AR-identity leakage that broke `main_flare.py` holdout.
Reports per-cube + aggregate TSS, also splits by encoder-seen vs encoder-novel
cubes to expose any encoder distribution-shift effect.

Persistence baseline (label[t-1]) reported as TSS lower bound — within-cube
labels are heavily auto-correlated over 24h forward windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.v5.wind_probe_head import build_probe
from solarflare_data.flare_dataset import (
    FlareFrameDataset, filter_labeled_harps)
from training.flare_trainer import (
    FlareState, build_flare_optimizer, save_flare_ckpt,
    train_one_epoch, validate, _binary_metrics)
from utils.device import resolve_device
from utils.run_logger import log_jsonl, log_meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/flare_e30.yaml")
    p.add_argument("--split-frac", type=float, default=0.7)
    p.add_argument("--cls", default=None)
    p.add_argument("--kind", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _tss_at_thr(logits, y, thr):
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    p = (logits >= thr).astype(np.int64)
    tp = float(((p == 1) & (y == 1)).sum()); fn = float(((p == 0) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum()); tn = float(((p == 0) & (y == 0)).sum())
    return tp / max(tp + fn, 1.0) - fp / max(fp + tn, 1.0)


def _persistence_tss(label, keep):
    pred = np.zeros_like(label, dtype=bool); pred[1:] = label[:-1]
    k = keep.copy(); k[0] = False
    y = label[k].astype(np.int64); p = pred[k].astype(np.int64)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    tp = float(((p == 1) & (y == 1)).sum()); fn = float(((p == 0) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum()); tn = float(((p == 0) & (y == 0)).sum())
    return tp / max(tp + fn, 1.0) - fp / max(fp + tn, 1.0)


def _per_cube_eval(head, ds, thr_val, train_keep, device):
    per = ds.per_cube_arrays(); pc, AL, AY = {}, [], []
    for h in ds.harps:
        feat = per[h]["feat"]; lab = per[h]["label"]; v = per[h]["valid"]
        em = v & ~train_keep[h]
        if em.sum() == 0:
            continue
        with torch.no_grad():
            z = torch.from_numpy(feat[em]).float().to(device)
            logits = head(z).cpu().numpy()
        y = lab[em].astype(np.int64)
        m = _binary_metrics(logits, y)
        m["n"], m["n_pos"] = int(y.size), int(y.sum())
        m["tss_fixed"] = _tss_at_thr(logits, y, thr_val)
        m["tss_persist"] = _persistence_tss(lab, em)
        pc[h] = m; AL.append(logits); AY.append(y)
    AL_np = np.concatenate(AL) if AL else np.zeros(0)
    AY_np = np.concatenate(AY) if AY else np.zeros(0)
    agg = _binary_metrics(AL_np, AY_np)
    agg["n"], agg["n_pos"] = int(AY_np.size), int(AY_np.sum())
    agg["tss_fixed"] = _tss_at_thr(AL_np, AY_np, thr_val)
    return {"per_cube": pc, "aggregate": agg}


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cls: cfg["label"]["cls"] = args.cls
    if args.kind: cfg["head"]["kind"] = args.kind
    if args.max_epochs is not None: cfg["training"]["epochs"] = int(args.max_epochs)
    if args.out_dir: cfg["logging"]["out_dir"] = args.out_dir

    device = resolve_device(args.device or cfg.get("device", {}).get("prefer", "cpu"))
    print(f"[info] device={device}")
    data_dir = Path(cfg["data"]["data_dir"])
    feat_tag = cfg["encoder"]["feat_tag"]
    cls = cfg["label"]["cls"]; window_hr = int(cfg["label"]["window_hr"])

    enc_seen = sorted(set(cfg["data"]["train_cubes"]) | set(cfg["data"]["val_cubes"]))
    manifest = json.loads(Path(cfg["data"]["manifest_path"]).read_text())
    all_ids = sorted([e["harp_id"] for e in manifest["cubes"]])
    enc_novel = [h for h in all_ids if h not in enc_seen]
    all_harps = filter_labeled_harps(sorted(enc_seen + enc_novel), data_dir, cls, window_hr)
    enc_seen_lab = [h for h in all_harps if h in set(enc_seen)]
    enc_novel_lab = [h for h in all_harps if h not in set(enc_seen)]
    print(f"[info] cubes: seen={enc_seen_lab}  novel={enc_novel_lab}")

    sf = float(args.split_frac)
    train_ds = FlareFrameDataset(all_harps, data_dir, feat_tag, cls, window_hr, ("train", sf))
    val_ds = FlareFrameDataset(all_harps, data_dir, feat_tag, cls, window_hr, ("eval", sf))
    pn, tn = train_ds.pos_count(); pv, vn = val_ds.pos_count()
    print(f"[info] split={sf} train={tn} pos={pn} ({100*pn/max(tn,1):.1f}%) "
          f"eval={vn} pos={pv} ({100*pv/max(vn,1):.1f}%) dim={train_ds.dim}")

    train_keep = {}
    for ci, h in enumerate(train_ds.harps):
        v = train_ds._valids[ci]
        cut = int(round(sf * v.sum()))
        valid_ts = np.where(v)[0]
        tk = np.zeros_like(v); tk[valid_ts[:cut]] = True
        train_keep[h] = tk

    pos_weight = (tn - pn) / max(pn, 1)
    print(f"[info] pos_weight={pos_weight:.2f}")
    pw_t = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    kind = cfg["head"]["kind"]
    head = build_probe(kind, dim=int(cfg["encoder"]["dim"]),
                       hidden=int(cfg["head"].get("hidden", 256)),
                       dropout=float(cfg["head"].get("dropout", 0.1))).to(device)
    opt = build_flare_optimizer(head, cfg)

    out_dir = Path(cfg["logging"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.jsonl"
    log_meta(run_log, event="run_start", config=cfg, device=str(device),
             split_frac=sf, cls=cls, window_hr=window_hr,
             enc_seen=enc_seen_lab, enc_novel=enc_novel_lab,
             train_frames=tn, val_frames=vn, train_pos=pn, val_pos=pv,
             pos_weight=float(pos_weight))

    state = FlareState()
    train_fn = log_jsonl(run_log, "train")(train_one_epoch)
    val_fn = log_jsonl(run_log, "val")(validate)
    for ep in range(int(cfg["training"]["epochs"])):
        state.epoch = ep
        tr = train_fn(head, train_loader, opt, device, state, pw_t)
        vr = val_fn(head, val_loader, device, state, pw_t)
        print(f"[ep{ep:>3}] tr loss={tr['loss']:.4f} TSS={tr['tss_best']:.3f}  "
              f"val loss={vr['loss']:.4f} TSS={vr['tss_best']:.3f} AUC={vr['auc']:.3f}")
        if vr["tss_best"] > state.best_tss:
            state.best_tss = vr["tss_best"]
            save_flare_ckpt(out_dir / "best.pt", head, kind, int(cfg["encoder"]["dim"]),
                            cls, window_hr, float(pos_weight), vr["thr_best"], cfg, state, vr)
        save_flare_ckpt(out_dir / "last.pt", head, kind, int(cfg["encoder"]["dim"]),
                        cls, window_hr, float(pos_weight), vr["thr_best"], cfg, state, vr)

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(best["state_dict"]); head.eval()
    thr_val = float(best["thr_best"])
    res = _per_cube_eval(head, val_ds, thr_val, train_keep, device)
    payload = {"split_frac": sf, "cls": cls, "window_hr": window_hr,
               "thr_val": thr_val, "best_val_tss": float(state.best_tss),
               "per_cube": res["per_cube"], "aggregate": res["aggregate"],
               "enc_seen": enc_seen_lab, "enc_novel": enc_novel_lab}
    (out_dir / "temporal_eval.json").write_text(json.dumps(payload, indent=2))
    a = res["aggregate"]
    print(f"[done] best val TSS={state.best_tss:.3f} thr={thr_val:.3f}")
    print(f"[done] AGG  n={a['n']} pos={a['n_pos']} AUC={a['auc']:.3f} "
          f"TSS={a['tss_best']:.3f} TSS@thr={a['tss_fixed']:.3f}")
    for h, m in res["per_cube"].items():
        f = "novel" if h in enc_novel_lab else "seen"
        print(f"  [{f:>5}] {h:>14} n={m['n']:>5} pos={m['n_pos']:>4} "
              f"AUC={m['auc']:.3f} TSS={m['tss_best']:.3f} @thr={m['tss_fixed']:.3f} "
              f"persist={m.get('tss_persist', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
