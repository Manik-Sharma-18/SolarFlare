"""Evaluate trained flare-classifier head on holdout (encoder-novel) cubes.

Loads `best.pt` from outputs_flare/<run>/, applies head to cached features,
computes per-cube + aggregate TSS / AUC / pos_rate. Threshold from best.pt is
val-optimal (frozen); also reports per-cube-optimal TSS as upper bound.

Compares to persistence baseline: lag-1 label (was there an M+ in the prior
window) — TSS lower bound for sanity.

Writes:
- outputs_flare/<run>/holdout_metrics.json
- outputs_flare/<run>/holdout_metrics.md  (table for thesis)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.v5.wind_probe_head import build_probe
from solarflare_data.flare_dataset import FlareFrameDataset, filter_labeled_harps


def _metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return {"tss_best": float("nan"), "thr_best": 0.0, "auc": float("nan"),
                "pos_rate": float(y.mean()) if y.size else float("nan"),
                "n": int(y.size), "n_pos": int(y.sum())}
    order = np.argsort(-logits)
    y_s = y[order].astype(np.float64)
    n_pos = float(y.sum()); n_neg = float(y.size - y.sum())
    tp = np.cumsum(y_s); fp = np.cumsum(1.0 - y_s)
    tpr = tp / n_pos; fpr = fp / n_neg
    tss = tpr - fpr
    k = int(np.argmax(tss))
    fpr_aug = np.concatenate([[0.0], fpr, [1.0]])
    tpr_aug = np.concatenate([[0.0], tpr, [1.0]])
    auc = float(np.trapezoid(tpr_aug, fpr_aug))
    return {"tss_best": float(tss[k]), "thr_best": float(logits[order][k]),
            "auc": auc, "pos_rate": float(y.mean()),
            "n": int(y.size), "n_pos": int(y.sum()),
            "tpr_at_best": float(tpr[k]), "fpr_at_best": float(fpr[k])}


def _metrics_at_thr(logits: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (logits >= thr).astype(np.int64)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return {"tss_fixed": float("nan"), "tpr_fixed": float("nan"), "fpr_fixed": float("nan")}
    tp = float(((pred == 1) & (y == 1)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1.0)
    fpr = fp / max(fp + tn, 1.0)
    return {"tss_fixed": float(tpr - fpr), "tpr_fixed": float(tpr), "fpr_fixed": float(fpr)}


def _persistence_tss(label: np.ndarray, valid: np.ndarray) -> dict:
    """Persistence baseline: predict label[t] = label[t-1]. Drop t=0."""
    if label.size < 2:
        return {"tss_persist": float("nan")}
    pred = np.zeros_like(label, dtype=bool)
    pred[1:] = label[:-1]
    keep = valid.copy(); keep[0] = False
    y = label[keep].astype(np.int64)
    p = pred[keep].astype(np.int64)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return {"tss_persist": float("nan"), "n_persist": int(y.size)}
    tp = float(((p == 1) & (y == 1)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum())
    tn = float(((p == 0) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1.0)
    fpr = fp / max(fp + tn, 1.0)
    return {"tss_persist": float(tpr - fpr), "n_persist": int(y.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pt in outputs_flare/<run>/")
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]; kind = ckpt["kind"]; dim = int(ckpt["dim"])
    cls = ckpt["cls"]; window_hr = int(ckpt["window_hr"])
    thr_val = float(ckpt["thr_best"])
    feat_tag = cfg["encoder"]["feat_tag"]
    train_cubes = set(cfg["data"]["train_cubes"])
    val_cubes = set(cfg["data"]["val_cubes"])

    manifest = json.loads(Path(args.manifest).read_text())
    all_ids = [e["harp_id"] for e in manifest["cubes"]]
    novel = sorted([h for h in all_ids if h not in train_cubes and h not in val_cubes])
    novel = filter_labeled_harps(novel, Path(args.data_dir), cls, window_hr)
    print(f"[info] cls={cls} window={window_hr}h  novel cubes (labeled): {novel}")

    head = build_probe(kind, dim=dim,
                       hidden=int(cfg["head"].get("hidden", 256)),
                       dropout=float(cfg["head"].get("dropout", 0.1)))
    head.load_state_dict(ckpt["state_dict"]); head.eval()

    ds = FlareFrameDataset(novel, args.data_dir, feat_tag, cls, window_hr)
    per = ds.per_cube_arrays()

    per_cube: dict[str, dict] = {}
    agg_logits, agg_y = [], []
    for h in novel:
        feat = per[h]["feat"]; lab = per[h]["label"]; v = per[h]["valid"]
        with torch.no_grad():
            z = torch.from_numpy(feat[v]).float()
            logits = head(z).numpy()
        y = lab[v].astype(np.int64)
        m_self = _metrics(logits, y)
        m_fix = _metrics_at_thr(logits, y, thr_val)
        m_per = _persistence_tss(lab, v)
        per_cube[h] = {**m_self, **m_fix, **m_per}
        agg_logits.append(logits); agg_y.append(y)
        print(f"  {h:>14}  n={m_self['n']:>5} pos={m_self['n_pos']:>4} "
              f"AUC={m_self['auc']:.3f}  TSS(self)={m_self['tss_best']:.3f}  "
              f"TSS(@val_thr)={m_fix['tss_fixed']:.3f}  TSS(persist)={m_per.get('tss_persist', float('nan')):.3f}")

    agg_logits_np = np.concatenate(agg_logits) if agg_logits else np.zeros(0)
    agg_y_np = np.concatenate(agg_y) if agg_y else np.zeros(0)
    agg_self = _metrics(agg_logits_np, agg_y_np)
    agg_fix = _metrics_at_thr(agg_logits_np, agg_y_np, thr_val)
    print(f"  {'AGGREGATE':>14}  n={agg_self['n']:>5} pos={agg_self['n_pos']:>4} "
          f"AUC={agg_self['auc']:.3f}  TSS(self)={agg_self['tss_best']:.3f}  "
          f"TSS(@val_thr)={agg_fix['tss_fixed']:.3f}")

    out = Path(args.ckpt).parent
    payload = {"per_cube": per_cube, "aggregate": {**agg_self, **agg_fix},
               "val_thr": thr_val, "cls": cls, "window_hr": window_hr,
               "ckpt": str(args.ckpt)}
    (out / "holdout_metrics.json").write_text(json.dumps(payload, indent=2))

    lines = [f"# Flare classifier holdout — cls={cls}, window={window_hr}h",
             f"ckpt: `{args.ckpt}`  val-optimal thr: {thr_val:.4f}", "",
             "| Cube | n | pos | pos% | AUC | TSS(self) | TSS@val_thr | TSS(persist) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for h, m in per_cube.items():
        pc = 100 * m["pos_rate"] if np.isfinite(m["pos_rate"]) else float("nan")
        lines.append(
            f"| {h} | {m['n']} | {m['n_pos']} | {pc:.1f}% | "
            f"{m['auc']:.3f} | {m['tss_best']:.3f} | "
            f"{m['tss_fixed']:.3f} | {m.get('tss_persist', float('nan')):.3f} |")
    lines += ["",
              f"**Aggregate:** n={agg_self['n']} pos={agg_self['n_pos']} "
              f"AUC={agg_self['auc']:.3f} TSS(self)={agg_self['tss_best']:.3f} "
              f"TSS(@val_thr)={agg_fix['tss_fixed']:.3f}",
              "",
              "vs Prithish XGBoost baseline: TSS 0.5–0.6 on full unseen holdout."]
    (out / "holdout_metrics.md").write_text("\n".join(lines))
    print(f"[done] → {out / 'holdout_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
