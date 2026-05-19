"""Per-cube affine calibration of frozen probe predictions.

Tests "scale-only mismatch" hypothesis: features encode correct temporal
structure but absolute scale drifts per cube (encoder trained with per-cube
z-score on inputs). Fits y = a*pred + b on leading 30% of each cube; reports
metrics on remaining 70%.

Inputs: same probe ckpts as probe_eval.
Outputs: outputs_probe/eval/probe_calibration.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.v5.wind_probe_head import build_probe
from solarflare_data.probe_dataset import (
    ProbeFrameDataset, TargetStats, split_cubes_for_probe,
)
from scripts.probe_eval_metrics import predict_cube, _r2, _pearson, _mae, _mape


def load_probe(path: Path, device: torch.device):
    p = torch.load(str(path), map_location=device, weights_only=False)
    head = build_probe(p["kind"], dim=int(p["dim"]),
                       hidden=int(p["config"]["probe"].get("hidden", 256)),
                       dropout=float(p["config"]["probe"].get("dropout", 0.1)))
    head.load_state_dict(p["state_dict"]); head.to(device).eval()
    return head, TargetStats.from_dict(p["stats"])


def fit_affine(pred_cal: np.ndarray, y_cal: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(pred_cal) & np.isfinite(y_cal) & (y_cal > 0)
    if m.sum() < 5: return 1.0, 0.0
    p, y = pred_cal[m], y_cal[m]
    a, b = np.polyfit(p, y, 1)
    return float(a), float(b)


def calibrate_cube(harp: str, ds: ProbeFrameDataset, head, stats, device,
                   cal_frac: float = 0.30) -> dict:
    arrs = ds.per_cube_arrays()[harp]
    feat, y, valid = arrs["feat"], arrs["y"], arrs["valid"]
    pred = predict_cube(head, feat, valid, stats, device)
    y_eval = y.copy(); y_eval[~np.isfinite(y_eval)] = np.nan; y_eval[y_eval <= 0] = np.nan

    valid_idx = np.where(valid)[0]
    if valid_idx.size < 10:
        return {"harp": harp, "n_valid": int(valid_idx.size), "skipped": True}
    n_cal = max(5, int(round(valid_idx.size * cal_frac)))
    cal_mask = np.zeros_like(valid)
    cal_mask[valid_idx[:n_cal]] = True
    eval_mask = valid & (~cal_mask)

    a, b = fit_affine(pred[cal_mask], y_eval[cal_mask])
    pred_cal = pred * a + b

    yt = y_eval[eval_mask]; yp_raw = pred[eval_mask]; yp_cal = pred_cal[eval_mask]
    return {
        "harp": harp, "n_valid": int(valid_idx.size), "n_cal": n_cal,
        "n_eval": int(eval_mask.sum()), "a": a, "b": b,
        "raw":  {"r2": _r2(yt, yp_raw), "r": _pearson(yt, yp_raw),
                  "mae": _mae(yt, yp_raw), "mape": _mape(yt, yp_raw)},
        "cal":  {"r2": _r2(yt, yp_cal), "r": _pearson(yt, yp_cal),
                  "mae": _mae(yt, yp_cal), "mape": _mape(yt, yp_cal)},
    }


def write_md(out_path: Path, results_by_kind: dict[str, list[dict]]) -> None:
    lines = ["# Per-cube affine calibration of probe predictions\n"]
    lines.append("Fit y = a·pred + b on leading 30% of each cube; eval on remaining 70%.")
    lines.append("Tests scale-mismatch hypothesis. Compares raw probe vs calibrated.\n")
    for kind, rows in results_by_kind.items():
        lines.append(f"\n## Probe = {kind}\n")
        lines.append("| Cube | n_eval | a | b | raw R² / MAPE | **cal R² / MAPE** |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            if r.get("skipped"):
                lines.append(f"| {r['harp']} | — | — | — | skipped | skipped |")
                continue
            lines.append(
                f"| {r['harp']} | {r['n_eval']} | {r['a']:+.3f} | {r['b']:+.2e} "
                f"| {r['raw']['r2']:+.3f} / {r['raw']['mape']:.1f}% "
                f"| **{r['cal']['r2']:+.3f} / {r['cal']['mape']:.1f}%** |"
            )
        # Aggregate medAPE.
        vals = [r["cal"]["mape"] for r in rows if not r.get("skipped") and np.isfinite(r["cal"]["mape"])]
        if vals:
            lines.append(f"\n**Median across cubes (calibrated medAPE):** {np.median(vals):.1f}%\n")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/probe_e09.yaml")
    ap.add_argument("--linear", default="outputs_probe/linear/best.pt")
    ap.add_argument("--mlp", default="outputs_probe/mlp/best.pt")
    ap.add_argument("--out-dir", default="outputs_probe/eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cal-frac", type=float, default=0.30)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["data"]["data_dir"])
    manifest_path = Path(cfg["data"]["manifest_path"])
    feat_tag = cfg["encoder"]["feat_tag"]
    abs_value = bool(cfg["data"]["target"]["abs_value"])
    clip = float(cfg["data"]["target"]["clip"])

    if cfg["data"].get("train_cubes") and cfg["data"].get("val_cubes"):
        train_harps = sorted(cfg["data"]["train_cubes"])
        val_harps = sorted(cfg["data"]["val_cubes"])
    else:
        train_harps, val_harps = split_cubes_for_probe(
            manifest_path, val_frac=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["data"]["split_seed"]),
            allowlist=cfg["data"].get("cube_allowlist"),
        )
    all_ids = sorted(json.loads(manifest_path.read_text())["cubes"], key=lambda e: e["harp_id"])
    novel_harps = [e["harp_id"] for e in all_ids
                   if e["harp_id"] not in set(train_harps) | set(val_harps)]
    target_harps = val_harps + novel_harps

    results = {}
    for kind, p in (("linear", args.linear), ("mlp", args.mlp)):
        if not Path(p).exists(): continue
        head, stats = load_probe(Path(p), device)
        ds = ProbeFrameDataset(target_harps, data_dir, feat_tag, abs_value, clip, stats)
        results[kind] = [calibrate_cube(h, ds, head, stats, device, args.cal_frac)
                          for h in target_harps]
        print(f"[{kind}] {len(results[kind])} cubes calibrated")
    write_md(out_dir / "probe_calibration.md", results)
    (out_dir / "probe_calibration.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[done] {out_dir}/probe_calibration.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
