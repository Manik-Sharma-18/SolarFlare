"""XGBoost ceiling check on cached E09 features.

Tests "linear capacity is the bottleneck" hypothesis. Same train/val/novel
splits as main_probe.py. Trains XGBRegressor on log-z normalized target,
evaluates R² / Pearson r / MAE / medAPE in original units.

Output: outputs_probe/eval/probe_xgb.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

from solarflare_data.probe_dataset import (
    ProbeFrameDataset, fit_target_stats, split_cubes_for_probe,
)
from scripts.probe_eval_metrics import _r2, _pearson, _mae, _mape


def stack_frames(ds: ProbeFrameDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack all valid (z, y_norm, y_orig) frames + cube tag per row."""
    cubes = ds.per_cube_arrays()
    Z, Y_NORM, Y_ORIG, TAG = [], [], [], []
    for h, d in cubes.items():
        feat, y, valid = d["feat"], d["y"], d["valid"]
        idx = np.where(valid)[0]
        Z.append(feat[idx])
        Y_ORIG.append(y[idx])
        Y_NORM.append(ds.stats.normalize(y[idx]))
        TAG.extend([h] * idx.size)
    return np.concatenate(Z), np.concatenate(Y_NORM), np.concatenate(Y_ORIG), TAG


def metrics_block(y_true: np.ndarray, y_pred_orig: np.ndarray) -> dict:
    return {"r2": _r2(y_true, y_pred_orig), "r": _pearson(y_true, y_pred_orig),
            "mae": _mae(y_true, y_pred_orig), "mape": _mape(y_true, y_pred_orig)}


def evaluate_split(model: xgb.XGBRegressor, ds: ProbeFrameDataset,
                   harps: list[str], stats) -> dict:
    cubes = ds.per_cube_arrays()
    per_cube = {}
    yt_all, yp_all = [], []
    for h in harps:
        d = cubes[h]
        feat, y, valid = d["feat"], d["y"], d["valid"]
        y_eval = y.copy(); y_eval[~valid] = np.nan; y_eval[y_eval <= 0] = np.nan
        if not valid.any():
            per_cube[h] = {"skipped": True}; continue
        pred_norm = model.predict(feat[valid])
        pred = np.full_like(y_eval, np.nan)
        pred[valid] = stats.invert(pred_norm.astype(np.float32))
        per_cube[h] = metrics_block(y_eval, pred)
        yt_all.append(y_eval); yp_all.append(pred)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    return {"per_cube": per_cube, "aggregate": metrics_block(yt, yp)}


def write_md(out_path: Path, results: dict, params: dict) -> None:
    lines = [f"# XGBoost probe on E09 features\n",
             f"Params: {json.dumps(params)}\n",
             "Metrics in original ⟨|wind|⟩ units. medAPE = median APE.\n",
             "## Aggregate\n",
             "| Eval set | n cubes | R² | Pearson r | MAE | medAPE |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["aggregate"]
        lines.append(f"| {name} | {len(r['per_cube'])} | {a['r2']:+.3f} | {a['r']:+.3f} | {a['mae']:.2e} | {a['mape']:.1f}% |")
    lines.append("\n## Per-cube\n")
    for name, r in results.items():
        lines.append(f"\n### {name}\n")
        lines.append("| Cube | R² | r | MAE | medAPE |")
        lines.append("|---|---|---|---|---|")
        for h, m in r["per_cube"].items():
            if m.get("skipped"): continue
            lines.append(f"| {h} | {m['r2']:+.3f} | {m['r']:+.3f} | {m['mae']:.2e} | {m['mape']:.1f}% |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/probe_e09.yaml")
    ap.add_argument("--out-dir", default="outputs_probe/eval")
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--early-stop", type=int, default=20)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["data"]["data_dir"])
    manifest_path = Path(cfg["data"]["manifest_path"])
    feat_tag = cfg["encoder"]["feat_tag"]
    abs_value = bool(cfg["data"]["target"]["abs_value"])
    clip = float(cfg["data"]["target"]["clip"])

    train_harps, val_harps = split_cubes_for_probe(
        manifest_path, val_frac=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["data"]["split_seed"]),
        allowlist=cfg["data"].get("cube_allowlist"),
    )
    all_ids = sorted(json.loads(manifest_path.read_text())["cubes"], key=lambda e: e["harp_id"])
    novel = [e["harp_id"] for e in all_ids
             if e["harp_id"] not in set(train_harps) | set(val_harps)]

    stats = fit_target_stats(train_harps, data_dir, abs_value=abs_value, clip=clip)
    print(f"[stats] mu={stats.mu:.4f} sigma={stats.sigma:.4f}")

    train_ds = ProbeFrameDataset(train_harps, data_dir, feat_tag, abs_value, clip, stats)
    val_ds = ProbeFrameDataset(val_harps, data_dir, feat_tag, abs_value, clip, stats)
    novel_ds = ProbeFrameDataset(novel, data_dir, feat_tag, abs_value, clip, stats)

    Xtr, Ytr, _, _ = stack_frames(train_ds)
    Xv, Yv, _, _ = stack_frames(val_ds)
    print(f"[data] train={Xtr.shape} val={Xv.shape}")

    params = {"n_estimators": args.n_trees, "max_depth": args.max_depth,
              "learning_rate": args.lr, "subsample": 0.8, "colsample_bytree": 0.8,
              "reg_lambda": 1.0, "objective": "reg:squarederror",
              "early_stopping_rounds": args.early_stop, "verbosity": 0}
    model = xgb.XGBRegressor(**params)
    model.fit(Xtr, Ytr, eval_set=[(Xv, Yv)], verbose=False)
    print(f"[train] best_iter={model.best_iteration} best_score={model.best_score:.4f}")

    results = {
        "encoder-val (harp_51)": evaluate_split(model, val_ds, val_harps, stats),
        "novel cubes (off-dist)": evaluate_split(model, novel_ds, novel, stats),
        "encoder-train (sanity)": evaluate_split(model, train_ds, train_harps, stats),
    }
    write_md(out_dir / "probe_xgb.md", results, params)
    (out_dir / "probe_xgb.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[done] {out_dir}/probe_xgb.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
