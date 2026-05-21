"""Rate-of-change probe: predict Δ⟨|wind|⟩(t) = y(t) - y(t-1).

Persistence baseline for Δ is trivially 0 (predict no change). Probe getting
positive R² here directly demonstrates features encode frame-to-frame change
beyond temporal autocorrelation.

Trains both linear (sklearn) and XGBoost on cached E09 features. Same splits
as main_probe.py. Δ is small in absolute units; we z-score it for training,
report metrics in original Δ units (no log transform — Δ can be negative).

Output: outputs_probe/eval/probe_delta.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml
from sklearn.linear_model import Ridge

from solarflare_data.probe_dataset import ProbeFrameDataset, fit_target_stats, split_cubes_for_probe
from scripts.probe_eval_metrics import _r2, _pearson, _mae


def build_delta_arrays(harps: list[str], data_dir: Path, feat_tag: str,
                       abs_value: bool, clip: float, stats) -> dict[str, dict]:
    """Per-cube z, Δy, valid_mask. Drops first frame and frames where y or y_prev invalid."""
    ds = ProbeFrameDataset(harps, data_dir, feat_tag, abs_value, clip, stats)
    out = {}
    for h, d in ds.per_cube_arrays().items():
        feat, y, valid = d["feat"], d["y"], d["valid"]
        T = y.shape[0]
        dy = np.full((T,), np.nan, dtype=np.float32)
        dy[1:] = y[1:] - y[:-1]
        dvalid = np.zeros((T,), dtype=bool)
        dvalid[1:] = valid[1:] & valid[:-1] & np.isfinite(y[1:]) & np.isfinite(y[:-1])
        out[h] = {"feat": feat, "dy": dy, "valid": dvalid}
    return out


def stack_for_train(per_cube: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    Z, DY = [], []
    for d in per_cube.values():
        idx = np.where(d["valid"])[0]
        Z.append(d["feat"][idx]); DY.append(d["dy"][idx])
    return np.concatenate(Z), np.concatenate(DY)


def evaluate(per_cube: dict[str, dict], predict_fn, dy_mu: float, dy_sigma: float) -> dict:
    """predict_fn: z[N,D] → ŷ_norm[N]. Returns aggregate + per-cube metrics."""
    per = {}
    yt_all, yp_all, ypers_all = [], [], []
    for h, d in per_cube.items():
        feat, dy, valid = d["feat"], d["dy"], d["valid"]
        if not valid.any():
            per[h] = {"skipped": True}; continue
        pred_norm = predict_fn(feat[valid])
        pred = np.full_like(dy, np.nan)
        pred[valid] = pred_norm * dy_sigma + dy_mu
        persist = np.zeros_like(dy); persist[~valid] = np.nan
        per[h] = {
            "probe": {"r2": _r2(dy, pred), "r": _pearson(dy, pred), "mae": _mae(dy, pred)},
            "persistence_zero": {"r2": _r2(dy, persist), "r": _pearson(dy, persist),
                                  "mae": _mae(dy, persist)},
        }
        yt_all.append(dy); yp_all.append(pred); ypers_all.append(persist)
    yt = np.concatenate(yt_all)
    aggregate = {
        "probe": {"r2": _r2(yt, np.concatenate(yp_all)),
                  "r": _pearson(yt, np.concatenate(yp_all)),
                  "mae": _mae(yt, np.concatenate(yp_all))},
        "persistence_zero": {"r2": _r2(yt, np.concatenate(ypers_all)),
                              "r": _pearson(yt, np.concatenate(ypers_all)),
                              "mae": _mae(yt, np.concatenate(ypers_all))},
    }
    return {"per_cube": per, "aggregate": aggregate}


def fmt_pair(m: dict) -> str:
    return f"{m['r2']:+.3f} / {m['r']:+.3f} / {m['mae']:.2e}"


def write_md(out_path: Path, results_by_model: dict[str, dict[str, dict]]) -> None:
    lines = ["# Rate-of-change probe — predicts Δ⟨|wind|⟩(t) = y(t) − y(t−1)\n",
             "Persistence-zero baseline = 'predict no change' (Δ=0). Probe wins ⇔ features carry frame-to-frame information.",
             "All metrics in original Δ units (signed, no log/abs). MAE is mean |Δ_true − Δ_pred|.\n"]
    for model_name, results in results_by_model.items():
        lines.append(f"\n## Model = {model_name}\n")
        lines.append("| Eval set | n cubes | persist-0 R² / r / MAE | **probe R² / r / MAE** |")
        lines.append("|---|---|---|---|")
        for set_name, r in results.items():
            a = r["aggregate"]
            lines.append(f"| {set_name} | {len(r['per_cube'])} | {fmt_pair(a['persistence_zero'])} | **{fmt_pair(a['probe'])}** |")
        lines.append("\n### Per-cube (probe model only)\n")
        for set_name, r in results.items():
            lines.append(f"\n#### {set_name}\n")
            lines.append("| Cube | persist-0 R² | probe R² | probe r | probe MAE |")
            lines.append("|---|---|---|---|---|")
            for h, m in r["per_cube"].items():
                if m.get("skipped"): continue
                lines.append(f"| {h} | {m['persistence_zero']['r2']:+.3f} | {m['probe']['r2']:+.3f} | {m['probe']['r']:+.3f} | {m['probe']['mae']:.2e} |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/probe_e09.yaml")
    ap.add_argument("--out-dir", default="outputs_probe/eval")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--xgb-trees", type=int, default=300)
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
    novel = [e["harp_id"] for e in all_ids if e["harp_id"] not in set(train_harps) | set(val_harps)]

    stats = fit_target_stats(train_harps, data_dir, abs_value=abs_value, clip=clip)
    train_pc = build_delta_arrays(train_harps, data_dir, feat_tag, abs_value, clip, stats)
    val_pc = build_delta_arrays(val_harps, data_dir, feat_tag, abs_value, clip, stats)
    novel_pc = build_delta_arrays(novel, data_dir, feat_tag, abs_value, clip, stats)

    Xtr, DYtr = stack_for_train(train_pc)
    Xv, DYv = stack_for_train(val_pc)
    dy_mu, dy_sigma = float(DYtr.mean()), float(DYtr.std() or 1.0)
    DYtr_n = (DYtr - dy_mu) / dy_sigma
    DYv_n = (DYv - dy_mu) / dy_sigma
    print(f"[delta] train_n={Xtr.shape[0]} val_n={Xv.shape[0]} dy_mu={dy_mu:.3e} sigma={dy_sigma:.3e}")

    ridge = Ridge(alpha=args.ridge_alpha).fit(Xtr, DYtr_n)
    xgb_m = xgb.XGBRegressor(n_estimators=args.xgb_trees, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              early_stopping_rounds=20, verbosity=0)
    xgb_m.fit(Xtr, DYtr_n, eval_set=[(Xv, DYv_n)], verbose=False)
    print(f"[xgb] best_iter={xgb_m.best_iteration} best_score={xgb_m.best_score:.4f}")

    results_by_model = {}
    for name, predict_fn in (("ridge", lambda x: ridge.predict(x)),
                              ("xgboost", lambda x: xgb_m.predict(x))):
        results_by_model[name] = {
            "encoder-val (harp_51)": evaluate(val_pc, predict_fn, dy_mu, dy_sigma),
            "novel cubes (off-dist)": evaluate(novel_pc, predict_fn, dy_mu, dy_sigma),
            "encoder-train (sanity)": evaluate(train_pc, predict_fn, dy_mu, dy_sigma),
        }
    write_md(out_dir / "probe_delta.md", results_by_model)
    (out_dir / "probe_delta.json").write_text(json.dumps(results_by_model, indent=2, default=float))
    print(f"[done] {out_dir}/probe_delta.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
