"""Eval helpers: per-cube/aggregate R²/Pearson r in original units, persistence
baseline, markdown writer. Split from probe_eval.py to honor 200-line cap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from solarflare_data.probe_dataset import ProbeFrameDataset, TargetStats


@torch.no_grad()
def predict_cube(head: torch.nn.Module, feats: np.ndarray, valid: np.ndarray,
                 stats: TargetStats, device: torch.device) -> np.ndarray:
    """Return [T] fp32, NaN where invalid frames."""
    T, D = feats.shape
    out = np.full((T,), np.nan, dtype=np.float32)
    if not valid.any():
        return out
    z = torch.from_numpy(feats[valid].astype(np.float32)).to(device)
    pred_norm = head(z).cpu().numpy()
    out[valid] = stats.invert(pred_norm.astype(np.float32))
    return out


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2: return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    if ss_tot < 1e-12: return float("nan")
    return 1.0 - ss_res / ss_tot


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2: return float("nan")
    yt, yp = y_true[m], y_pred[m]
    if yt.std() < 1e-12 or yp.std() < 1e-12: return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 1: return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median APE (more robust than mean for log-distributed targets)."""
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if m.sum() < 1: return float("nan")
    return float(np.median(np.abs(y_true[m] - y_pred[m]) / y_true[m]) * 100.0)


def _persistence_pred(y_true: np.ndarray) -> np.ndarray:
    out = np.full_like(y_true, np.nan)
    if y_true.size == 0: return out
    out[1:] = y_true[:-1]
    out[0] = y_true[0]
    return out


def evaluate_set(name: str, harps: list[str], heads: dict[str, tuple[torch.nn.Module, TargetStats]],
                 data_dir: Path, feat_tag: str, abs_value: bool, clip: float,
                 device: torch.device) -> dict:
    rows = {"set": name, "n_cubes": len(harps), "harps": harps, "metrics": {}}
    concat_true: list[np.ndarray] = []
    concat_persist: list[np.ndarray] = []
    concat_preds: dict[str, list[np.ndarray]] = {k: [] for k in heads}
    per_cube: dict[str, dict] = {}

    any_stats = next(iter(heads.values()))[1]
    ds = ProbeFrameDataset(harps, data_dir, feat_tag, abs_value, clip, any_stats)
    cubes = ds.per_cube_arrays()

    for h in harps:
        d = cubes[h]
        feat, y, valid = d["feat"], d["y"], d["valid"]
        y_eval = y.copy(); y_eval[~np.isfinite(y_eval)] = np.nan; y_eval[y_eval <= 0] = np.nan
        persist = _persistence_pred(y_eval)
        cube_metrics = {"persistence": {"r2": _r2(y_eval, persist), "r": _pearson(y_eval, persist),
                                         "mae": _mae(y_eval, persist), "mape": _mape(y_eval, persist)}}
        for kind, (head, stats) in heads.items():
            pred = predict_cube(head, feat, valid, stats, device)
            cube_metrics[kind] = {"r2": _r2(y_eval, pred), "r": _pearson(y_eval, pred),
                                   "mae": _mae(y_eval, pred), "mape": _mape(y_eval, pred)}
            concat_preds[kind].append(pred)
        per_cube[h] = cube_metrics
        concat_true.append(y_eval)
        concat_persist.append(persist)

    yt = np.concatenate(concat_true)
    yp_pers = np.concatenate(concat_persist)
    aggregate = {"persistence": {"r2": _r2(yt, yp_pers), "r": _pearson(yt, yp_pers),
                                 "mae": _mae(yt, yp_pers), "mape": _mape(yt, yp_pers)}}
    for kind in heads:
        yp = np.concatenate(concat_preds[kind])
        aggregate[kind] = {"r2": _r2(yt, yp), "r": _pearson(yt, yp),
                            "mae": _mae(yt, yp), "mape": _mape(yt, yp)}
    rows["per_cube"] = per_cube
    rows["aggregate"] = aggregate
    return rows


def write_metrics_md(out_path: Path, eval_results: list[dict], heads_order: list[str]) -> None:
    lines = ["# Wind-flux probe evaluation\n"]
    lines.append("Predictions inverted to original ⟨|wind|⟩ units (Mx/cm²-equivalent).\n")
    cols = ["persistence"] + heads_order
    metric_cells = lambda m, c: f"{m[c]['r2']:+.3f} | {m[c]['r']:+.3f} | {m[c]['mae']:.2e} | {m[c]['mape']:.1f}%"
    header = "| Eval set | n cubes | " + " | ".join(f"{c} R² | r | MAE | medAPE" for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in range(2 + 4 * len(cols))) + "|"
    lines.append("\n## Aggregate (all frames in set, concatenated). MAE in original ⟨|wind|⟩ units; medAPE = median |y−ŷ|/y × 100%.\n")
    lines.append(header); lines.append(sep)
    for r in eval_results:
        a = r["aggregate"]
        cells = [metric_cells(a, c) for c in cols]
        lines.append(f"| {r['set']} | {r['n_cubes']} | " + " | ".join(cells) + " |")

    lines.append("\n## Per-cube\n")
    for r in eval_results:
        lines.append(f"\n### {r['set']}\n")
        lines.append("| Cube | " + " | ".join(f"{c} R² | r | MAE | medAPE" for c in cols) + " |")
        lines.append("|" + "|".join("---" for _ in range(1 + 4 * len(cols))) + "|")
        for h, m in r["per_cube"].items():
            cells = [metric_cells(m, c) for c in cols]
            lines.append(f"| {h} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")
