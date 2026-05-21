"""Multi-horizon linear probes on E30 features + time-series overlays.

Fits closed-form ridge regression on cached E30 features mapping
features at frame t to spatial-mean abs winding flux at frame t+h,
for h in {12 min (lag-1), 1 h (lag-5), 3 h (lag-15)}. Trains a single
global ridge per horizon on the concatenated 13 train cubes; evaluates
on val + novel cubes.

Outputs:
  thesis/assets/figures/probe_timeseries.pdf            — h=12min overlay
  thesis/assets/figures/probe_multi_horizon.pdf         — 4-cube x 3-horizon panel
  thesis/assets/figures/probe_multi_horizon_metrics.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from solarflare_data.wind_target import cache_path as wind_cache_path  # noqa: E402

DATA = REPO / "data"
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FEAT_TAG = "E30"
ABS_VALUE = True
CLIP = 1.0e7

TRAIN_CUBES = ["harp_8", "harp_26", "harp_43", "harp_45", "harp_49",
               "harp_54", "harp_83", "harp_116", "harp_156", "harp_219",
               "harp_274", "harp_316", "harp_318"]
VAL_CUBES = ["harp_86", "harp_221", "harp_11930"]
NOVEL_CUBES = ["harp_17", "harp_51", "harp_245", "harp_may2024", "harp_nov2025"]

HORIZONS = [
    ("12 min", 1),
    ("1 h", 5),
    ("3 h", 15),
]
CADENCE_MIN = 12

OVERLAY_CUBES = ["harp_11930", "harp_49", "harp_17", "harp_may2024"]


def load_cube(harp: str):
    feat = np.load(DATA / f"{harp}_feat_{FEAT_TAG}.npy").astype(np.float32)
    valid_feat = np.load(DATA / f"{harp}_feat_{FEAT_TAG}_valid.npy").astype(bool)
    y_path = wind_cache_path(DATA / f"{harp}.zarr", abs_value=ABS_VALUE, clip=CLIP)
    y = np.load(y_path).astype(np.float32)
    T = min(feat.shape[0], valid_feat.shape[0], y.shape[0])
    feat, valid_feat, y = feat[:T], valid_feat[:T], y[:T]
    keep = valid_feat & np.isfinite(y) & np.all(np.isfinite(feat), axis=1)
    return feat, y, keep


def build_pairs(harp: str, h: int):
    feat, y, keep = load_cube(harp)
    T = y.shape[0]
    if T <= h:
        return np.zeros((0, feat.shape[1])), np.zeros(0), np.zeros(0, dtype=bool)
    # (z_t, y_{t+h}) for t in [0, T-h)
    z = feat[:T - h]
    yt = y[h:]
    valid = keep[:T - h] & keep[h:]
    return z[valid], yt[valid], valid


def fit_ridge(Z, y, lam=100.0):
    # log-space target for stability (matches probe trainer's log10(y+1) norm)
    yl = np.log10(np.maximum(y, 0.0) + 1.0)
    mu, sd = yl.mean(), yl.std() + 1e-9
    yz = (yl - mu) / sd
    D = Z.shape[1]
    A = Z.T @ Z + lam * np.eye(D, dtype=np.float64)
    b = Z.T @ yz
    w = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    return w.astype(np.float32), float(mu), float(sd)


def predict(w, mu, sd, Z):
    yz = Z @ w
    yl = yz * sd + mu
    return (np.power(10.0, yl) - 1.0).astype(np.float32)


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pearson(y, yhat):
    if y.size < 2:
        return float("nan")
    sa, sb = np.std(y), np.std(yhat)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0, 1])


def medape(y, yhat):
    denom = np.maximum(np.abs(y), 1.0)
    return float(np.median(np.abs(y - yhat) / denom) * 100.0)


def train_global(h):
    Zs, ys = [], []
    for c in TRAIN_CUBES:
        z, y, _ = build_pairs(c, h)
        if z.size:
            Zs.append(z); ys.append(y)
    Z = np.concatenate(Zs, axis=0); y = np.concatenate(ys, axis=0)
    return fit_ridge(Z, y, lam=100.0)


def fit_log_affine(y_true, y_pred):
    """Fit log10(y) = a*log10(yhat) + b on positive samples."""
    mask = (y_true > 1.0) & (y_pred > 1.0)
    if mask.sum() < 4:
        return 1.0, 0.0
    lx = np.log10(y_pred[mask]); ly = np.log10(y_true[mask])
    A = np.stack([lx, np.ones_like(lx)], axis=1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_log_affine(y_pred, a, b):
    safe = np.maximum(y_pred, 1.0)
    return np.power(10.0, a * np.log10(safe) + b).astype(np.float32)


def eval_cube(harp, h, w, mu, sd, cal_frac=0.3):
    feat, y_full, keep = load_cube(harp)
    T = y_full.shape[0]
    if T <= h:
        return None
    yhat_all = np.full(T, np.nan, dtype=np.float32)
    pred = predict(w, mu, sd, feat[:T - h])
    yhat_all[h:T] = pred  # prediction targets index t+h
    yhat_all[~keep] = np.nan
    # lag-h persistence: predict y_{t+h} from y_t
    y_pers = np.full(T, np.nan, dtype=np.float32)
    y_pers[h:T] = y_full[:T - h]
    y_pers[~keep] = np.nan

    # Split keep mask into leading cal_frac for fitting log-affine, trailing
    # rest for evaluation. Apply per-cube fit on probe predictions only.
    valid_idx = np.where(keep & np.isfinite(yhat_all) & np.isfinite(y_full)
                         & np.isfinite(y_pers))[0]
    if valid_idx.size < 8:
        return None
    n_cal = max(4, int(round(cal_frac * valid_idx.size)))
    cal_idx = valid_idx[:n_cal]
    eval_idx = valid_idx[n_cal:]
    if eval_idx.size < 4:
        return None
    a, b = fit_log_affine(y_full[cal_idx], yhat_all[cal_idx])
    yhat_cal = np.full(T, np.nan, dtype=np.float32)
    yhat_cal[h:T] = apply_log_affine(yhat_all[h:T], a, b)
    yhat_cal[~keep] = np.nan

    # Metrics on the evaluation slice only
    y_true = y_full[eval_idx]
    y_probe_raw = yhat_all[eval_idx]
    y_probe_log = yhat_cal[eval_idx]
    y_persist = y_pers[eval_idx]
    metrics = {
        "n_cal": int(n_cal),
        "n_eval": int(eval_idx.size),
        "log_a": a, "log_b": b,
        "probe_raw_r2": r2(y_true, y_probe_raw),
        "probe_raw_medape": medape(y_true, y_probe_raw),
        "probe_log_r2": r2(y_true, y_probe_log),
        "probe_log_r":  pearson(y_true, y_probe_log),
        "probe_log_medape": medape(y_true, y_probe_log),
        "persist_r2": r2(y_true, y_persist),
        "persist_medape": medape(y_true, y_persist),
    }
    return {
        "metrics": metrics,
        "y_true": y_full, "y_probe_raw": yhat_all, "y_probe_log": yhat_cal,
        "y_persist": y_pers, "eval_idx": eval_idx, "cal_idx": cal_idx,
        "horizon_steps": h,
    }


def plot_overlay(cube_results, fname, title):
    n = len(cube_results)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.2 * n + 0.4), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (harp, res) in zip(axes, cube_results):
        y = res["y_true"]; yp_raw = res["y_probe_raw"]
        yp_log = res["y_probe_log"]; ypers = res["y_persist"]
        t = np.arange(len(y)) * CADENCE_MIN / 60.0  # hours
        ax.plot(t, y, color="#222", linewidth=0.7, label="actual $\\langle|W|\\rangle(t)$")
        ax.plot(t, ypers, color="#d62728", linewidth=0.7, alpha=0.7,
                label="lag-1 persistence")
        ax.plot(t, yp_log, color="#1f77b4", linewidth=0.9, alpha=0.9,
                label="probe + log-affine")
        ev = res["eval_idx"]
        ax.axvspan(t[ev[0]], t[ev[-1]], color="#cccccc", alpha=0.18, zorder=0)
        m = res["metrics"]
        ax.set_title(
            f"{harp} (eval slice shaded) — probe$_{{\\rm log}}$ "
            f"$R^2={m['probe_log_r2']:+.2f}$, "
            f"persistence $R^2={m['persist_r2']:+.2f}$",
            fontsize=9)
        ax.set_ylabel(r"$\langle|W|\rangle$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)
        if ax is axes[0]:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        if ax is axes[-1]:
            ax.set_xlabel("time (h)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def main():
    # Train one global ridge per horizon
    horizon_models = {}
    for label, h in HORIZONS:
        w, mu, sd = train_global(h)
        horizon_models[label] = (h, w, mu, sd)
        print(f"trained h={label} ({h} step) ridge on {len(TRAIN_CUBES)} train cubes")

    # Single-horizon overlay (h=12min) on selected cubes
    h_label, h = HORIZONS[0][0], HORIZONS[0][1]
    h_steps, w, mu, sd = horizon_models[h_label]
    short_overlay = []
    for c in OVERLAY_CUBES:
        try:
            res = eval_cube(c, h_steps, w, mu, sd)
            if res is not None:
                short_overlay.append((c, res))
        except FileNotFoundError as e:
            print(f"  skip {c}: {e}")
    plot_overlay(short_overlay, "probe_timeseries.pdf",
                 "Spatial-mean winding flux: E30 linear probe vs lag-1 persistence "
                 "(h = 12 min)")

    # Multi-horizon grid: cubes x horizons. Show probe+log-affine vs persistence.
    fig, axes = plt.subplots(len(OVERLAY_CUBES), len(HORIZONS),
                             figsize=(13.0, 2.2 * len(OVERLAY_CUBES) + 0.6),
                             sharex=False)
    all_metrics = {}
    for j, (label, _) in enumerate(HORIZONS):
        h_steps, w, mu, sd = horizon_models[label]
        for i, c in enumerate(OVERLAY_CUBES):
            ax = axes[i, j]
            try:
                res = eval_cube(c, h_steps, w, mu, sd)
            except FileNotFoundError:
                res = None
            if res is None:
                ax.text(0.5, 0.5, f"missing\n{c}", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([]); continue
            y = res["y_true"]; yp = res["y_probe_log"]; ypers = res["y_persist"]
            t = np.arange(len(y)) * CADENCE_MIN / 60.0
            ev = res["eval_idx"]
            ax.axvspan(t[ev[0]], t[ev[-1]], color="#cccccc", alpha=0.16, zorder=0)
            ax.plot(t, y, color="#222", linewidth=0.55,
                    label="actual" if (i == 0 and j == 0) else None)
            ax.plot(t, ypers, color="#d62728", linewidth=0.55, alpha=0.7,
                    label=f"persistence (lag {h_steps})" if (i == 0 and j == 0) else None)
            ax.plot(t, yp, color="#1f77b4", linewidth=0.7, alpha=0.9,
                    label="probe + log-affine" if (i == 0 and j == 0) else None)
            m = res["metrics"]
            all_metrics.setdefault(label, {})[c] = m
            ax.set_title(f"{c} — $h={label}$  "
                         f"probe$_{{\\rm log}}$ $R^2={m['probe_log_r2']:+.2f}$ | "
                         f"pers $R^2={m['persist_r2']:+.2f}$",
                         fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3, linewidth=0.4)
            if i == len(OVERLAY_CUBES) - 1:
                ax.set_xlabel("time (h)", fontsize=7)
            if j == 0:
                ax.set_ylabel(r"$\langle|W|\rangle$", fontsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("E30 linear probe vs lag-h persistence across three "
                 "forecast horizons (12 min, 1 h, 3 h)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "probe_multi_horizon.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")

    # Aggregate metrics across all val + novel cubes for each horizon
    summary = {}
    for label, _ in HORIZONS:
        h_steps, w, mu, sd = horizon_models[label]
        rows = {}
        for c in VAL_CUBES + NOVEL_CUBES:
            res = eval_cube(c, h_steps, w, mu, sd)
            if res is not None:
                rows[c] = res["metrics"]
        summary[label] = rows
    (OUT / "probe_multi_horizon_metrics.json").write_text(
        json.dumps(summary, indent=2))
    print(f"wrote {(OUT / 'probe_multi_horizon_metrics.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
