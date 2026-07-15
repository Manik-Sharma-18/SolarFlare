# M4 — PIL-localized winding + flare history (2026-07-06)

Ported the two data-free levers from Williams/MacTaggart 2026 (arXiv 2512.14840)
into the leave-AR-out winding XGBoost, plus a novel spatial one.

## New feature families (`scripts/flare_xgb/`)
- **`pil.py` — PIL-localized winding** (novel, data-free δL_c proxy). Spatial stats
  (Gini, q99, concentration, entropy) of |wind| restricted to a dilated
  neighbourhood of SIGN FLIPS. Sign taken on a σ=4 px gaussian-smoothed copy —
  raw winding sign field is near-white (area-frac 0.79), smoothing → 0.37 so PILs
  are large-scale inversion lines. 8 base feats × temporal expansion = 120 cols.
- **`history.py` — flare history**. Log-weighted GOES score (C=10, M=100, X=1000
  × magnitude) trailing 12/24h, excluding last 1h. Join harp→noaa (labels_meta)
  →`hek_cache/<noaa>_*.csv`. 20/25 cubes match; 5 misses are 0-M+X (quiet). 4 feats.
- **`dataset.py`** — also added rolling kurtosis/skew (3/6/12h) = burstiness (Arm B).

## Results (M/24h, 21355 frames, 25 ARs, 11 flaring, GroupKFold)
Within-winding TSS: scalar 0.180 | spatial(global) 0.264 | **pil 0.363** | history 0.453.
- PIL beats global hand-stats by **+0.10 TSS** — best pure-winding feature.
- Dilution: spatial+pil+history (289 feats) 0.393 < history alone (4 feats) 0.453.
  Fixed-hparam XGB overfits when feature count explodes on 25 groups.

vs SHARP (5-fold, TSS / AUC):
| group | TSS | AUC |
|---|---|---|
| pil | 0.408 | 0.737 |
| history | 0.450 | 0.702 |
| history+pil | 0.488 | 0.824 |
| sharp | 0.422 | 0.773 |
| sharp+pil | 0.415 | 0.783 |
| sharp+history | 0.475 | 0.802 |
| **sharp+history+pil** | **0.495** | 0.821 |

Group-bootstrap Δ (2000×, AUC primary — TSS-at-best too noisy at N=25):
- **ΔAUC history+pil vs history = +0.124 [+0.029, +0.280] SIGNIFICANT.**
- ΔAUC sharp+history+pil vs sharp+history +0.019 [-0.032,+0.102] n.s.
- ΔAUC sharp+pil vs sharp +0.011 n.s. All TSS deltas n.s.

## Verdict
1. **Thesis holds vs the cheap baselines**: PIL-localized winding significantly
   improves ranking over flare-history alone and beats global hand-stats. The 2D
   spatial structure carries flare signal the scalar-mean throws away — confirmed.
2. **Blocked by SHARP overlap**: PIL winding does NOT add over the full SHARP suite
   (TOTUSJH/R_VALUE already encode PIL current). Same wall as M2/M3, now with CI.
   → need REAL current-carrying δL_c, not the total-winding PIL proxy.
3. **Blocked by N=25**: TSS CIs all include 0. Pre-registered ceiling (M0).
4. Best model sharp+history+pil TSS 0.495 approaches MacTaggart holdout 0.524.

Both blockers (2 + 3) fixed by one action → `.planning/plan_artop_pipeline.md`.
Reproduce: `python -m scripts.flare_xgb.run`, `python -m scripts.flare_xgb.sharp_eval`.
