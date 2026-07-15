# M2 Track A — scalar XGBoost baseline + spatial-vs-scalar ablation

**Date:** 2026-06-26 · leave-AR-out · `scripts/flare_xgb/` · ≥M1.0 / 24 h

## Setup
- 25 labeled HARP cubes (11 flaring, 14 quiet), 21 355 valid frames, pos_rate 0.160.
- Per-frame features from the **total winding map** (M0): SCALAR group (signed
  mean/sum, asinh-mean = what prior work integrates to) vs SPATIAL group (Gini,
  entropy, sign-imbalance, cancel-ratio, concentration, quantiles) + within-AR
  rolling/lag temporal context. XGBoost, GroupKFold OOF by AR, scale_pos_weight.
- TSS = max(tpr − fpr) over thresholds on the pooled OOF predictions.

## Result — spatial beats the spatial-mean (thesis confirmed)

| Group | n_feat | TSS (5-fold) | AUC |
|---|---|---|---|
| SCALAR | 21 | 0.250 | 0.623 |
| **SPATIAL** | 77 | **0.307** | **0.671** |
| ALL | 98 | 0.271 | 0.651 |

**Robustness (5 seeds × fold counts):** Δ(spatial−scalar) = +0.056 / +0.056 /
+0.037 for splits 4/5/6, seed σ ≤ 0.008. Ordering stable in every config.
Absolute TSS wobbles 0.25–0.38 with fold count (only 11 flaring ARs → fold
composition dominates variance), but spatial always wins.

## Reading
- The +0.04–0.06 TSS / +0.05 AUC lift comes from **scalar summaries of the map**
  alone — no 2D structure yet. Direct support for M3 (feed the actual maps).
- **ALL < SPATIAL**: adding the scalar-mean family on top of spatial *hurts*
  (0.307→0.271). The integrated mean is noise once you have the distribution
  shape — consistent with the M0 cancellation result (|Σw|/Σ|w| ≈ 0.034).
- `cancel_ratio` / `sign_imbalance` / `gini_abs` are the new axes; they're
  exactly what the spatial mean destroys.

## Honest caveats (for the PhD audience)
- TSS 0.31 ≪ Williams 0.524, expected: (a) 25 small ARs vs 232; (b) **total
  winding only — no current-carrying δL_c**, their strongest feature; (c)
  per-frame OOF TSS, not their once-daily operational aggregation (which lifts
  TSS); (d) crude map summaries, not the 2D maps.
- Not yet comparable to 0.524. The *comparable* claim needs Track B data or M3.
- harp_11930 is all-positive (290/290) — adds base-rate, no within-AR signal.

## Next
- **M3**: ConvLSTM/CNN on the 2D winding maps → the gap should widen. This is
  the publishable test.
- Optional: once-daily operational aggregation to make TSS comparable to 0.524.
- Re-export windCur/windPot for δL_c (closes the biggest feature gap vs Williams).
