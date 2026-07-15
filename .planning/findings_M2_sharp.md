# M2-SHARP — does spatial winding add over the SHARP baseline?

**Date:** 2026-06-26 · leave-AR-out · ≥M/24h · same folds · `scripts/flare_xgb/sharp_eval.py`

## Result

| Group | n_feat | TSS | AUC |
|---|---|---|---|
| scalar (winding mean) | 21 | 0.258 | 0.631 |
| spatial (winding stats) | 77 | 0.340 | 0.675 |
| **SHARP** | 13 | **0.422** | **0.773** |
| SHARP + spatial winding | 90 | 0.425 | 0.782 |
| all | 111 | 0.422 | 0.785 |

SHARP fetched from JSOC (`hmi.sharp_cea_720s`, drms metadata query, no FITS),
top-13 Bobra & Couvidat params, aligned to frames (coverage ~1.0).

## The hard finding (report honestly)

1. **SHARP alone (0.422/0.773) beats winding-spatial (0.340/0.675) by a wide
   margin** (+0.082 TSS, +0.098 AUC). The field-standard scalar baseline is much
   stronger than our winding features.
2. **Spatial winding adds essentially NOTHING over SHARP:** SHARP→SHARP+spatial
   is +0.003 TSS / +0.009 AUC — within noise. `all` doesn't help either.
3. ⇒ The novelty claim "spatial winding maps add forecasting skill" is **NOT
   supported once SHARP is the baseline** — at this data scale, with total
   winding only.

## Why (physics, not just stats)

TOTUSJH = total unsigned **current helicity**, the #1 SHARP feature — a
topological quantity in the *same family* as winding/helicity. ABSNJZH (net
current helicity) too. **SHARP already contains integrated helicity-like
scalars**, so our total-winding features are largely redundant with it.

This sharpens the importance of the **current-carrying** decomposition:
- We have **total winding only** (M0). Total winding ≈ dominated by the
  potential/redundant part already in SHARP.
- Williams' result was specifically that the **current-carrying** topological
  flux (δL_c) adds over SHARP. That component — which we lack — is the only
  part with a real chance of beating SHARP. δL_c re-export (windCur/windPot) is
  now the load-bearing next step, not optional.

## Honest status of the thesis
- "spatial winding > scalar-mean winding": HOLDS (0.340 vs 0.258) but is a
  within-winding result — scientifically minor.
- "winding adds over SHARP" (the claim that matters for novelty): **FAILS** with
  total winding. Needs δL_c, or richer spatial features, or more data to revisit.
- Silver lining: our SHARP baseline (0.422 leave-AR-out, 25 ARs) is solid and
  in range of Williams' 0.524 (232 ARs) — the pipeline is sound.

## Next options
1. **δL_c re-export** (windCur/windPot via ARTop) — the current-carrying
   component is the only winding piece that could add over SHARP. Highest value.
2. Reframe: pitch as "SHARP-competitive flare forecasting + a winding-structure
   analysis" rather than "winding beats SHARP".
3. Spatial SHARP: SHARP params are scalars; is there a *spatial* SHARP-map angle?
   (Probably out of scope.)
