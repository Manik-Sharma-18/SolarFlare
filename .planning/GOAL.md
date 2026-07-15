# GOAL — Spatial Winding Flux for Flare Forecasting

**Created:** 2026-06-26 · **Branch:** Version_4 · **Status:** DRAFT (confirm before locking)

## One-sentence goal

Forecast ≥M-class solar flares (12–24 h horizon) from the **2D spatiotemporal
structure of magnetic winding flux density maps**, and show it adds skill over
the SHARP-parameter baseline — a direction unexplored in the literature.

## The thesis (why this is novel & physically motivated)

1. Winding flux density is a **signed, nonlocal** field, **concentrated at the
   PIL**. (Physics: `.planning/winding_flux_physics.md` §1–3.)
2. Everyone in the literature reduces it to a **spatial-mean / integrated
   scalar** before forecasting (ARTop topological-time-series papers I & II).
   The mean suffers +/− **cancellation** — the authors themselves avoid global
   totals for this reason — and **dilutes** the PIL-localized signal.
3. **No prior work feeds the 2D winding map to a deep model.** Verified by
   adversarial lit search (`winding_flux_physics.md` §6). The gap is real.
4. The 2D map therefore carries flare-precursor structure the scalar discards.

## Evidence base (already established)

- Field *forecasting* is a dead end (EDA F1–F6): near-white spatial PSD,
  lag-1h corr 0.003. → do NOT forecast the field; *classify* from it.
- Flare *classification* from topological scalars works: leave-AR-out AUC 0.75,
  TSS 0.38 (winding-only). SHARP-only published leave-AR-out ≈ TSS 0.5.

## Benchmark to beat: Williams et al. 2026 (ApJ 999, 87 / arXiv 2512.14840)

The direct competitor and the bar. ARTop topological **scalar** time series →
XGBoost, ≥M1.0 within 24 h. **This is exactly our scalar baseline + the SOTA we
must exceed.** Match their protocol so numbers are directly comparable.

**Their data:** 232 SHARP regions (207 train / 25 val / 12 holdout), Cycle 24,
NOAA 11070–13924, 720 s cadence, >1e5 AR-observations. 2142 C+ flares, 384 M/X,
36 X. Public "living dataset".
**Their features:** δL'_c, δH'_c, δL', δH', v_z·δL'_c, v_z·δH'_c, v_z|B_z|,
accumulated L_c/H_c + engineered: rolling 2 h mean/std, lags 0.2/0.4 h, SES,
excess kurtosis 3/6/12 h, flare-history 12/24 h, flare-score kernels, limb %.
**Label:** binary ≥M1.0 / 24 h. **Split:** leave-AR-out, full sequences per AR,
val stratified to include AR 11158/11429/12673. **Imbalance:** scale_pos_weight,
~8 % positive.
**Result:** validation TSS 0.804, **holdout TSS 0.524** (limb-excluded 0.521),
holdout acc 72.6 %. Optimal threshold 0.476.

**⇒ Target: beat holdout TSS 0.524 under their exact protocol.**

**Their stated weaknesses = our attack surface:**
1. **All scalar — zero spatial structure used.** ← our entire thesis (M3).
2. Limb/projection corrupts winding; crude ±60° cut doesn't help. ← a spatial
   model can *learn* the foreshortening signature (it's spatial) (M3/M4).
3. Newly-rotated ARs: integrated topology buildup missed. ← a spatial *snapshot*
   gives instantaneous topology state without needing the emergence history.
4. Hard digital label misclassifies accumulated C/small-M activity. ← can
   revisit label / multiclass.
5. No winding-vs-SHARP-vs-helicity ablation. ← we provide it (M2).

## Milestones (proposed — confirm/edit)

**M0 — Confirm the data (blocking).**
Resolve what `wind[H,W,T]` actually stores (total dL/dt vs windCur vs δ-map).
Read ARTop export / ask senior. Everything gates on this.

**M1 — Spatial-statistics baseline (fast, low-risk first result).**
Extract per-frame spatial features beyond the mean: Gini, spatial entropy,
moments of |winding|, PIL-localized winding fraction, +/− patch separation.
→ XGBoost ≥M/12h classifier, leave-AR-out. Compare vs (a) spatial-mean scalar,
(b) SHARP top-6. Deliverable: "spatial stats beat the mean and add to SHARP."

**M2 — Scalar baseline + ablation (the bar). Two tracks.**
- **Track A (start now, no blockers):** on our **27 winding cubes** (23 GB, ARTop
  already paid), derive Williams-style scalars, XGBoost ≥M1.0/24 h leave-AR-out.
  Ablation they omitted: winding vs helicity vs SHARP vs combined. Self-contained
  on this hardware (14 cores, 64 GB).
- **Track B (async, optional):** email Williams/MacTaggart for the derived living
  dataset (no public link found). If it arrives → reproduce holdout TSS 0.524 +
  spatial comparison on their 232 regions for the headline number.
- **Do NOT recompute ARTop on 232 regions** — months on 14 cores + TB download.
  Infeasible. Our 27 maps suffice to prove spatial > scalar.

**M3 — ConvLSTM spatiotemporal encoder (high-novelty core = beat 0.524).**
Repurpose V4 ConvLSTM as an *encoder* of winding-map sequences → ≥M1.0/24 h
label (not a field forecaster). Cancellation-aware channels
(+winding / −winding / |winding|). Hybrid: spatial-encoder embedding ⊕ Williams
scalars → must exceed the M2 control. This is the publishable delta: spatial
structure adds skill the integrated scalar cannot.

**M4 — Interpretability / physics validation.**
Saliency / attention over maps — does the model attend to the PIL? Validates
the physical claim; reviewer-facing.

## Non-goals (explicit)
- No next-frame field forecasting. No autoregressive winding rollout.
- No JEPA / transformers / pretrained TSFM curve fitting.
- No within-cube / random-split evaluation — leave-AR-out only.

## Open questions
- Do we have decomposed maps (windCur/windPot/emergence/braiding) saved, or
  only the total? If only total, request re-export — flare signal concentrates
  in the current-carrying component.
- Horizon: 12 h vs 24 h primary? Class threshold ≥M vs ≥C?
- Which cubes have reliable HEK M/X labels for leave-AR-out folds?
