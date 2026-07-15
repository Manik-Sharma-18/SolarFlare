# Winding Flux — Physics Breakdown & ML Strategy

**Date:** 2026-06-26 · **Branch:** Version_4 · Author: astrophysics audit

Pairs with [[eda_winding_flux]] (statistical EDA) — this doc adds the physics
that *explains* those EDA findings and reframes the ML target.

## 1. What winding flux IS

Magnetic **winding** L = renormalised magnetic helicity. Helicity H weights
field-line linkage by field *strength* Bz; winding replaces Bz with the sign
indicator σz ∈ {−1,0,+1}. So winding measures pure field-line *entanglement*
(topology), stripped of flux-strength confound. (MacTaggart & Prior 2021;
Raphaldini+ 2022; ARTop, Alielden+ 2023.)

**Winding flux density** (per-pixel integrand, what a pixel map holds):

    dL/dt(x) = −σz(x) ∫_P σz(y) K[u(x),u(y)] d²y          (nonlocal!)
    K = (1/2π) e_z · [(u(x)−u(y)) × r] / |r|²,   r = y−x

Helicity analog uses Bz instead of σz. Key property: **nonlocal** — each pixel
value is an integral over the whole active-region plane. Not a local field.

**Decompositions ARTop produces** (all per-pixel maps, all in `data/`-land):
- Emergence vs braiding: u = u_e + u_b. u_b = v∥ (footpoint shuffling),
  u_e = −(vz/Bz)B∥ (vertical advection of twisted tube). → dL_b/dt, dL_e/dt.
- Current vs potential: B = B_p + B_c. → windCur (dL_c/dt), windPot (dL_p/dt).
  δL = ∫(|dL_c/dt| − |dL_p/dt|) d²y. δL>0 ⇒ free-energy-bearing topology.

## 2. How it's derived here (magnetogram → zarr pixel)

SHARP/HMI vector magnetogram time series → DAVE4VM-style velocity inversion →
ARTop (Julia) computes winding-flux density maps at 12-min cadence.
`unit_conversion.md`: `deltaLflux = sum(|windCur|−|windPot|)·dx·dy`,
hardcoded dx=dy=360 km ⇒ 129 600 km²/px. Output units G·km² family.

`data/<harp>.zarr : wind[H,W,T]` = **most likely the total winding-flux density
map per frame** (sign-carrying pseudoscalar, per-pixel max ~1e7). ⚠️ CONFIRM
whether `wind` = total dL/dt, or windCur, or windCur−windPot — changes framing.
Loader: `harp_loader._densify_harp_cube` clips |w|>1e8→0, drops Time==0,
transposes [H,W,T]→[T,H,W]. Norm: signed_asinh, softening 1e3 (probe-proven).

## 3. What signals it can / cannot have

| Signal type | Present? | Why (physics) |
|---|---|---|
| Temporal periodicity / QPP | **NO** | No mechanism for a clock in topology injection; literature reports none; our FFT/Welch/STFT/detrend all null (top1 power frac <0.20, residual_std_frac 0.88–0.95). |
| Episodic injection spikes | **YES** | Reconnection + flux emergence are discrete events. ḋL spikes breach 6–10 h moving-avg ±3σ envelope **6–24 h before flares** (winding-indicator paper). This is THE signal. |
| Smooth accumulation (ramp) | YES | δL builds up positively over days in flaring ARs; stays negative in quiet ARs. Free-energy proxy. |
| Sign / chirality | YES | Pseudoscalar; hemispheric helicity preference. Sign of net winding input is physical, not noise. |
| Spatial coarse envelope (low-k ≳14–23 Mm) | YES | EDA F3/F4: lag-corr 0.64 @5 h. The only *frame-forecastable* component. |
| Per-pixel high-k field | **NO** | Nonlocal kernel + σz texture ⇒ near-white PSD (F5), lag-1h corr 0.003 (F1). Heavy ±cancellation — papers integrate it away explicitly. |

**Why the EDA nulls are physically inevitable:** winding density is a *nonlocal
convolution of a sign field*. That manufactures fine-scale ± texture with no
local persistence (explains F1/F5) and no temporal line (explains F6). The
physics literally never promised a forecastable per-pixel field.

## 4. The reframing (central finding)

**V4 forecasts the wrong object.** Frame-to-frame regression of the 2D winding
density fights nonlocal sign-texture noise — the predictable part (low-k
envelope) is a thin slice, and the *physically meaningful* part (flare
precursor) is destroyed by spatial averaging into MSE.

The published, validated signal pipeline is:
1. **Spatially integrate** the density map each frame → scalar winding-input
   rate dL/dt(t) (and its current-carrying part, and δL).
2. **Moving-average detrend** (6–10 h running mean) → residual.
   *(Exactly the operation just prototyped in `temporal.py` — physically
   motivated, not arbitrary. It found no periodicity because the signal is an
   event point-process, not an oscillation.)*
3. **Spike / anomaly detection** vs ±2.5–3σ envelope → flare precursor flag.

This is why [[flare_classification_task]] works (leave-AR-out AUC 0.75,
TSS 0.38 from FastTOP scalar CSVs) while curve regression hit a ceiling
([[winding_flux_signal_nature]], [[pretrained_tsfm_curve_ceiling]]). The scalar
topological time series carries real, leave-AR-out-generalising signal; the
2D-field forecast does not.

## 5. ML strategy — ranked by physics-expected yield

1. **Scalar event forecasting (primary).** Target = δL(t) / dL_c/dt(t)
   spatially-integrated series. Frame as: (a) anomaly detection on MA-residual,
   (b) ≥M/12–24 h flare classification. Leverages the 6–24 h precursor window.
   Inputs: the decomposed scalars (emergence, braiding, current, potential) as
   multichannel 1D series. This is the winning lane.
2. **Decomposition-aware channels.** Don't feed only total `wind`. Feed/derive
   windCur, windPot, dL_b, dL_e as separate inputs — the flare signal lives in
   the *current-carrying* component, swamped in the total.
3. **Multi-task head.** Jointly predict (i) next-step coarse low-k envelope
   (the only forecastable field part) + (ii) δL scalar + (iii) flare label.
   Shared encoder, physics-separated heads. Couples the two viable signals.
4. **Spatial: forecast only low-k.** If keeping a field head, predict the
   asinh field at ≥4× coarsening / low-pass the loss (EDA verdict 1). Never
   spend MSE on high-k σz-texture.
5. **Conservative scalar.** Net winding input ≈ ∮ boundary term — physics gives
   a near-conserved accumulation; predicting *increments* (not levels) of L(t)
   may be better-conditioned than predicting the density field.

## 6. Literature gap & novelty (2026-06-26 search)

**Q: SHARP + winding combined paper?** YES, exactly one, very recent:
Williams/MacTaggart "Topologically Derived Time Series for Flare Forecasting
II" (XGBoost, ApJ 2026) combines ARTop winding+helicity *scalars* with SHARP
rolling-stats. Holdout TSS **0.524**. BUT: no ablation isolating winding's
marginal lift over SHARP — the "how much does topology add" question is still
formally open. Paper I (data-set prep) is scalar time series only.

**Q: spatial (2D) winding flux maps used in ML? → NO. Real gap.** Verified
adversarially (7 searches):
- Topological papers I & II **reduce maps to scalars** (δL, δL_c, v_z·δL_c…);
  2D maps appear only in *retrospective validation figures*, never as model
  input. They explicitly forecast on spatially-integrated series.
- Connectivity-based helicity flux density (Pariat 2013, NOAA 11158) produces
  2D helicity maps but **pure-physics studies — no ML on them surfaced**.
- CNN flare papers feed **raw magnetograms / SHARP**, not winding/helicity flux
  *density* maps.
- Caveat: absence-of-evidence, not proof of negative. But the gap is clean
  enough to claim "unexplored" to reviewers.

**Why spatial-mean is provably lossy (the physics pitch):** winding density is
signed (±); the spatial mean suffers massive +/− cancellation — the literature
itself avoids global totals, using *local balance* Δ_L "to minimize cancellation
artifacts." And winding is "largely dominated by strong transversal field near
the PIL" — i.e. the flare signal is *spatially concentrated at the PIL* and the
mean dilutes it across the whole patch. Both say: the map carries signal the
scalar throws away.

**Novel directions (ranked, defensible):**
1. **Spatial winding statistics** beyond the mean — Gini / spatial entropy /
   moments of |winding|, PIL-localized winding fraction, +/− patch separation.
   Cheap, interpretable, new scalar features. Show they beat spatial-mean and
   add to SHARP. (Low-risk first paper.)
2. **ConvLSTM/CNN directly on winding flux density maps → flare classifier.**
   First ML use of spatial winding maps. **Repurposes the V4 ConvLSTM backbone**
   — not as a field *forecaster* (dead end, §4) but as a spatiotemporal
   *encoder* of winding maps → flare label. Reuses the whole codebase.
3. **Cancellation-aware channels:** feed +winding / −winding / |winding|
   separately to bypass the mean-cancellation loss. Physics-motivated.
4. **Saliency/attention over maps** — does the model attend to the PIL? Physics
   validation + interpretability (reviewer catnip).

This reframes V4: spatiotemporal ConvLSTM on winding maps → ≥M flare forecast is
both the EDA-supported lane AND a genuine literature gap.

## 7. Open / to confirm

- ⚠️ Confirm what `wind` channel actually stores (total vs windCur vs δ-map).
  Read the ARTop export script / ask senior. Gate everything on this.
- Do we have the *decomposed* maps (windCur/windPot/emergence/braiding) saved,
  or only the total? If only total, request a re-export — the decomposition is
  where the flare signal concentrates.
- Build scalar series δL(t), dL_c/dt(t) per cube; reproduce the ±3σ envelope
  spike detector; check spikes lead our HEK M/X labels by 6–24 h.
- CWT on the scalar series (intermittent, not periodic) — only transform that
  can localise episodic injection in time-frequency; the one remaining probe
  worth running after the FFT-family nulls.
