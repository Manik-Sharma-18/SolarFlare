# Auto-EDA Plan — Winding Flux Signal Characterisation

**Date:** 2026-06-11 · **Branch:** Version_4 · **Status:** DONE — package at
`scripts/eda/`, full 28-cube report at `outputs/eda/EDA_REPORT.md`.

## Sweep results (28 cubes, 2026-06-11)

- Pilot findings hold globally, but **spread is huge**: predictability
  ceiling at 1 h (asinh+32 px) ranges 0.04–0.69 across cubes, median ≈ 0.27.
- **Test cubes are the hardest**: harp_245 = 0.32, harp_274 = 0.05,
  harp_49 = 0.01 (lag-1h, asinh+32px). Two of three test cubes sit at the
  bottom of the predictability distribution — bounds achievable test CSI.
- **Frame-registration drift: REFUTED** (`scripts/eda/registration.py`,
  2026-06-11). Suspects (harp_1028/274/43/83/833/892) show large pairwise
  "shifts", but three tests kill the hypothesis: (a) median-reference
  iterative registration on harp_1028 finds only ~1 px true corrections and
  leaves XL lag-1 coherence flat (0.1464 → 0.1477); (b) the method *does*
  recover synthetic ±20 px jitter injected into coherent harp_11930
  (XL lag1 0.744 → 0.795, clean 0.796) — so null result on harp_1028 is
  real; (c) even genuine 20 px jitter costs only ~0.05 XL coherence —
  misalignment cannot explain ceilings of 0.01–0.15. Measured step size is
  a *symptom* of decorrelation (cross-corr peak wanders on decorrelated
  frames): step_px_p95 anti-correlates with predictability ceiling across
  all 28 cubes (harp_11930 0.64 px / ceiling 0.69; harp_49 62.7 px / 0.01).
  Low-ceiling cubes are intrinsically decorrelated — envelope persists,
  spatial pattern shuffles. No loader-side registration; don't pursue.
- **Flare precursors**: `power_L14-56Mm` and `power_ratio_L_S` are best
  single features where M/X labels exist (AUC 0.86–0.96 on harp_11149,
  245, 11930, 1028; 0.78–0.89 on 2748, 892). Candidate aux inputs for the
  dual-head classifier. Caveat: within-cube AUC, autocorrelated labels.
- **Amplitude outliers**: harp_8 (11.3×median q99), harp_1028 (6×). Per-cube
  norm absorbs this; never use global norm.
- asinh-space spatial slope ≈ k^-1.9 (consistent across cubes); XL band
  carries 64–94% of asinh-field power.

## 0. Pilot probe results (harp_11930 + harp_116, longest contiguous 720 s runs)

Probe run inline (numpy-only, venv python). Findings, both cubes consistent:

| # | Finding | Evidence |
|---|---------|----------|
| F1 | **Raw per-pixel field is unpredictable.** Lag-1 h pixel correlation of raw signed `w` is ≈ 0.003. | lag-corr probe |
| F2 | **asinh transform recovers signal.** Same correlation after `asinh(w/1e3)`: 0.18–0.34. Pearson on raw is destroyed by kurtosis ≈ 5.7×10⁵ heavy tails. Pipeline softening = 1000 is the right order. | transform probe |
| F3 | **Predictability lives at low spatial frequency.** k-band lag coherence (harp_11930): k∈[0,4) → 0.73 @12 min, 0.64 @5 h; k∈[4,16) → 0.41/0.26; k∈[16,64) → 0.09/0.01; k≥64 → 0.00. Scales below ~14 Mm are stochastic texture at 12-min cadence. | k-band coherence |
| F4 | **Smoothing raises the predictability ceiling monotonically.** asinh + box r=2 px → lag-1h 0.34–0.51; r=8 px (6 Mm) → 0.46–0.62; r=32 px (23 Mm) → 0.48–0.69, holding at 5 h (0.38–0.66). | blur probe |
| F5 | **Spatial PSD is near-white** (slope k^−0.37): most raw variance sits in the unpredictable high-k band. Plain full-res MSE spends nearly all gradient on noise. | radial PSD |
| F6 | **Temporal spectrum is red / aperiodic.** Top "periods" = segment length & harmonics (38.8 h, 19.4 h…), no discrete line; 12-min cadence Nyquist (24 min) cannot resolve p-modes (5 min) — none expected in winding flux anyway. Cadence is not the lever; spatial scale is. | temporal PSD |

**Headline:** the forecastable object is the *asinh-compressed, spatially coarse
(≳6–23 Mm) envelope*, not the full-resolution signed field.

## 1. Auto-EDA module design

Package: `scripts/eda/` (200-line cap per file). One CLI: `python -m scripts.eda.run
[--cubes all|name,...] [--out outputs/eda/]`. Per-cube JSON + cross-cube markdown
report + PNG figures. CPU-only, full 21 cubes ≲ 1 h locally.

| Stage | Module | Computes |
|-------|--------|----------|
| A integrity | `integrity.py` | gaps/sentinels, NaN counts, contiguous-segment table, value distribution (quantiles, kurtosis), harp_8-style outlier census per cube |
| B temporal | `temporal.py` | FFT/Welch on scalar series (mean abs-w, asinh-mean, q99) per contiguous segment; ACF e-folding time; flag any discrete period |
| C spatial | `spatial.py` | radial PSD per frame; slope time-series; slope vs flare-label windows (does spectrum steepen pre-flare?) |
| D coherence | `coherence.py` | k-band lag-correlation matrix (bands × lags 12 min–24 h) per cube → **predictability-horizon map**, the central artefact |
| E transforms | `transforms.py` | grid {raw, abs, asinh, signed_asinh} × blur {0,2,8,32 px} → lag-corr table = model-free predictability ceiling per representation |
| F target-link | `target_link.py` | band-power features vs flare labels (C/M/X × 6/12/24 h) and vs future coarse-frame energy; mutual-information ranking |
| G units | `units.py` | cross-cube amplitude comparability; ARTop hardcodes dx=dy=360 km (see `unit_conversion.md`) ignoring CDELT/foreshortening — regress per-cube amplitude vs disk position to detect systematic scale error |
| report | `report.py` | merge JSONs → `outputs/eda/EDA_REPORT.md` + figures |

Run order A→G; D and E are the decision-driving stages, run them first if pressed.

## 2. Feature-engineering verdicts (ranked by expected impact)

1. **Low-pass or downsample the prediction target.** Predict the asinh field at
   4–8× coarser resolution (or low-pass the loss target at k≈16). F3/F5: full-res
   MSE optimises noise. Likely also explains S31-style staircase divergence —
   autoregressive feedback of unpredictable high-k texture compounds.
2. **Spectral loss weighting.** If full-res output must stay: weight MSE per
   k-band by measured coherence (Stage D output gives the weights directly).
   Cheap arm: Laplacian-pyramid loss with weights ∝ band predictability.
3. **Keep asinh normalisation** (already in pipeline, `softening=1000` validated
   by F2). Do not train any arm on raw/linear scale.
4. **Magnitude-envelope auxiliary channel.** Sign flickers, |w| envelope persists
   (F4). Feed `asinh(|w|)` blurred at r≈8 px as extra input channel; optional
   auxiliary head predicting it.
5. **Longer horizons are viable for the coarse field.** k<4 coherence 0.64 at
   5 h → multi-hour coarse forecasts are signal-supported; full-res multi-step
   is not. Evaluate staircase metrics on the low-pass field.
6. **Unit conversion (Mm²/px) is a constant multiplier** — fold into
   normalisation, fix labels for reports (`G·Mm²`), no effect on learning.
   The real risk is the *ignored CDELT/foreshortening variation* across HARPs
   (Stage G checks); if systematic, add per-cube scale correction at load time.

## 3. Open questions for EDA to answer

- Does PSD slope or band-power ratio shift before M/X flares? (Stage C/F — would
  give a physics-grounded classifier feature beyond pixel rates.)
- Optimal blur radius per cube size — fixed 8 px vs fixed physical 6 Mm?
- Is harp_116's lower coherence (0.48 vs 0.69) size- or activity-driven? Needs
  the full 21-cube sweep to see the spread.

## 4. Next actions

1. Build `scripts/eda/` stages D, E first (decision-driving), then A–C, F, G.
2. Run full 21-cube sweep locally; commit `EDA_REPORT.md`.
3. Design S-series arms from verdicts 1–2 (low-pass target / pyramid loss) with
   bootstrap CSI eval; one knob per arm per [[s30b_six_fix_bundle_toxic]].
