# M5 — Why winding/helicity scalars are unforecastable (noise EDA, 2026-07-07)

Ran variation + forecastability EDA on the 17-HARP FastTOP scalar series
(`Temporal_Data/HARP_*_Production/FastTOP_Pipeline_Summary.csv`). Question:
the scalar winding/helicity series vary 4–8× more than unsigned flux — is that
signal or noise, and why can't we forecast it?

## Variation (median over 17 HARPs) — winding varies far more than flux
| qty | temporal CV | roughness mean\|Δ\|/scale | dyn range | excess kurtosis | sign-flip |
|---|---|---|---|---|---|
| unsigned flux (ref) | 0.44 | 0.02 | 1.7 | −0.30 | 1% |
| winding (current) | 2.09 | 1.47 | 8.9 | **81** | 43% |
| helicity (current) | 3.45 | 1.11 | 12.6 | **102** | 24% |

Ratio vs flux: winding/hel = 4–8× CV, **55–78× roughness**, 5–8× dyn range.

## Forecastability — the variation is NOISE, not slow structure
| qty | acf@12min | acf@2h | PSD β | noise frac | persist R² | Δincr acf |
|---|---|---|---|---|---|---|
| unsigned flux | **0.97** | 0.83 | −1.58 (red) | 2.5% | **+0.95** | 0.00 |
| free energy | 0.98 | 0.81 | −1.52 (red) | 2.1% | +0.95 | −0.07 |
| winding total | 0.07 | 0.06 | −0.18 | 70% | −0.87 | −0.47 |
| **winding current** | **0.006** | 0.02 | **−0.05 (white)** | **79%** | **−0.99** | −0.50 |
| helicity current | 0.33 | 0.11 | −0.41 | 56% | −0.35 | −0.45 |

Three smoking guns:
1. **Zero autocorrelation** — WindCur lag-1 (one 12-min frame) = 0.006 → decorrelates
   in a single step. Flux = 0.97, >0.4 out to 12h. No temporal memory to extrapolate.
2. **White spectrum** — WindCur PSD slope ≈ 0 (flat = equal power all freqs = noise
   by definition). Flux β=−1.6 (red = smooth low-freq trend = forecastable).
3. **Negative persistence R²** — naive "next=current" scores −0.99 on WindCur, worse
   than a flat constant. Increments anti-correlated (−0.5): every +jump → −jump.
   Violent mean-reverting jitter, not a random walk. Worst case for autoregression.

## Meaning
- **Instantaneous winding value = white noise** → the curve/staircase is unforecastable
  by construction. This is the MECHANISTIC PROOF of the forecasting ceiling every model
  hit (ConvLSTM, TimesFM/Chronos, CNN-LSTM hybrid). Not a modelling failure — the target
  has no extrapolable structure. See [[pretrained_tsfm_curve_ceiling]], [[winding_flux_signal_nature]].
- **Helicity-current keeps real memory** (acf1 0.33 vs winding 0.006; β −0.41 vs −0.05) →
  why helicity is the key classification carrier (AUC 0.60→0.72 when hel twins added).
- **Escape = detect a regime shift in the noise STATISTICS, not forecast the value.**
  Variance, burst-rate, kurtosis (81–102), sign-imbalance, PIL concentration shift before
  flares. That is classification. Justifies MacTaggart's kurtosis/burstiness features and
  our whole pivot [[flare_classification_task]] [[m4_pil_history_result]].

One line: **winding is a white-noise reservoir; the flare precursor is in the SHAPE of
the noise distribution, not in where the value is heading.**

Repro: scratchpad/{var_analysis,forecastability}.py → var_bars.png, forecastability.png,
{var,forecast}_metrics.csv. HARP_8 is the pathological-value cube (overlay spike ok, but
use HARP_49 for clean PSD).
