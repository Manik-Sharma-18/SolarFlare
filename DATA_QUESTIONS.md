# Solar Flare Prediction — Data Audit & Open Questions

**Status:** Current model (V4 transfer-learning ConvLSTM) paused. Architecture being revised from scratch. This document captures everything known about the input data and lists open questions for the data provider.

**Date compiled:** 2026-04-25
**Working tree:** `Version_4` branch (commit `5ef18c9`)

---

## 1. Current Model Recap (paused)

**Task.** Predict next 4 frames of magnetic winding flux from 10 input frames. Spatiotemporal regression, encoder-decoder.

### V4 architecture

| Stage | Component | Details |
|---|---|---|
| Input | dual channel | ch0 = asinh-normalized flux, ch1 = sigmoid extreme indicator @ p99.5 |
| Encoder | 3 SA-ConvLSTM cells | channels [32, 64, 128], kernel 5, stride-2 downsample between L1→L2 |
| Decoder | autoregressive ConvLSTM | 4 steps, teacher forcing 0.0, residual head: `pred = prev + delta_scale * delta` |
| Skip | spatial attention gate | sigmoid mask on encoder L1 hidden state before concat |
| Temporal attn | over encoder temporal outputs | weighted query of past timesteps |
| Output | Conv2D 16→1 | predicts delta, added to prev frame |
| Regularization | MC Dropout 0.15 | uncertainty estimation optional |

### Loss (simplified in V4)

```
Total = 1.0 * L1 + 1.0 * (1 - SSIM)
```

Extreme / temporal-diff / asymmetric / temporal-var terms all DISABLED — were confusing optimizer with 6 competing terms.

### Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 1 (RNN sequential timestep loop dominates cost) |
| Grad clip (max norm) | 0.5 |
| AMP | on |
| Scheduler | constant |
| Patience | 3 |
| Epochs | 5 (debug runs) |
| Flare oversampling | 5x for sequences with >0.5% extreme pixel density |
| Split | whole-file 70 / 20 / 10 (train / test / val) |
| Augmentation | balanced (h/v flips) |

### V4 transfer learning pipeline (`configs/`)

1. **Pretrain** on SDO/HMI SHARP magnetograms — single channel, 256×256 patches, 50 epochs, lr=3e-4, tf=0.5
2. **Finetune** on winding flux — freeze encoder 5 epochs, then unfreeze with 0.1× LR, reinit input conv (C=1 → C=2)

**Status: pretrain data dir `data_magnetogram/` empty — pipeline coded, never run.**

### Lineage

| Version | Commit | Change |
|---|---|---|
| v2 | older | ConvLSTM baseline |
| v3 | `cb0860e` | SA-ConvLSTM + ALiBi + delta_scale + extreme loss |
| v3.1 | `f64d405` | ALiBi removed (counteracted positional encoding) |
| v4 | `5ef18c9` | Transfer-learning pipeline + loss simplified for diagnostics |

### Known pain points (from `improvements.md`)

- Per-timestep MAE spread only ~5% → model collapses to persistence
- t+4 errors ≈ t+1 errors → no genuine temporal dynamics learned
- Quiet-sun dominance: most pixels near 0 even after asinh
- Only 14 winding-flux files (3 different grid shapes)
- Cross-campaign leakage risk if same AR appears in multiple campaigns

---

## 2. Data Audit — what we know

### File inventory

```
data/                      14 raw .npy files, ~20 GB total
data_processed/            14 cubes + metadata.json (asinh-normalized, dense)
data_magnetogram/          EMPTY — pretrain target, not yet acquired
data_magnetogram_processed/ EMPTY
```

### Raw file format

| Property | Value |
|---|---|
| Container | NumPy structured array, single `.npy` per file |
| Fields | `X`, `Y`, `windTotal`, `time` |
| Numeric dtype | `float64` (f8) — 8 bytes per value |
| Time dtype | `datetime64[s]` — second precision |
| Layout | Sparse (X, Y, time, value) tuples — full grid × time tensor flattened to 1D |
| NaN / Inf | **zero** in samples checked |
| Compression | none — raw float64 |

Per-file row counts range 32M to 47M. Each file = `T × H × W` flattened.

### Temporal properties

| Property | Value |
|---|---|
| Cadence | **720 s = 12 min**, exact across all checked files |
| Duration per file | 4 to 24 hours (19 to 120 timestamps) |
| Total timestamps | **1370** unique across all 14 files |
| Total observation time | **274 hours ≈ 11.4 days** |
| Campaigns | May 2024 (4 files), Oct 2024 (3 files), Nov 2025 (7 files) |
| Within-campaign continuity | consecutive days within campaign, but inter-campaign gap = months |

### Spatial properties

| Property | Value |
|---|---|
| Grid resolution | ~0.36 units/pixel (X 0.3604, Y 0.3608 — units assumed Mm) |
| **Three distinct grid shapes** | (440, 884) — May 2024 |
| | (627, 877) — Oct 2024 |
| | (437, 1042) — Nov 2025 |
| Crop unifies to | 437 × 877 (center-crop) |
| Lost pixels via crop | up to ~30% of Nov 2025 spatial extent |
| Domain extent | X up to 318–375 Mm, Y up to 157–225 Mm |
| Grid uniformity | regular Cartesian, fully populated (sparse storage but dense logical grid) |

### Value distribution (`windTotal`)

| Stat | Value |
|---|---|
| Range | ±2×10⁸ (global max ≈ 1.65e8, min ≈ -1.84e8) |
| Mean | ≈ -113 (essentially zero on this scale) |
| Std | ~76,000 |
| Median | 0 |
| Zeros | 8% of pixels (likely outside active region mask) |
| p1 / p99 | -16k / +17k |
| p99.9 | 1.2 × 10⁵ |
| p99.99 | 1.1 × 10⁶ |
| p99.999 | 8.0 × 10⁶ |
| Extreme threshold @ p99.5 (raw) | 28,092 |
| Skew | symmetric around 0, **extremely heavy-tailed** (4 orders of magnitude between p99 and max) |

### Normalization stats (precomputed in `data_processed/metadata.json`)

```
method:                       asinh
softening:                    1000.0
scale:                        7.635   (= arcsinh(p99.99 / 1000))
center:                       0.0
extreme_threshold (raw):      28,092
extreme_threshold_norm:       0.528
data_min:                     -1.84e8
data_max:                     +1.65e8
data_mean:                    -113.2
data_std:                     75,971
```

### Storage cost

| Form | Size |
|---|---|
| Raw structured (fp64, sparse) | ~20 GB |
| Dense cubes (fp32) equiv | ~5–6 GB |
| Per-frame fp32 dense | ~1.5 MB (437 × 877 × 4 bytes) |

---

## 3. Open Questions for Data Provider

### A. Physical / instrumental

1. **Source instrument & pipeline.** Is `windTotal` derived from SDO/HMI vector magnetograms? Which inversion / disambiguation code (HMI VFISV, ME0, etc.)? What spatial registration / reprojection pipeline upstream?
2. **Physical units of `windTotal`.** Mx²/cm⁴? G²·m⁻¹? Dimensional formula behind "winding flux" as computed here? Prior–Berger formulation, Berger–Field gauge-invariant, integrated over volume reduced to surface?
3. **Units of X, Y.** Megameters on solar surface? Pixel index × resolution? CEA-projected coordinates or heliographic?
4. **Pixel resolution exact.** 0.36 Mm/pixel matches HMI 0.5"/pixel × ~720 km — confirm? Or arbitrary CEA grid?
5. **Cadence justification.** 12 min. Resampled from HMI 12-min native cadence, or downsampled from 90s NRT? Any temporal smoothing / averaging window applied?
6. **Active region tracking.** Are spatial frames de-rotated to fixed Carrington frame, or do they track AR center over disk? Different grid sizes per campaign suggest per-AR cutout — confirm.
7. **Which active regions?** AR / NOAA designations for May 2024, Oct 2024, Nov 2025 campaigns? (May 2024 likely = AR 13664, the X8.7 / G5 storm?)
8. **Flare events ground truth.** Which GOES X-ray flares occurred during these windows? Magnitude (C/M/X class), peak times, exact GOES-class? Currently no labels — only proxy via flux density.

### B. Data quality / preprocessing

9. **Why three grid sizes?** Different ARs cut at different bounding boxes? Resolution difference? Intentional tiling per campaign?
10. **Are zeros physical or padding?** 8% exact-zero values. Masked-out (off-disk, low SNR), or genuinely zero winding flux?
11. **Noise floor.** Instrumental noise level for `windTotal`? Below what value should we treat as noise?
12. **Heavy tail values > 10⁷.** Real flare-ribbon signal, or saturated / spurious pixels (cosmic rays, bad pixels, inversion failures)?
13. **NaN policy upstream.** Was NaN-filling applied before delivery (zero NaN in raw)? Does zero replace NaN, or are they genuine?
14. **Data quality flags.** Any QUALITY / CONFIDENCE arrays available alongside? HMI ships per-pixel quality bits.
15. **fp64 necessity.** Values fit easily in fp32 (max ~2e8). fp64 chosen by default, or is precision actually needed below ~1e-7 relative? (Could downcast to fp32, halve storage, no info loss if no.)
16. **Solar B-angle / ephemeris.** Were observations corrected for solar B0 / P-angle? Limb-darkening? Foreshortening at high latitudes?

### C. Coverage / extension

17. **More data available?** How many additional ARs / time windows can be provided? **Critical bottleneck — only ~274 hours total now.**
18. **Magnetogram pairing.** Can corresponding LoS / vector magnetograms be provided for pretraining? Currently `data_magnetogram/` empty.
19. **Pre-flare windows.** Are sequences specifically chosen to cover pre-flare periods, or random AR samples? Need positive/negative balance for forecasting eval.
20. **Class labels.** Per-frame or per-window flare labels: "flare in next N hours, magnitude class"? If not provided, what flare catalog do we cross-reference (GOES, NOAA SWPC, HEK)?
21. **Quiet-AR controls.** Sequences from quiet ARs that did NOT flare, for negative class? Currently unclear what fraction of data covers quiet vs active.

### D. Splits / leakage

22. **Cross-campaign AR leakage.** May 2024 and Oct 2024 might be same AR observed in different Carrington rotations. Confirm AR identity to avoid temporal leakage in train/test split.
23. **Recommended split policy.** Provider's recommendation: split by AR, by date, by Carrington rotation? Anything we should specifically hold out?
24. **Dataset version / DOI.** Is this dataset versioned? Any published reference (paper, data release note) we should cite?

### E. Companion data (would unlock more)

25. **Coronal context.** Are AIA EUV channels (171, 193, 211, 304 Å) available for the same windows? Useful for multi-modal input.
26. **In-situ.** Solar wind / SEP measurements (DSCOVR, ACE, GOES SEM) for the same windows?
27. **Photospheric drivers.** SHARP keywords (USFLUX, MEANGBT, R_VALUE, etc.) per timestamp? These are standard flare-prediction features — could be auxiliary inputs.

---

## 4. Implications for Next-Gen Architecture

Findings that should drive the redesign:

- **Data is fp64 but information content fits fp32.** Drop to fp32 in preprocessing, halve memory.
- **Heavy tail spans 8 orders of magnitude.** Asinh works but the choice of softening (1000) and scale percentile (99.99) are tuned per-dataset — must be re-tuned if data extended.
- **3 grid shapes break naive batching.** Center-crop to 437×877 loses up to 30% of Nov 2025 area. Better: per-AR patch extraction at fixed FOV around AR centroid.
- **1370 timestamps / 274 hours is tiny for sequence learning.** Will not support large architectures from scratch — pretraining (or pure unsupervised objectives like masked-frame modeling) is mandatory.
- **12-min cadence + 14 frames input/output = 168 min context window.** Flare buildup typically hours-days. Architecture must accept much longer context, or use multiscale temporal sampling.
- **8% hard zeros + heavy tail.** Loss must handle dual regime: quiet-sun (predict zero correctly) + extreme (predict large values correctly). Single L1 + SSIM can't.
- **Zero ground-truth flare labels.** Current "extreme = above p99.5 flux" is a proxy, not a true forecasting target. Must align with GOES X-ray catalog before claiming "flare prediction."
- **Same AR in multiple campaigns = leakage hazard.** Confirm AR identity before any new train/test split.
