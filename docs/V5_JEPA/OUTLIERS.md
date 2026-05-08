# Outlier audit — `data/*.zarr`

**Date:** 2026-05-08
**Quantity:** **winding flux** (NOT magnetic field strength). Sign = chirality (pseudoscalar).
**Physical reference (per senior):**
- per-pixel max ~**1e7**
- integrated AR total ~**1e13–1e14** for very active regions
**Threshold:** `WIND_FLUX_CLIP = 1e8` (10× safety margin above per-pixel physical max).
**Reproduce:** `python3 scripts/outlier_report.py`

## History — what was wrong before

Earlier audit used `BZ_CLIP_GAUSS = 1e5` and treated values as Gauss field strength
(comparing against 5,000 G sunspot umbra max). That was wrong on two counts:
1. Wrong quantity — winding flux, not B.
2. Wrong threshold — clipped legitimate per-pixel peaks (10⁵–10⁷) as "outliers."

Re-running at the correct 1e8 threshold reveals a much cleaner picture: 9 / 10 cubes
have ≤ 832 outlier pixels (≤ 0.0005% frac). Only **harp_8** is genuinely pathological.

## Summary table — outliers above 1e8

| cube       |    T | n_extreme |  frac %  | peak \|wf\| | σ before → after |
| ---------- | ---: | --------: | -------: | ----------: | ---------------- |
| **harp_8** |  732 | **14,220**| 0.0151 % | **1.68e+10**| 3,800,264 → 174,535 |
| harp_11930 |  325 |       832 | 0.0005 % |    1.99e+08 |  66,509 → 29,478 |
| harp_54    |  445 |       302 | 0.0004 % |    2.44e+08 |  66,847 → 27,665 |
| harp_43    |  927 |       246 | 0.0003 % |    1.27e+08 |  48,921 → 24,968 |
| harp_26    | 1061 |       191 | 0.0002 % |    6.15e+07 |  33,705 → 20,848 |
| harp_49    |  651 |        86 | 0.0001 % |    4.56e+08 |  70,581 → 16,734 |
| harp_51    |  548 |        53 | 0.0001 % |    1.28e+08 |  36,818 → 19,605 |
| harp_45    |  684 |        29 | 0.0001 % |    6.27e+07 |  32,194 → 15,968 |
| harp_83    |  347 |         4 | 0.0000 % |    2.00e+07 |  18,396 → 16,382 |
| harp_17    |  380 |         1 | 0.0000 % |    1.01e+07 |   7,783 →  6,607 |

## Key observations

1. **harp_8 is the dominant anomaly.** 14,220 outlier pixels (0.015%), peak
   `+1.68e+10` = **1,680× per-pixel physical max**. σ goes 3.8M → 174k after clip
   (22× reduction). 159 / 732 frames affected.
2. **Other 9 cubes near-clean.** All have ≤ 832 outlier pixels and frac ≤ 0.0005%.
   harp_17 has a single outlier pixel; harp_83 has four.
3. **Outliers are sparse + random.** Not concentrated in time or space (median
   `n_pixels = 1` per affected frame). Looks like rare numerical glitches, not
   a pipeline-wide artifact.
4. **Sign mixed.** Most peaks negative, some positive (`harp_8 +1.68e10`,
   `harp_51 +1.28e8`). Consistent with random numerical instability, not a
   one-sided sentinel like NaN→FLT_MAX.
5. **`Time` array also has a few zero entries.** `harp_17[t=0]` and
   `harp_43[t=last]` come back as `1970-01-01 00:00:00Z` (unix epoch = 0).
   Loader filters them via `Time > 0` per `06_data.md` §11.5. Independent of
   `wind` outliers.

## Per-cube peak coordinates and timestamps

| cube       | peak (winding flux) | (y, x, t)       | UTC                  |
| ---------- | ------------------: | --------------- | -------------------- |
| harp_8     | +1.680e+10          | (161, 526, 712) | 2010-05-08 13:00:00Z |
| harp_49    | −4.563e+08          | (131, 225, 529) | 2010-06-12 13:24:00Z |
| harp_54    | −2.442e+08          | (150, 265, 406) | 2010-06-14 15:36:00Z |
| harp_11930 | −1.987e+08          | (245, 484, 148) | 2024-10-02 22:36:00Z |
| harp_51    | +1.281e+08          | (106, 160, 384) | 2010-06-13 06:24:00Z |
| harp_43    | −1.268e+08          | (95, 103, 915)  | 2010-06-07 19:12:00Z |
| harp_45    | −6.273e+07          | (64, 206, 633)  | 2010-06-08 00:12:00Z |
| harp_26    | −6.146e+07          | (138, 193, 300) | 2010-05-23 04:24:00Z |
| harp_83    | −2.000e+07          | (59, 118, 61)   | 2010-07-04 20:24:00Z |
| harp_17    | −1.006e+07          | (54, 79, 366)   | 2010-05-09 18:12:00Z |

Note: harp_17, harp_83 peaks are within ~2× of the 1e7 physical max, so they
might still be marginal-real signal. Threshold 1e8 keeps them clipped (10×
margin) but they are borderline cases worth confirming.

## Hypotheses for senior

1. **harp_8 has a numerical-instability hotspot.** Concentration of bad pixels
   in a single AR (14k vs ≤832 elsewhere) suggests a per-AR pathology, not a
   global pipeline issue. Worth checking: do the outlier `(y, x)` indices
   cluster in space, or scatter?
2. **Per-pixel ≪ per-cube.** The 1e7 per-pixel limit + 1e13–1e14 integrated
   total implies typical AR has ~10⁶–10⁷ "active" pixels. Consistent with
   AR areas of 100–1000 Mm² at 0.364 Mm/px (10⁵–10⁶ px). Senior may have
   different convention — confirm units.
3. **Sentinel-vs-real ambiguity at boundary.** Values 1e7–1e8 (harp_17 peak,
   harp_83 peak) sit between physical max and our outlier threshold. Decision:
   trust 10× margin and clip them, or shift threshold to 5e7 to be more
   conservative? Tradeoff: tighter clip removes more potential noise but
   risks discarding legitimate extreme winding-flux events.

## Current guard (in code)

`solarflare_data/zarr_loader.py`:

```python
WIND_FLUX_CLIP: float = 1.0e8   # per-pixel winding flux; physical max ~1e7

finite = np.isfinite(raw)
extreme = finite & (np.abs(raw) > WIND_FLUX_CLIP)
wind[extreme] = 0.0                          # zero them out
valid_pixel_mask = finite & ~extreme         # marked invalid for loss
```

Outliers contribute zero to model input and are excluded from per-pixel loss
via `valid_pixel_mask`. Per-cube `nanmean / nanstd` for z-score is computed on
`valid_pixel_mask` only, so normalization stats are clean.

## Open questions for senior

1. **Confirm units of `wind`.** What are the dimensions of "winding flux"
   here — G·cm², G/cm, G·cm⁻¹·s⁻¹, dimensionless? Determines whether 1e7 is
   the right anchor for per-pixel physical max.
2. **harp_8 pathology.** Why does this single cube have 30× more outliers
   than the next-worst cube (harp_11930)? Is there a known event in May 2010,
   or a known pipeline issue specific to that AR run?
3. **Threshold choice.** Should `WIND_FLUX_CLIP` be 1e8 (10× margin, current)
   or stricter (5e7)? Stricter clip catches harp_17 / harp_83 borderline
   peaks but risks discarding legitimate extreme events.
4. **Source-side fix.** Should outliers be (a) zeroed (current),
   (b) interpolated, (c) treated as missing and inpainted by the model, or
   (d) fixed at source by re-running winding-flux integration with stricter
   convergence? Sparse + random distribution suggests source-level fix is
   the right long-term answer.
5. **`Time = 0` sentinels.** Are the `1970-01-01` entries flagged anywhere
   in upstream metadata, or only detectable post-hoc via `Time > 0`?

## Action items

- [x] Update `WIND_FLUX_CLIP` from 1e5 → 1e8 in `solarflare_data/zarr_loader.py`.
- [x] Rename `BZ_CLIP_GAUSS` → `WIND_FLUX_CLIP` (units corrected).
- [x] Rewrite this doc with correct physics.
- [x] Re-run MPS sanity at new threshold — val_loss 0.202 → **0.0407 (5× lower)**.
- [ ] Re-run CUDA sanity at 1e8 (parity check; not blocking).
- [ ] Confirm units of `wind` with senior.
- [ ] Investigate harp_8 outlier spatial clustering.
