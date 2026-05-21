# Wind flux clipping — `WIND_FLUX_CLIP = 1e8`

**Quantity:** winding flux (NOT magnetic field strength). Sign = chirality (pseudoscalar).
**Physical reference (per senior 2026-05-08):** per-pixel max ~**1e7**; integrated AR total ~**1e13–1e14**.
**Threshold:** `WIND_FLUX_CLIP = 1e8` (10× safety margin above per-pixel physical max).
**Code:** `solarflare_data/zarr_loader.py`. **Audit script:** `scripts/outlier_report.py`.
**Evidence:** 5-ep MPS no-mask val 0.202 → **0.0407** after 1e5 → 1e8 (5× better, see E03 + F2).

---

## History — what was wrong before

Earlier code used `BZ_CLIP_GAUSS = 1e5` treating values as Gauss field strength (vs 5,000 G umbra max). Wrong twice:
1. Wrong quantity — winding flux, not B.
2. Wrong threshold — clipped legitimate per-pixel peaks (10⁵–10⁷) as "outliers."

Re-running at 1e8 reveals a clean picture: 9/10 cubes ≤832 outlier pixels (≤0.0005% frac). Only **harp_8** genuinely pathological.

---

## Outliers above 1e8 (per cube)

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

Audit predates 2026-05 ingest (harp_may2024, harp_nov2025). Re-run when next ingest happens.

---

## Observations

1. **harp_8 dominant.** 14,220 outlier px (0.015%), peak `+1.68e+10` = **1,680× physical max**. σ 3.8M → 174k after clip. 159/732 frames affected.
2. **9 other cubes near-clean.** All ≤832 outlier px, frac ≤0.0005%.
3. **Outliers sparse + random.** Not clustered in time/space (median 1 px per affected frame). Rare numerical glitches, not pipeline artifact.
4. **Sign mixed.** Some +, some −. Not a one-sided sentinel.
5. **`Time = 0` entries** appear in `harp_17[t=0]` + `harp_43[t=last]` (unix epoch). Loader filters via `Time > 0` (`06_data.md` §11.5). Independent of `wind` outliers.

---

## Current guard (code)

`solarflare_data/zarr_loader.py`:

```python
WIND_FLUX_CLIP: float = 1.0e8   # per-pixel winding flux; physical max ~1e7

finite = np.isfinite(raw)
extreme = finite & (np.abs(raw) > WIND_FLUX_CLIP)
wind[extreme] = 0.0                          # zero them out
valid_pixel_mask = finite & ~extreme         # marked invalid for loss
```

Outliers contribute zero to model input + excluded from per-pixel loss via `valid_pixel_mask`. Per-cube `nanmean/nanstd` for z-score uses `valid_pixel_mask` only.

---

## Open questions for senior

1. **Units of `wind`.** G·cm², G/cm, G·cm⁻¹·s⁻¹, dimensionless? Anchors per-pixel 1e7.
2. **harp_8 pathology.** Why 30× more outliers than next-worst (harp_11930)? Known May 2010 event or pipeline issue specific to that AR?
3. **Threshold tightness.** 1e8 (current, 10× margin) vs 5e7 (catches harp_17/harp_83 borderline peaks but risks discarding real extremes)?
4. **Source-side fix.** Zero (current) / interpolate / mask-and-inpaint / fix upstream winding-flux integration? Sparse-random distribution suggests source fix long-term.
5. **`Time = 0` sentinels.** Upstream metadata flag, or only post-hoc via `Time > 0`?
