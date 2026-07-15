# M0 — `wind` cube semantics (RESOLVED 2026-06-26)

## Answer

`data/<harp>.zarr : wind[H,W,T]` = **total winding flux density** per pixel
(ARTop `windTotal`), a **signed pseudoscalar**. NOT windCur, NOT windCur−windPot.

Evidence:
- Upstream raw field is literally named `windTotal` (README, MODEL_ARCHITECTURE,
  loader). zarr carries no metadata attrs (empty) — naming is the provenance.
- Signed & bipolar: valid frame (harp_11930 t=180) frac_neg 0.541 / frac_pos
  0.459, range −8.40e6 … +6.65e6. Consistent with per-pixel max ~1e7.
- Decomposition maps (windCur / windPot / emergence / braiding) **NOT on disk** —
  no cur/pot/emerg/braid files in `data/`. Only the total is stored.

## Headline result (supports the whole thesis)

**Spatial cancellation `|Σw| / Σ|w| = 0.034`** on a valid frame. The signed
spatial sum/mean — the scalar every prior paper forecasts — **cancels to ~3.4%
of the total winding magnitude**. ~97% of the signal magnitude is destroyed by
+/− cancellation when you integrate. Direct empirical motivation for keeping the
2D map. Pre-registration: confirm this ratio holds cube-wide in M1.

## Implications

1. **Limitation:** Williams' strongest features are the *current-carrying*
   components (δL_c). We have **total only** → cannot replicate δL_c without
   ARTop re-export (windCur/windPot) or an author data request. Track A uses
   total winding; flag δL_c as a future upgrade (Track B / re-export).
2. **Cancellation-aware channels are mandatory** (not optional): feed
   +winding / −winding / |winding| separately, since the signed mean self-erases.
3. asinh softening 1e3 still applies (heavy tails, max ~8e6).

## Label landscape (for M2 leave-AR-out)

25 labeled cubes (harp_may2024, harp_nov2025 unlabeled). Per-frame boolean
≥class within {6,12,24}h from HEK.
- **11 flaring cubes** (≥1 M/X event), **14 quiet** (0 events).
- Biggest: harp_892 (30M/4X), harp_1028 (18M/2X), harp_11930 (12M/2X),
  harp_833 (10M/4X), harp_2748 (6M/6X).
- Small N vs Williams 232 → Track A = proof-of-concept of the *mechanism*
  (spatial > scalar), not a SOTA-scale number. Report leave-AR-out TSS with
  honest CI; the 232-region claim needs Track B data.
