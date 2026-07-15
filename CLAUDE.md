# SolarFlare — Project Context (group_id: "manik")

## Goal (2026-06-26 — fresh start)

**Forecast ≥M solar flares from the SPATIAL (2D) structure of magnetic winding
flux density maps** — the literature gap. Everyone integrates winding to a 1D
scalar before forecasting; we keep the map and feed its spatiotemporal structure
to a model, recovering PIL-localized, sign-resolved injection that spatial-mean
cancels away. Measure whether it adds skill over SHARP parameters.

See `.planning/GOAL.md` (authoritative) and `.planning/winding_flux_physics.md`
(physics + literature-gap analysis). EDA evidence in `.planning/eda_winding_flux.md`.

### Abandoned directions (do not reintroduce)
- **Field forecasting** (predict next-K winding-flux frames). EDA-proven dead end:
  winding density is a nonlocal sign-field convolution → near-white spatial PSD,
  lag-1h corr 0.003. The forecastable object is not the per-pixel field.
- **V5 JEPA** (preserved on `v5-jepa-lora`). No JEPA / transformers / peft / HF.
- Old V4 ablation sweep (S0–S46), dual-head curve regression — archived at
  `archive/old_v4_goal/`. Curve regression hit a physical ceiling.

## Why this goal (one line)
Winding density is signed + PIL-concentrated; the spatial mean cancels the
signal. The 2D map carries flare-precursor structure the scalar throws away,
and no prior work has fed winding maps to a deep model. See physics doc §4–6.

## Data
- `data/*.zarr` — HARP active-region cubes, 12-min cadence, single channel
  `wind[H,W,T]` (winding-flux density, signed pseudoscalar) + `Time[T]` (unix s).
- `Time == 0` = sentinel (missing frame), filter before windowing.
- Per-pixel physical max |w| ≈ 1e7; clip guard 1e8 in loader.
- ⚠️ **CONFIRM** what `wind` stores: total dL/dt vs windCur vs windCur−windPot.
  Gates everything. Check ARTop export script / ask senior.
- Flare labels: `data/<harp>_labels_{C,M,X}_{6,12,24}h.npy` from HEK.
- SHARP scalars (TOTUSJH, USFLUX, R_VALUE…) — to fetch from JSOC for baseline.

## Key entry points (current code, V4 backbone reused as encoder)
| File | Purpose |
|---|---|
| `models/cells/convlstm_cell.py` | Base ConvLSTM cell (repurpose: map encoder). |
| `models/sa_convlstm.py` | Self-attention ConvLSTM (MPS-safe manual bmm path). |
| `solarflare_data/harp_loader.py` | Densify zarr → [T,H,W], clip, per-cube norm. |
| `scripts/eda/` | EDA package (temporal/spatial/coherence/cadence probes). |
| `.planning/winding_flux_physics.md` | Physics + lit-gap + novelty roadmap. |

## Gotchas (still live)
- **MPS attention quirk** — SDPA NaN under no_grad+attn_mask; SA-ConvLSTM uses
  manual bmm+softmax. Don't unify without MPS regression test.
- **harp_8 outlier** — values to 1.68e10, ~14k pathological px. Clip + valid-mask.
- **asinh norm** — signed_asinh softening=1e3 (probe-proven), NOT 1e6. Never
  train on raw/linear scale.
- **venv interpreter** — `/Volumes/T9/IndraAstra/.venv` symlink broke (homebrew
  bumped python 3.14.5→3.14.6). Run via `/opt/homebrew/bin/python3.14` with
  `PYTHONPATH=/Volumes/T9/IndraAstra/.venv/lib/python3.14/site-packages`. Repoint
  the symlink to fix permanently.

### Slot vocabulary (training)
- "mps"/"mini" → `mini_mps` (Mac Mini only; do NOT use studio_mps). Crashes on
  ≥10-epoch runs — low-priority confirms only.
- "cuda"/"5060ti"/"remote" → `5060ti_cuda` (all long arms here).
- "cpu" → `mini_cpu`. Controller needs torch on PATH (activate venv first).

## Conventions
- 200-line cap per file (IndraAstra-wide, see `/Volumes/T9/IndraAstra/CLAUDE.md`).
- Planning artefacts under `.planning/`.
- Eval: **leave-AR-out** always. Within-cube AUC is inflated (autocorrelated
  12-min samples leak across split) — report leave-AR-out TSS, not random-split.
