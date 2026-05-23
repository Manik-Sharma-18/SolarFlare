# SolarFlare — Project Context (group_id: "manik")

## Where we are (2026-05-21)

- **Active branch:** `Version_4`. ConvLSTM-based solar winding-flux forecaster.
- **V5 JEPA abandoned** (preserved on `v5-jepa-lora`). Don't reintroduce JEPA / V-JEPA-2-AC / transformers / peft / huggingface_hub.
- Predecessor V3 also available on `Version_3`.

## Architecture (V4)

- **Backbone:** ConvLSTM + self-attention ConvLSTM (`models/sa_convlstm.py`), spatial attention (`models/attention.py`), predictor head (`models/predictor.py`), MC-dropout uncertainty (`models/uncertainty.py`).
- **Task:** sequence-to-sequence winding-flux frame prediction (predict next-K frames from past-N).
- **Transfer learning:** pretrain on magnetograms (`configs/pretrain_magnetogram.yaml`), fine-tune on winding flux (`configs/finetune_winding_flux.yaml`).
- See `README.md` and `architecture.md` for full description; `improvements.md` for change history.

## Key entry points

| File | Purpose |
|---|---|
| `main.py` | Training entry point. |
| `inference.py` | Eval / forecast generation. |
| `preprocess_data.py` | Build training tensors from raw cubes. |
| `generate_comparison.py` | v3 vs v2 comparison plots. |
| `models/convlstm.py` | Base ConvLSTM cell. |
| `models/sa_convlstm.py` | Self-attention variant (SAM mechanism, MPS-safe). |
| `configs/finetune_winding_flux.yaml` | Main config. |
| `.planning/STATE.md` | Project state, milestone tracker. |
| `.planning/PROJECT.md` | Authoritative project reference. |

## Data

- `data/*.zarr` — 21 active-region cubes, 12-min cadence, single-channel `wind` + `Time` (unix epoch seconds).
- Layout: `wind[H, W, T]`, `Time[T]`.
- `Time == 0` = sentinel (missing frame), filter before windowing.
- Per-pixel physical max $|w| \approx 10^7$; clip guard $10^8$ in loader.

## Gotchas

- **MPS attention quirks** — SDPA returns NaN under `torch.no_grad` with `attn_mask`; SA-ConvLSTM uses manual `bmm + softmax` path. Don't unify without MPS regression test.
- **harp_8 outlier** — pixel values up to $1.68 \times 10^{10}$, ~14k pathological pixels. Clip + valid-mask before loss.
- **Center-crop, not bilinear resize** — spatial dim alignment uses center-crop to 437×877 per quick-03 plan.

## V5 leftover (untracked on V4)

Working tree contains V5 outputs / models / TinyTeX / email_data — gitignored on V4, preserved on `v5-jepa-lora`. Safe to delete if disk needed. Don't import from `models/v5/`.

## Session Start

1. Check current `.planning/STATE.md` for active milestone + last activity.
2. Read `.planning/PROJECT.md` and `.planning/codebase/ARCHITECTURE.md`.
3. Pick up from `last_activity` recorded in STATE.

## Conventions

- 200-line cap per file (IndraAstra-wide, see `/Volumes/T9/IndraAstra/CLAUDE.md`).
- Planning artefacts under `.planning/phases/<NN>-name/` with `NN-MM-PLAN.md` + `NN-MM-SUMMARY.md` pattern.
