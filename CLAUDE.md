# SolarFlare — Project Context (group_id: "manik")

> **First read on a fresh session:** [`docs/V5_JEPA/INDEX.md`](docs/V5_JEPA/INDEX.md) (entry-point hub), then `09_progress.md` for narrative. This file bootstraps only.

## Where we are (2026-05-12)

- **Active branch:** `v5-jepa-lora`. V4 stays on `Version_4` as baseline.
- **Active architecture:** **V5.0 Path B — JEPA-from-scratch** (V-JEPA-2-AC template at small scale: ViT context + EMA target + block-causal predictor, ~55 M total / 33 M trainable, smooth-L1 in embedding space).
- **Path A (LoRA on Surya) abandoned.** HelioSpectFormer hard-locked to `img_size=4096 / 60-min / 13ch`; AR cubes are variable-HxW / 12-min / 1ch. Don't re-introduce `transformers` / `peft` / `huggingface_hub`.
- **Latest results** — see [`INDEX.md`](docs/V5_JEPA/INDEX.md) best-results table.
  Headline: E09 sanity mask-ON slow curric **val 0.00831 ep98 CONFIRMED**. E12-E15 mask-policy ablation + E16 bigger-backbone arm running.

## Key entry points

| File | Purpose |
|---|---|
| `main_v5.py` | Config-driven entry. `--config <yaml> --max-epochs N --device cuda\|mps\|cpu`. |
| `configs/v5_path_a.yaml` | Path B full spec (filename kept for compat). ViT-Small, 20 epochs. |
| `configs/v5_sanity.yaml` | MPS-friendly tiny config (dim=192, 4 cubes). |
| `models/v5/jepa_model.py` | `V5JEPAModel` — context + EMA target + predictor. |
| `training/jepa_trainer.py` | Single LR group; `update_target_ema()` after each step. |
| `solarflare_data/zarr_loader.py` | Lazy zarr + `WIND_FLUX_CLIP=1e8` sentinel guard. |
| `docs/V5_JEPA/INDEX.md` | **First read.** Hub: best results, concepts, active research, dead ends. |
| `docs/V5_JEPA/09_progress.md` | Narrative source of truth. |
| `docs/V5_JEPA/12_experiments.md` | Live run log + summary table. |

## Run commands

```bash
# Local MPS smoke
python3 scripts/smoke_test_v5.py --config configs/v5_sanity.yaml

# Local MPS sanity
python3 main_v5.py --config configs/v5_sanity.yaml --max-epochs 5

# CUDA via training queue (5060ti)
echo manik > .sf_user                       # one-time
scripts/launch_slot.sh 5060ti_cuda main_v5.py \
  --config configs/v5_sanity.yaml --max-epochs 5
ssh 5060ti -t /usr/bin/tmux attach -t sf-<user>-5060ti_cuda-main_v5
```

`solarflare-training` skill (`.claude/skills/`) documents the multi-machine queue (slot list, audit rules, monitor commands).

## Gotchas

- **MPS `F.scaled_dot_product_attention` returns NaN** with `attn_mask` under `torch.no_grad`. `models/v5/predictor.py` routes MPS through manual `(q@kᵀ)·scale → masked_fill → softmax → @v`. CUDA keeps SDPA. Don't unify without testing val_loss on MPS. (F6)
- **Quantity is winding flux, NOT magnetic field.** Per-pixel max ~1e7. `WIND_FLUX_CLIP=1e8` in `zarr_loader.py` (10× margin). Old `BZ_CLIP_GAUSS=1e5` destroyed real signal. harp_8 has 14k pathological px up to 1.68e10. Full audit: [`docs/V5_JEPA/concepts/wind_flux_clipping.md`](docs/V5_JEPA/concepts/wind_flux_clipping.md). (F2)
- **`launch_slot.sh` injects `--device <device>`** as first script arg. Scripts launched on a slot must accept it (main_v5 does).
- **CUDA audit blocks launches** missing `pin_memory=True` / `non_blocking=True` / no fp16 GradScaler markers. main_v5 carries `# CUDA-5060ti-validated` marker; actual transfers in `training/jepa_trainer.py`.
- **Sync before queuing 5060ti experiments.** Run `scripts/sync_verify.sh --slot 5060ti_cuda --level both --fix` before submitting any job to 5060ti_cuda. Queue submissions bypass sync check; sync failure at launch time silently kills the run.

## Data

- `data/*.zarr` — **21 AR cubes**, 12-min cadence, single-channel `wind` + `Time`. Per-cube frame counts: see `docs/V5_JEPA/06_data.md`.
- `data/manifest.json` — generated via `scripts/build_zarr_manifest.py`.
- Locked priors: pixel scale 0.364 Mm/px constant; sign = chiral pseudoscalar; no metadata beyond `wind` + `Time`. Full lock list: `docs/V5_JEPA/00_overview.md`.

## Session Start

1. `mcp__graphiti__search_memory_facts("recent V5 JEPA experiments findings", group_ids=["manik"])`
2. Read [`docs/V5_JEPA/INDEX.md`](docs/V5_JEPA/INDEX.md) then `09_progress.md`.
3. `scripts/slot_status.sh` — verify queue matches real tmux state. Investigate ghost RUNNING entries before launching replacements.

## Session End

1. Append completed runs to `docs/V5_JEPA/12_experiments.md` (table row + detail section). Promote CONFIRMED findings to `12_experiments_findings.md`.
2. Run `docs-sync` skill — verify no stale claims / 200-line cap breaches / undocumented run.jsonl entries.
3. `mcp__graphiti__add_memory` — one episode per completed experiment (config, result, verdict). Always `group_id="manik"`.
4. `git commit` if milestone-worthy. No commit for transient state.

## Evidence Tags (use in 12_experiments.md, 09_progress.md, commits)

- **CONFIRMED** — clean run, val curve, control present.
- **HYPOTHESIS** — post-hoc explanation, single-run, or cited-only.
- **STALE** — true on prior arch/scale that changed.

Rules: post-hoc failure explanations = `HYPOTHESIS`, not facts. "KILLED" requires `CONFIRMED` evidence; else `KILLED (unvalidated)`. Base changes → retest load-bearing results before citing.

## Conventions (project-specific)

- 200-line cap per file (IndraAstra-wide, see `/Volumes/T9/IndraAstra/CLAUDE.md`).
- Cube-level holdout for splits — no AR-identity leakage across train/val.
- D4 chiral aug: H/V flip, 90°, 270° flip sign; 180° preserves; identity passes through. See `docs/V5_JEPA/06_data.md` §11.4.

## Maintaining This File

- Current truth — git tracks history. No dated log entries here.
- Target ≤95 lines. Cold-path content (experiment bullets, data table, pending work) lives in `docs/V5_JEPA/INDEX.md`.
- Confirm structural changes with user before editing. Wording/data-table updates no confirmation needed.
