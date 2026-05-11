# SolarFlare — Project Context

> **First read on a fresh session:** `docs/V5_JEPA/09_progress.md` is the
> source of truth for current state. This file just bootstraps you to it.

## Where we are (2026-05-11)

- **Active branch:** `v5-jepa-lora`. V4 stays on `Version_4` as baseline.
- **Active architecture:** **V5.0 Path B — JEPA-from-scratch** (V-JEPA-2-AC
  template at small scale: ViT context + EMA target + block-causal predictor,
  ~55 M total / 33 M trainable, smooth-L1 in embedding space).
- **Path A (LoRA on Surya) is abandoned.** HelioSpectFormer hard-locked to
  `img_size=4096 / 60-min / 13ch`; AR cubes are variable-HxW / 12-min / 1ch.
  Don't re-introduce `transformers` / `peft` / `huggingface_hub`.
- **Sanity status:** All green. Full experiment log: `docs/V5_JEPA/12_experiments.md`.
  - No-mask MPS 5-ep: val 0.0407 (1e8 clip fix, 5× over 1e5 baseline).
  - Mask-ON MPS 50-ep: val 0.0689 ep49, not converged (18-ep curriculum plateau; fix in progress).
  - Mask-OFF MPS 50-ep: val 0.0125 ep38, saturated ep24+ (tiny-dataset limit).
  - path_a CUDA 5060ti (dim=384, 21 cubes): ep5 val 0.127, resuming ep6→19 (E08, running).
  - Mask-ON MPS 100-ep slow curriculum (E09): running on mini_mps.

## Key entry points

| File | Purpose |
|---|---|
| `main_v5.py` | Config-driven entry. `--config <yaml> --max-epochs N --device cuda\|mps\|cpu`. |
| `configs/v5_path_a.yaml` | Path B full spec (filename kept for compat). ViT-Small, 20 epochs. |
| `configs/v5_sanity.yaml` | MPS-friendly tiny config (dim=192, 4 cubes). |
| `models/v5/jepa_model.py` | `V5JEPAModel` — context + EMA target + predictor. |
| `training/jepa_trainer.py` | Single LR group; `update_target_ema()` after each step. |
| `solarflare_data/zarr_loader.py` | Lazy zarr + `WIND_FLUX_CLIP=1e8` sentinel guard. |
| `docs/V5_JEPA/09_progress.md` | **Always check this first.** |

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

The `solarflare-training` skill (in `.claude/skills/`) documents the full
multi-machine queue. See `SKILL.md` for slot list, audit rules, monitor
commands.

## Gotchas

- **MPS `F.scaled_dot_product_attention` returns NaN** with `attn_mask` under
  `torch.no_grad`. `models/v5/predictor.py` routes MPS through a manual
  `(q@kᵀ)·scale → masked_fill → softmax → @v` path. CUDA keeps SDPA. Don't
  unify these without testing val_loss on MPS.
- **Quantity is winding flux, NOT magnetic field strength.** Per senior:
  per-pixel physical max ~1e7; integrated AR total ~1e13–1e14. Earlier
  `BZ_CLIP_GAUSS=1e5` clip was destroying real signal (10⁵–10⁷ legitimate
  peaks). Replaced by `WIND_FLUX_CLIP=1e8` in `zarr_loader.py` (10× safety
  margin). harp_8 still has 14k pathological pixels up to 1.68e10 = 1,680×
  physical max. See `docs/V5_JEPA/OUTLIERS.md` for full audit.
- **`launch_slot.sh` injects `--device <device>`** as first script arg. Any
  script launched on a slot must accept it (main_v5 does).
- **CUDA audit blocks launches** missing `pin_memory=True` / `non_blocking=True`
  / no fp16 GradScaler markers. main_v5 carries `# CUDA-5060ti-validated`
  marker because the actual transfers live in `training/jepa_trainer.py`.

## Data

- `data/*.zarr` — **21 AR cubes**, 12-min cadence, single-channel `wind` + `Time`.
  | Cube | Frames | Cube | Frames |
  |---|---|---|---|
  | harp_8 | 200 | harp_86 | 566 |
  | harp_17 | 91 | harp_116 | 213 |
  | harp_26 | 215 | harp_156 | 220 |
  | harp_43 | 219 | harp_219 | 80 |
  | harp_45 | 125 | harp_221 | 77 |
  | harp_49 | 239 | harp_245 | 465 |
  | harp_51 | 169 | harp_274 | 228 |
  | harp_54 | 256 | harp_316 | 123 |
  | harp_83 | 126 | harp_318 | 161 |
  | harp_11930 | 627 | harp_may2024 | 440 |
  | harp_nov2025 | 437 | | |
- `data/manifest.json` — generated via `scripts/build_zarr_manifest.py`.
- Locked priors: pixel scale 0.364 Mm/px constant; sign convention chiral
  pseudoscalar; no metadata beyond `wind` + `Time`. See
  `docs/V5_JEPA/00_overview.md` for the full lock list.

## Conventions (project-specific)

- 200-line cap per file (IndraAstra-wide, see `/Volumes/T9/IndraAstra/CLAUDE.md`).
- Cube-level holdout for splits — no AR-identity leakage across train/val.
- D4 chiral aug: H/V flip, 90°, 270° flip sign; 180° preserves; identity passes
  through. See `docs/V5_JEPA/06_data.md` §11.4.

## Pending work

1. ~~Mask catalog~~ **DONE** — `solarflare_data/mask_catalog.py`, 5 strategies, curriculum. Validated MPS 5-ep.
2. **path_a full run** — E08 running on 5060ti (ep6→19, resumes from ep5 val 0.127). Check `outputs_v5/run.jsonl`.
3. **Slow curriculum convergence** — E09 running on mini_mps (100 ep, tail_only→ep25, full mix→ep65). Check `outputs_v5_mini_mask_on_slow/run.jsonl`.
4. **Eval suite** — pixel-decoder ablation, CSI/HSS, persistence baseline. Blocked on path_a convergence.
5. **Encoder feature cache** — cache target embeddings to disk once architecture settles. ~2× speedup.
6. **Strategy A follow-up** — visible-only encoder (true V-JEPA sparse tokens). Separate PR.
