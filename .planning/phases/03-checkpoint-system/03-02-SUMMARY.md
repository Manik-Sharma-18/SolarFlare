---
phase: 03-checkpoint-system
plan: 02
subsystem: checkpoint-integration
tags: [checkpoint, resume, normalization, inference, trainer, config]
depends_on: [03-01]
provides: [resume_from_config, self_contained_inference, centralized_checkpoint_saves, two_checkpoint_management]
affects: [04-01, 05-01]
tech_stack:
  added: []
  patterns: [two-checkpoint-rolling-delete, normalization-in-checkpoint, dual-config-format-fallback]
key_files:
  created: []
  modified: [training/trainer.py, main.py, inference.py, config.yaml, utils/config_validator.py]
decisions:
  - "Latest checkpoint rolling delete: old latest deleted when new one saved; best_model.pt never deleted by rolling logic"
  - "Inference handles both nested (config.model.X) and flat (config.X) checkpoint config formats"
  - "normalization_params from checkpoint preferred; metadata.json as legacy fallback in inference"
metrics:
  duration: "~4 min"
  completed: "2026-02-04"
---

# Phase 3 Plan 2: Checkpoint Integration Summary

Wire centralized checkpoint I/O into trainer.py (resume + two-checkpoint management), main.py (normalization threading + resume_from), inference.py (self-contained loading from checkpoint), config.yaml (resume_from field), and config_validator.py (resume_from validation).

## Tasks Completed

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Rewrite trainer.py checkpoint logic with resume support | b0521c2 | All torch.save replaced with save_checkpoint(), resume via load_checkpoint_for_resume(), best+latest in checkpoints/ subfolder, emergency uses centralized save |
| 2 | Add resume_from to config, validate, thread normalization | 2eb41ad | config.yaml resume_from field, config_validator checks file exists, main.py passes normalization_params and resume_from, test eval loads from checkpoints/best_model.pt |
| 3 | Update inference.py for self-contained checkpoint loading | 9f52933 | load_model returns (model, normalization_params), uses load_checkpoint_for_inference, handles nested+flat config formats, metadata.json fallback |

## What Was Built

### training/trainer.py
- **Import:** `save_checkpoint`, `load_checkpoint_for_resume` from `utils.checkpoint`
- **Signature:** `train_model()` now accepts `normalization_params: dict = None`
- **Resume:** At training start, if `config['resume_from']` set, calls `load_checkpoint_for_resume()` to restore model/optimizer/scheduler/scaler/patience/history, sets `start_epoch`
- **Checkpoint dir:** Creates `checkpoints/` subfolder under save_dir
- **Best model:** Saved via `save_checkpoint()` to `checkpoints/best_model.pt` when val_loss improves
- **Latest checkpoint:** Rolling `checkpoint_epoch_NNN_valloss_X.XXXX.pt` saved every epoch; old latest deleted
- **Emergency:** Uses `save_checkpoint(emergency=True)` with full state in `checkpoints/` subfolder
- **Removed:** Old `load_checkpoint()` function (was inline `torch.load`)

### main.py
- **Import:** `from utils.checkpoint import load_checkpoint` replaces `from training.trainer import load_checkpoint`
- **normalization_params:** Extracted from metadata, passed to `train_model()`
- **resume_from:** Added to `train_config` dict
- **Test eval:** Loads from `checkpoints/best_model.pt` via centralized `load_checkpoint()`
- **run_inference():** Uses centralized `load_checkpoint()` instead of trainer's old function

### inference.py
- **load_model():** Returns `(model, normalization_params)` tuple instead of just model
- **Self-contained:** Uses `load_checkpoint_for_inference()` -- normalization extracted from checkpoint
- **Config format:** Handles both nested `config.model.X` and flat `config.X` formats
- **load_normalization():** Kept for backward compatibility with deprecation note
- **__main__:** Prefers checkpoint normalization, falls back to metadata.json

### config.yaml
- **resume_from:** New top-level field, defaults to `null` (start fresh)

### utils/config_validator.py
- **resume_from validation:** If set and non-empty, checks file exists (error if not), warns if not `.pt` extension

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **Rolling latest delete safety**: The rolling delete logic explicitly checks `latest_checkpoint_path != checkpoints_dir / 'best_model.pt'` to never accidentally delete the best model file.

2. **Dual config format in inference**: Since train_config is a flat merge but future configs may be nested, inference.py tries `config.get('model', {}).get('X')` first (nested), falling back to `config.get('X')` (flat). This handles both old and new checkpoint formats.

3. **normalization_params precedence**: Checkpoint normalization is preferred over metadata.json. This makes inference self-contained without metadata.json for new checkpoints while supporting legacy ones.

## Verification Results

- No `torch.save` in trainer.py: PASS
- No `torch.load` in trainer.py or inference.py: PASS
- `save_checkpoint` calls in trainer.py (best, latest, emergency): PASS (3 call sites)
- `load_checkpoint_for_resume` in trainer.py: PASS
- `checkpoints_dir` subfolder setup: PASS
- `latest_checkpoint_path` rolling management: PASS
- `resume_from` in config.yaml: PASS (defaults to null)
- `resume_from` validation in config_validator.py: PASS
- `normalization_params` threaded through main.py: PASS
- `load_checkpoint_for_inference` in inference.py: PASS
- All imports succeed: PASS (`python -c "from training.trainer import train_model; from inference import load_model; print('OK')"`)

## Next Phase Readiness

Phase 3 complete. The checkpoint system is fully integrated:
- Trainer saves/loads via centralized module with atomic writes
- Resume restores full training state (model, optimizer, scheduler, scaler, patience, history)
- Inference is self-contained (no metadata.json dependency for new checkpoints)
- Config validation catches missing resume files at startup

Ready for Phase 4 (Data Pipeline Hardening).
