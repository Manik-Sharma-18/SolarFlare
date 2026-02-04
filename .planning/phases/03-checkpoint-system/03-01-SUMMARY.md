---
phase: 03-checkpoint-system
plan: 01
subsystem: checkpoint-io
tags: [checkpoint, torch-save, atomic-write, cross-device, versioning]
depends_on: [01-01]
provides: [save_checkpoint, load_checkpoint, load_checkpoint_for_resume, load_checkpoint_for_inference, CHECKPOINT_VERSION]
affects: [03-02, 03-03]
tech_stack:
  added: []
  patterns: [atomic-write-via-os-replace, cpu-normalized-tensors, versioned-checkpoint-dict]
key_files:
  created: [utils/checkpoint.py]
  modified: [utils/__init__.py]
decisions:
  - "_DummyGradScaler (private) imported directly from utils.device since no public alias exists"
  - "load_checkpoint always maps to CPU regardless of device arg; caller moves to target device"
  - "Config diff reports removals and additions, not just value changes"
metrics:
  duration: "~2 min"
  completed: "2026-02-04"
---

# Phase 3 Plan 1: Checkpoint I/O Module Summary

Centralized checkpoint save/load module with atomic writes, version validation, CPU tensor normalization, architecture mismatch detection, optimizer device remapping, and config diffing.

## Tasks Completed

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Implement save_checkpoint with atomic writes and CPU tensor normalization | c79bd27 | _atomic_save (tempfile + os.replace), _move_optimizer_state_to_cpu, save_checkpoint with full schema |
| 2 | Implement load_checkpoint with version validation, architecture mismatch, config diffing | 30389c9 | load_checkpoint, load_checkpoint_for_resume, load_checkpoint_for_inference, _diff_configs, _check_state_dict_compatibility, _optimizer_to_device |

## What Was Built

### utils/checkpoint.py

**Constants:**
- `CHECKPOINT_VERSION = 1` -- integer version stamp for format compatibility

**Private helpers:**
- `_move_optimizer_state_to_cpu()` -- recursive tensor-to-CPU walker for optimizer state
- `_atomic_save()` -- tempfile in same dir + torch.save + os.fsync + os.replace
- `_diff_configs()` -- recursive config dict comparison returning human-readable diffs
- `_check_state_dict_compatibility()` -- compares model vs checkpoint state dict keys, raises RuntimeError with diff
- `_optimizer_to_device()` -- moves optimizer state tensors to target device after loading

**Public functions:**
- `save_checkpoint()` -- builds full checkpoint dict (version, model, optimizer, scheduler, scaler, early stopping, normalization, config, history, emergency metadata), moves all tensors to CPU, writes atomically
- `load_checkpoint()` -- validates version, loads with map_location='cpu' and weights_only=False
- `load_checkpoint_for_resume()` -- restores model/optimizer/scheduler/scaler, diffs config, returns (start_epoch, best_val_loss, patience_counter, history, normalization_params)
- `load_checkpoint_for_inference()` -- loads checkpoint, resolves device, returns (checkpoint, device)

### utils/__init__.py

Added exports: save_checkpoint, load_checkpoint, load_checkpoint_for_resume, load_checkpoint_for_inference, CHECKPOINT_VERSION

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **_DummyGradScaler import**: The class is private (`_DummyGradScaler`) in device.py. Imported directly since no public alias exists. This is an internal cross-module reference within the utils package.

2. **load_checkpoint device parameter**: Kept `device=None` parameter for API symmetry but always uses `map_location='cpu'`. The caller (load_checkpoint_for_resume/inference) handles device placement.

3. **Config diff completeness**: _diff_configs reports key removals (`<removed>`) and additions (`<absent>`) in addition to value changes, providing full visibility into config drift.

## Verification Results

- All public functions importable: PASS
- `os.replace` atomic write pattern present: PASS
- `weights_only=False` explicit (no FutureWarning): PASS
- `map_location='cpu'` cross-device loading: PASS
- Version check rejects mismatched versions: PASS (RuntimeError with clear message)
- Architecture mismatch shows key diff: PASS
- CPU tensor normalization for model and optimizer: PASS

## Next Phase Readiness

Ready for 03-02 (trainer wiring): trainer.py can now call `save_checkpoint()` and `load_checkpoint_for_resume()` instead of inline `torch.save`/`torch.load`. The emergency checkpoint code in trainer.py should also be updated to use `save_checkpoint(emergency=True)`.
