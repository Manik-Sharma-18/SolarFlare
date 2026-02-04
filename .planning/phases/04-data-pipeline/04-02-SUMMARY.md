---
phase: "04-data-pipeline"
plan: "02"
subsystem: "data-loading"
tags: ["dataloader", "splitting", "normalization", "multiprocessing", "mmap"]
dependency-graph:
  requires: ["04-01"]
  provides: ["whole-file-splitting", "platform-aware-dataloaders", "training-only-normalization"]
  affects: ["04-03", "05-mps-compatibility"]
tech-stack:
  added: []
  patterns: ["whole-file-split-assignment", "mmap-normalization-sampling", "platform-aware-multiprocessing", "none-filtering-collate"]
key-files:
  created: []
  modified: ["solarflare_data/loader.py", "solarflare_data/__init__.py"]
key-decisions:
  - id: "04-02-split"
    decision: "Whole-file split via seeded random.Random shuffle, remainder assigned to val"
  - id: "04-02-norm"
    decision: "Normalization from training files only, mmap sampled every 100th value"
  - id: "04-02-spawn"
    decision: "macOS uses spawn multiprocessing context; Linux uses default fork"
  - id: "04-02-collate"
    decision: "skip_none_collate filters None from __getitem__ errors, returns None if all fail"
metrics:
  duration: "~3 min"
  completed: "2026-02-04"
---

# Phase 04 Plan 02: Loader Rewrite Summary

Whole-file split assignment with seeded shuffle, training-only mmap normalization, platform-aware DataLoader factory with spawn on macOS, conditional pin_memory, worker seeding, and None-filtering collate.

## Performance

- Duration: ~3 min
- Tasks: 2/2 complete
- No blockers encountered

## Accomplishments

1. Rewrote `load_and_prepare_data` with whole-file splitting -- entire .npy files assigned to train/test/val, never split within a file
2. Added `assign_files_to_splits` function with seeded `random.Random` shuffle and configurable ratios
3. Normalization computed from training files only via mmap subsampling (every 100th value)
4. Rewrote `create_dataloaders` with platform-aware multiprocessing (`spawn` on macOS, default `fork` on Linux)
5. `pin_memory=True` only when device.type == "cuda"
6. `_seed_worker` seeds numpy and stdlib random per worker using PyTorch recommended pattern
7. `_skip_none_collate` filters None samples from dataset error handling
8. `persistent_workers=True` when `num_workers > 0`
9. Rewrote `load_preprocessed_data` with same whole-file splitting pattern
10. Updated `__init__.py` to export `create_dataloaders` and `assign_files_to_splits`

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Rewrite loader.py | 8fad88e | solarflare_data/loader.py |
| 2 | Update __init__.py exports | 8abfeea | solarflare_data/__init__.py |

## Files Modified

- `solarflare_data/loader.py` -- Complete rewrite of splitting, loading, and DataLoader creation
- `solarflare_data/__init__.py` -- Added create_dataloaders and assign_files_to_splits exports

## Decisions Made

1. **Whole-file split via seeded shuffle:** `random.Random(seed).shuffle(indices)` then partition by ratio boundaries. Remainder goes to val; negative val guard reduces train.
2. **Training-only normalization:** Only mmap training file indices, sample every 100th value, feed to existing `_compute_norm_params`. Val/test never contribute to normalization stats.
3. **Dataset norm_method mapping:** `robust` and `fixed` norm methods map to `"linear"` in SolarFluxDataset (which applies `(x - center) / scale`); `asinh` maps to `"asinh"`.
4. **macOS spawn context:** `multiprocessing_context="spawn"` on Darwin avoids fork-safety issues with mmap file descriptors. Only set when `num_workers > 0`.
5. **skip_none_collate:** Returns `None` when entire batch is None (training loop must handle). Filters individual None samples silently.
6. **Structured array handling:** Detected at load time; converted to temp .npy cubes for mmap access by SolarFluxDataset.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- 04-03 (integration/config wiring) can proceed -- loader API is stable
- `load_and_prepare_data` signature changed (split_ratios replaces train_split/val_split, augmentation replaces augment_train, seed added) -- main.py callers will need updating in 04-03
- `create_dataloaders` now requires `device` parameter for pin_memory -- callers need updating
