---
phase: "04-data-pipeline"
plan: "03"
subsystem: "config-integration"
tags: ["config", "validation", "main-entry", "backward-compat", "data-pipeline"]
dependency-graph:
  requires: ["04-01", "04-02"]
  provides: ["data-pipeline-wiring", "new-config-fields", "backward-compat-conversion"]
  affects: ["05-mps-compatibility"]
tech-stack:
  added: []
  patterns: ["deprecation-warning-conversion", "config-defaulting"]
key-files:
  created: []
  modified: ["config.yaml", "main.py", "utils/config_validator.py"]
decisions:
  - id: "04-03-01"
    description: "Old train_split/val_split auto-converts to split_ratios with deprecation warning"
  - id: "04-03-02"
    description: "Old augment boolean converts to augmentation mode (true->balanced, false->none)"
  - id: "04-03-03"
    description: "num_workers moved from training section to data section in config.yaml"
metrics:
  duration: "~2 min"
  completed: "2026-02-04"
---

# Phase 04 Plan 03: Config Integration Summary

**One-liner:** Wire new data pipeline (split_ratios, stride, augmentation, num_workers) into main.py, config.yaml, and config validator with backward compatibility for old fields.

## What Was Done

### Task 1: Update config.yaml with new data pipeline fields
- Replaced `train_split`/`val_split` with `split_ratios: [0.7, 0.2, 0.1]`
- Replaced `augment: true` with `augmentation: "none"` (none/balanced/aggressive)
- Added `stride: 1` for sliding window control
- Moved `num_workers` from `training:` section to `data:` section
- Commit: `6e9891e`

### Task 2: Update main.py and config_validator.py
- Updated `run_training()` to extract and pass `augmentation`, `stride`, `split_ratios`, `num_workers`, `device`, and `seed` to loader functions
- Added validation for `augmentation` (must be none/balanced/aggressive)
- Added validation for `split_ratios` (3-element list, each > 0, sum ~1.0)
- Added validation for `stride` (positive integer) and `num_workers` (non-negative integer)
- Added backward compatibility: old `train_split`/`val_split` auto-converts to `split_ratios` with deprecation warning
- Added backward compatibility: old `augment` boolean auto-converts to augmentation mode string
- Removed old `train_split`/`val_split` required-field validation
- Commit: `4bb49cf`

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 04-03-01 | Old train_split/val_split auto-converts to split_ratios | Users with old configs get deprecation warning, not a crash |
| 04-03-02 | augment=true maps to "balanced" mode | Preserves existing behavior (h/v flips) for old configs |
| 04-03-03 | num_workers moved to data section | Logically belongs with data loading, not training hyperparams |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

- Config validator accepts valid new config fields
- Config validator rejects invalid augmentation modes with clear error
- Config validator rejects split_ratios not summing to ~1.0
- Config validator rejects negative stride
- Backward compatibility confirmed: old train_split/val_split converts correctly
- Backward compatibility confirmed: old augment boolean converts correctly
- main.py imports cleanly with no old field references in run_training()

## Next Phase Readiness

Phase 04 (Data Pipeline) is now complete. All three plans executed:
- 04-01: Dataset class rewrite (SolarFluxDataset with mmap, deterministic augmentation)
- 04-02: Loader rewrite (whole-file splitting, training-only normalization, platform-aware workers)
- 04-03: Config integration (wiring, validation, backward compat)

Ready for Phase 05 (MPS Compatibility). No blockers.
