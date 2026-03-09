---
phase: quick-02
plan: 01
subsystem: data-pipeline, training, evaluation
tags: [bugfix, threshold, dual-channel, flare-detection]
dependency_graph:
  requires: []
  provides: [correct-extreme-threshold, working-dual-channel, spatial-density-flare-detection]
  affects: [config.yaml, dataset.py, loader.py, losses.py, trainer.py, main.py, tests]
tech_stack:
  added: []
  patterns: [spatial-density-criterion, normalized-threshold-passthrough]
key_files:
  created: []
  modified:
    - config.yaml
    - preprocess_data.py
    - data_processed/metadata.json
    - utils/config_validator.py
    - solarflare_data/dataset.py
    - solarflare_data/loader.py
    - main.py
    - training/losses.py
    - training/trainer.py
    - tests/test_losses.py
    - tests/test_metrics.py
    - tests/test_config.py
    - generate_comparison.py
decisions:
  - "99th percentile threshold (0.277) replaces incorrect 99.5th percentile (0.3456)"
  - "Spatial density criterion (>2% pixels) replaces np.any() for flare detection"
  - "Normalized threshold (0.3143) stored in metadata for backward compatibility"
metrics:
  duration_seconds: 2577
  completed: "2026-03-09"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 13
  tests_passed: 206
  tests_failed: 0
---

# Quick Task 2: Fix Broken Extreme Threshold Summary

Corrected three fundamental bugs in the extreme threshold system: wrong threshold value (0.3456 -> 0.277), dead dual-channel indicator (raw-space threshold against normalized data), and useless np.any() flare detection replaced with spatial density criterion (>2% of pixels above threshold).

## Changes Made

### Task 1: Config, metadata, and preprocessing fixes (39393e3)

- Changed `extreme_threshold` from 0.3456 to 0.277 in both `loss` and `evaluation` sections of config.yaml
- Changed `extreme_threshold_percentile` from 99.5 to 99 in config.yaml
- Added `flare_density_threshold: 0.02` to the `data` section of config.yaml
- Updated `preprocess_data.py` to compute and store `extreme_threshold_normalized` alongside the raw threshold
- Added `extreme_threshold_normalized: 0.3143` to `data_processed/metadata.json`
- Updated default value comment in `utils/config_validator.py`

### Task 2: Dual-channel and flare detection fixes (3332e2e)

- Fixed `load_preprocessed_data()` in loader.py to read `extreme_threshold_normalized` from metadata instead of the raw value (28070), with backward-compat fallback computation
- Fixed `load_and_prepare_data()` to compute normalized threshold from raw when using asinh normalization
- Replaced `np.any(output_frames > extreme_threshold)` in `build_index()` with spatial density criterion: `np.abs(output_frames) > threshold` mean fraction exceeding `flare_density_threshold`
- Added `flare_density_threshold` parameter throughout: `build_index()`, both loader functions, and `main.py`
- Dual-channel indicator now produces meaningful values (0.018-0.962 range) instead of all zeros

### Task 3: Default threshold updates across codebase (08cf9ea)

- Updated all default `threshold=0.3456` to `0.277` in:
  - `WeightedMAELoss.__init__`, `AsymmetricExtremeLoss.__init__`, `CompositeLoss.__init__`
  - `get_loss_function()` factory fallback
  - `validate()` signature default
  - `train_model()` eval config fallback
  - `main.py` fallback defaults (2 locations)
- Updated all test files to use 0.277: `test_losses.py`, `test_metrics.py`, `test_config.py`
- Updated `generate_comparison.py` threshold references (3 occurrences)

## Verification Results

1. `grep -rn "0\.3456" --include="*.py" --include="*.yaml" .` returns 0 matches
2. `python3 -m pytest tests/ -x -q` -- 206 passed, 1 skipped
3. Dual-channel indicator produces values min=0.018, max=0.962 for normalized flux (not all zeros)
4. Flare detection with spatial density: 0% flagged for low-density data, 57% for high-density data

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Fix config, metadata, and preprocessing | 39393e3 | config.yaml, preprocess_data.py, data_processed/metadata.json |
| 2 | Fix dual-channel indicator and spatial density flare detection | 3332e2e | solarflare_data/dataset.py, solarflare_data/loader.py, main.py |
| 3 | Update all default thresholds and tests | 08cf9ea | training/losses.py, trainer.py, tests/*.py, generate_comparison.py |
