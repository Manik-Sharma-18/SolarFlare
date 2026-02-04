---
phase: 02-config-validation-error-handling
plan: 01
subsystem: config-validation
tags: [validation, error-handling, config, data-loading]

dependency_graph:
  requires: [01-01, 01-02]
  provides: [validate_config, ConfigValidationError, DataValidationError, error_handling config section]
  affects: [02-02, 03-01, 06-01]

tech_stack:
  added: []
  patterns: [accumulate-all-errors, pre-flight-scan, fail-fast-at-startup]

key_files:
  created:
    - utils/config_validator.py
  modified:
    - utils/__init__.py
    - main.py
    - config.yaml
    - solarflare_data/loader.py

decisions:
  - id: 02-01-A
    description: ConfigValidationError accumulates all errors into a list, raises once at end
    rationale: Users fix everything in one pass instead of iterative fix-run cycles
  - id: 02-01-B
    description: Pre-flight scan uses mmap_mode='r' for minimal I/O during validation
    rationale: Avoids loading full files into memory just to check structure
  - id: 02-01-C
    description: Warnings use logging module, errors accumulate as strings
    rationale: Warnings are non-fatal and benefit from log-level filtering; errors must be collected

metrics:
  duration: ~4 min
  completed: 2026-02-03
---

# Phase 02 Plan 01: Config Validation & Data Pre-flight Summary

Startup config validation and data file pre-flight scanning so bad configurations are caught with clear messages before wasting compute, and corrupted data files produce actionable failure reports.

## What Was Done

### Task 1: Config validation module (9415105)

Created `utils/config_validator.py` with `validate_config(config)` that:

- Validates all required keys with type checks (device, seed, data, model, training, loss, normalization)
- Performs cross-field validation: dual_channel vs input_channels mismatch, AMP on CPU
- Emits warnings (via logging) for unusual-but-valid values: high LR, high batch size, patience > epochs, dropout without uncertainty
- Accumulates ALL errors before raising a single `ConfigValidationError`
- Wired into `main.py` `run_training()` immediately after config load, before `seed_everything()`
- Added `error_handling` section to `config.yaml` with `max_consecutive_nan`, `grad_norm_warning_threshold`, `data_failure_threshold`

### Task 2: Data file pre-flight scan (fb14cbd)

Added pre-flight validation to `solarflare_data/loader.py`:

- `_preflight_scan_npy()`: memory-maps each .npy file, verifies structured array with required fields (X, Y, time, windTotal)
- `_preflight_scan_npz()`: loads each .npz file, verifies 'data' key exists
- Both functions enforce configurable `failure_threshold` (default 10%)
- If threshold exceeded: raises `DataValidationError` with per-file error details
- If under threshold: logs warnings for failed files, continues with valid ones
- Added `failure_threshold` parameter to both `load_and_prepare_data()` and `load_preprocessed_data()`
- Wired threshold from `config.error_handling.data_failure_threshold` in `main.py`

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 02-01-A | Accumulate all config errors, raise once | One fix-run cycle instead of many |
| 02-01-B | mmap_mode='r' for pre-flight scan | Minimal I/O, no full file load |
| 02-01-C | Warnings via logging, errors via list | Non-fatal vs collected pattern |

## Verification Results

All 8 verification checks passed:
1. Config validator imports successfully
2. dual_channel + input_channels mismatch caught
3. Multiple errors (bad device + negative LR + bad split) reported at once
4. lr=1.0 warns but does not raise
5. DataValidationError imports successfully
6. failure_threshold parameter present on load_and_prepare_data
7. validate_config called before seed_everything in main.py
8. config.yaml has error_handling section with all three keys

## Next Phase Readiness

Plan 02-02 (NaN/gradient handling + graceful shutdown) can proceed. The `error_handling` config section created here provides the `max_consecutive_nan` and `grad_norm_warning_threshold` values that 02-02 will consume in the training loop.
