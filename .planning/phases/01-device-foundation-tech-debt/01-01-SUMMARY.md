---
phase: 01-device-foundation-tech-debt
plan: 01
subsystem: device-management
tags: [torch, device-detection, mps, cuda, amp, config]
dependency-graph:
  requires: []
  provides: [resolve_device, get_amp_context, get_grad_scaler, clear_device_cache, device-config, seed-config]
  affects: [01-02, 01-03, phase-02, phase-03, phase-05]
tech-stack:
  added: []
  patterns: [priority-based-device-detection, explicit-error-on-unavailable-device]
key-files:
  created: []
  modified: [utils/device.py, utils/__init__.py, config.yaml]
decisions:
  - id: device-api-shape
    summary: "Four public functions replace single get_device(); device resolved once at startup"
  - id: mps-grad-scaler
    summary: "MPS always gets DummyGradScaler; real GradScaler only for CUDA"
  - id: config-device-field
    summary: "Top-level string field device: auto replaces nested device.use_cuda boolean"
metrics:
  duration: "~5 minutes"
  completed: 2026-02-03
---

# Phase 01 Plan 01: Cross-Platform Device API Summary

**Rewritten utils/device.py with CUDA > MPS > CPU detection, MPS-aware AMP/scaler, device cache cleanup; config.yaml migrated from use_cuda boolean to device: auto string.**

## What Was Done

### Task 1: Rewrite utils/device.py with cross-platform device API
**Commit:** `54426bc`

Completely rewrote `utils/device.py` replacing the old `get_device(use_cuda: bool)` with four public functions:

- **`resolve_device(device_config)`** - Accepts "auto", "cuda", "mps", "cpu". Auto mode detects CUDA > MPS > CPU. Forced device that is unavailable raises `RuntimeError` with clear message (no silent fallback). Prints single startup log line with device details.
- **`get_amp_context(use_amp, device)`** - Returns `torch.amp.autocast` for any device type when AMP enabled, `nullcontext()` when disabled.
- **`get_grad_scaler(use_amp, device)`** - Returns real `GradScaler` only for CUDA+AMP; `_DummyGradScaler` for MPS, CPU, or AMP-disabled.
- **`clear_device_cache(device)`** - Dispatches to `torch.cuda.empty_cache()`, `torch.mps.empty_cache()`, or no-op based on device type.

Updated `utils/__init__.py` to export `resolve_device` and `clear_device_cache` (replacing old `get_device` export).

### Task 2: Update config.yaml device section
**Commit:** `f45c5ac`

- Replaced nested `device: { use_cuda: true }` with top-level `device: "auto"` string field
- Added `seed: 42` top-level field for reproducibility (consumed by Plan 02)
- No other config sections modified

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing Python dependencies for verification**
- **Found during:** Task 1 verification
- **Issue:** The miniforge3 Python environment did not have torch, numpy, or matplotlib installed, which blocked import verification (utils/__init__.py imports from metrics and visualization modules)
- **Fix:** Installed torch, numpy, matplotlib via miniforge3 pip
- **Files modified:** None (environment setup only)

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| AMP context uses `device.type` directly | Cleaner than if/elif chain; torch.amp.autocast supports all device types |
| `_log_device` as private helper | Keeps resolve_device focused; log format isolated for future changes |
| `platform.processor()` for MPS chip name | Standard library approach; returns "arm" on Apple Silicon (sufficient for identification) |
| ValueError for invalid config, RuntimeError for unavailable device | Different error types for different failure modes (user config error vs hardware mismatch) |

## Verification Results

All five verification criteria passed:
1. All four function imports succeed
2. `resolve_device('auto')` returns `mps` on Apple Silicon Mac
3. `resolve_device('INVALID')` raises `ValueError` with clear message
4. `config.yaml` parses with `device` as string and `seed` as integer
5. Old `get_device` function no longer importable

## Next Phase Readiness

Plan 01-02 (wire device into main.py, add seeding) can proceed immediately. The new API is the foundation all downstream plans depend on.
