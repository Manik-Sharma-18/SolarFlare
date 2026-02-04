---
phase: 05
plan: 02
subsystem: inference-memory
tags: [mps, uncertainty, welford, memory, gc]
dependency_graph:
  requires: []
  provides: [memory-efficient-uncertainty, gc-cache-clearing]
  affects: [05-03, 06-01]
tech_stack:
  added: []
  patterns: [welford-online-algorithm, gaussian-confidence-intervals]
key_files:
  created: []
  modified: [models/uncertainty.py, utils/device.py]
decisions:
  - id: "05-02-welford"
    summary: "Welford's online algorithm for O(1) MC Dropout memory"
  - id: "05-02-gaussian-ci"
    summary: "Gaussian approximation (mean +/- z*std) replaces torch.quantile"
  - id: "05-02-gc-collect"
    summary: "gc.collect() before device cache clear on MPS/CUDA"
metrics:
  duration: ~1 min
  completed: 2026-02-04
---

# Phase 5 Plan 2: Memory-Efficient Uncertainty and Cache Clearing Summary

**One-liner:** Welford's online algorithm for O(1) MC Dropout uncertainty; gc.collect() before device cache clearing on MPS/CUDA.

## What Was Done

### Task 1: Rewrite predict_with_uncertainty with Welford's online algorithm
- Replaced O(N) `predictions.append()` + `torch.stack()` with Welford's running mean/variance accumulators
- Added `_welford_update()` helper for reuse across both functions
- `predict_with_uncertainty` now maintains `count`, `mean`, `m2` tensors -- constant memory regardless of `n_samples`
- `predict_with_confidence_intervals` replaced `torch.quantile` (broken on MPS) with Gaussian approximation: `mean +/- z * std`
- Z-score lookup table for common confidence levels (0.90, 0.95, 0.99) with `torch.erfinv` fallback for arbitrary levels
- Model mode restored via `try/finally` for exception safety
- Function signatures and return types unchanged
- **Commit:** 769bd70

### Task 2: Enhance clear_device_cache with gc.collect()
- Added `import gc` to `utils/device.py`
- `gc.collect()` called before `torch.cuda.empty_cache()` and `torch.mps.empty_cache()`
- Releases dangling Python tensor references before device runtime reclaims memory
- Critical for MPS which has known memory leak issues with unreferenced tensors
- CPU branch remains no-op (OS/Python GC manages naturally)
- **Commit:** 97b6468

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 05-02-welford | Welford's online algorithm for MC Dropout | O(1) memory vs O(N) stacking; prevents OOM on large models |
| 05-02-gaussian-ci | Gaussian approximation replaces torch.quantile | torch.quantile produces wrong results on MPS; CLT valid for n>=20 |
| 05-02-gc-collect | gc.collect() before empty_cache on MPS/CUDA | Releases Python-side tensor references; prevents memory leaks in long runs |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- `predict_with_uncertainty` returns correct shapes, non-negative std
- `predict_with_confidence_intervals` returns upper >= lower bounds
- Zero `torch.stack` usage in uncertainty.py
- Zero `torch.quantile` usage in uncertainty.py (only docstring mention)
- `gc.collect` present in device.py
- `clear_device_cache` works without errors

## Next Phase Readiness

- Memory-efficient uncertainty estimation ready for MPS testing
- Cache clearing enhanced for long training stability
- No blockers for remaining Phase 5 plans
