---
phase: 01-device-foundation-tech-debt
plan: 03
subsystem: inference-cleanup
tags: [inference, device-detection, tech-debt, dead-code-removal]
dependency-graph:
  requires: [01-01]
  provides: [device-aware-inference, legacy-cleanup]
  affects: [phase-02, phase-03]
tech-stack:
  added: []
  patterns: [auto-device-detection-at-call-site, device-inference-from-model-params]
key-files:
  created: []
  modified: [inference.py]
  deleted: [ConvLSTM.py]
decisions:
  - id: predict-device-from-model
    summary: "predict() infers device from model.parameters() when device=None, avoiding redundant device passing"
  - id: load-model-auto-detect
    summary: "load_model() calls resolve_device('auto') when device=None for zero-config usage"
metrics:
  duration: "~1 minute"
  completed: 2026-02-03
---

# Phase 01 Plan 03: Delete ConvLSTM.py and Fix inference.py Summary

**Deleted 698-line legacy ConvLSTM.py prototype; replaced all hardcoded 'cuda' strings in inference.py with resolve_device() auto-detection and torch.device parameters.**

## What Was Done

### Task 1: Delete legacy ConvLSTM.py
**Commit:** `defbbec`

Removed `ConvLSTM.py` (698 lines) from the project root using `git rm`. This was a legacy prototype containing duplicated implementations of ConvLSTMCell, ConvLSTM, SolarFluxPredictor, training loop, data loading, and visualization -- all of which are properly organized in `models/convlstm.py`, `models/predictor.py`, and other modules.

Verified no Python files in the project imported from the root-level `ConvLSTM.py`. All existing references to `ConvLSTM` are properly scoped to `models/convlstm.py` via relative imports.

### Task 2: Fix inference.py to use resolve_device()
**Commit:** `e8c6bc7`

Updated `inference.py` to eliminate all hardcoded `'cuda'` device strings:

- **Added import:** `from utils.device import resolve_device`
- **`load_model()`:** Changed `device: str = 'cuda'` to `device: torch.device = None`. When None, calls `resolve_device("auto")` for automatic CUDA/MPS/CPU detection.
- **`predict()`:** Changed `device: str = 'cuda'` to `device: torch.device = None`. When None, infers device from `next(model.parameters()).device` -- avoiding redundant device passing.
- **`__main__` block:** Replaced `device = 'cuda' if torch.cuda.is_available() else 'cpu'` with `device = resolve_device("auto")`.

All existing functionality preserved: `load_model`, `predict`, `load_raw_data`, `normalize`, `unnormalize`, `load_normalization`.

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| `predict()` infers device from model params when None | Eliminates need to pass device separately; model already knows its device |
| `load_model()` auto-detects when None | Zero-config default makes inference script work on any platform without changes |

## Verification Results

All five verification criteria passed:
1. `ConvLSTM.py` does not exist at project root
2. `grep "'cuda'" inference.py` returns no matches
3. `grep "resolve_device" inference.py` returns 3 matches (import + 2 call sites)
4. `python -c "import inference"` succeeds without error
5. No broken imports from ConvLSTM.py deletion (all references are to models/convlstm.py)

## Next Phase Readiness

Phase 01 is now complete (plans 01, 02, 03 all done). The device foundation is fully established:
- `utils/device.py` provides the cross-platform API (01-01)
- `main.py` uses it for training (01-02)
- `inference.py` uses it for prediction (01-03)
- Legacy dead code removed (01-03)

Phase 02 (Training Loop Hardening) can proceed.
