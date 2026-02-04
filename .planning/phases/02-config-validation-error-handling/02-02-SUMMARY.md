---
phase: 02-config-validation-error-handling
plan: 02
subsystem: training-safety
tags: [nan-detection, gradient-monitoring, signal-handling, emergency-checkpoint, training-loop]

dependency_graph:
  requires:
    - phase: 02-01
      provides: error_handling config section with max_consecutive_nan and grad_norm_warning_threshold
    - phase: 01-01
      provides: DummyGradScaler for MPS/CPU, get_grad_scaler, clear_device_cache
  provides:
    - NaN/Inf loss detection with batch skipping in train_epoch
    - NaNLossError exception after consecutive NaN threshold
    - Gradient norm monitoring with configurable warning threshold
    - Graceful shutdown via SIGINT/SIGTERM with emergency checkpoint
    - DummyGradScaler NaN gradient guard for MPS/CPU
  affects: [03-01, 06-01]

tech_stack:
  added: []
  patterns: [nan-skip-continue, consecutive-threshold-abort, signal-handler-with-restore, emergency-checkpoint-metadata]

key_files:
  created: []
  modified:
    - training/trainer.py
    - utils/device.py
    - main.py

key_decisions:
  - "NaN detected before backward pass; batch skipped entirely (no optimizer step)"
  - "Consecutive NaN threshold (configurable) triggers NaNLossError with emergency checkpoint"
  - "Gradient norm warning threshold uses clip_grad_norm_ return value (no extra computation)"
  - "Shutdown check happens between epochs (not mid-batch) for simplicity"
  - "Emergency checkpoints use EMERGENCY_ prefix and emergency=True metadata flag"
  - "Signal handlers restored in finally block to prevent leaked handlers"

patterns-established:
  - "NaN-skip pattern: check loss before backward, skip on NaN/Inf, count consecutive"
  - "Emergency checkpoint pattern: EMERGENCY_ prefix + emergency=True + reason field"
  - "Signal handler lifecycle: save old handlers, register new, restore in finally"

duration: ~3 min
completed: 2026-02-03
---

# Phase 02 Plan 02: NaN/Gradient Handling + Graceful Shutdown Summary

**NaN-safe training loop with gradient norm monitoring, consecutive NaN abort, and SIGINT/SIGTERM emergency checkpoints**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-03T18:22:51Z
- **Completed:** 2026-02-03T18:26:14Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- NaN/Inf loss detected before backward pass, batch skipped with warning, consecutive threshold triggers abort
- Gradient norm monitoring logs warnings when norms exceed configurable threshold
- Ctrl+C and SIGTERM save emergency checkpoint after current epoch, second Ctrl+C force-quits
- DummyGradScaler guards against NaN gradients on MPS/CPU paths
- Validation also handles NaN batches gracefully (skip and log)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add NaN/Inf detection and gradient norm monitoring** - `c2a11ad` (feat)
2. **Task 2: Add graceful shutdown with emergency checkpoint** - `47c0fb0` (feat)

## Files Created/Modified
- `training/trainer.py` - NaN detection, gradient monitoring, NaNLossError, signal handlers, emergency checkpoints
- `utils/device.py` - DummyGradScaler NaN gradient guard in step()
- `main.py` - error_handling config wired to train_config, SystemExit handling for graceful shutdown

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 02-02-A | NaN check before backward pass, not after | Prevents wasted computation and gradient corruption |
| 02-02-B | Shutdown check between epochs, not mid-batch | Simpler than passing shutdown state through train_epoch; epochs complete fast enough |
| 02-02-C | EMERGENCY_ prefix + emergency=True metadata | Both human-readable (filename) and machine-readable (flag) identification |
| 02-02-D | NaN abort saves emergency checkpoint before re-raising | Preserves model state even on NaN abort for debugging |

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 2 is now complete (both plans done). Phase 3 (Checkpoint System) can proceed. The emergency checkpoint pattern established here provides the foundation for the full checkpoint system -- atomic writes, resume logic, and cross-device portability will build on the checkpoint dict structure used here.

---
*Phase: 02-config-validation-error-handling*
*Completed: 2026-02-03*
