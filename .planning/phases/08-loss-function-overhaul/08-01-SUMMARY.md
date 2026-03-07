---
phase: 08-loss-function-overhaul
plan: 01
subsystem: training
tags: [pytorch, loss-functions, temporal-dynamics, tdd, asymmetric-loss, weighted-mae]

# Dependency graph
requires:
  - phase: 07-evaluation-metrics
    provides: "Evaluation metrics (CSI, HSS, contingency tables) using threshold 0.3456"
provides:
  - "Fixed WeightedMAELoss with absolute threshold binary weighting"
  - "AsymmetricExtremeLoss class for underestimation penalty above threshold"
  - "compute_temporal_diff_loss function for temporal dynamics matching"
  - "compute_temporal_var_penalty function for variation incentivization"
  - "apply_temporal_weights function for per-timestep weighting"
  - "Comprehensive unit tests for all loss components (13 new tests)"
affects: [08-02-composite-forward, 08-03-training-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Absolute threshold binary weighting for extreme region detection"
    - "Asymmetric loss with directional clamp for underestimation penalty"
    - "Temporal difference via consecutive frame subtraction on 5D tensors"
    - "Variation penalty capped at target level via torch.min"
    - "Per-timestep weight broadcasting via (1,1,T,1,1) reshape"

key-files:
  created: []
  modified:
    - "training/losses.py"
    - "tests/test_losses.py"

key-decisions:
  - "Used same threshold (0.3456) for loss as evaluation -- consistent definition of extreme"
  - "WeightedMAELoss default extreme_weight changed from 2.0 to 3.0 per LOSS-06 spec"
  - "T<=1 edge case returns 0.0 tensor for temporal functions (graceful degradation)"

patterns-established:
  - "5D temporal tensor (B,C,T,H,W) with T-dim slicing for frame-to-frame operations"
  - "Binary threshold mask as float for differentiable weighting"
  - "Directional clamp pattern: (target-pred).clamp(min=0) for underestimation"

requirements-completed: [LOSS-01, LOSS-02, LOSS-03, LOSS-04, LOSS-05]

# Metrics
duration: 3min
completed: 2026-03-08
---

# Phase 8 Plan 01: Loss Components Summary

**TDD implementation of 5 loss building blocks: fixed WeightedMAE with absolute threshold, AsymmetricExtremeLoss for underestimation penalty, temporal diff/variation/weighting helper functions**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T19:30:19Z
- **Completed:** 2026-03-07T19:33:36Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Fixed WeightedMAELoss to use absolute threshold binary weighting instead of per-sample relative normalization, ensuring consistent penalty across all samples
- Created AsymmetricExtremeLoss that penalizes underestimation 2x more than overestimation in extreme regions (above 0.3456), with symmetric loss below threshold
- Implemented three temporal helper functions: temporal diff loss (L1 on frame-to-frame changes), temporal variation penalty (negative loss capped at target variation), and per-timestep weighting (broadcasting weights to 5D tensor)
- Added 13 new test cases covering all new components with edge cases (T=1, boundary values, identical dynamics)

## Task Commits

Each task was committed atomically:

1. **Task 1: RED - Write failing tests for all loss components** - `e51dcec` (test)
2. **Task 2: GREEN - Implement all loss components to pass tests** - `1760267` (feat)

_Note: TDD tasks - test commit (RED) followed by implementation commit (GREEN)_

## Files Created/Modified
- `training/losses.py` - Fixed WeightedMAELoss, added AsymmetricExtremeLoss class, added compute_temporal_diff_loss, compute_temporal_var_penalty, and apply_temporal_weights functions
- `tests/test_losses.py` - Updated TestWeightedMAE (4 tests), added TestAsymmetricExtremeLoss (3 tests), TestTemporalDiffLoss (3 tests), TestTemporalVarPenalty (3 tests), TestTemporalWeighting (2 tests)

## Decisions Made
- Used same threshold (0.3456) for loss classification as evaluation metrics -- ensures the model optimizes for the same definition of "extreme" that CSI/HSS measure
- WeightedMAELoss default extreme_weight changed from 2.0 to 3.0 per LOSS-06 minimum spec
- T<=1 edge case in temporal functions returns 0.0 tensor rather than raising an error, allowing graceful degradation for non-temporal inputs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 5 loss building blocks ready for CompositeLoss.forward() integration in Plan 08-02
- AsymmetricExtremeLoss, compute_temporal_diff_loss, compute_temporal_var_penalty, apply_temporal_weights are importable from training.losses
- Full test suite passes (137 passed, 1 skipped, 0 failures)

## Self-Check: PASSED

All files verified present. All commit hashes verified in git log.

---
*Phase: 08-loss-function-overhaul*
*Completed: 2026-03-08*
