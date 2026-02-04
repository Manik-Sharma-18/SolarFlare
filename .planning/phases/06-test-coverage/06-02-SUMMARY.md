---
phase: 06-test-coverage
plan: 02
subsystem: testing
tags: [pytest, ssim, ms-ssim, convlstm, loss-functions, model-shapes, mps]

# Dependency graph
requires:
  - phase: 06-01
    provides: "Test infrastructure (conftest, markers, base fixtures)"
  - phase: 05-01
    provides: "MPS-safe SSIM with tiling and kernel caching"
provides:
  - "19 loss function unit tests covering SSIM, MS-SSIM, Gaussian kernel, WeightedMAE, CompositeLoss, factory"
  - "15 model forward pass shape tests covering all config combinations and MPS smoke test"
  - "Bug fix: double-preprocess in non-downsample forward path"
affects: [06-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Test classes grouped by module/feature (TestSSIM, TestCompositeLoss, etc.)"
    - "autouse fixture for kernel cache cleanup between tests"
    - "Helper _make_model() factory for tiny test models"

key-files:
  created:
    - "tests/test_losses.py"
    - "tests/test_model.py"
  modified:
    - "models/predictor.py"

key-decisions:
  - "use_ms_ssim=False in CompositeLoss tests to keep spatial dims small (32x32)"
  - "Tiny model channels=[4,8,16] for fast test execution (<1s total)"

patterns-established:
  - "Loss tests: autouse kernel cache clearing to prevent test pollution"
  - "Model tests: inline model creation (not fixtures) to avoid memory accumulation"

# Metrics
duration: 3min
completed: 2026-02-04
---

# Phase 6 Plan 02: Loss & Model Tests Summary

**34 unit tests for loss functions (SSIM, MS-SSIM, WeightedMAE, CompositeLoss) and model forward pass shapes with bug fix for double-preprocess path**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-04T14:33:34Z
- **Completed:** 2026-02-04T14:36:34Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- 19 loss function tests covering all functions, edge cases (NaN, identical inputs), and factory
- 15 model shape tests covering single/dual channel, parametrized t_in/t_out, downsample, teacher forcing, dropout, MPS
- Discovered and fixed bug in SolarFluxPredictor: preprocess conv2d applied twice in non-downsample path

## Task Commits

Each task was committed atomically:

1. **Task 1: Write loss function unit tests** - `c43671d` (test)
2. **Task 2: Write model forward pass shape tests** - `581c37d` (test + fix)

## Files Created/Modified
- `tests/test_losses.py` - 19 tests: SSIM (4), MS-SSIM (2), Gaussian kernel (3), WeightedMAE (2), CompositeLoss (4), get_loss_function (4)
- `tests/test_model.py` - 15 tests: shape (9 incl parametrized), output (3), utilities (2), MPS smoke (1)
- `models/predictor.py` - Fixed double-preprocess bug in non-downsample forward path

## Decisions Made
- Used `use_ms_ssim=False` in CompositeLoss tests to allow 32x32 spatial dims (MS-SSIM needs 64x64 for downsampling levels)
- Tiny model channels=[4,8,16] keeps all 15 model tests under 1 second total

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed double-preprocess in non-downsample forward path**
- **Found during:** Task 2 (model forward pass tests)
- **Issue:** When `downsample_input=False`, `self.preprocess` was applied twice: once in the else-branch (line 204) converting input_channels->c1, then again at line 210 which expected input_channels but received c1 channels. This caused a RuntimeError on all non-downsample model configurations.
- **Fix:** Removed the first preprocess call in the non-downsample branch; input is now just reshaped, letting the common preprocess (line 210) handle the channel transformation.
- **Files modified:** models/predictor.py
- **Verification:** All 12 previously-failing model tests now pass
- **Committed in:** 581c37d (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Critical bug fix -- non-downsample model path was completely broken. No scope creep.

## Issues Encountered
None beyond the bug fix documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 34 new tests all passing on CPU; MPS smoke test passes on Apple Silicon
- Combined with 06-01's 32 tests, total test suite now at 66 tests
- Ready for 06-03 (training loop and pipeline integration tests)

---
*Phase: 06-test-coverage*
*Completed: 2026-02-04*
