---
phase: 08-loss-function-overhaul
plan: 02
subsystem: training
tags: [pytorch, loss-functions, temporal-dynamics, composite-loss, config-validation]

# Dependency graph
requires:
  - phase: 08-loss-function-overhaul
    plan: 01
    provides: "WeightedMAELoss, AsymmetricExtremeLoss, temporal diff/var/weighting functions"
provides:
  - "Restructured CompositeLoss with 6-component temporal-aware forward pass"
  - "Two-phase computation: temporal terms on 5D tensor before flatten, spatial terms after"
  - "Per-timestep weighted L1 loss with configurable weights [1.0, 1.5, 2.0, 2.5]"
  - "Updated config.yaml with extreme_weight=3.0 and all temporal/asymmetric params"
  - "Updated get_loss_function factory reading all new config keys"
  - "Config validator for new loss keys with cross-check warning"
affects: [08-03-training-loop-logging]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase loss computation: temporal terms on 5D before flatten, spatial on 4D after"
    - "Per-timestep weighting via element-wise L1 error multiplication (not pred/target modification)"
    - "6-component return_components dict for per-component loss logging"

key-files:
  created: []
  modified:
    - "training/losses.py"
    - "config.yaml"
    - "utils/config_validator.py"
    - "tests/test_losses.py"
    - "tests/test_config.py"

key-decisions:
  - "Per-timestep weights applied to element-wise L1 error (not pred/target) to avoid changing SSIM behavior"
  - "WeightedMAE extreme_weight parameter in CompositeLoss constructor used as both loss weight and MAE internal weight when > 1.0"
  - "ssim_weight reduced from 0.5 to 0.3 per research recommendation to allow temporal terms more influence"

patterns-established:
  - "Two-phase forward: compute temporal_diff/temporal_var on original 5D shape, then flatten to 4D for SSIM/MAE/asymmetric"
  - "4D backward compatibility: temporal terms return 0.0 tensors when input is 4D"
  - "Config cross-check: warn when loss.extreme_threshold differs from evaluation.extreme_threshold"

requirements-completed: [LOSS-01, LOSS-02, LOSS-03, LOSS-06]

# Metrics
duration: 3min
completed: 2026-03-08
---

# Phase 8 Plan 02: CompositeLoss Restructuring Summary

**Restructured CompositeLoss with two-phase temporal-aware forward pass: 6 loss components (L1, SSIM, WeightedMAE, temporal diff, temporal var, asymmetric) with per-timestep weighting and extreme_weight=3.0**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T19:35:59Z
- **Completed:** 2026-03-07T19:38:54Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 5

## Accomplishments
- Restructured CompositeLoss.forward() with two-phase computation: temporal terms (temporal_diff, temporal_var) computed on 5D tensor before flattening, spatial terms (SSIM, WeightedMAE, asymmetric) computed on flattened 4D tensor
- Per-timestep weights [1.0, 1.5, 2.0, 2.5] applied to element-wise L1 error (not to pred/target directly), preserving SSIM behavior while weighting later timesteps higher
- Updated config.yaml with extreme_weight=3.0 (LOSS-06), ssim_weight=0.3, and all new temporal/asymmetric parameters
- Updated get_loss_function factory to read all 6 new config keys (temporal_diff_weight, temporal_var_lambda, asymmetric_weight, asymmetric_alpha, extreme_threshold, temporal_weights)
- Added config validation for all new loss keys with cross-check warning when loss and evaluation extreme thresholds differ
- 7 new tests added (4 CompositeLoss, 1 factory, 2 config validation), full suite passes (143 passed, 1 skipped)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for CompositeLoss restructuring** - `9c3c18d` (test)
2. **Task 1 GREEN: Implement CompositeLoss restructuring** - `88c3581` (feat)

_Note: TDD task - test commit (RED) followed by implementation commit (GREEN)_

## Files Created/Modified
- `training/losses.py` - Restructured CompositeLoss with 6-component forward, updated get_loss_function factory with new config keys
- `config.yaml` - Updated loss section: extreme_weight=3.0, ssim_weight=0.3, added temporal_diff_weight, temporal_var_lambda, temporal_weights, asymmetric_weight, asymmetric_alpha, extreme_threshold
- `utils/config_validator.py` - Added validation for 6 new loss config keys with type/range checks and cross-field warning
- `tests/test_losses.py` - Updated test_composite_loss_components for 8 keys, added 5D temporal terms test, 4D backward compat test, temporal weights test, factory new params test
- `tests/test_config.py` - Added test_loss_temporal_config_valid and test_loss_temporal_weights_wrong_type

## Decisions Made
- Per-timestep weights applied to element-wise L1 error rather than modifying pred/target directly -- this avoids changing SSIM behavior and follows research Open Question 3 recommendation
- WeightedMAE internal extreme_weight set to match CompositeLoss extreme_weight when > 1.0, otherwise defaults to 3.0 -- ensures the WeightedMAE penalty scales consistently
- ssim_weight reduced from 0.5 to 0.3 per research recommendation to give temporal terms more influence in the total loss

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CompositeLoss with 6-component forward ready for per-component loss logging in Plan 08-03
- return_components dict provides all 8 keys (total + 6 components + ssim_val) needed for training loop logging
- Config has all parameters set to research-recommended defaults
- Full test suite passes (143 passed, 1 skipped, 0 failures)

## Self-Check: PASSED

All files verified present. All commit hashes verified in git log.

---
*Phase: 08-loss-function-overhaul*
*Completed: 2026-03-08*
