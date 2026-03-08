---
phase: 09-training-policy
plan: 01
subsystem: data
tags: [WeightedRandomSampler, flare-detection, class-imbalance, oversampling, pytorch]

# Dependency graph
requires:
  - phase: 08-loss-overhaul
    provides: extreme_threshold config value (0.3456) used for flare detection
provides:
  - build_index with flare flag detection (extreme_threshold param)
  - create_dataloaders with WeightedRandomSampler integration
  - Config cross-check for flare_oversample_weight + augmentation
  - train_flare_flags in metadata for downstream use
affects: [09-training-policy, 10-temporal-architecture, 11-integration]

# Tech tracking
tech-stack:
  added: [torch.utils.data.WeightedRandomSampler]
  patterns: [tuple-return for index+metadata, conditional sampler vs shuffle]

key-files:
  created: []
  modified:
    - solarflare_data/dataset.py
    - solarflare_data/loader.py
    - utils/config_validator.py
    - tests/test_data_pipeline.py
    - tests/test_config.py
    - main.py

key-decisions:
  - "Flare detection scans output frames only (not input) to identify sequences the model needs to predict"
  - "Augmented copies share same flare flag as base window to avoid inconsistency"
  - "Sampler uses replacement=True with num_samples=len(weights) to maintain epoch length"
  - "Flare threshold sourced from evaluation.extreme_threshold (0.3456) when oversampling enabled"

patterns-established:
  - "Tuple return pattern: build_index returns (index, metadata) for extensibility"
  - "Conditional sampler: WeightedRandomSampler replaces shuffle only when weight > 1.0"
  - "Config cross-check pattern: warn on conflicting settings (oversample + no augmentation)"

requirements-completed: [TRAIN-04]

# Metrics
duration: 8min
completed: 2026-03-08
---

# Phase 9 Plan 01: Flare-Aware Weighted Sampling Summary

**WeightedRandomSampler oversamples flare-containing sequences 3x via output-frame extreme detection in build_index**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-08T11:37:59Z
- **Completed:** 2026-03-08T11:46:00Z
- **Tasks:** 3 (TDD: RED, GREEN, REFACTOR)
- **Files modified:** 6

## Accomplishments
- build_index returns (index, flare_flags) tuple with extreme_threshold param for flare detection
- WeightedRandomSampler conditionally replaces shuffle=True when flare_oversample_weight > 1.0
- Config cross-check warns when oversampling enabled without augmentation diversity
- Full backward compatibility maintained -- all 162 existing tests pass unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: RED - failing tests** - `353b3e7` (test)
2. **Task 2: GREEN - implement flare sampling** - `4e71220` (feat)
3. **Task 3: REFACTOR - wire through loader/main** - `8f08b19` (refactor)

_TDD flow: RED (17 failing tests) -> GREEN (all pass) -> REFACTOR (end-to-end wiring)_

## Files Created/Modified
- `solarflare_data/dataset.py` - build_index returns (index, flare_flags) with extreme_threshold param
- `solarflare_data/loader.py` - create_dataloaders with WeightedRandomSampler, _build_sampler_weights helper
- `utils/config_validator.py` - Cross-check: flare_oversample_weight + augmentation=none warning
- `tests/test_data_pipeline.py` - TestFlareDetection (9 tests), TestWeightedSampler (5 tests), flare_npy_files fixture
- `tests/test_config.py` - 3 config cross-check tests for flare oversample warning
- `main.py` - Wire flare_extreme_threshold and flare_oversample_weight from config

## Decisions Made
- Flare detection scans output frames only (not input), because the goal is to oversample sequences where the model needs to predict extreme events
- Augmented copies of the same window share the same flare flag, since augmentation is a spatial transform that does not change whether extremes are present
- Sampler uses replacement=True with num_samples=len(weights) to keep epoch length unchanged while oversampling flare sequences
- Flare threshold sourced from evaluation.extreme_threshold (default 0.3456) when flare_oversample_weight > 1.0, None otherwise

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Wired flare threshold through loader functions and main.py**
- **Found during:** Task 3 (REFACTOR)
- **Issue:** build_index had the extreme_threshold param but load_and_prepare_data and load_preprocessed_data did not pass it, making flare detection inoperative end-to-end
- **Fix:** Added flare_extreme_threshold param to both loader functions, wired main.py to pass evaluation.extreme_threshold when flare_oversample_weight > 1.0
- **Files modified:** solarflare_data/loader.py, main.py
- **Verification:** All 162 tests pass
- **Committed in:** 8f08b19

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix was necessary for the feature to work end-to-end. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Flare-aware weighted sampling is ready for use via `data.flare_oversample_weight: 3.0` in config
- train_flare_flags stored in metadata for logging/analysis
- Ready for Phase 9 Plan 02 (training policy implementation)

## Self-Check: PASSED

All 7 files verified present. All 3 commits verified in git log.

---
*Phase: 09-training-policy*
*Completed: 2026-03-08*
