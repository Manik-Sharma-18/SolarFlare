---
phase: quick-03
plan: 01
subsystem: data
tags: [center-crop, spatial-normalization, numpy, loader]

# Dependency graph
requires:
  - phase: none
    provides: n/a
provides:
  - Center-crop spatial normalization replacing bilinear resize
  - _center_crop() helper function in loader.py
  - crop_size config key (replaces target_size)
affects: [training-pipeline, data-loading, model-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [center-crop for physical-scale-preserving spatial normalization]

key-files:
  created:
    - tests/test_center_crop.py
  modified:
    - solarflare_data/loader.py
    - main.py
    - config.yaml

key-decisions:
  - "Center-crop to 437x877 (largest common rectangle) instead of bilinear resize to 448x896"
  - "Pure numpy slicing for crop -- no torch import needed in crop path"
  - "Renamed target_size -> crop_size throughout pipeline for semantic clarity"

patterns-established:
  - "Physical scale preservation: spatial normalization via crop, not resize"

requirements-completed: [QUICK-03]

# Metrics
duration: 3min
completed: 2026-03-09
---

# Quick Task 3: Fix Spatial Dimension Mismatch by Center-Crop Summary

**Center-crop to 437x877 replacing bilinear resize, preserving 0.36 Mm/pixel physical scale across all 14 data cubes**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-09T17:02:12Z
- **Completed:** 2026-03-09T17:05:21Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced bilinear interpolation (F.interpolate) with numpy center-crop in both load paths (raw and preprocessed)
- All 14 data cubes now normalize to 437x877 preserving identical 0.36 Mm/pixel physical spacing
- Eliminated torch dependency in spatial normalization path (pure numpy slicing)
- 8 new tests verify center-crop correctness for all three cube dimension groups
- All 214 tests pass (206 existing + 8 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace bilinear resize with center-crop in loader.py and update config pipeline** - `03619cc` (fix)
2. **Task 2: Verify center-crop correctness with a targeted smoke test** - `dca00b3` (test)

## Files Created/Modified
- `solarflare_data/loader.py` - Added _center_crop() helper, replaced F.interpolate with center-crop in both load_and_prepare_data() and load_preprocessed_data()
- `main.py` - Updated config reading from target_size to crop_size, passes crop_size to both loader functions
- `config.yaml` - Changed target_size: [448, 896] to crop_size: [437, 877] with updated comments
- `tests/test_center_crop.py` - 8 unit tests covering output shape per group, center alignment, no-op, error handling, temporal dim preservation, data integrity

## Decisions Made
- Center-crop to 437x877 (largest common rectangle across all three cube groups) preserves 0.36 Mm/pixel scale identically
- Pure numpy array slicing for crop -- removes torch dependency from the spatial normalization path
- Renamed target_size to crop_size throughout the pipeline for semantic clarity (crop != resize)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Steps
- Retrain model with center-cropped data to evaluate impact on CSI and temporal variation metrics
- Previous model checkpoints are invalid (different spatial dimensions) -- full retraining required

## Self-Check: PASSED

All 4 files verified present. Both commit hashes (03619cc, dca00b3) verified in git log.

---
*Quick Task: 03-fix-spatial-dimension-mismatch*
*Completed: 2026-03-09*
