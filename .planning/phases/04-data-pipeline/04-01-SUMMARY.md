---
phase: 04-data-pipeline
plan: 01
subsystem: data
tags: [dataset, mmap, augmentation, sliding-window, normalization]
depends_on:
  requires: []
  provides: [mmap-dataset, build-index, augmentation-constants]
  affects: [04-02, 04-03]
tech-stack:
  added: []
  patterns: [lazy-mmap-per-worker, index-multiplication-augmentation, on-the-fly-normalization]
key-files:
  created: []
  modified: [solarflare_data/dataset.py, solarflare_data/__init__.py]
key-decisions:
  - id: 04-01-01
    decision: Augmentation is deterministic via index multiplication, not random per-sample
    rationale: Reproducibility and correctness -- each epoch sees the same augmented samples in the same order
  - id: 04-01-02
    decision: mmap handles opened lazily in _get_mmap, not in __init__
    rationale: Each DataLoader worker gets its own file descriptor, safe for both spawn and fork
  - id: 04-01-03
    decision: __getitem__ returns None on read error instead of raising
    rationale: Custom collate_fn in loader.py will filter None samples, preventing single-file corruption from crashing training
duration: ~2 min
completed: 2026-02-04
---

# Phase 4 Plan 1: Mmap Dataset Rewrite Summary

Memory-mapped lazy-open Dataset with precomputed sliding-window index and deterministic augmentation via index multiplication (AUG_NONE/HFLIP/VFLIP/ROT90/ROT180/ROT270).

## Performance

- Duration: ~2 minutes
- Single task, executed cleanly

## Accomplishments

- Rewrote `SolarFluxDataset` to store only file paths and index tuples (no in-memory arrays)
- Implemented `_get_mmap()` for lazy mmap handle opening per worker process
- Added 6 deterministic augmentation types as module-level constants
- Implemented `_apply_augmentation()` static method for spatial transforms
- Added on-the-fly normalization support (asinh and linear methods)
- Preserved dual-channel extreme indicator computation from original
- Created `build_index()` function generating correct sample multipliers: 1x (none), 3x (balanced), 6x (aggressive)
- Updated `__init__.py` exports for new public API

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Rewrite SolarFluxDataset with lazy mmap and index multiplication | a51de6f | solarflare_data/dataset.py, solarflare_data/__init__.py |

## Files Modified

- `solarflare_data/dataset.py` -- Complete rewrite: SolarFluxDataset class + build_index function
- `solarflare_data/__init__.py` -- Updated exports to include build_index and AUG_* constants

## Decisions Made

1. **Deterministic augmentation via index multiplication** -- Each (file_idx, window_start) gets separate index entries for each aug_type. No randomness in __getitem__.
2. **Lazy mmap per worker** -- _get_mmap caches handles in instance dict populated on first access. Safe for spawn/fork multiprocessing.
3. **None return on read error** -- __getitem__ catches exceptions and returns None with a warning log. Downstream collate_fn (in loader.py) handles filtering.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- build_index and SolarFluxDataset are ready for consumption by 04-02 (DataLoader factory)
- Augmentation constants exported for use in config validation
- norm_params/norm_method interface ready for pipeline integration
