# Phase 4: Data Pipeline - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Handle 10-50GB datasets without loading everything into RAM. Memory-mapped loading, safe multi-worker DataLoader, device-aware memory pinning, and whole-file train/test/val splitting. Augmentation modes for solar/satellite imagery.

</domain>

<decisions>
## Implementation Decisions

### Dataset structure
- Each .npy file is one geographic tile/region containing many timesteps
- Individual training samples extracted via sliding window over the time dimension
- Default window stride = 1 (maximum overlap), configurable in config.yaml
- Precompute full index of all valid (file_idx, window_start) pairs at dataset initialization — enables len(), random access, and proper shuffling

### Train/test/val splitting
- Whole-file assignment: entire .npy files go to one split, no splitting within files
- Ratio-based random assignment, seeded with the global training seed for reproducibility
- Default ratios: 70% train, 20% test, 10% validation — configurable via `split_ratios: [0.7, 0.2, 0.1]` in config.yaml
- Extra files from rounding go to training set
- Files are shuffled by seed then assigned in order to splits

### Memory-mapped loading
- Always use mmap (no in-memory fallback), even for small datasets — one code path
- Mmap lifetime and tensor copy strategy at Claude's discretion (optimize for safety with multi-worker DataLoader)

### Worker process safety
- Spawn on macOS, fork on Linux — avoids inherited file descriptor issues on macOS
- num_workers configurable in config.yaml, default 0 (no workers) — user opts in
- prefetch_factor left at PyTorch default (2)
- On worker errors: skip the bad sample and log a warning, training continues

### Augmentation
- Two modes beyond none: **balanced** and **aggressive**
- Balanced: entire datacubes flipped horizontally and vertically — triples training data (original + h-flip + v-flip)
- Aggressive: balanced plus 90° rotations — further multiplies training data
- Configurable in config.yaml, default: `augmentation: none`
- Augmentation applies to training split only — val/test see original data
- How augmented copies are generated (on-the-fly index entries vs pre-materialized) at Claude's discretion

### Claude's Discretion
- Mmap lifetime management (keep mapped per epoch vs lazy open)
- Tensor copy strategy (immediate copy vs view until collate)
- Augmentation implementation approach (index multiplication vs disk materialization)
- pin_memory logic (already specified in success criteria: True only on CUDA)

</decisions>

<specifics>
## Specific Ideas

- "If 10 data.npy files, then 7 train, 2 test, 1 val instead of splitting each .npy file with some percentages"
- Augmentation in balanced mode triples data by applying flips to the entire datacube (not per-sample random transforms)
- Aggressive mode adds 90° rotations on top of balanced flips

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-data-pipeline*
*Context gathered: 2026-02-04*
