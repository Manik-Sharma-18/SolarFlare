# Phase 5: MPS Compatibility & Memory Optimization - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

MPS training produces numerically correct results for all operations, and long training runs do not accumulate memory. This covers: MPS-safe op alternatives for broken native ops, memory-efficient uncertainty estimation via Welford's algorithm, device-aware cache management, and SSIM correctness on MPS with large tensor support. Adding new model architectures, new loss functions, or new training features is out of scope.

</domain>

<decisions>
## Implementation Decisions

### Op Fallback Strategy
- Always use MPS-safe alternatives on MPS — no runtime detection of native op correctness
- CUDA and CPU keep using native ops; alternatives are MPS-only
- Log once at INFO level at startup that MPS alternative ops are active
- All MPS-safe op alternatives live in a dedicated `utils/mps_ops.py` module

### Uncertainty Estimation Rework
- MC Dropout forward pass count is configurable via config.yaml, with a sensible default (e.g., 20)
- Welford's online algorithm only — no option to store all forward passes in memory
- Compute mean + std only from MC passes (no quantiles needed)
- Process one sample at a time through all MC passes before moving to next — minimal peak memory

### Memory Cleanup Policy
- Clear device caches every epoch
- No memory usage warnings or thresholds — if OOM, user adjusts batch size
- Device-specific cache clearing: `torch.mps.empty_cache()` on MPS, `torch.cuda.empty_cache()` on CUDA, skip on CPU

### SSIM Computation
- Channel-loop fallback on MPS only (loop over channels, conv2d per channel); CUDA/CPU use native grouped convolution
- Gaussian kernel cached per device in a module-level dict
- Tile large spatial inputs to prevent OOM — split into tiles, compute SSIM per tile, average
- Tiling threshold is configurable via config.yaml

### Claude's Discretion
- Whether to call `gc.collect()` alongside cache clearing (based on whether dangling references are a realistic concern)
- Default values for MC Dropout pass count and SSIM tiling threshold
- Exact Welford's implementation details
- Tile size and overlap strategy for SSIM tiling

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-mps-compatibility*
*Context gathered: 2026-02-04*
