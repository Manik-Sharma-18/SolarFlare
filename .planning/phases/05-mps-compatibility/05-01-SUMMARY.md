# Phase 5 Plan 1: MPS-Safe Ops and SSIM Rework Summary

**One-liner:** MPS-safe op wrappers (safe_outer, safe_quantile) and SSIM reworked with kernel caching, channel-loop MPS fallback, and tiled large-tensor computation.

## Metadata

- **Phase:** 05-mps-compatibility
- **Plan:** 01
- **Duration:** ~3 min
- **Completed:** 2026-02-04

## What Was Done

### Task 1: Create utils/mps_ops.py with device-dispatched op wrappers
- Created `utils/mps_ops.py` with three public functions:
  - `safe_outer(a, b)`: broadcast multiply on MPS, `torch.outer` on CUDA/CPU
  - `safe_quantile(tensor, q, dim)`: sort-based interpolation on MPS, `torch.quantile` on CUDA/CPU
  - `is_mps(tensor_or_device)`: detect MPS device from tensor or device object
- One-time INFO log on first MPS op usage
- Exported from `utils/__init__.py`
- **Commit:** 6f09b3a

### Task 2: Rework SSIM in losses.py with kernel caching, MPS channel-loop, and tiling
- Added `_KERNEL_CACHE` dict for Gaussian kernel reuse (keyed by size, sigma, device)
- Replaced `g.outer(g)` with `safe_outer(g, g)` in kernel creation
- Added `_ssim_conv2d` helper: per-channel conv2d loop on MPS (avoids grouped conv bugs), native grouped conv on CUDA/CPU
- Added `_tiled_ssim` for spatial tensors above threshold (default 256): 16px overlap, valid-region averaging to avoid double-counting
- `CompositeLoss` accepts `ssim_tiling_threshold` parameter, passed through to ssim/ms_ssim
- `get_loss_function` reads `ssim_tiling_threshold` from config
- WeightedMAELoss and overall CompositeLoss structure unchanged
- **Commit:** 63bd544

## Verification Results

| Check | Result |
|-------|--------|
| `safe_outer` shape (5,5) on CPU | PASS |
| `ssim(x, x) ~= 1.0` on CPU | PASS (1.0000) |
| `_KERNEL_CACHE` has 1 entry after 2 calls | PASS |
| `CompositeLoss(x, x)` ssim_val ~= 1.0 | PASS (1.0000) |
| Tiled SSIM (300x300, threshold=128) ~= 1.0 | PASS (1.0000) |
| No `torch.outer` in losses.py | PASS |
| `_KERNEL_CACHE` exists in losses.py | PASS |

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Overlap of 16px for tiling | Matches window_size//2 padding; prevents boundary artifacts |
| Cache key uses `str(device)` | Ensures different MPS/CPU devices get separate cached kernels |
| `_ssim_conv2d` extracts conv dispatch | Keeps ssim() readable; single dispatch point for all 5 conv calls |

## Key Files

### Created
- `utils/mps_ops.py` -- MPS-safe op alternatives (safe_outer, safe_quantile, is_mps)

### Modified
- `utils/__init__.py` -- Added mps_ops exports
- `training/losses.py` -- Kernel caching, MPS channel-loop conv, tiled SSIM

## Dependencies

- **Requires:** Phase 1 (device detection via `utils.device`)
- **Provides:** MPS-correct SSIM loss, reusable MPS op wrappers
- **Affects:** Phase 5 plans 02-03 (will use mps_ops for additional MPS safety)
