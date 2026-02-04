# Phase 5: MPS Compatibility & Memory Optimization - Research

**Researched:** 2026-02-04
**Domain:** PyTorch MPS backend correctness, memory-efficient uncertainty estimation, SSIM computation
**Confidence:** MEDIUM

## Summary

This phase addresses four interconnected concerns: (1) MPS-broken ops that need safe alternatives, (2) memory-blowing uncertainty estimation that needs Welford's algorithm, (3) epoch-boundary cache cleanup for long runs, and (4) SSIM correctness and OOM prevention on MPS with large tensors.

The existing codebase has specific touchpoints: `losses.py` uses `g.outer(g)` for Gaussian kernel creation and `F.conv2d(..., groups=C)` for SSIM; `models/uncertainty.py` stacks all N predictions in memory then computes mean/std (O(N) memory); `predict_with_confidence_intervals()` calls `torch.quantile()` which is broken on MPS; and `trainer.py` already calls `clear_device_cache(device)` every epoch but does not call `gc.collect()`.

The research confirms: `torch.quantile` on MPS produces silently wrong results (not just unsupported -- it runs but gives garbage when `dim` is not None). Grouped conv2d on MPS has a history of correctness bugs including silent wrong results with tensor views. MPS memory leaks are a known ongoing issue, particularly with LSTM models, and `gc.collect()` before `torch.mps.empty_cache()` is the community-standard workaround. Welford's algorithm is well-established with simple PyTorch implementation requiring only three running accumulators.

**Primary recommendation:** Create `utils/mps_ops.py` with device-dispatched wrappers for all MPS-broken ops, rewrite `predict_with_uncertainty` to use Welford's online accumulation, add channel-loop SSIM fallback + tiling for large tensors, and add `gc.collect()` before cache clearing on MPS.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch (torch) | 2.x (user's installed) | All computation | Only dependency needed |
| gc (stdlib) | built-in | Force garbage collection before MPS cache clear | Addresses MPS dangling reference leaks |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| welford-torch | N/A (do NOT use) | PyTorch Welford's | Decision: implement inline, 10 lines of code, no external dep needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom SSIM | pytorch-msssim (VainF) | External dep, may also have MPS grouped conv issue |
| Custom SSIM | fused-ssim | Native MPS Metal kernels, but adds compile dependency |
| Custom Welford | welford-torch PyPI | Unnecessary dependency for ~10 lines of code |

**Installation:** No new packages needed. All changes use PyTorch and stdlib only.

## Architecture Patterns

### Recommended Project Structure
```
utils/
  mps_ops.py          # NEW: MPS-safe op alternatives (quantile, outer, ssim conv)
  device.py           # MODIFY: add gc.collect() to clear_device_cache for MPS
  metrics.py           # unchanged
training/
  losses.py           # MODIFY: device-aware SSIM with channel-loop + tiling + kernel cache
  trainer.py          # unchanged (already calls clear_device_cache)
models/
  uncertainty.py      # MODIFY: Welford's algorithm, remove torch.quantile usage
config.yaml          # MODIFY: add ssim_tile_threshold, uncertainty.n_samples already exists
```

### Pattern 1: Device-Dispatched Op Wrapper
**What:** Functions in `mps_ops.py` that check device type and route to MPS-safe or native implementation
**When to use:** Any op known to be broken or produce incorrect results on MPS
**Example:**
```python
# utils/mps_ops.py
import torch

def safe_quantile(tensor: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """MPS-safe quantile via sort+index. Native torch.quantile on CUDA/CPU."""
    if tensor.device.type == "mps":
        sorted_tensor = torch.sort(tensor, dim=dim).values
        idx = int(round(q * (tensor.shape[dim] - 1)))
        return sorted_tensor.select(dim, idx)
    return torch.quantile(tensor, q, dim=dim)

def safe_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MPS-safe outer product via broadcast multiply."""
    if a.device.type == "mps":
        return a.unsqueeze(-1) * b.unsqueeze(0)
    return torch.outer(a, b)
```

### Pattern 2: Welford's Online Algorithm for MC Dropout
**What:** Accumulate mean and variance online without storing all predictions
**When to use:** MC Dropout uncertainty estimation (replaces torch.stack of N predictions)
**Example:**
```python
# Welford's online algorithm for running mean + variance
count = 0
mean = torch.zeros_like(first_pred)
m2 = torch.zeros_like(first_pred)

for i in range(n_samples):
    pred = model(x, teacher_forcing_ratio=0.0)
    count += 1
    delta = pred - mean
    mean += delta / count
    delta2 = pred - mean
    m2 += delta * delta2

variance = m2 / count  # or m2 / (count - 1) for sample variance
std = torch.sqrt(variance + 1e-8)
```

### Pattern 3: Channel-Loop SSIM Convolution on MPS
**What:** Loop over channels with groups=1 conv2d instead of groups=C
**When to use:** MPS device only; CUDA/CPU use native grouped convolution
**Example:**
```python
def _ssim_conv2d(x: torch.Tensor, window: torch.Tensor, n_channels: int, padding: int) -> torch.Tensor:
    """Device-aware SSIM convolution."""
    if x.device.type == "mps":
        # Channel-loop fallback: one conv2d per channel
        single_window = window[:1]  # (1, 1, K, K)
        return torch.cat([
            F.conv2d(x[:, c:c+1], single_window, padding=padding)
            for c in range(n_channels)
        ], dim=1)
    else:
        return F.conv2d(x, window, padding=padding, groups=n_channels)
```

### Pattern 4: Tiled SSIM for Large Spatial Tensors
**What:** Split large spatial inputs into overlapping tiles, compute SSIM per tile, average
**When to use:** When spatial dimensions exceed configurable threshold (e.g., 256x256)
**Example:**
```python
def tiled_ssim(pred, target, tile_size=256, overlap=16, **ssim_kwargs):
    """Compute SSIM in spatial tiles to prevent OOM on large tensors."""
    H, W = pred.shape[-2:]
    if H <= tile_size and W <= tile_size:
        return ssim(pred, target, **ssim_kwargs)

    stride = tile_size - overlap
    ssim_values = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            pred_tile = pred[..., y_start:y_end, x_start:x_end]
            target_tile = target[..., y_start:y_end, x_start:x_end]
            ssim_values.append(ssim(pred_tile, target_tile, **ssim_kwargs))

    return torch.stack(ssim_values).mean()
```

### Pattern 5: Module-Level Gaussian Kernel Cache
**What:** Cache Gaussian kernels per device to avoid recomputation
**When to use:** SSIM computation (kernel is constant, only varies by window_size/sigma/device)
**Example:**
```python
_gaussian_kernel_cache: dict = {}  # (size, sigma, device_str) -> tensor

def _get_cached_gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    key = (size, sigma, str(device))
    if key not in _gaussian_kernel_cache:
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(-1) * g.unsqueeze(0)  # broadcast outer (MPS-safe)
        _gaussian_kernel_cache[key] = kernel
    return _gaussian_kernel_cache[key]
```

### Anti-Patterns to Avoid
- **Stacking all MC predictions then computing stats:** O(N) memory growth. Use Welford's online accumulation instead.
- **Using torch.quantile on MPS:** Produces silently wrong results. Use sort+index alternative.
- **Using groups=C in conv2d on MPS for SSIM:** Risk of silent correctness bugs. Use channel-loop fallback.
- **Calling torch.mps.empty_cache() without gc.collect():** Python garbage collector may hold references to MPS tensors that prevent the allocator from freeing memory.
- **Creating Gaussian kernel every forward pass:** Wasteful; kernel is deterministic for given (size, sigma, device). Cache it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Running mean/variance | Naive sum-of-squares accumulator | Welford's algorithm | Catastrophic cancellation with large means; Welford avoids subtraction of large numbers |
| MPS-safe quantile | CPU transfer round-trip | Sort + index on device | Avoids device transfer overhead; stays on MPS |
| SSIM Gaussian kernel | Recompute each call | Module-level cache dict | Kernel is deterministic; caching avoids repeated allocation |

**Key insight:** The MPS backend has silent correctness bugs (not just missing ops). Operations that run without error can still produce wrong results. Always-use-alternative on MPS is the only safe strategy (as per user decision).

## Common Pitfalls

### Pitfall 1: torch.quantile Silent Wrong Results on MPS
**What goes wrong:** `torch.quantile(tensor, q, dim=dim)` runs without error on MPS but produces incorrect values because MPS sort implementation mishandles strided tensor views created internally by quantile.
**Why it happens:** Internal `unsqueeze(-1).transpose(dim, -1).sort()` creates non-contiguous views that MPS sort does not handle correctly.
**How to avoid:** Never call torch.quantile on MPS tensors. Use sort+index alternative in mps_ops.py. The decision locks this: "always use MPS-safe alternatives on MPS."
**Warning signs:** Confidence intervals that seem unreasonably narrow or wide; quantile values outside expected range.

### Pitfall 2: Welford's Division by Zero on First Sample
**What goes wrong:** If you compute std before at least 2 samples, you get division by zero or NaN.
**Why it happens:** Variance = M2 / (count - 1) for sample variance, or M2 / count for population variance. With count=0, both fail.
**How to avoid:** Use population variance (M2 / count) since we always have n_samples >= 2. Add epsilon to sqrt: `torch.sqrt(variance + 1e-8)`.
**Warning signs:** NaN in uncertainty maps.

### Pitfall 3: MPS Memory Leak in LSTM Training Loops
**What goes wrong:** Memory usage grows monotonically over epochs despite calling empty_cache.
**Why it happens:** Known MPS backend bug with LSTM operations (missing autorelease in MPS LSTM kernel). ConvLSTM used in this project is custom but may trigger similar MPS memory management issues.
**How to avoid:** Call `gc.collect()` before `torch.mps.empty_cache()` to ensure Python-side references are cleared first. This is the community-standard workaround.
**Warning signs:** Activity Monitor showing increasing GPU memory over epochs on Mac.

### Pitfall 4: SSIM Tiling Edge Effects
**What goes wrong:** Tile boundaries produce artifacts in SSIM computation because the Gaussian window at edges has partial support.
**Why it happens:** The 11x11 Gaussian window needs 5 pixels of context on each side. At tile boundaries, SSIM values are computed with truncated windows.
**How to avoid:** Use overlapping tiles (overlap >= window_size // 2, i.e., >= 6 pixels). Only count the non-overlapping center region of each tile toward the final average, or simply average all tiles accepting small boundary bias.
**Warning signs:** SSIM values that change significantly when tile_size changes.

### Pitfall 5: Gaussian Kernel Cache Memory Leak Across Devices
**What goes wrong:** Module-level cache dict holds references to tensors on specific devices indefinitely.
**Why it happens:** If user switches devices during a session, old kernels on previous devices remain cached.
**How to avoid:** This is acceptable for this project (single device per run). The cache is tiny (one 11x11 float32 tensor per device = 484 bytes). Not worth adding LRU eviction.
**Warning signs:** None in practice.

### Pitfall 6: Grouped Conv Producing Wrong SSIM Values on MPS
**What goes wrong:** `F.conv2d(x, window, groups=C)` may produce silently incorrect results on MPS for certain tensor layouts.
**Why it happens:** MPS has known bugs with convolution on tensor views/chunks (issue #169342) and with large channel counts (issue #142836).
**How to avoid:** Decision locked: channel-loop fallback on MPS. Loop over C channels with groups=1.
**Warning signs:** ssim(x, x) returning value != 1.0 on MPS.

### Pitfall 7: MC Dropout Mode Management
**What goes wrong:** Model is set to train() mode for MC Dropout but not restored to eval() mode afterward.
**Why it happens:** Exception during forward passes or early return.
**How to avoid:** Use try/finally to restore model.eval(). The existing code already does this partially but the Welford rewrite needs to maintain the pattern.
**Warning signs:** BatchNorm (if ever added) using running stats vs batch stats incorrectly.

## Code Examples

### Welford's Algorithm for MC Dropout (Complete)
```python
# Source: Wikipedia Algorithms for calculating variance + PyTorch adaptation
def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC Dropout with Welford's online algorithm. O(1) memory in n_samples."""
    was_training = model.training
    model.train()  # Enable dropout

    try:
        with torch.no_grad():
            # First pass to initialize shapes
            first_pred = model(x, teacher_forcing_ratio=0.0)
            count = 1
            mean = first_pred.clone()
            m2 = torch.zeros_like(first_pred)

            for i in range(1, n_samples):
                pred = model(x, teacher_forcing_ratio=0.0)
                count += 1
                delta = pred - mean
                mean += delta / count
                delta2 = pred - mean
                m2 += delta * delta2

        variance = m2 / count
        std = torch.sqrt(variance + 1e-8)
        return mean, std
    finally:
        if not was_training:
            model.eval()
```

### MPS-Safe Quantile via Sort+Index
```python
# Source: PyTorch issue #101878 workaround
def safe_quantile(tensor: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """Quantile computation safe for all devices including MPS."""
    if tensor.device.type == "mps":
        sorted_vals = torch.sort(tensor, dim=dim).values
        n = tensor.shape[dim]
        # Linear interpolation index
        idx = q * (n - 1)
        low = int(idx)
        high = min(low + 1, n - 1)
        frac = idx - low
        low_val = sorted_vals.select(dim, low)
        high_val = sorted_vals.select(dim, high)
        return low_val + frac * (high_val - low_val)
    return torch.quantile(tensor, q, dim=dim)
```

### Device-Aware Cache Clearing (Updated)
```python
# Source: PyTorch MPS memory leak issues #145374, #154329, #81610
import gc

def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        gc.collect()  # Clear Python references first
        torch.mps.empty_cache()
    # CPU: no-op
```

### SSIM with Channel-Loop Fallback and Kernel Cache
```python
_kernel_cache: dict = {}

def _get_gaussian_kernel(size, sigma, device):
    key = (size, sigma, str(device))
    if key not in _kernel_cache:
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.unsqueeze(-1) * g.unsqueeze(0)  # broadcast (MPS-safe)
        _kernel_cache[key] = kernel_2d
    return _kernel_cache[key]

def ssim(pred, target, window_size=11, sigma=1.5, data_range=2.0, size_average=True):
    C = pred.size(1)
    kernel_2d = _get_gaussian_kernel(window_size, sigma, pred.device)

    if pred.device.type == "mps":
        # Channel-loop fallback
        window_single = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        def conv(x):
            return torch.cat([
                F.conv2d(x[:, c:c+1], window_single, padding=window_size // 2)
                for c in range(C)
            ], dim=1)
    else:
        window = kernel_2d.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)  # (C, 1, K, K)
        def conv(x):
            return F.conv2d(x, window, padding=window_size // 2, groups=C)

    # Standard SSIM computation using conv()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1, mu2 = conv(pred), conv(target)
    # ... (rest of SSIM formula same as current)
```

## State of the Art

| Old Approach (Current Code) | New Approach (This Phase) | Why |
|---------------------------|---------------------------|-----|
| `torch.stack(predictions)` then `.mean()/.std()` | Welford's online accumulation | O(1) vs O(N) memory in n_samples |
| `torch.quantile()` in uncertainty | Sort+index alternative (or remove quantile entirely) | quantile broken on MPS |
| `g.outer(g)` for Gaussian kernel | `g.unsqueeze(-1) * g.unsqueeze(0)` broadcast | outer may have MPS issues; broadcast is universally safe |
| `F.conv2d(..., groups=C)` for SSIM | Channel-loop on MPS, grouped on CUDA/CPU | Grouped conv has silent correctness bugs on MPS |
| Create Gaussian kernel every SSIM call | Module-level cache keyed by (size, sigma, device) | Avoids repeated allocation |
| `torch.mps.empty_cache()` alone | `gc.collect()` then `torch.mps.empty_cache()` | Python GC releases references MPS allocator cannot see |
| No tiling for large SSIM inputs | Tiled SSIM above configurable threshold | Prevents OOM on 512x512+ inputs |

**Deprecated/outdated:**
- `predict_with_confidence_intervals()` in uncertainty.py uses `torch.quantile` -- per user decision, only mean+std are needed from MC passes (no quantiles). This function can be removed or reworked to use sort+index if kept.

## Open Questions

1. **torch.outer actual MPS status**
   - What we know: DEV-02 requirement says it needs broadcast alternative. No specific MPS bug report found for torch.outer. It likely works via decomposition into element-wise ops.
   - What's unclear: Whether it silently produces wrong results in any edge case on MPS.
   - Recommendation: Use broadcast multiply anyway (decision locked: "always use MPS-safe alternatives on MPS"). Cost is zero -- broadcast is equivalent and universally safe.

2. **SSIM tiling overlap size**
   - What we know: Window size is 11, so half-window is 5. Overlap of 6+ pixels ensures every pixel has full Gaussian window support in at least one tile.
   - What's unclear: Whether averaging overlapping regions introduces measurable bias vs. the non-tiled computation.
   - Recommendation: Default overlap = 16 pixels (generous), tile_size = 256. This is a Claude's Discretion item. For the data sizes in this project (likely <= 512x512), tiling may rarely activate but is cheap insurance.

3. **SSIM tiling threshold default**
   - What we know: This is Claude's Discretion. Spatial dims of 512+ are large for MPS.
   - Recommendation: Default `ssim_tile_threshold: 256` in config.yaml. This means 256x256 and below compute SSIM directly; above 256 in either dimension triggers tiling. Conservative but safe.

4. **MC Dropout default n_samples**
   - What we know: Config already has `uncertainty.n_samples: 20`. This is reasonable -- literature suggests 10-50 passes.
   - Recommendation: Keep default at 20. With Welford's, memory cost is independent of n_samples.

5. **Whether to keep predict_with_confidence_intervals()**
   - What we know: Decision says "compute mean + std only from MC passes (no quantiles needed)."
   - Recommendation: Remove it or leave it with a deprecation comment + fix using sort+index for any future use. Removing is cleaner.

## Sources

### Primary (HIGH confidence)
- PyTorch GitHub issue [#101878](https://github.com/pytorch/pytorch/issues/101878) -- torch.quantile on MPS produces wrong results (OPEN, unfixed)
- PyTorch GitHub issue [#145374](https://github.com/pytorch/pytorch/issues/145374) -- LSTM MPS memory leak, partially fixed then regressed
- PyTorch GitHub issue [#169342](https://github.com/pytorch/pytorch/issues/169342) -- MPS batch inference with chunk+conv produces wrong results (Dec 2025)
- PyTorch GitHub issue [#142836](https://github.com/pytorch/pytorch/issues/142836) -- MPS incorrect output from conv with large dimensions
- Wikipedia [Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) -- Welford's algorithm reference

### Secondary (MEDIUM confidence)
- PyTorch GitHub issue [#154329](https://github.com/pytorch/pytorch/issues/154329) -- MPS memory leak (May 2025)
- PyTorch GitHub issue [#81610](https://github.com/pytorch/pytorch/issues/81610) -- M1 MPS memory leak with gc.collect workaround
- Community pattern: `gc.collect()` + `torch.mps.empty_cache()` widely recommended across multiple issues and forums
- VainF/pytorch-msssim [ssim.py](https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py) -- reference SSIM implementation using grouped conv

### Tertiary (LOW confidence)
- torch.outer MPS status: No specific bug found, but broadcast alternative is trivially equivalent and universally safe
- SSIM tiling overlap value of 16: Based on reasoning about Gaussian window size, not empirically validated

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- PyTorch only, no new deps
- Architecture (mps_ops.py pattern): HIGH -- device-dispatch is straightforward
- Welford's algorithm: HIGH -- well-established, simple implementation
- SSIM channel-loop fallback: MEDIUM -- workaround pattern is standard but MPS grouped conv status may have improved in latest PyTorch
- SSIM tiling: MEDIUM -- pattern is sound but overlap/threshold defaults are best-guess
- Memory cleanup (gc.collect): MEDIUM -- widely recommended workaround but does not fully solve all MPS memory leaks
- Pitfalls: HIGH -- backed by multiple open PyTorch issues

**Research date:** 2026-02-04
**Valid until:** 2026-03-04 (MPS backend is fast-moving; re-check if PyTorch version changes)
