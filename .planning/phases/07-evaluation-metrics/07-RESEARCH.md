# Phase 7: Evaluation Metrics - Research

**Researched:** 2026-03-07
**Domain:** Meteorological verification metrics, temporal prediction evaluation, PyTorch validation instrumentation
**Confidence:** HIGH

## Summary

Phase 7 instruments comprehensive evaluation metrics into the existing validation loop. The work is entirely self-contained: no new dependencies, no model architecture changes, no loss function changes. All metrics use standard PyTorch operations (tensor math, comparisons, reductions) and reuse the existing `ssim()` function from `training/losses.py`.

The core challenge is restructuring `validate()` in `training/trainer.py` to return a structured metrics dict instead of a `(loss, mae_per_timestep)` tuple, then wiring all new metrics (CSI, HSS, persistence baseline, standalone SSIM, peak flux error, temporal variation ratio) into the accumulation loop. The critical correctness concern is CSI/HSS computation: counts (TP, FP, FN, TN) must be accumulated across all batches and computed once from totals to avoid batch-averaging bias.

The extreme threshold for CSI/HSS binary classification has been computed: the 99.5th percentile of `|values|` in raw space (30,019.39) maps to approximately **0.3456** in normalized asinh space. All metrics operate on normalized values, so this threshold applies directly to model predictions and targets.

**Primary recommendation:** Extend `utils/metrics.py` with pure-function metric computations, restructure `validate()` to return a dict, and update `train_model()` and `main.py` to consume the new dict format. Keep all metric functions stateless (inputs in, scalars out) for testability.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Extreme threshold for CSI/HSS: fixed absolute value, configurable in config.yaml (e.g., `extreme_threshold: <value>`). Default derived from actual data distribution. Binary per-pixel classification. Persistence predictions also get CSI/HSS computed.
- Persistence baseline: last input frame repeated for all T output timesteps. Skill = `(1 - model_MAE / persistence_MAE) * 100`. Per-timestep granularity. Persistence also gets CSI/HSS.
- Metrics output: extend existing `training_history.json` with new keys. Console summary per epoch; full per-timestep only at final epoch or verbose mode. Update `utils/visualization.py` for new metric plots. Per-timestep CSI/HSS logged.
- All metrics computed on normalized (asinh) values -- no denormalization.
- CSI/HSS: accumulate TP, FP, FN counts across all validation batches, compute once from totals.
- Temporal variation ratio: `mean(|frame[t+1] - frame[t]|)`, ratio = pred_variation / target_variation.
- Standalone SSIM: single-scale SSIM (not MS-SSIM).
- Peak flux error: `|max(pred) - max(target)|` value only, no spatial offset.
- `validate()` returns a structured metrics dict (not tuple).
- No special memory safeguards needed -- accumulate scalar counts and running sums only.

### Claude's Discretion
- Exact data-derived threshold value for CSI/HSS
- Internal metric computation order and helper function organization
- Visualization subplot layout and styling for new metric plots
- How verbose mode is toggled (config flag, CLI arg, or log level)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Per-timestep MAE, RMSE, and correlation during validation | Extend existing `compute_metrics()` to add RMSE and correlation per-timestep; current code only does total RMSE/correlation, not per-timestep |
| EVAL-02 | CSI computed and logged per epoch using extreme threshold | New `compute_csi()` function using accumulated TP/FP/FN counts; threshold = 0.3456 in normalized space |
| EVAL-03 | HSS computed and logged per epoch | New `compute_hss()` function using accumulated TP/FP/FN/TN counts; shares contingency table with CSI |
| EVAL-04 | Persistence baseline MAE and skill-over-persistence per epoch | Persistence = last input frame repeated T times; skill = `(1 - model_MAE / persistence_MAE) * 100` |
| EVAL-05 | SSIM logged as standalone validation metric | Reuse existing `ssim()` from `training/losses.py`; call per-batch on (B*T, C, H, W) reshaped tensors |
| EVAL-06 | Peak flux error logged per epoch | `abs(pred.amax(dim=(-2,-1)) - target.amax(dim=(-2,-1)))` per timestep, averaged across batches |
| EVAL-07 | Temporal variation ratio logged per epoch | `mean(abs(pred[:,t+1] - pred[:,t]))` / `mean(abs(target[:,t+1] - target[:,t]))` accumulated across batches |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | existing (project uses torch) | All tensor operations for metrics | Already the project framework; no new deps |
| NumPy | existing | History serialization, visualization data | Already used throughout |
| Matplotlib | existing | New metric plots | Already used in `utils/visualization.py` |

### Supporting
No new libraries needed. All metrics are computed with standard PyTorch tensor operations:
- `torch.abs`, `torch.mean`, `torch.sqrt`, `torch.sum` for MAE, RMSE
- `torch.gt`, `torch.le` for binary thresholding (CSI/HSS contingency table)
- `torch.amax` for peak flux extraction
- Existing `ssim()` function from `training/losses.py` for standalone SSIM

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-computed CSI/HSS | torchmetrics | Would add a dependency for two simple formulas; not worth it for this project size |
| Hand-computed SSIM | torchmetrics.SSIM | Already have a working, MPS-compatible SSIM function; reuse it |
| Custom persistence | N/A | Persistence baseline is trivial (last frame repeat); no library needed |

**Installation:**
```bash
# No new packages needed
```

## Architecture Patterns

### Recommended Project Structure
```
utils/
  metrics.py          # EXTEND: add per-timestep RMSE, correlation, CSI, HSS, persistence, peak flux, temporal variation
training/
  trainer.py          # MODIFY: validate() returns dict, train_model() stores new keys
utils/
  visualization.py    # EXTEND: plot new metrics alongside existing loss curves
config.yaml           # ADD: extreme_threshold key under new evaluation section
utils/
  config_validator.py  # EXTEND: validate extreme_threshold parameter
```

### Pattern 1: Stateless Metric Functions
**What:** Each metric is a pure function: tensors in, scalars or small dicts out. No class state, no side effects.
**When to use:** All new metric functions in `utils/metrics.py`.
**Example:**
```python
# Source: project convention (existing compute_metrics, compute_rmse, compute_correlation follow this pattern)
def compute_csi(tp: int, fp: int, fn: int) -> float:
    """CSI = TP / (TP + FP + FN). Returns 0.0 if denominator is 0."""
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0
```

### Pattern 2: Accumulate-then-Compute for Batch Metrics
**What:** For metrics that must not be batch-averaged (CSI/HSS), accumulate raw counts across all batches, then compute the metric once from totals.
**When to use:** CSI, HSS, SSIM (weighted average), temporal variation ratio.
**Example:**
```python
# In validate() loop:
total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
for batch in dataloader:
    # ... get predictions ...
    binary_pred = (predictions.abs() > threshold).long()
    binary_target = (Y_target.abs() > threshold).long()
    # Per-timestep accumulation
    for t in range(T):
        p = binary_pred[:, :, t]
        g = binary_target[:, :, t]
        total_tp += (p * g).sum().item()
        total_fp += (p * (1 - g)).sum().item()
        total_fn += ((1 - p) * g).sum().item()
        total_tn += ((1 - p) * (1 - g)).sum().item()

# After loop:
csi = compute_csi(total_tp, total_fp, total_fn)
hss = compute_hss(total_tp, total_fp, total_fn, total_tn)
```

### Pattern 3: Structured Metrics Dict Return
**What:** `validate()` returns a flat dict with all metrics instead of a tuple.
**When to use:** The refactored `validate()` function.
**Example:**
```python
# validate() returns:
{
    'val_loss': float,
    'val_mae_per_timestep': list[float],       # [t1, t2, t3, t4]
    'val_rmse_per_timestep': list[float],
    'val_correlation_per_timestep': list[float],
    'val_csi': float,                           # overall (all timesteps pooled)
    'val_csi_per_timestep': list[float],
    'val_hss': float,
    'val_hss_per_timestep': list[float],
    'val_ssim': float,
    'val_ssim_per_timestep': list[float],
    'persistence_mae_per_timestep': list[float],
    'persistence_skill_per_timestep': list[float],  # percentage
    'persistence_csi': float,
    'persistence_hss': float,
    'peak_flux_error_per_timestep': list[float],
    'temporal_variation_ratio': float,
}
```

### Pattern 4: Backward-Compatible History Extension
**What:** New keys are added to the `history` dict alongside existing keys. Existing keys (`train_loss`, `val_loss`, `val_mae_per_timestep`, `learning_rate`) remain unchanged.
**When to use:** In `train_model()` when merging validation results into history.
**Example:**
```python
# train_model() merges metrics dict into history:
val_metrics = validate(model, val_loader, ...)
history['train_loss'].append(train_loss)
history['val_loss'].append(val_metrics['val_loss'])
history['val_mae_per_timestep'].append(val_metrics['val_mae_per_timestep'])
# New keys:
history['val_csi'].append(val_metrics['val_csi'])
history['val_hss'].append(val_metrics['val_hss'])
# ... etc
```

### Anti-Patterns to Avoid
- **Batch-averaging CSI/HSS:** Computing CSI per batch then averaging gives incorrect results because CSI is non-linear. Always accumulate TP/FP/FN across all batches first.
- **Denormalizing for metrics:** All metrics should operate on normalized (asinh) values. Denormalization would create inconsistency with the loss function.
- **Putting metric logic in trainer.py:** Keep metric computation functions in `utils/metrics.py`. The trainer calls them but does not implement the math.
- **Using classes for simple metrics:** CSI, HSS, persistence are simple formulas. Pure functions are cleaner than stateful metric classes for this use case.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSIM computation | Custom SSIM metric function | `training.losses.ssim()` | Already MPS-compatible, tiling-aware, and tested. Import and call directly |
| Data range for SSIM | Guessed value | `ssim_data_range: 2.0` from config | Data is normalized to approximately [-1, 1], so range is 2.0. Already configured |

**Key insight:** This phase is about wiring and orchestration, not novel algorithms. Every metric formula is well-known and simple. The complexity is in correct accumulation, clean integration with the existing training loop, and backward-compatible output.

## Common Pitfalls

### Pitfall 1: CSI/HSS Batch Averaging
**What goes wrong:** Computing CSI per batch, then averaging batch CSIs gives a biased estimate, especially with small batch size (this project uses batch_size=1).
**Why it happens:** CSI = TP/(TP+FP+FN) is non-linear in its components. The average of ratios is not the ratio of averages.
**How to avoid:** Accumulate TP, FP, FN, TN as integers across all batches. Compute CSI/HSS once from the totals after the full validation pass.
**Warning signs:** CSI values that fluctuate wildly between epochs; CSI values that don't match manual spot-checks.

### Pitfall 2: Per-Timestep Accumulation Dimensions
**What goes wrong:** Incorrect dimension indexing when accumulating per-timestep metrics from (B, C, T, H, W) tensors.
**Why it happens:** The tensor layout is (B, C, T, H, W) where C=1 for output_channels. Forgetting to handle the C dimension or confusing T with C.
**How to avoid:** Always explicitly index: `pred[:, :, t, :, :]` for timestep t. Use `output_channels` slicing consistently: `Y_target = Y_out[:, :output_channels]`.
**Warning signs:** Shape mismatch errors; metrics arrays with wrong length.

### Pitfall 3: Persistence Baseline Shape Mismatch
**What goes wrong:** The last input frame has shape `(B, C, 1, H, W)` but needs to be compared against `(B, C, T, H, W)` target.
**Why it happens:** Forgetting to expand/repeat the frame across the T dimension.
**How to avoid:** `persistence = X_in[:, :output_channels, -1:, :, :].expand_as(Y_target)` -- use `-1:` (keeps dim) not `-1` (removes dim), then `expand_as`.
**Warning signs:** Broadcasting errors or silently wrong MAE values.

### Pitfall 4: Extreme Threshold in Wrong Space
**What goes wrong:** Using the raw-space threshold (30,019) instead of the normalized-space threshold (~0.3456) for binary classification.
**Why it happens:** The metadata stores the raw-space threshold. All validation happens in normalized space.
**How to avoid:** Store and use the threshold in normalized space. The default should be computed from data: `np.arcsinh(raw_threshold / softening) / scale`. Make this configurable in config.yaml.
**Warning signs:** CSI/HSS are always 0.0 (threshold too high in normalized space) or always 1.0 (too low).

### Pitfall 5: Validate Return Value Breaking Callers
**What goes wrong:** Changing `validate()` from returning `(loss, mae_per_timestep)` to returning a dict breaks `train_model()` and `main.py` unpacking.
**Why it happens:** Multiple callers destructure the return value as a tuple.
**How to avoid:** Update ALL call sites simultaneously. Search for `validate(` in the codebase. There are exactly 2 call sites: `trainer.py:train_model()` (line ~394) and `main.py:run_training()` (line ~201).
**Warning signs:** `TypeError: cannot unpack non-sequence dict` at runtime.

### Pitfall 6: Division by Zero in Persistence Skill
**What goes wrong:** `persistence_MAE` could theoretically be 0 if the target exactly matches the last input frame for all timesteps.
**Why it happens:** Unlikely but possible with constant-value regions.
**How to avoid:** Guard: `skill = (1 - model_mae / persistence_mae) * 100 if persistence_mae > 1e-8 else 0.0`.
**Warning signs:** NaN or Inf values in persistence skill.

### Pitfall 7: SSIM on 5D Tensors
**What goes wrong:** The existing `ssim()` function expects 4D input `(B, C, H, W)`, but predictions are `(B, C, T, H, W)`.
**Why it happens:** SSIM is a spatial metric, not temporal. Need to reshape.
**How to avoid:** Reshape to `(B*T, C, H, W)` before calling `ssim()`, same pattern used in `CompositeLoss.forward()`. Can also compute per-timestep by calling per frame.
**Warning signs:** RuntimeError about expected 4D input.

## Code Examples

Verified patterns from the existing codebase and standard meteorological definitions.

### CSI (Critical Success Index)
```python
# Source: NOAA/EUMETRAIN meteorological verification standards
# CSI = TP / (TP + FP + FN)
# Range: [0, 1], 1 = perfect, excludes correct negatives
def compute_csi(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0
```

### HSS (Heidke Skill Score)
```python
# Source: EUMETRAIN / NOAA meteorological verification
# HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
# Range: [-inf, 1], 0 = no skill (random), 1 = perfect
def compute_hss(tp: int, fp: int, fn: int, tn: int) -> float:
    num = 2 * (tp * tn - fp * fn)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / denom if denom > 0 else 0.0
```

### Binary Thresholding for Contingency Table
```python
# Predictions and targets in normalized asinh space
# threshold ~0.3456 for 99.5th percentile of |values|
def accumulate_contingency(pred, target, threshold):
    """Accumulate TP/FP/FN/TN counts from a batch.

    Args:
        pred: (B, C, T, H, W) predictions
        target: (B, C, T, H, W) targets
        threshold: absolute value threshold in normalized space

    Returns:
        per_timestep_counts: list of (tp, fp, fn, tn) per timestep
    """
    binary_pred = (pred.abs() > threshold).long()
    binary_target = (target.abs() > threshold).long()
    T = pred.shape[2]
    counts = []
    for t in range(T):
        p = binary_pred[:, :, t]
        g = binary_target[:, :, t]
        tp = (p * g).sum().item()
        fp = (p * (1 - g)).sum().item()
        fn = ((1 - p) * g).sum().item()
        tn = ((1 - p) * (1 - g)).sum().item()
        counts.append((int(tp), int(fp), int(fn), int(tn)))
    return counts
```

### Persistence Baseline
```python
# Source: CONTEXT.md decision -- last input frame repeated for all T output timesteps
def compute_persistence_prediction(X_in, output_channels, T_out):
    """Create persistence prediction from last input frame.

    Args:
        X_in: (B, C_in, T_in, H, W) input tensor
        output_channels: number of output channels to use
        T_out: number of output timesteps

    Returns:
        persistence: (B, output_channels, T_out, H, W) persistence prediction
    """
    last_frame = X_in[:, :output_channels, -1:, :, :]  # (B, C_out, 1, H, W)
    return last_frame.expand(-1, -1, T_out, -1, -1)     # (B, C_out, T, H, W)
```

### Standalone SSIM in Validation
```python
# Reuse existing ssim() from training/losses.py
from training.losses import ssim

def compute_ssim_per_timestep(pred, target, data_range=2.0):
    """Compute single-scale SSIM per timestep.

    Args:
        pred: (B, C, T, H, W) predictions
        target: (B, C, T, H, W) targets

    Returns:
        list of SSIM values per timestep
    """
    B, C, T, H, W = pred.shape
    ssim_values = []
    for t in range(T):
        val = ssim(
            pred[:, :, t], target[:, :, t],
            data_range=data_range, size_average=True
        )
        ssim_values.append(val.item())
    return ssim_values
```

### Peak Flux Error
```python
# Source: CONTEXT.md decision -- |max(pred) - max(target)| per timestep
def compute_peak_flux_error(pred, target):
    """Peak flux error per timestep.

    Args:
        pred: (B, C, T, H, W)
        target: (B, C, T, H, W)

    Returns:
        list of mean peak flux errors per timestep
    """
    T = pred.shape[2]
    errors = []
    for t in range(T):
        # Max over spatial dims (H, W) per sample
        pred_max = pred[:, :, t].amax(dim=(-2, -1))  # (B, C)
        target_max = target[:, :, t].amax(dim=(-2, -1))
        error = (pred_max - target_max).abs().mean().item()
        errors.append(error)
    return errors
```

### Temporal Variation Ratio
```python
# Source: CONTEXT.md decision -- aligns with Phase 8 temporal diff loss
def compute_temporal_variation_ratio(pred, target):
    """Temporal variation ratio = pred_variation / target_variation.

    Args:
        pred: (B, C, T, H, W) predictions (T >= 2)
        target: (B, C, T, H, W) targets

    Returns:
        ratio: float (1.0 = perfect, <1 = too smooth, >1 = too noisy)
    """
    T = pred.shape[2]
    if T < 2:
        return 1.0  # Cannot compute temporal variation with single frame

    pred_diffs = (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean().item()
    target_diffs = (target[:, :, 1:] - target[:, :, :-1]).abs().mean().item()

    if target_diffs < 1e-8:
        return 0.0  # Target has no variation
    return pred_diffs / target_diffs
```

### Refactored validate() Signature
```python
def validate(
    model, dataloader, device,
    loss_fn=None, use_amp=True, show_progress=True,
    output_channels=1,
    extreme_threshold=0.3456,  # NEW: configurable threshold
    ssim_data_range=2.0,       # NEW: from config
) -> dict:
    """Validate model with comprehensive metrics.

    Returns:
        Dict with all metric keys (see Architecture Patterns section).
    """
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `validate()` returns `(loss, mae)` tuple | Returns structured dict | Phase 7 | All callers must update; enables extensibility |
| Only MAE tracked per-timestep | MAE, RMSE, correlation, CSI, HSS per-timestep | Phase 7 | Comprehensive view of model quality |
| No baseline comparison | Persistence baseline with skill percentage | Phase 7 | Quantifies "does the model beat doing nothing?" |
| SSIM only inside loss function | SSIM as standalone metric + inside loss | Phase 7 | Can track structural quality independently of loss weighting |

**Note on current model performance:**
- Val MAE per timestep (epoch 25): [0.1054, 0.1080, 0.1091, 0.1101]
- The model's temporal variation ratio is known to be ~6% (from project research), meaning predictions are very smooth/near-persistence. This metric will quantify that directly.

## Open Questions

1. **Exact normalized threshold value**
   - What we know: raw threshold = 30,019.39 (99.5th percentile of |values|), which maps to ~0.3456 in asinh space
   - What's unclear: Whether this threshold should be validated against the actual preprocessed data distribution (the calculation uses sampled values)
   - Recommendation: Use 0.3456 as default, make configurable in config.yaml. The implementation task should verify by inspecting a few validation batches that the threshold produces a reasonable extreme/non-extreme split (roughly 0.5% extreme pixels).

2. **Verbose mode implementation**
   - What we know: Per-timestep breakdown should only print at final epoch or in verbose mode
   - What's unclear: Whether to use a config flag, CLI arg, or log level
   - Recommendation: Add `verbose_metrics: false` to config.yaml under a new `evaluation` section. Simple boolean check in the epoch summary print logic. Keeps it consistent with existing config-driven approach.

3. **Per-timestep vs pooled CSI/HSS for persistence**
   - What we know: Model CSI/HSS should be per-timestep. CONTEXT.md says persistence "also gets CSI/HSS computed"
   - What's unclear: Whether persistence CSI/HSS should also be per-timestep or just a single pooled value
   - Recommendation: Per-timestep for consistency (persistence skill likely degrades over timesteps too).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | None (uses pytest defaults + conftest.py markers) |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Per-timestep MAE, RMSE, correlation in validation | unit | `python -m pytest tests/test_metrics.py::test_per_timestep_mae_rmse_correlation -x` | No -- Wave 0 |
| EVAL-02 | CSI computation from contingency table | unit | `python -m pytest tests/test_metrics.py::test_csi_computation -x` | No -- Wave 0 |
| EVAL-03 | HSS computation from contingency table | unit | `python -m pytest tests/test_metrics.py::test_hss_computation -x` | No -- Wave 0 |
| EVAL-04 | Persistence baseline and skill-over-persistence | unit | `python -m pytest tests/test_metrics.py::test_persistence_baseline -x` | No -- Wave 0 |
| EVAL-05 | Standalone SSIM as validation metric | unit | `python -m pytest tests/test_metrics.py::test_standalone_ssim -x` | No -- Wave 0 |
| EVAL-06 | Peak flux error computation | unit | `python -m pytest tests/test_metrics.py::test_peak_flux_error -x` | No -- Wave 0 |
| EVAL-07 | Temporal variation ratio computation | unit | `python -m pytest tests/test_metrics.py::test_temporal_variation_ratio -x` | No -- Wave 0 |
| ALL | validate() returns correct dict structure | integration | `python -m pytest tests/test_metrics.py::test_validate_returns_dict -x` | No -- Wave 0 |
| ALL | train_model history contains new keys | integration | `python -m pytest tests/test_metrics.py::test_history_new_keys -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_metrics.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_metrics.py` -- new file covering all EVAL-* requirements (unit tests for pure metric functions + integration test for validate() dict return)
- [ ] No framework install needed (pytest already configured)
- [ ] No conftest changes needed (existing fixtures sufficient; may add a `tiny_predictions` fixture)

## Data Context

### Key Numbers for Implementation
| Property | Value | Source |
|----------|-------|--------|
| Tensor shape (predictions) | (B, 1, 4, H, W) | output_channels=1, t_out=4 |
| Tensor shape (input) | (B, 2, 10, H, W) | input_channels=2 (dual), t_in=10 |
| Spatial dims (with downsample) | ~220x442 | Original ~440x884, 2x downsampled |
| Batch size | 1 | config.yaml |
| Validation samples | ~40-80 (1 file, ~70 windows) | 10 files, 10% val split |
| Normalized data range | [-0.944, 1.000] | asinh normalization |
| Extreme threshold (normalized) | ~0.3456 | 99.5th percentile of abs(values) |
| Extreme threshold (raw) | 30,019.39 | metadata.json |
| Current val MAE range | 0.105 - 0.110 | training_history.json epoch 25 |

### Config.yaml New Section
```yaml
# Evaluation metrics configuration
evaluation:
  extreme_threshold: 0.3456    # Normalized space threshold for CSI/HSS binary classification
  verbose_metrics: false        # Print per-timestep breakdown every epoch (vs only final)
```

## Sources

### Primary (HIGH confidence)
- `training/trainer.py` -- existing validate() and train_model() implementation (lines 141-476)
- `utils/metrics.py` -- existing compute_metrics(), compute_rmse(), compute_correlation()
- `training/losses.py` -- existing ssim() function (lines 92-142), CompositeLoss 5D reshape pattern (lines 346-349)
- `data_processed/metadata.json` -- normalization params and extreme threshold
- `config.yaml` -- current configuration structure
- [EUMETRAIN CSI definition](https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos2/uos2_ko4.htm) -- CSI = TP/(TP+FP+FN)
- [EUMETRAIN HSS definition](https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos3/uos3_ko1.htm) -- HSS = 2(TP*TN-FP*FN)/((TP+FN)(FN+TN)+(TP+FP)(FP+TN))

### Secondary (MEDIUM confidence)
- Computed normalized threshold (0.3456) via `arcsinh(30019.39/1000) / 11.85` -- verified numerically but should be cross-checked against actual preprocessed validation data

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all PyTorch built-ins
- Architecture: HIGH -- existing codebase patterns well-understood, clear integration points
- Pitfalls: HIGH -- identified from actual code inspection (batch averaging, shape mismatches, caller breakage)
- Metric formulas: HIGH -- standard meteorological verification (NOAA/EUMETRAIN)
- Threshold value: MEDIUM -- computed numerically, should be verified against actual data

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable -- no external dependencies that could change)
