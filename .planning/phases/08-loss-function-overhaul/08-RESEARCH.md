# Phase 8: Loss Function Overhaul - Research

**Researched:** 2026-03-08
**Domain:** PyTorch custom loss functions for spatiotemporal prediction, temporal dynamics incentivization
**Confidence:** HIGH

## Summary

Phase 8 overhauls the composite loss function to penalize temporal stationarity, reward frame-to-frame dynamics, and apply stronger penalties for missing extreme solar flare regions. The current model produces near-persistence predictions (variation ratio 0.056, CSI 0.05), and the loss function is the primary lever to break this pattern before architecture changes in Phase 10.

All required operations use standard PyTorch primitives (tensor slicing, F.l1_loss, element-wise ops, .clamp()). No external libraries are needed. The existing `CompositeLoss` class in `training/losses.py` provides the `return_components=True` pattern that naturally extends to support new loss terms. The key architectural challenge is that `CompositeLoss.forward()` currently flattens the temporal dimension at lines 346-349 (B,C,T,H,W -> B*T,C,H,W) before computing any loss, which destroys the temporal information needed for temporal_diff, temporal_var, and per-timestep weighting. The forward method must be restructured to compute temporal-aware terms BEFORE flattening, then compute spatial-only terms (L1, MS-SSIM) after flattening.

**Primary recommendation:** Restructure `CompositeLoss.forward()` to compute temporal loss terms on the 5D tensor first, then flatten for spatial loss terms. Extend the existing `return_components` dict pattern for all 6 components. Wire per-epoch component values into `training_history.json` via the existing history dict pattern.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Temporal dynamics is the #1 priority -- temporal_diff and temporal_var get highest relative weights among new terms
- All 6 component weights (L1, MS-SSIM, WeightedMAE, temporal_diff, temporal_var, asymmetric) designed together as a coherent system -- rebalance existing weights if needed to make room for temporal terms
- Extreme weight set to 3.0 (LOSS-06 minimum spec)
- Per-timestep temporal weights: [1.0, 1.5, 2.0, 2.5] (linear ramp, as specified in LOSS-02)
- Asymmetric penalty applies ONLY above the extreme threshold -- normal regions use symmetric loss (matches LOSS-05 spec)
- Asymmetric alpha = 2.0 (underestimating a flare costs 2x overestimating)
- Fixed WeightedMAE uses binary weighting: base_weight=1.0 below threshold, extreme_weight=3.0 above threshold. No magnitude scaling within each zone.
- Default lambda = 0.1 (gentle nudge toward variation)
- Temporal variation penalty applies equally to ALL frame-to-frame transitions (no timestep weighting -- LOSS-02 already handles that signal separately)
- Temporal variation penalty applies to ALL pixels globally, not just extreme regions
- Capped at target variation: penalty = -lambda * min(pred_variation, target_variation). No reward for exceeding target's natural variation level. Prevents noisy/jittery predictions.
- Compact console summary per epoch: total loss + temporal_diff + temporal_var + extreme (3 new terms most relevant to v3.0 goals)
- Full breakdown of all 6 components saved to training_history.json every epoch
- Multi-panel loss breakdown plot added to training visualization (each component as subplot or overlaid lines)

### Claude's Discretion
- Exact starting weights for all 6 components (must prioritize temporal dynamics, rebalance coherently)
- Whether loss extreme threshold differs from eval threshold (0.3456)
- CompositeLoss restructuring to handle temporal-aware terms (current forward() flattens temporal dim early)
- Loss breakdown plot layout and styling
- Compact console line format

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LOSS-01 | Temporal difference loss term: L1(pred[t+1]-pred[t], target[t+1]-target[t]) | Compute on 5D tensor before flatten; produces T-1 diff frames; use F.l1_loss; verified in PyTorch 2.10.0 |
| LOSS-02 | Per-timestep temporal weighting (configurable, default [1.0, 1.5, 2.0, 2.5]) | Apply weights via broadcasting (1,1,T,1,1) to the loss tensor before mean reduction; must apply BEFORE temporal flatten |
| LOSS-03 | Temporal variation penalty: -lambda * mean(\|pred[t+1]-pred[t]\|) capped at target variation | Negative loss term rewards variation; capped via min(pred_var, target_var); lambda=0.1 default |
| LOSS-04 | Fix WeightedMAE to use absolute threshold (not per-sample relative) | Replace lines 283-284 in WeightedMAELoss; use binary mask (target.abs() > threshold) with fixed weights 1.0/3.0 |
| LOSS-05 | Asymmetric loss for underestimation above extreme threshold (alpha=2.0) | New AsymmetricExtremeLoss class; underestimation = (target - pred).clamp(min=0); applied only where target.abs() > threshold |
| LOSS-06 | Extreme weight increased to 3.0+ in default config | Update config.yaml loss.extreme_weight from 1.0 to 3.0 |
| LOSS-07 | Each loss component logged individually during training | Extend return_components dict; train_epoch captures components; history dict gets 6 new keys; compact console print; multi-panel plot |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | All loss computation, tensor ops | Already installed; all operations (slicing, F.l1_loss, clamp, broadcasting) verified on this version |
| matplotlib | (installed) | Loss component breakdown plot | Already used for all training visualization |

### Supporting
No new dependencies required. All loss function operations use PyTorch built-in tensor operations.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom temporal diff | kornia.losses | Unnecessary dependency; the temporal diff is 3 lines of PyTorch |
| Custom asymmetric loss | FocalLoss from torchvision | Focal loss addresses class imbalance differently; asymmetric underestimation penalty is simpler and more targeted |

**Installation:**
```bash
# No new packages needed
```

## Architecture Patterns

### Recommended Code Structure
```
training/
  losses.py          # Extended: CompositeLoss, WeightedMAELoss (fixed), AsymmetricExtremeLoss (new)
  trainer.py         # Modified: train_epoch captures components, train_model logs them

utils/
  visualization.py   # Extended: loss component breakdown plot

config.yaml          # Extended: new loss config keys
```

### Pattern 1: Temporal-First Loss Computation
**What:** Compute temporal-aware loss terms on the original 5D tensor (B,C,T,H,W), then flatten temporal dim for spatial-only terms.
**When to use:** Any time the loss function needs access to frame-to-frame information.
**Example:**
```python
# Source: verified against existing CompositeLoss.forward() structure
def forward(self, pred, target, return_components=False):
    # pred, target: (B, C, T, H, W)

    # --- TEMPORAL TERMS (need 5D) ---
    # Per-timestep weighting (LOSS-02)
    if pred.dim() == 5:
        B, C, T, H, W = pred.shape
        tw = torch.tensor(self.temporal_weights[:T], device=pred.device)
        tw = tw.view(1, 1, T, 1, 1)

        # Temporal difference loss (LOSS-01)
        pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        temporal_diff_loss = F.l1_loss(pred_diffs, target_diffs)

        # Temporal variation penalty (LOSS-03)
        pred_var = pred_diffs.abs().mean()
        target_var = target_diffs.abs().mean()
        temporal_var_loss = -self.temporal_var_lambda * torch.min(pred_var, target_var)

        # Apply per-timestep weights, then flatten for spatial terms
        pred_weighted = pred * tw
        target_weighted = target * tw
        pred_flat = pred_weighted.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        target_flat = target_weighted.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        pred_flat = pred
        target_flat = target
        temporal_diff_loss = torch.tensor(0.0, device=pred.device)
        temporal_var_loss = torch.tensor(0.0, device=pred.device)

    # --- SPATIAL TERMS (use flattened 4D) ---
    l1_loss = F.l1_loss(pred_flat, target_flat)
    ssim_loss = 1.0 - ssim_or_ms_ssim(pred_flat, target_flat)
    extreme_loss = self.weighted_mae(pred_flat, target_flat)
    asymmetric_loss = self.asymmetric_extreme(pred_flat, target_flat)

    # Combine
    total = (self.l1_weight * l1_loss
             + self.ssim_weight * ssim_loss
             + self.extreme_weight * extreme_loss
             + self.temporal_diff_weight * temporal_diff_loss
             + temporal_var_loss  # Already scaled by lambda, sign is negative
             + self.asymmetric_weight * asymmetric_loss)
```

### Pattern 2: Binary Threshold Weighting (Fixed WeightedMAE)
**What:** Replace per-sample relative normalization with absolute threshold binary mask.
**When to use:** When weight assignment must be consistent across all samples regardless of batch content.
**Example:**
```python
# Source: CONTEXT.md locked decision + existing WeightedMAELoss structure
def forward(self, pred, target):
    abs_error = torch.abs(pred - target)
    # Binary weighting: 1.0 below threshold, extreme_weight above
    extreme_mask = (target.abs() > self.threshold).float()
    weights = self.base_weight + (self.extreme_weight - self.base_weight) * extreme_mask
    weighted_error = weights * abs_error
    return weighted_error.mean()
```

### Pattern 3: Asymmetric Extreme Penalty
**What:** Apply heavier penalty for underestimating extreme regions, symmetric elsewhere.
**When to use:** When false negatives (missed flares) are operationally worse than false positives.
**Example:**
```python
# Source: IMPROVEMENT_NOTES section 1.3 + CONTEXT.md locked decisions
def forward(self, pred, target):
    above_threshold = (target.abs() > self.threshold).float()
    underestimation = (target - pred).clamp(min=0)  # positive where pred < target
    overestimation = (pred - target).clamp(min=0)    # positive where pred > target
    # Above threshold: alpha * under + over; Below threshold: standard MAE
    asym_error = above_threshold * (self.alpha * underestimation + overestimation)
    normal_error = (1 - above_threshold) * (underestimation + overestimation)
    return (asym_error + normal_error).mean()
```

### Pattern 4: Component Logging Integration
**What:** Capture per-component loss values in train_epoch and propagate to training history.
**When to use:** LOSS-07 requires individual logging of all 6 components.
**Example:**
```python
# In train_epoch:
if isinstance(loss_fn, CompositeLoss):
    components = loss_fn(predictions, Y_target, return_components=True)
    loss = components['total']
    # Accumulate each component for epoch average
    for key in component_keys:
        component_sums[key] += components[key].item()
else:
    loss = loss_fn(predictions, Y_target)

# In train_model, after train_epoch returns:
# Add component averages to history dict
for key in component_keys:
    history[f'train_{key}'].append(avg_components[key])
```

### Anti-Patterns to Avoid
- **Flattening temporal dim before computing temporal terms:** The current code at lines 346-349 flattens immediately. This must be deferred until after temporal_diff and temporal_var are computed.
- **Using per-sample normalization in WeightedMAE:** Lines 283-284 normalize by max_target, making weights inconsistent across samples. Use absolute threshold instead.
- **Applying temporal variation penalty to flattened tensors:** The penalty needs frame-to-frame diffs, which require the T dimension to exist.
- **Negative loss without capping:** Raw `-lambda * pred_var` could grow unboundedly, incentivizing noisy predictions. The cap at target_var prevents this.
- **Per-timestep weighting after flatten:** Once (B,C,T,H,W) is reshaped to (B*T,C,H,W), there is no way to identify which frames correspond to which timestep. Weighting must happen before reshaping.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| L1 loss | Custom mean absolute error | `F.l1_loss()` | PyTorch built-in, optimized, gradient-correct |
| SSIM computation | Custom SSIM | Existing `ssim()` / `ms_ssim()` in losses.py | Already implemented with MPS compatibility, tiling, caching |
| Gradient clipping | Manual norm computation | `torch.nn.utils.clip_grad_norm_()` | Already used in train_epoch |
| Config-driven construction | Hardcoded loss params | `get_loss_function(config)` factory | Existing pattern; extend, don't replace |

**Key insight:** Every new loss term (temporal_diff, temporal_var, asymmetric) uses only basic PyTorch tensor operations. No custom autograd functions are needed. Standard PyTorch operations propagate gradients correctly through slicing, clamping, and element-wise multiplication.

## Common Pitfalls

### Pitfall 1: Temporal Flatten Before Temporal Terms
**What goes wrong:** Computing temporal_diff on already-flattened (B*T,C,H,W) is impossible -- the T dimension no longer exists.
**Why it happens:** The existing code flattens at the start of forward(). Copy-paste from existing code pattern.
**How to avoid:** Restructure forward() into two phases: temporal terms on 5D, then flatten for spatial terms.
**Warning signs:** temporal_diff_loss is always 0 or NaN; temporal_var_loss has no effect on training.

### Pitfall 2: Per-Timestep Weights Applied to Wrong Dimension
**What goes wrong:** Weights [1.0, 1.5, 2.0, 2.5] broadcast to the wrong axis, weighting channels or spatial dims instead of timesteps.
**Why it happens:** 5D tensor (B,C,T,H,W) has T at dim=2. Easy to confuse with dim=1 (channels).
**How to avoid:** Always reshape weights to (1,1,T,1,1) to explicitly match the temporal dimension.
**Warning signs:** Per-timestep MAE does not show expected gradient (later timesteps not improving faster).

### Pitfall 3: Unbounded Temporal Variation Penalty
**What goes wrong:** Without capping, `-lambda * pred_var` rewards arbitrarily large prediction variation, causing noisy/jittery outputs.
**Why it happens:** Negative loss terms are inherently unstable if unconstrained.
**How to avoid:** Cap with `torch.min(pred_var, target_var)` -- no reward for exceeding target's natural variation.
**Warning signs:** Training loss goes negative; predictions become noisy/oscillating rather than smooth dynamics.

### Pitfall 4: Asymmetric Loss Applied Below Threshold
**What goes wrong:** Applying alpha penalty to normal regions causes the model to overpredict everywhere, not just in extreme regions.
**Why it happens:** Missing the threshold mask in the asymmetric term.
**How to avoid:** Multiply asymmetric error by `above_threshold` mask. Normal regions use standard symmetric error.
**Warning signs:** Model produces systematically high predictions across the entire spatial domain, not just active regions.

### Pitfall 5: train_epoch Returns Only Total Loss
**What goes wrong:** Per-component values are computed but never returned to the training loop, so they cannot be logged.
**Why it happens:** train_epoch currently returns `(avg_loss, consecutive_nan_count)`. Adding component returns requires signature change.
**How to avoid:** Return component averages as a third element (dict), or modify train_epoch to accept a component accumulator.
**Warning signs:** training_history.json has empty lists for component keys.

### Pitfall 6: Loss Weight Magnitudes Cause Gradient Imbalance
**What goes wrong:** If temporal_diff_weight is too high relative to L1, the model optimizes only for matching deltas and ignores absolute accuracy.
**Why it happens:** The 6 loss components have different natural scales. L1 and SSIM are O(0.01-0.1); temporal_diff may be larger or smaller depending on data.
**How to avoid:** Start with conservative weights, verify each component's gradient magnitude during the first few batches. Log component values to see relative scales.
**Warning signs:** One component dominates (>80% of total loss); other components stagnate.

### Pitfall 7: Temporal Variation Penalty Gradient Sign
**What goes wrong:** The negative sign in `-lambda * min(pred_var, target_var)` means the gradient INCREASES the total loss when variation increases. This is correct (we want to minimize total loss, and a negative term decreases it when variation increases). But if implemented as `+lambda * min(...)`, the penalty is reversed.
**Why it happens:** Confusion about the sign convention.
**How to avoid:** The penalty is SUBTRACTED from the total loss. Double-check: with higher pred_var (up to target_var), total loss should DECREASE.
**Warning signs:** Temporal variation ratio decreases rather than increases during training.

## Code Examples

### Fixed WeightedMAELoss (LOSS-04)
```python
# Source: existing WeightedMAELoss refactored per CONTEXT.md decision
class WeightedMAELoss(nn.Module):
    def __init__(self, base_weight: float = 1.0, extreme_weight: float = 3.0,
                 threshold: float = 0.3456):
        super().__init__()
        self.base_weight = base_weight
        self.extreme_weight = extreme_weight
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_error = torch.abs(pred - target)
        extreme_mask = (target.abs() > self.threshold).float()
        weights = self.base_weight + (self.extreme_weight - self.base_weight) * extreme_mask
        return (weights * abs_error).mean()
```

### AsymmetricExtremeLoss (LOSS-05)
```python
# Source: IMPROVEMENT_NOTES section 1.3 + CONTEXT.md locked decisions
class AsymmetricExtremeLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, threshold: float = 0.3456):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        above_threshold = (target.abs() > self.threshold).float()
        underestimation = (target - pred).clamp(min=0)
        overestimation = (pred - target).clamp(min=0)
        # Above threshold: asymmetric; below: standard MAE
        error = above_threshold * (self.alpha * underestimation + overestimation) \
                + (1 - above_threshold) * (underestimation + overestimation)
        return error.mean()
```

### Temporal Difference Loss (LOSS-01)
```python
# Source: verified PyTorch 2.10.0 tensor operations
def compute_temporal_diff_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on frame-to-frame changes. Input: (B, C, T, H, W)."""
    pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    return F.l1_loss(pred_diffs, target_diffs)
```

### Temporal Variation Penalty (LOSS-03)
```python
# Source: CONTEXT.md locked decision for capping behavior
def compute_temporal_var_penalty(pred: torch.Tensor, target: torch.Tensor,
                                  lambda_val: float = 0.1) -> torch.Tensor:
    """Negative loss: rewards variation up to target level. Input: (B, C, T, H, W)."""
    pred_diffs = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    pred_var = pred_diffs.abs().mean()
    target_var = target_diffs.abs().mean()
    # Cap: no reward for exceeding target variation
    capped_var = torch.min(pred_var, target_var)
    return -lambda_val * capped_var
```

### Per-Timestep Weighting (LOSS-02)
```python
# Source: verified broadcasting on (B,C,T,H,W) tensors
def apply_temporal_weights(loss_tensor: torch.Tensor,
                           weights: list) -> torch.Tensor:
    """Apply per-timestep weights to a 5D loss tensor before reduction."""
    T = loss_tensor.shape[2]
    tw = torch.tensor(weights[:T], device=loss_tensor.device, dtype=loss_tensor.dtype)
    tw = tw.view(1, 1, T, 1, 1)
    return (loss_tensor * tw).mean()
```

### Compact Console Summary (LOSS-07)
```python
# Source: existing print patterns in trainer.py (f-strings, .6f/.4f format)
print(f"  Loss: {total:.6f} | TDiff: {temporal_diff:.4f}"
      f" | TVar: {temporal_var:.4f} | Extreme: {extreme:.4f}")
```

## Design Decisions Within Claude's Discretion

### Loss Weight Recommendation

Based on the constraint that temporal dynamics is priority #1, and the existing weights are L1=1.0, SSIM=0.5, extreme=1.0 (total=2.5):

**Recommended initial weights for all 6 components:**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| L1 | 1.0 | Unchanged baseline; provides stable gradient signal |
| MS-SSIM | 0.3 | Reduced from 0.5; spatial structure preserved by L1, free weight budget for temporal terms |
| WeightedMAE | 3.0 | Locked per LOSS-06 spec; up from 1.0 |
| temporal_diff | 1.0 | Equal to L1; high priority per user decision |
| temporal_var | -0.1 (lambda) | Locked per user decision; gentle nudge |
| asymmetric | 0.5 | Moderate; supplements WeightedMAE for extreme regions |

**Rationale:** The old total effective weight was ~2.5 (L1=1.0, SSIM=0.5, extreme=1.0). The new total is ~5.8 excluding temporal_var (which subtracts). The temporal terms (diff=1.0 + var=0.1) represent ~19% of total loss, ensuring temporal dynamics get strong gradient signal. WeightedMAE at 3.0 ensures extreme regions dominate over L1. SSIM reduced because its primary role (spatial structure) overlaps with L1, and weight budget is needed for temporal terms.

**Confidence: MEDIUM** -- these are starting points. Optimal weights depend on the actual gradient magnitudes of each term, which can only be determined empirically. The first training run should log component values to verify balance.

### Extreme Threshold for Loss

**Recommendation:** Use the same threshold as evaluation (0.3456).

**Rationale:** Using the same threshold ensures the loss function optimizes for the same definition of "extreme" that CSI/HSS measure. A different threshold would create a mismatch: the model could optimize well for loss-extreme but poorly for eval-extreme. The 0.3456 threshold corresponds to the 99.5th percentile in normalized space, which is the physically motivated boundary.

**Confidence: HIGH** -- using consistent thresholds across loss and evaluation is standard practice in binary-event prediction. If the threshold were too aggressive (capturing too few pixels), the loss gradient would be too sparse. At 99.5th percentile, approximately 0.5% of pixels are extreme, which provides enough gradient signal while maintaining focus on rare events.

### CompositeLoss Restructuring

**Recommendation:** Two-phase forward pass within the same method.

The forward method should:
1. Check if input is 5D (has temporal dim)
2. If 5D: compute temporal_diff, temporal_var on full 5D tensor. Apply per-timestep weights. Then flatten to 4D.
3. Compute L1, SSIM, WeightedMAE, asymmetric on 4D tensor
4. Combine all terms with weights

This avoids splitting into separate classes and maintains backward compatibility (4D input still works for cases without temporal dimension).

### Loss Breakdown Plot

**Recommendation:** Add a new subplot row to `plot_training_history()`, expanding the layout from 2x3 to 3x3.

New row 2 would contain:
- (2,0): All 6 loss components overlaid on one axes (log scale y-axis), one line per component
- (2,1): Temporal terms only (temporal_diff + temporal_var) on linear scale
- (2,2): Extreme terms only (WeightedMAE + asymmetric) on linear scale

This keeps the existing 6 subplots intact (backward compatible) and adds focused views of the new terms.

**Confidence: HIGH** -- follows the existing visualization pattern and provides actionable diagnostic views.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-sample relative normalization for extreme weighting | Absolute threshold binary weighting | This phase | Consistent penalty across samples; model gets same signal regardless of batch content |
| Single total loss logged | Per-component loss breakdown | This phase | Diagnostic ability to identify which loss term is driving training behavior |
| No temporal loss terms | Temporal difference + variation penalty | This phase | Direct gradient signal for frame-to-frame dynamics |
| Symmetric error for all regions | Asymmetric penalty above threshold | This phase | Model learns to err on side of predicting flares when uncertain |

**Deprecated/outdated:**
- `WeightedMAELoss` relative normalization (lines 283-284): Being replaced with absolute threshold per LOSS-04

## Open Questions

1. **Optimal loss weight balance**
   - What we know: Starting weights recommended above based on analysis of gradient priorities
   - What's unclear: Actual gradient magnitudes of each component at initialization; whether temporal_diff naturally produces values at the same scale as L1
   - Recommendation: Log component values during first epoch to verify relative scales; adjust weights if any component is >80% of total

2. **Interaction between temporal_diff and per-timestep weighting**
   - What we know: temporal_diff operates on T-1 difference frames; per-timestep weights apply to T frames before flattening
   - What's unclear: Should temporal_diff also be weighted by timestep? (e.g., later diffs weighted more)
   - Recommendation: Per CONTEXT.md, temporal_var applies equally to all transitions (no timestep weighting). Apply same principle to temporal_diff for consistency -- LOSS-02 handles per-timestep focus via the main loss terms.

3. **Per-timestep weighting application strategy**
   - What we know: Weights must apply before temporal flatten
   - What's unclear: Whether to weight the entire loss tensor (pred and target scaled) or only the error tensor
   - Recommendation: Weight the loss per-element before reduction. Compute element-wise L1 without reduction, multiply by temporal weights, then take mean. This is more principled than scaling pred/target (which would change SSIM behavior).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (installed, tests/ directory exists) |
| Config file | tests/conftest.py (shared fixtures including base_config, device) |
| Quick run command | `python -m pytest tests/test_losses.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LOSS-01 | Temporal diff loss produces non-zero gradient for different frame-to-frame changes | unit | `python -m pytest tests/test_losses.py::TestTemporalDiffLoss -x` | Wave 0 |
| LOSS-02 | Per-timestep weighting makes later timesteps contribute more to loss | unit | `python -m pytest tests/test_losses.py::TestTemporalWeighting -x` | Wave 0 |
| LOSS-03 | Temporal var penalty is negative, capped at target variation | unit | `python -m pytest tests/test_losses.py::TestTemporalVarPenalty -x` | Wave 0 |
| LOSS-04 | WeightedMAE uses absolute threshold, not per-sample normalization | unit | `python -m pytest tests/test_losses.py::TestWeightedMAE -x` | Existing (needs update) |
| LOSS-05 | Asymmetric loss penalizes underestimation more above threshold | unit | `python -m pytest tests/test_losses.py::TestAsymmetricExtremeLoss -x` | Wave 0 |
| LOSS-06 | extreme_weight=3.0 in config | unit | `python -m pytest tests/test_config.py -x` | Existing (may need update) |
| LOSS-07 | CompositeLoss return_components includes all 6 terms | unit | `python -m pytest tests/test_losses.py::TestCompositeLoss -x` | Existing (needs update) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_losses.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_losses.py::TestTemporalDiffLoss` -- covers LOSS-01 (temporal diff returns non-zero for different dynamics, zero for identical dynamics)
- [ ] `tests/test_losses.py::TestTemporalWeighting` -- covers LOSS-02 (later timesteps weighted more; weight shape broadcast correctness)
- [ ] `tests/test_losses.py::TestTemporalVarPenalty` -- covers LOSS-03 (negative value; capped at target variation; zero for T=1)
- [ ] `tests/test_losses.py::TestAsymmetricExtremeLoss` -- covers LOSS-05 (asymmetric > symmetric for underestimation above threshold; symmetric below threshold)
- [ ] Update `tests/test_losses.py::TestWeightedMAE` -- covers LOSS-04 (verify absolute threshold behavior; same weight regardless of batch max)
- [ ] Update `tests/test_losses.py::TestCompositeLoss::test_composite_loss_components` -- covers LOSS-07 (all 6 component keys present in return dict)
- [ ] `tests/test_losses.py::TestCompositeLoss::test_composite_loss_5d_temporal_terms` -- verify temporal terms computed correctly on 5D input
- [ ] Update `tests/test_losses.py::TestGetLossFunction` -- covers LOSS-06 + new config keys (temporal_diff_weight, etc.)

## Sources

### Primary (HIGH confidence)
- `training/losses.py` -- examined full source code of CompositeLoss, WeightedMAELoss, get_loss_function (425 lines)
- `training/trainer.py` -- examined train_epoch, validate, train_model for integration points (743 lines)
- `config.yaml` -- examined current loss config section (lines 59-67)
- `utils/visualization.py` -- examined plot_training_history for extension points (163 lines)
- `tests/test_losses.py` -- examined existing 19 tests for coverage gaps (224 lines)
- PyTorch 2.10.0 tensor operations -- verified temporal diff, per-timestep weighting, clamping, binary masking all work correctly

### Secondary (MEDIUM confidence)
- `.planning/IMPROVEMENT_NOTES.md` -- loss function improvement proposals (sections 1.1, 1.2, 1.3, 7.1, 7.4, 7.7)
- `.planning/research/TEMPORAL_ARCHITECTURES.md` -- delta prediction analysis confirming loss function is primary lever
- `.planning/phases/08-loss-function-overhaul/08-CONTEXT.md` -- user decisions constraining implementation

### Tertiary (LOW confidence)
- None. All findings verified against existing codebase and PyTorch documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all PyTorch built-ins, no new dependencies, verified on installed version
- Architecture: HIGH -- extension of existing patterns (CompositeLoss, return_components, config factory), restructuring approach verified
- Pitfalls: HIGH -- identified from direct code analysis of existing forward() method and training loop integration
- Loss weights: MEDIUM -- starting points based on analysis; empirical validation needed during training

**Research date:** 2026-03-08
**Valid until:** indefinite (pure PyTorch operations, no version-sensitive APIs)
