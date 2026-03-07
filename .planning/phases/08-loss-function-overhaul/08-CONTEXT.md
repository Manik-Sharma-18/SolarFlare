# Phase 8: Loss Function Overhaul - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

The loss function directly penalizes temporal stationarity, rewards capturing frame-to-frame dynamics, and applies stronger penalties for missing extreme regions. Adds temporal difference loss, temporal weighting, variation penalty, fixes WeightedMAE, and adds asymmetric extreme penalty. Each loss component is logged individually. Requirements: LOSS-01 through LOSS-07.

</domain>

<decisions>
## Implementation Decisions

### Loss balance & priorities
- Temporal dynamics is the #1 priority — temporal_diff and temporal_var get highest relative weights among new terms
- All 6 component weights (L1, MS-SSIM, WeightedMAE, temporal_diff, temporal_var, asymmetric) designed together as a coherent system — rebalance existing weights if needed to make room for temporal terms
- Extreme weight set to 3.0 (LOSS-06 minimum spec)
- Per-timestep temporal weights: [1.0, 1.5, 2.0, 2.5] (linear ramp, as specified in LOSS-02)

### Extreme region definition
- Claude's discretion on whether loss threshold matches eval threshold (0.3456) or uses a separate value, based on data distribution analysis
- Asymmetric penalty applies ONLY above the extreme threshold — normal regions use symmetric loss (matches LOSS-05 spec)
- Asymmetric alpha = 2.0 (underestimating a flare costs 2x overestimating)
- Fixed WeightedMAE uses binary weighting: base_weight=1.0 below threshold, extreme_weight=3.0 above threshold. No magnitude scaling within each zone.

### Temporal variation penalty
- Default lambda = 0.1 (gentle nudge toward variation)
- Applies equally to ALL frame-to-frame transitions (no timestep weighting — LOSS-02 already handles that signal separately)
- Applies to ALL pixels globally, not just extreme regions
- Capped at target variation: penalty = -lambda * min(pred_variation, target_variation). No reward for exceeding target's natural variation level. Prevents noisy/jittery predictions.

### Component reporting
- Compact console summary per epoch: total loss + temporal_diff + temporal_var + extreme (3 new terms most relevant to v3.0 goals)
- Full breakdown of all 6 components saved to training_history.json every epoch
- Multi-panel loss breakdown plot added to training visualization (each component as subplot or overlaid lines)

### Claude's Discretion
- Exact starting weights for all 6 components (must prioritize temporal dynamics, rebalance coherently)
- Whether loss extreme threshold differs from eval threshold (0.3456)
- CompositeLoss restructuring to handle temporal-aware terms (current forward() flattens temporal dim early)
- Loss breakdown plot layout and styling
- Compact console line format

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `training/losses.py:CompositeLoss`: Existing composite loss with `return_components=True` pattern — extend with new terms
- `training/losses.py:WeightedMAELoss`: Bug to fix (uses per-sample relative normalization at line 283-284) — replace with absolute threshold binary weighting
- `training/losses.py:get_loss_function()`: Factory reads config dict — extend with new parameters
- `training/losses.py:ssim()` and `ms_ssim()`: SSIM infrastructure unchanged, reuse as-is

### Established Patterns
- `CompositeLoss.forward(return_components=True)` returns dict of individual loss terms — extend pattern for new components
- Config-driven loss construction via `get_loss_function(config)` — add new config keys for temporal/asymmetric params
- Console logging uses `print()` with f-strings for epoch summaries — extend for compact loss component line
- `training_history.json` stores per-epoch metrics as lists — add new loss component keys

### Integration Points
- `CompositeLoss.forward()` currently flattens temporal dim (B,C,T,H,W -> B*T,C,H,W) at line 347-349 — must restructure to preserve temporal info for temporal_diff and temporal_var terms before flattening for spatial terms
- `training/trainer.py:train_epoch()` calls loss function — must capture and log per-component values
- `config.yaml` loss section — add temporal_diff_weight, temporal_var_lambda, asymmetric_alpha, temporal_weights, extreme_threshold
- `utils/visualization.py` — add loss component breakdown plot

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Key principle from v3.0 thesis: the model currently produces near-persistence predictions (variation ratio 0.056, CSI 0.05). The loss function overhaul is the primary mechanism to break this pattern.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-loss-function-overhaul*
*Context gathered: 2026-03-08*
