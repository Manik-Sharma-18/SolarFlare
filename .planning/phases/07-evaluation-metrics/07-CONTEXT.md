# Phase 7: Evaluation Metrics - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Instrument comprehensive, per-timestep evaluation metrics into the validation loop: CSI, HSS, persistence baseline with skill-over-persistence, standalone SSIM, peak flux error, and temporal variation ratio. Every subsequent phase (loss, training, architecture) depends on these metrics to measure impact.

</domain>

<decisions>
## Implementation Decisions

### Extreme threshold for CSI/HSS
- Fixed absolute value, configurable in config.yaml (e.g., `extreme_threshold: <value>`)
- Default value derived from actual data distribution (Claude inspects validation data to pick a physically meaningful default)
- Binary classification: pixel is extreme (1) or not (0) -- standard meteorological approach
- Spatial per-pixel classification: CSI measures WHERE flares occur, not just whether a frame contains one
- Persistence predictions also get CSI/HSS computed for comparison

### Persistence baseline
- Definition: last input frame repeated for all T output timesteps ("nothing changes" baseline)
- Skill-over-persistence: percentage improvement = `(1 - model_MAE / persistence_MAE) * 100`
- Per-timestep granularity: skill at t=1, t=2, t=3, t=4 separately (shows degradation at longer horizons)
- Persistence also gets CSI/HSS computed alongside model CSI/HSS

### Metrics output and reporting
- Extend existing `training_history.json` with new keys (val_csi, val_hss, val_ssim, persistence_skill, peak_flux_error, temporal_variation_ratio, etc.)
- Console output: summary line per epoch with key metrics (CSI, HSS, persistence skill, SSIM). Full per-timestep breakdown only at final epoch or verbose mode
- Update `utils/visualization.py` to plot new metrics alongside existing loss curves
- Per-timestep CSI/HSS logged (consistent with per-timestep MAE/RMSE/correlation)

### Metric computation approach
- All metrics computed on normalized (asinh) values -- no denormalization during validation
- CSI/HSS: accumulate TP, FP, FN counts across all validation batches, compute once from totals (avoids batch-averaging bias)
- Temporal variation ratio: mean absolute difference `mean(|frame[t+1] - frame[t]|)`, ratio = pred_variation / target_variation (aligns with Phase 8 temporal diff loss)
- Standalone SSIM: single-scale SSIM (not MS-SSIM). MS-SSIM stays in loss function only
- Peak flux error: value only `|max(pred) - max(target)|`, no spatial offset tracking
- `validate()` returns a structured metrics dict (not tuple) -- clean, extensible, train_model() merges into history
- No special memory safeguards needed -- accumulate scalar counts and running sums only

### Claude's Discretion
- Exact data-derived threshold value for CSI/HSS
- Internal metric computation order and helper function organization
- Visualization subplot layout and styling for new metric plots
- How verbose mode is toggled (config flag, CLI arg, or log level)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils/metrics.py`: Has `compute_metrics` (MAE), `compute_rmse`, `compute_correlation` -- extend with CSI, HSS, persistence, peak flux, temporal variation
- `training/losses.py:ssim()`: Working single-scale SSIM function with MPS support and tiling -- reuse directly as standalone metric
- `training/losses.py:CompositeLoss.forward(return_components=True)`: Returns individual loss components -- pattern to follow for metric decomposition

### Established Patterns
- `training/trainer.py:validate()` currently returns `(avg_loss, avg_mae_per_timestep)` -- refactor to return metrics dict
- `training/trainer.py:train_model()` stores history as `{'train_loss': [], 'val_loss': [], 'val_mae_per_timestep': [], 'learning_rate': []}` -- extend with new metric keys
- Console logging uses `print()` with f-strings (not logger) for epoch summaries
- `utils/visualization.py` exists for training plots -- extend for new metrics

### Integration Points
- `validate()` in `training/trainer.py` is the main hook -- all new metrics computed here
- `train_model()` history dict stores per-epoch metrics -- add new keys
- `config.yaml` structure needs `extreme_threshold` parameter
- `utils/config_validator.py` may need to validate the new threshold config

</code_context>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches. Key principle: "you cannot improve what you cannot measure" -- these metrics are the foundation for all v3.0 improvements.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-evaluation-metrics*
*Context gathered: 2026-03-07*
