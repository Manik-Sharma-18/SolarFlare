# Phase 11: Integration Testing & Validation - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Full training run with all v3.0 features enabled simultaneously (SA-ConvLSTM, temporal attention, attention gates, temporal loss, cosine LR, flare oversampling, balanced augmentation). Produce a diagnostic comparison report documenting v3.0 vs v2.0 on all evaluation metrics. Validates that all prior phase work (7-10) integrates correctly and delivers measurable improvement.

</domain>

<decisions>
## Implementation Decisions

### v2.0 baseline strategy
- Use known v2.0 values from `outputs copy/diagnostic_results.json` (test split only)
- Hardcode v2.0 baseline numbers directly in the comparison script/report — no separate v2.0 config file
- v2.0 test baselines: MAE [0.102, 0.109, 0.112, 0.114], RMSE [0.145, 0.153, 0.156, 0.157], correlation [0.565, 0.508, 0.483, 0.467], persistence skill [2.9%, 4.7%, 5.2%, 5.1%], CSI 0.051, HSS 0.092, var ratio 0.060

### Comparison report format
- Markdown report at project root: `COMPARISON.md`
- Metric comparison tables with delta columns (v3.0 value, v2.0 value, change, % change)
- Bar charts / line plots comparing per-timestep metrics (MAE, RMSE, correlation, CSI) side by side — saved as PNGs referenced from markdown
- Include sample prediction visualizations (input -> predicted -> ground truth frames) showing qualitative temporal dynamics improvement
- Summary verdict at top: PASS (key metrics improved) / MIXED (some improved, some regressed) / REGRESSION (key metrics worse)
- Tradeoffs documented neutrally — no single metric trumps others, all reported honestly

### Success thresholds
- No fixed minimum targets — "any improvement" over v2.0 on temporal variation ratio and CSI counts as success
- Key metrics for verdict: temporal variation ratio (v2.0: 0.060), CSI (v2.0: 0.051), persistence skill (v2.0: 2.9-5.2%)
- MAE/RMSE regression is acceptable if temporal metrics improve — documented as tradeoff, not failure
- Per-timestep breakdown in report shows whether improvement holds across all horizons or degrades at longer timesteps

### Run strategy
- Single training run, seed 42, 50 epochs with cosine schedule
- Smoke test first: short 3-5 epoch run to verify all v3.0 features work together without crashes, NaN, or errors
- Full 50-epoch run after smoke test passes
- Device: MPS (Mac GPU), use_amp: false (already configured)
- If issues arise: diagnose root cause and fix (learning rate, loss imbalance, attention collapse), then re-run — investigation is part of the validation process

### Claude's Discretion
- Smoke test epoch count (3-5 range)
- Chart styling and layout for comparison visualizations
- Report prose structure and section ordering
- How to select representative prediction samples for qualitative comparison
- Whether to include attention entropy analysis in the report

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `main.py:run_training()`: Complete training pipeline — data loading, model creation, training, testing, visualization. Runs end-to-end with current config.yaml
- `outputs copy/diagnostic_results.json`: Full v2.0 baseline with per-timestep MAE, RMSE, correlation, persistence skill, CSI, HSS, var ratio across train/val/test splits
- `training/trainer.py:validate()`: Returns comprehensive metrics dict (val_loss, CSI, HSS, SSIM, persistence skill, temporal variation ratio, per-timestep breakdowns)
- `utils/visualization.py:plot_training_history()`: Existing training curve plotter — extend or create comparison-specific plots
- `utils/visualization.py:visualize_predictions()`: Generates prediction vs ground truth frames

### Established Patterns
- `config.yaml` already has all v3.0 defaults configured (SA-ConvLSTM, temporal loss, cosine LR, balanced augmentation, flare oversampling)
- `test_results.json` captures all metrics from test set evaluation — structured JSON for programmatic comparison
- `training_history.json` stores per-epoch metrics — available for loss curve comparison
- 11 existing test files in `tests/` covering model, losses, metrics, attention, checkpoint, config, data pipeline, SA-ConvLSTM, device

### Integration Points
- `main.py` is the entry point — run it to execute the full training pipeline
- `outputs/test_results.json` is the v3.0 results source after training completes
- Comparison report script reads both v2.0 baseline (hardcoded) and v3.0 results (from outputs/)
- Visualization PNGs saved alongside COMPARISON.md at project root (or outputs/)

</code_context>

<specifics>
## Specific Ideas

- v2.0 diagnostic data already exists at `outputs copy/diagnostic_results.json` — comprehensive per-timestep breakdown across all splits
- The v2.0 model produces only 6% of target's frame-to-frame variation (pred_variation: 0.006 vs target: 0.105) — this is the primary metric that should show dramatic improvement with temporal loss and SA-ConvLSTM
- v2.0 correlation degrades from 0.565 (t=1) to 0.467 (t=4) — the report should show whether v3.0's temporal attention and attention gates slow this degradation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-integration-testing-validation*
*Context gathered: 2026-03-09*
