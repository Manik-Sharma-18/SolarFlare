---
gsd_state_version: 1.0
milestone: v3.0
milestone_name: Temporal Dynamics & Flare Detection
status: executing
stopped_at: Completed 09-02-PLAN.md
last_updated: "2026-03-08T11:56:46.462Z"
last_activity: 2026-03-08 — Completed 09-02 v3.0 training policy defaults
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-07)

**Core value:** Transform the model from near-persistence predictions into a genuine temporal forecaster with strong flare detection
**Current focus:** Phase 9 - Training Policy

## Current Position

Phase: 9 of 11 (Training Policy)
Plan: 2 of 2 in current phase (09-02 complete, phase complete)
Status: Executing
Last activity: 2026-03-08 — Completed 09-02 v3.0 training policy defaults

Progress: [██████████] 100%

## Accumulated Context

### Decisions

See PROJECT.md Key Decisions table for full history.

- [v3.0 roadmap]: Metrics first (Phase 7), then loss (8), training (9), architecture (10), integration (11) -- you cannot improve what you cannot measure, and loss fixes must precede architecture scaling
- [v3.0 research]: All features use PyTorch built-ins; no new dependencies required
- [07-01]: Lazy import of ssim from training.losses to avoid circular dependency risk
- [07-01]: All metric functions pure/stateless; contingency table returns Python ints for direct CSI/HSS formula use
- [Phase 07]: validate() returns dict instead of tuple; contingency tables accumulated across all batches before CSI/HSS
- [Phase 07]: Evaluation config section optional with defaults (threshold=0.3456, verbose=false); per-timestep breakdown only at final epoch
- [Phase 08-01]: Used same threshold (0.3456) for loss as evaluation -- consistent definition of extreme
- [Phase 08-01]: WeightedMAELoss default extreme_weight changed from 2.0 to 3.0 per LOSS-06 spec
- [Phase 08-01]: T<=1 edge case returns 0.0 tensor for temporal functions (graceful degradation)
- [Phase 08-02]: Per-timestep weights applied to element-wise L1 error (not pred/target) to preserve SSIM behavior
- [Phase 08-02]: ssim_weight reduced from 0.5 to 0.3 per research recommendation for temporal term influence
- [Phase 08-02]: Config cross-check warning when loss.extreme_threshold differs from evaluation.extreme_threshold
- [Phase 08-02]: Per-timestep weights applied to element-wise L1 error (not pred/target) to preserve SSIM behavior
- [Phase 08]: Component tracking uses isinstance detection rather than flag parameter for simple API
- [Phase 09-01]: Flare detection scans output frames only (not input) -- oversamples what model needs to predict
- [Phase 09-01]: Sampler uses replacement=True with num_samples=len(weights) to maintain epoch length
- [Phase 09-01]: Flare threshold sourced from evaluation.extreme_threshold (0.3456) when oversampling enabled
- [Phase 09-02]: Plan 01 already wired flare flags and oversample weight through main.py; Plan 02 added config defaults and diagnostic logging only

### Pending Todos

None.

### Blockers/Concerns

- Known quirk: predictor forward() double-preprocess when downsample_input=False
- MPS runtime validation deferred (no CI hardware)
- [Research]: Optimal loss term weights (temporal_diff, temporal_var, asymmetric_alpha) unknown -- determine empirically during Phase 8
- [Research]: Overfitting risk with 4x parameter increase on 568 samples -- monitor train-val gap in Phase 10, fallback to [24,48,96] channels

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Create model architecture document with visuals for non-ML audience | 2026-02-17 | b1a2396 | [001-model-architecture-doc](./quick/001-model-architecture-doc/) |

## Session Continuity

Last session: 2026-03-08T11:53:32.363Z
Stopped at: Completed 09-02-PLAN.md
Resume file: None
