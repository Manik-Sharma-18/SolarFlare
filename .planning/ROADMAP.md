# Roadmap: SolarFlare v3.0

## Overview

SolarFlare v3.0 transforms the model from near-persistence predictions (variation ratio 0.056, CSI 0.05) into a genuine temporal forecaster with strong flare detection. The roadmap proceeds in strict dependency order: first instrument evaluation metrics so every change is measurable, then overhaul the loss function to directly attack broken temporal dynamics, then tune training policy to amplify loss improvements, then scale architecture with attention and wider channels, and finally validate everything end-to-end against the v2.0 baseline.

## Milestones

- ✅ **v2.0 Stabilization & Cross-Platform** - Phases 1-6 (shipped 2026-02-04)
- **v3.0 Temporal Dynamics & Flare Detection** - Phases 7-11 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

v2.0 Stabilization & Cross-Platform (Phases 1-6) - SHIPPED 2026-02-04

See .planning/MILESTONES.md for v2.0 details.



### v3.0 Temporal Dynamics & Flare Detection

- **Phase 7: Evaluation Metrics** - Instrument CSI, HSS, persistence baseline, and per-timestep metrics into the validation loop
- **Phase 8: Loss Function Overhaul** - Add temporal difference loss, temporal weighting, variation penalty, fix WeightedMAE, and asymmetric extreme penalty
- **Phase 9: Training Policy** - Enable cosine LR, balanced augmentation, eliminate teacher forcing, and add class-imbalanced sampling
- **Phase 10: Architecture Scaling** - SA-ConvLSTM cells, spatial/temporal attention, wider channels, kernel 5, MC dropout, delta head scaling
- **Phase 11: Integration Testing & Validation** - Full training run with all v3.0 features, diagnostic comparison against v2.0 baseline

## Phase Details

### Phase 7: Evaluation Metrics

**Goal**: Every training run produces comprehensive, per-timestep evaluation metrics that quantify temporal dynamics, flare detection skill, and improvement over persistence baseline
**Depends on**: Nothing (first v3.0 phase; builds on existing v2.0 training loop)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07
**Success Criteria** (what must be TRUE):

1. After a validation epoch, per-timestep MAE, RMSE, and correlation are logged for each output frame
2. CSI and HSS appear in epoch logs computed against the extreme threshold, giving a single number for flare detection quality
3. Persistence baseline MAE is computed each epoch, and skill-over-persistence percentage is reported alongside model MAE
4. SSIM is logged as a standalone validation metric (not buried inside composite loss), and peak flux error and temporal variation ratio are reported per epoch
**Plans:** 2 plans

Plans:
- [x] 07-01-PLAN.md — TDD: Implement all metric computation functions (CSI, HSS, persistence, SSIM, peak flux, temporal variation, per-timestep RMSE/correlation) with tests
- [ ] 07-02-PLAN.md — Wire metrics into validate(), update callers, add evaluation config and metric visualization

### Phase 8: Loss Function Overhaul

**Goal**: The loss function directly penalizes temporal stationarity, rewards capturing frame-to-frame dynamics, and applies stronger penalties for missing extreme regions
**Depends on**: Phase 7 (metrics must exist to measure loss changes)
**Requirements**: LOSS-01, LOSS-02, LOSS-03, LOSS-04, LOSS-05, LOSS-06, LOSS-07
**Success Criteria** (what must be TRUE):

1. Temporal difference loss is active: the model is penalized for incorrect frame-to-frame changes (L1 on predicted deltas vs target deltas)
2. Later timesteps receive higher loss weight (configurable per-timestep weights default to [1.0, 1.5, 2.0, 2.5]), visible in per-component loss logs
3. WeightedMAE uses an absolute extreme threshold (not per-sample relative normalization), and asymmetric loss applies a configurable alpha penalty for underestimating extreme regions
4. Each loss component (L1, MS-SSIM, WeightedMAE, temporal_diff, temporal_var, asymmetric) is logged individually during training, not just the total
5. Temporal variation penalty encourages the model to produce frame-to-frame variation, with a configurable lambda weight
**Plans:** 3 plans

Plans:
- [x] 08-01-PLAN.md — TDD: Implement loss components (fix WeightedMAE, AsymmetricExtremeLoss, temporal diff/var/weighting functions)
- [x] 08-02-PLAN.md — Restructure CompositeLoss with 6-component temporal-aware forward, update config and factory
- [ ] 08-03-PLAN.md — Wire per-component loss logging into training loop and visualization

### Phase 9: Training Policy

**Goal**: Training configuration maximizes the effectiveness of the new loss function through better learning rate scheduling, data exposure, and honest autoregressive predictions
**Depends on**: Phase 8 (loss function must be stable before changing training regime)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05
**Success Criteria** (what must be TRUE):

1. Cosine annealing LR scheduler is active with eta_min=1e-6, and the learning rate curve is visible in training logs
2. Data augmentation includes both horizontal and vertical flips (3x effective dataset size), and flare-containing sequences are oversampled 3x via WeightedRandomSampler
3. Teacher forcing is eliminated (tf_start=0.0): the model produces all output frames autoregressively from step 1
4. Training runs for 50+ epochs with the cosine schedule, and the model converges (validation loss stabilizes or decreases)
**Plans:** 2 plans

Plans:
- [x] 09-01-PLAN.md — TDD: Implement flare-aware weighted sampling (build_index flare detection, WeightedRandomSampler integration, config cross-check)
- [ ] 09-02-PLAN.md — Update config.yaml with v3.0 training defaults and wire flare sampling through main.py

### Phase 10: Architecture Scaling

**Goal**: The model has sufficient representational capacity -- through attention mechanisms, wider channels, and larger kernels -- to exploit the improved loss and training regime
**Depends on**: Phase 9 (training policy must be stable before increasing model capacity)
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05, ARCH-06, ARCH-07
**Success Criteria** (what must be TRUE):

1. SA-ConvLSTM cells with self-attention memory replace standard ConvLSTM cells, and the encoder stores all hidden states for temporal attention access
2. Spatial attention gates on skip connections learn to focus on active regions (attention weights are not collapsed to uniform or zero)
3. Model uses channels [32, 64, 128] and kernel size 5 (both configurable), with a learned delta head scaling parameter
4. MC Dropout at 0.15 is active during training for regularization, and the model trains and evaluates without errors on CUDA, MPS, and CPU
**Plans:** 1/3 plans executed

Plans:
- [ ] 10-01-PLAN.md — TDD: Create SA-ConvLSTM module and attention modules (TemporalAttention, AttentionGate) with tests
- [ ] 10-02-PLAN.md — Wire SA-ConvLSTM, temporal attention, attention gate, and delta_scale into predictor; add architecture tests
- [ ] 10-03-PLAN.md — Update config.yaml defaults, optimizer param groups, and attention entropy logging

### Phase 11: Integration Testing & Validation

**Goal**: All v3.0 features work together in a full training run and produce measurably better predictions than the v2.0 baseline
**Depends on**: Phase 10 (all feature phases must be complete)
**Requirements**: None (validation phase -- validates all prior requirements end-to-end)
**Success Criteria** (what must be TRUE):

1. A full training run (50+ epochs) completes without errors, crashes, or NaN losses with all v3.0 features enabled simultaneously
2. Temporal variation ratio is significantly higher than v2.0 baseline (0.056), indicating the model produces genuine frame-to-frame dynamics
3. CSI for flare detection is significantly higher than v2.0 baseline (0.05), and skill-over-persistence exceeds the v2.0 margin of 3-9%
4. A diagnostic comparison report documents v3.0 vs v2.0 on all evaluation metrics (MAE, RMSE, CSI, HSS, SSIM, persistence skill, temporal variation ratio, peak flux error)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 7 -> 8 -> 9 -> 10 -> 11


| Phase                     | Plans Complete | Status      | Completed |
| ------------------------- | -------------- | ----------- | --------- |
| 7. Evaluation Metrics     | 1/2            | In Progress | -         |
| 8. Loss Function Overhaul | 2/3            | In Progress | -         |
| 9. Training Policy        | 1/2            | In Progress | -         |
| 10. Architecture Scaling  | 1/3 | In Progress|  |
| 11. Integration Testing   | 0/TBD          | Not started | -         |
