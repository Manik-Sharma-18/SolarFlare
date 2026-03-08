# Requirements: SolarFlare v3.0

**Defined:** 2026-03-07
**Core Value:** Transform the model from near-persistence predictions into a genuine temporal forecaster with strong flare detection

## v3.0 Requirements

### Evaluation & Metrics

- [x] **EVAL-01**: Training loop logs per-timestep MAE, RMSE, and correlation during validation
- [x] **EVAL-02**: CSI (Critical Success Index) computed and logged per epoch using extreme threshold
- [x] **EVAL-03**: HSS (Heidke Skill Score) computed and logged per epoch
- [x] **EVAL-04**: Persistence baseline MAE computed and skill-over-persistence reported per epoch
- [x] **EVAL-05**: SSIM logged as standalone validation metric (separate from composite loss)
- [x] **EVAL-06**: Peak flux error (predicted max vs actual max) logged per epoch
- [x] **EVAL-07**: Temporal variation ratio (pred variation / target variation) logged per epoch

### Loss Function

- [x] **LOSS-01**: Temporal difference loss term added: L1(pred[t+1]-pred[t], target[t+1]-target[t])
- [x] **LOSS-02**: Per-timestep temporal weighting applied to loss (configurable weights, default [1.0, 1.5, 2.0, 2.5])
- [x] **LOSS-03**: Temporal variation penalty added: -lambda * mean(|pred[t+1]-pred[t]|) with configurable lambda
- [x] **LOSS-04**: WeightedMAE fixed to use absolute extreme threshold instead of per-sample relative normalization
- [x] **LOSS-05**: Asymmetric loss penalty for underestimation of extreme regions (configurable alpha, applied above threshold only)
- [x] **LOSS-06**: Extreme weight increased to 3.0+ in default config
- [x] **LOSS-07**: Each loss component logged separately during training (not just total)

### Architecture

- [ ] **ARCH-01**: SA-ConvLSTM cells replace standard ConvLSTM cells (channel-attention variant with Self-Attention Memory)
- [ ] **ARCH-02**: Learned delta head scaling parameter (nn.Parameter, initialized to match typical delta magnitude)
- [ ] **ARCH-03**: Spatial attention gates on skip connections (Attention U-Net pattern: Conv2d + Sigmoid)
- [ ] **ARCH-04**: Model channels widened to [32, 64, 128] (configurable)
- [ ] **ARCH-05**: Kernel size increased to 5 (configurable)
- [ ] **ARCH-06**: MC Dropout enabled at 0.15 for regularization
- [ ] **ARCH-07**: Encoder stores all hidden states (not just final) for attention access

### Training Policy

- [ ] **TRAIN-01**: Cosine LR scheduler enabled (cosine annealing with eta_min=1e-6)
- [ ] **TRAIN-02**: Balanced augmentation enabled (horizontal + vertical flips, 3x effective dataset)
- [ ] **TRAIN-03**: Teacher forcing eliminated (tf_start=0.0)
- [x] **TRAIN-04**: Class-imbalanced sampling via WeightedRandomSampler (flare-containing sequences oversampled 3x)
- [ ] **TRAIN-05**: Training epochs increased to leverage more compute (50+ epochs with cosine schedule)

## Future Requirements

### Data Enhancement (v3.1+)
- **DATA-01**: Multi-quantity magnetogram input as additional channel
- **DATA-02**: Progressive temporal curriculum (t_out: 1 -> 2 -> 4)
- **DATA-03**: Temporal difference input channels (frame-to-frame diffs as explicit input)
- **DATA-04**: Acquire additional winding flux data cubes

### Architecture (v4.0+)
- **ARCH-10**: Transfer learning from HMI magnetogram sequences
- **ARCH-11**: ConvLSTM + Transformer hybrid temporal encoder
- **ARCH-12**: Temporal convolutions for longer input sequences

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full transformer (ViT/TimeSformer) | 568 samples is 20-100x too few; ConvLSTM inductive biases essential |
| Multi-GPU / distributed training | Single device focus |
| Hyperparameter tuning (Optuna/Ray) | Manual tuning sufficient for v3.0 |
| torch.compile | Poor MPS support |
| Progressive temporal curriculum | Try simpler temporal fixes first; defer complexity |
| Magnetogram input | Data not yet available |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-01 | Phase 7 | Complete |
| EVAL-02 | Phase 7 | Complete |
| EVAL-03 | Phase 7 | Complete |
| EVAL-04 | Phase 7 | Complete |
| EVAL-05 | Phase 7 | Complete |
| EVAL-06 | Phase 7 | Complete |
| EVAL-07 | Phase 7 | Complete |
| LOSS-01 | Phase 8 | Complete |
| LOSS-02 | Phase 8 | Complete |
| LOSS-03 | Phase 8 | Complete |
| LOSS-04 | Phase 8 | Complete |
| LOSS-05 | Phase 8 | Complete |
| LOSS-06 | Phase 8 | Complete |
| LOSS-07 | Phase 8 | Complete |
| TRAIN-01 | Phase 9 | Pending |
| TRAIN-02 | Phase 9 | Pending |
| TRAIN-03 | Phase 9 | Pending |
| TRAIN-04 | Phase 9 | Complete |
| TRAIN-05 | Phase 9 | Pending |
| ARCH-01 | Phase 10 | Pending |
| ARCH-02 | Phase 10 | Pending |
| ARCH-03 | Phase 10 | Pending |
| ARCH-04 | Phase 10 | Pending |
| ARCH-05 | Phase 10 | Pending |
| ARCH-06 | Phase 10 | Pending |
| ARCH-07 | Phase 10 | Pending |

**Coverage:**
- v3.0 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-07 after initial definition*
