# Architecture Integration: v3.0 Temporal Dynamics & Flare Detection

**Project:** SolarFlare v3.0
**Researched:** 2026-03-07
**Confidence:** HIGH (based on direct codebase inspection)

## Current Architecture

```
models/predictor.py      SolarFluxPredictor
  Encoder: 2x ConvLSTM [16, 32] + 2x downsample
  Latent:  1x ConvLSTM [64]
  Decoder: mirror encoder with skip connections (line 263)
  Output:  Conv2d -> delta, pred = input + delta (line 269)

training/losses.py       CompositeLoss (L1 + SSIM + WeightedMAE)
training/trainer.py      Training loop with validation
solarflare_data/
  dataset.py             SolarFluxDataset (mmap, sliding windows)
  loader.py              Data loading, splits, normalization
utils/metrics.py         compute_metrics (not wired into training)
```

## Integration Map

### 1. Loss Function Changes (training/losses.py)

**Modified: CompositeLoss.forward()**
- Add temporal difference loss term: `L1(pred[t+1]-pred[t], target[t+1]-target[t])`
- Add temporal weighting: multiply per-timestep losses by `[1.0, 1.5, 2.0, 2.5]`
- Add temporal variation penalty: `-lambda * mean(|pred[t+1]-pred[t]|)`
- Accept new config weights: `temporal_diff_weight`, `timestep_weights`, `temporal_var_weight`

**Modified: WeightedMAELoss.forward()**
- Replace relative normalization (`max_target` per sample) with absolute threshold
- Use pre-computed extreme threshold from norm_params (~30,019 raw)
- Add asymmetric penalty: `alpha * max(0, target-pred)` for underestimation

**New: get_loss_function() updates**
- Pass new config fields through to CompositeLoss constructor

### 2. Model Architecture Changes (models/predictor.py)

**New module: SpatialAttentionGate**
```python
class SpatialAttentionGate(nn.Module):
    # Conv2d(channels, 1, kernel_size=1) + Sigmoid
    # Applied before skip connection at line 263
    # attention = sigmoid(conv(encoder_features))
    # skip = skip * attention
```

**New module: TemporalAttention**
```python
class TemporalAttention(nn.Module):
    # Linear projection of encoder hidden states
    # Softmax attention weights over T_in timesteps
    # Weighted sum of encoder outputs -> decoder init
```

**Modified: SolarFluxPredictor.__init__()**
- Accept `use_spatial_attention`, `use_temporal_attention`, `delta_head_scale` config
- Add SpatialAttentionGate instances (one per skip connection)
- Add TemporalAttention module
- Add learnable delta scale parameter (nn.Parameter)

**Modified: SolarFluxPredictor.forward()**
- Apply spatial attention before skip connections
- Apply temporal attention to encoder outputs before decoder
- Multiply delta by learned scale parameter

**Config-only changes:**
- `channels: [32, 64, 128]` -- just constructor arg
- `kernel_size: 5` -- just constructor arg
- `dropout_rate: 0.15` -- existing support

### 3. Evaluation Changes (utils/metrics.py + training/trainer.py)

**New in utils/metrics.py:**
- `compute_csi(pred, target, threshold)` -- Critical Success Index
- `compute_hss(pred, target, threshold)` -- Heidke Skill Score
- `compute_persistence_baseline(input_seq, target)` -- persistence MAE
- `compute_peak_flux_error(pred, target)` -- peak flux comparison
- `compute_temporal_variation(pred, target)` -- variation ratio

**Modified: training/trainer.py validate_epoch()**
- Call all metrics during validation
- Log per-timestep MAE, RMSE, correlation, CSI, HSS, persistence skill, SSIM
- Return expanded metrics dict

### 4. Data Pipeline Changes (solarflare_data/)

**Modified: loader.py**
- Add `_tag_extreme_sequences()` -- scan each window for extreme pixels
- Create `WeightedRandomSampler` with higher weights for flare-containing sequences
- Pass sampler to DataLoader (replaces shuffle=True for train)

**Modified: dataset.py**
- No changes to __getitem__ -- sampling changes are at DataLoader level

### 5. Training Config Changes (config.yaml)

All new loss, model, and training parameters added to config schema.
Config validation in main.py extended for new fields.

## Data Flow Changes

```
BEFORE (v2.0):
  input(B,C,T_in,H,W) -> encoder -> latent -> decoder(+skip) -> delta -> pred

AFTER (v3.0):
  input(B,C,T_in,H,W) -> encoder -> [temporal_attention] -> latent
    -> decoder(+[spatial_attention]*skip) -> delta*[learned_scale] -> pred

  loss = L1 + SSIM + WeightedMAE(asymmetric) + temporal_diff + temporal_var
         (all per-timestep weighted by [1.0, 1.5, 2.0, 2.5])
```

## Build Order (Dependency-Aware)

```
Phase 7: Evaluation metrics & persistence baseline
  (independent, enables measuring all subsequent changes)

Phase 8: Loss function overhaul
  (temporal diff, temporal weighting, temporal var penalty, fix WeightedMAE, asymmetric)
  Depends on: metrics to measure impact

Phase 9: Training policy changes
  (cosine LR, balanced aug, eliminate TF, class-imbalanced sampling)
  Depends on: improved loss functions

Phase 10: Architecture scaling
  (spatial attention, temporal attention, wider channels, kernel 5, delta head, MC dropout)
  Depends on: stable loss + training policy

Phase 11: Integration testing & validation
  (full training run, diagnostic comparison, checkpoint compat)
  Depends on: all previous phases
```

*Note: Phase numbering continues from v2.0's 6 phases.*

## Key Integration Risks

1. **Temporal difference loss + temporal variation penalty** may conflict -- need careful weight tuning
2. **Spatial attention + gradient checkpointing** -- attention gates add to computation graph; verify checkpointing still works
3. **WeightedRandomSampler replaces shuffle=True** -- ensure reproducibility with seed
4. **Wider channels [32,64,128] + batch_size=1** -- verify MPS memory fits (should be fine with downsampled input)
5. **Multiple new loss terms** -- total loss magnitude changes; may need re-tuning LR

---
*Research completed: 2026-03-07*
