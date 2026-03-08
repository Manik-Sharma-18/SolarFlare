# Solar Flare Prediction Model — Improvement Notes

**Date:** 2026-02-04
**Context:** ConvLSTM encoder-decoder, 568 training samples from 7 files, 4-6hr cadence, M4 Pro MPS
**Core goal:** Predict solar flare occurrence (extreme winding flux regions)

---

## 1. Loss Function Improvements

### 1.1 Fix WeightedMAE relative normalization
**Current:** `WeightedMAE` normalizes by `max_target` per sample — weighting is inconsistent across samples. A frame with a strong flare weights its extreme pixels differently than a frame with a weak one.
**Proposed:** Use the pre-computed absolute extreme threshold (~30,019 raw / 99.5th percentile) as a fixed cutoff. Apply a fixed multiplier (5-10x) to pixels above threshold.
**Why:** Makes the penalty for missing flare regions consistent regardless of frame content. The model gets the same strong signal whether the flare is the largest thing in the frame or not.
**File:** `training/losses.py` — `WeightedMAELoss.forward()`

### 1.2 Increase extreme_weight in composite loss
**Current:** L1 (1.0) + SSIM (0.5) + extreme (1.0) — extreme term is only 40% of total loss.
**Proposed:** Increase `extreme_weight` from 1.0 to 3.0-5.0 in `config.yaml`.
**Why:** The model optimizes to predict quiet-sun regions well because that's most of the image area. Flare regions are spatially tiny. The extreme loss term must dominate for the model to allocate capacity toward flare prediction.
**File:** `config.yaml` — `loss.extreme_weight`

### 1.3 Asymmetric loss penalty
**Current:** Errors are symmetric — overestimating flux is penalized the same as underestimating.
**Proposed:** Penalize underestimation of high-flux regions more than overestimation. For pixels above extreme threshold: `loss = alpha * max(0, target - pred) + max(0, pred - target)` where alpha > 1 (e.g., 2-3x).
**Why:** A missed flare is operationally worse than a false alarm. The model should err on the side of predicting flares when uncertain.

---

## 2. Evaluation Metrics

### 2.1 Wire existing metrics into training loop
**Current:** `utils/metrics.py` has `compute_rmse`, `compute_correlation`, and per-timestep MAE — but none are logged during validation.
**Proposed:** Call these in the validation loop of `training/trainer.py` and log them per epoch.
**Why:** Without these, you can't distinguish "model is learning temporal dynamics" from "model is doing fancy spatial interpolation." Per-timestep MAE specifically reveals error compounding (t+4 should be worse than t+1 — if they're similar, the model is just copying).

### 2.2 Add Critical Success Index (CSI) / Threat Score
**Proposed:** Binarize predictions and targets above extreme threshold. Compute TP/(TP+FP+FN).
**Why:** Standard metric in weather/space weather forecasting. Directly answers "did the model predict high flux where high flux actually occurred?" Loss values alone cannot answer this.

### 2.3 Add Heidke Skill Score (HSS)
**Proposed:** HSS = 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
**Why:** Measures improvement over random chance. Critical because flares are rare events — a model that always predicts "no flare" gets high accuracy but HSS=0. This metric exposes that failure mode.

### 2.4 Peak flux error
**Proposed:** Per frame, compare predicted max flux vs actual max flux.
**Why:** Simple, interpretable, directly relevant to flare forecasting. If the model consistently underestimates peak flux, it's smoothing out the flares.

### 2.5 Persistence baseline comparison
**Proposed:** Compute all metrics for a "persistence" model (predict last input frame for all future steps). Report model metrics relative to persistence.
**Why:** At 4-6hr cadence, winding flux evolves meaningfully between frames, so persistence is beatable — but it's still the null hypothesis. If the model can't beat persistence, it hasn't learned dynamics. Every result should be reported as "X% improvement over persistence."

### 2.6 SSIM as standalone validation metric
**Current:** SSIM is buried inside the composite loss, not reported independently.
**Proposed:** Log SSIM value during validation as a standalone number.
**Why:** SSIM measures structural similarity — tells you if the model preserves spatial patterns (active region shapes, polarity boundaries) even if absolute magnitudes are off.

---

## 3. Architecture Improvements

### 3.1 Spatial attention gate
**Current:** Skip connection at `predictor.py:263` passes encoder features to decoder uniformly.
**Proposed:** Add a learned attention mask before the skip connection:
```python
attention = sigmoid(conv(encoder_features))  # (B, 1, H, W)
skip = skip * attention
```
**Why:** Lets the model learn to focus on active regions rather than processing quiet-sun and flare regions equally. Well-established technique from Attention U-Net (medical imaging), designed for exactly this problem — small regions of interest in large images. Adds ~few hundred parameters.
**File:** `models/predictor.py` — before line 263

### 3.2 Wider kernel size
**Current:** `kernel_size: 3` gives 3x3 receptive field per ConvLSTM timestep.
**Proposed:** Increase to `kernel_size: 5`.
**Why:** Flare precursors (flux emergence, shearing) can span larger spatial scales than 3x3. A wider kernel lets each ConvLSTM step capture broader spatial context. Moderate speed cost (~20-30% slower per step).
**File:** `config.yaml` — `model.kernel_size`

### 3.3 Increase channel capacity
**Current:** `channels: [16, 32, 64]`
**Proposed:** `channels: [32, 64, 128]`
**Why:** More capacity to represent complex flux dynamics. The current model may lack representational power for the nonlinear dynamics of flare buildup. Risk: overfitting with 568 samples. Mitigate with dropout and augmentation. Monitor train/val loss gap.
**File:** `config.yaml` — `model.channels`

### 3.4 Temporal attention over input sequence
**Current:** Encoder processes all 10 input frames sequentially; final hidden state encodes everything.
**Proposed:** Add attention weights over the encoder's temporal outputs before passing to decoder. The decoder queries encoder outputs to determine which input timesteps are most relevant.
**Why:** Not all 10 input frames (40-60 hours) are equally predictive. The last 2-3 frames before a flare contain the most signal. Temporal attention lets the model learn which frames to emphasize rather than relying on recurrent memory alone.

### 3.5 Multi-scale decoder
**Current:** Decoder operates at a single spatial scale.
**Proposed:** Add parallel decoder branches at different spatial resolutions, merge at output.
**Why:** Flare precursors have features at multiple scales — small emerging bipoles (fine scale) and large active region shearing (coarse scale). A multi-scale decoder can capture both.
**Priority:** Lower — adds complexity. Consider after simpler improvements are tested.

---

## 4. Training Policy Improvements

### 4.1 Class-imbalanced sampling (oversampling flare sequences)
**Current:** `stride: 1` extracts all sliding windows equally. Most windows are quiet-sun.
**Proposed:** Tag sequences where target frames contain pixels above extreme threshold. Use a `WeightedRandomSampler` to sample flare-containing sequences 3-5x more often.
**Why:** The model sees boring-to-boring transitions overwhelmingly more than flare buildup. Oversampling rebalances training focus without requiring new data. This is the highest-impact data-side improvement.
**File:** `solarflare_data/loader.py` or `solarflare_data/dataset.py`

### 4.2 Reduce teacher forcing
**Current:** `tf_start: 0.5` linearly decays to 0.
**Proposed:** Drop `tf_start` to 0.2 or 0.0.
**Why:** Teacher forcing masks a key weakness: the model never learns to propagate its own errors during flare buildup. If it slightly underestimates flux at t+1, that error compounds and it misses the flare at t+3. Lower teacher forcing forces the model to be robust to its own autoregressive errors.
**File:** `config.yaml` — `training.tf_start`

### 4.3 Enable learning rate scheduler
**Current:** `scheduler.type: "none"` — flat LR at 1e-4.
**Proposed:** Switch to `type: "cosine"` with `cosine_eta_min: 1e-6`.
**Why:** Cosine annealing lets the model explore broadly early in training, then fine-tune in later epochs. With a flat LR, the model may oscillate around a minimum instead of settling into it. Particularly helps in the last 5-10 epochs.
**File:** `config.yaml` — `training.scheduler.type`

### 4.4 Balanced augmentation
**Current:** `augmentation: "none"`
**Proposed:** Switch to `augmentation: "balanced"` (horizontal + vertical flips).
**Why:** 4x effective dataset (568 -> ~2272 samples). ConvLSTMs don't have built-in spatial invariance, so flips teach the model that flux evolution is orientation-independent. Don't use "aggressive" (rotations) — marginal gain over flips doesn't justify the speed cost and potential interpolation artifacts. Note: augmentation helps spatial diversity but doesn't create new temporal dynamics — the 7 independent evolution trajectories remain the real data bottleneck.
**File:** `config.yaml` — `data.augmentation`

### 4.5 More training data
**Current:** 7 training files, 568 samples.
**Proposed:** Acquire 2-3 additional winding flux data cubes.
**Why:** The single highest-impact improvement possible. No amount of augmentation or architectural changes substitutes for more independent temporal sequences. Each new file provides genuinely new flux evolution dynamics the model hasn't seen.

---

## 5. Speed / Efficiency Improvements

### 5.1 Enable AMP (CUDA only)
**Current:** `use_amp: false`
**Proposed:** Set `use_amp: true` when training on CUDA.
**Why:** ~20% speedup with negligible accuracy impact. Does not help on MPS (uses DummyGradScaler).
**File:** `config.yaml` — `training.use_amp`

### 5.2 Increase num_workers (CUDA only)
**Current:** `num_workers: 0` — data loading in main process.
**Proposed:** Set to 2-4 when on CUDA/Linux. Keep at 0 on macOS MPS (spawn context overhead).
**Why:** Overlaps data loading with GPU computation. On macOS the spawn multiprocessing context adds overhead that negates the benefit.
**File:** `config.yaml` — `data.num_workers`

---

## 6. Uncertainty & Interpretability

### 6.1 Enable MC Dropout for flare confidence
**Current:** `dropout_rate: 0.0`, uncertainty disabled.
**Proposed:** Set `dropout_rate: 0.1-0.2`, enable uncertainty estimation during inference.
**Why:** For flare prediction, knowing *confidence* is as important as the prediction itself. MC Dropout produces uncertainty maps — if the model is uncertain in a region that turns out to be a flare, that's valuable information. Also acts as regularization during training.
**File:** `config.yaml` — `model.dropout_rate`, `uncertainty.enabled`

---

## 7. Temporal Dynamics Improvements

**Diagnosis:** Per-timestep MAE spread is only 5% (t+1: 0.105 → t+4: 0.110). The model produces near-identical predictions across all 4 output steps — it has learned spatial structure but not temporal evolution. Root cause: residual prediction (`pred = input + delta`) + L1 loss creates a strong incentive to predict near-zero deltas, defaulting to persistence.

### 7.1 Temporal difference loss
**Current:** Loss only compares `pred[t] vs target[t]` frame-by-frame.
**Proposed:** Add a loss term on the *rate of change*: `L_diff = L1(pred[t+1] - pred[t], target[t+1] - target[t])` for consecutive predicted/target frame pairs. Add as a new weighted component in the composite loss.
**Why:** Forces the model to match how flux evolves between frames, not just absolute values. If ground truth shows flux increasing in a region, the model must predict that increase — it can't hide behind a static prediction. This is the single most impactful change for temporal dynamics.
**File:** `training/losses.py` — new term in `CompositeLoss`

### 7.2 Feed temporal differences as input channels
**Current:** Encoder sees 10 raw frames. Must implicitly learn what's changing by comparing frames through ConvLSTM memory.
**Proposed:** Compute frame-to-frame differences `diff[t] = frame[t] - frame[t-1]` (9 difference frames from 10 inputs) and concatenate as additional input channels or a parallel input stream.
**Why:** Gives the model direct access to "velocity" — where flux is increasing/decreasing and how fast. The ConvLSTM currently has to discover this from raw frames, which is hard. Analogous to providing optical flow in video prediction. The model still has raw frames for absolute magnitude, but now also has the derivative.
**File:** `solarflare_data/dataset.py` — compute in `__getitem__`, feed to encoder

### 7.3 Eliminate teacher forcing
**Current:** `tf_start: 0.5` — half the decoder steps early in training receive ground truth previous frame instead of the model's own prediction.
**Proposed:** Set `tf_start: 0.0`.
**Why:** Teacher forcing masks weak temporal dynamics. The model can produce bad t+1, get corrected by ground truth, and produce decent t+2 — hiding that its temporal model is broken. With tf=0, the model must always use its own predictions as decoder input, forcing it to learn robust autoregressive dynamics. Training will be noisier initially but produces a more honest model.
**File:** `config.yaml` — `training.tf_start`

### 7.4 Temporal weighting — penalize later timesteps more
**Current:** All 4 predicted timesteps contribute equally to the loss.
**Proposed:** Weight later timesteps more heavily: `weights = [1.0, 1.5, 2.0, 2.5]` or exponential `[1, 2, 4, 8]`. Apply per-timestep before summing in loss.
**Why:** t+1 is easy (close to persistence) and dominates the gradient. t+4 is where dynamics matter but contributes equally. Heavier weighting on later steps forces the model to allocate capacity toward harder, further-out predictions.
**File:** `training/losses.py` — `CompositeLoss.forward()` before temporal flatten

### 7.5 Progressive temporal training (curriculum)
**Current:** Model trains on t_out=4 from the start.
**Proposed:** Train in stages — Stage 1: t_out=1 (single-step dynamics), Stage 2: t_out=2, Stage 3: t_out=4. Load best checkpoint from previous stage.
**Why:** 4-step autoregressive prediction from scratch is hard — early in training, the t+4 gradient signal is pure noise. By starting with t_out=1, the model first learns reliable single-step dynamics, then extends. Established technique in sequence-to-sequence training.
**File:** Training script wrapper + `config.yaml` — `data.t_out`

### 7.6 Shorten input sequence (t_in)
**Current:** `t_in: 10` (40-60 hours of history).
**Proposed:** Try `t_in: 5` or `t_in: 6` (20-36 hours).
**Why:** The ConvLSTM must compress 10 frames into fixed-size hidden states. Older frames (48-60hrs ago) dilute the signal from recent frames where the dynamics most predictive of the next 16-24 hours are visible. Shorter input = more focused context. Test empirically — if performance doesn't drop with t_in=5, the extra frames were noise.
**File:** `config.yaml` — `data.t_in`

### 7.7 Temporal variation penalty
**Current:** No penalty for static predictions across timesteps.
**Proposed:** Add regularization: `L_var = -lambda * mean(|pred[t+1] - pred[t]|)` with lambda=0.1-0.3. This is a negative loss — it rewards the model for predicting change.
**Why:** Without this, the safe strategy is "predict the same thing for all 4 steps." The penalty nudges the model away from static predictions. Lambda should be small to avoid instability.
**File:** `training/losses.py` — new term in `CompositeLoss`

### 7.8 Delta head normalization
**Current:** Output head (`predictor.py:269`) produces delta directly. Typical delta magnitudes are very small (~0.01), making the network operate in a poor numerical range where near-zero is always safe.
**Proposed:** Normalize target deltas during training so typical delta magnitude is ~1.0. Or add a learnable scale parameter to the output head: `delta = scale * raw_delta` where scale is initialized to match typical delta magnitude.
**Why:** Neural networks learn better when targets are O(1). When the expected output is ~0.01, the network naturally converges to near-zero because the loss landscape is flat around zero. Rescaling makes non-trivial deltas easier to learn.
**File:** `models/predictor.py` — output head at line 269

---

## Priority Order (Recommended Implementation Sequence)

**Phase A — Quick wins (config changes only):**
1. Increase `extreme_weight` to 3.0 (config.yaml)
2. Enable `augmentation: "balanced"` (config.yaml)
3. Enable cosine LR scheduler (config.yaml)
4. Set `tf_start: 0.0` — eliminate teacher forcing (config.yaml)
5. Try `t_in: 5` — shorter, more focused input (config.yaml)

**Phase B — Metrics & evaluation (code changes, no architecture change):**
6. Wire existing metrics into validation loop
7. Add CSI, HSS, peak flux error metrics
8. Implement persistence baseline comparison
9. Log SSIM as standalone metric

**Phase C — Temporal dynamics (loss & input changes):**
10. Add temporal difference loss (match rate of change, not just absolutes)
11. Feed temporal differences as input channels (explicit velocity signal)
12. Add temporal weighting — penalize later timesteps more heavily
13. Add temporal variation penalty (reward predicting change)

**Phase D — Extreme region focus (loss & data pipeline):**
14. Fix WeightedMAE to use absolute threshold
15. Add asymmetric loss penalty (penalize missed flares more)
16. Implement class-imbalanced sampling (WeightedRandomSampler)

**Phase E — Architecture changes (significant code changes):**
17. Add spatial attention gate
18. Increase kernel_size to 5 or channels to [32, 64, 128]
19. Add temporal attention over encoder outputs
20. Delta head normalization (rescale targets to O(1))
21. Enable MC Dropout (0.1-0.2)

**Phase F — Training curriculum (multi-stage):**
22. Progressive temporal training (t_out: 1 → 2 → 4)

**Phase G — Data acquisition (external):**
23. Acquire more winding flux data cubes
