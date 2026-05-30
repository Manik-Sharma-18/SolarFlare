# SimpleConvLSTM Baseline — Findings (S0 / S1)

**Task:** sequence-to-sequence winding-flux forecasting — predict the next
4 frames (t+1…t+4) from the previous 10, per HARP active-region cube.

**Model under test (S0/S1):** a deliberately minimal 2-layer ConvLSTM
encoder–forecaster (flat hidden 64, kernel 3, BatchNorm, 1×1 conv head,
~0.9 M parameters). Single input channel (flux only), plain **L1** loss,
no attention, no residual connection, no extreme weighting. This is the
canonical nowcasting baseline (Shi et al. 2015) against which the heavier
6-layer attention model is being judged.

| Arm | Difference | Epochs | Purpose |
|-----|-----------|--------|---------|
| **S0** | augmentation on (D4 flips) | 15 (early-stopped @10) | baseline |
| **S1** | augmentation off | 4 | isolate value of augmentation |

Both trained on the RTX 5060 Ti, per-cube z-score normalisation, cube-level
train/val/test split (no active-region leakage).

---

## Headline result: the model collapses to a near-constant field

Both arms reconstruct the *smooth* field but have **zero flare skill**.

### Per-epoch validation (S0)

| Epoch | Val L1 | SSIM | CSI | HSS |
|------:|-------:|-----:|----:|----:|
| 1 | 0.106 | 0.095 | 0.0003 | −0.0002 |
| **4 (best)** | **0.0387** | **0.525** | 0.0000 | 0.0000 |
| 10 (stop) | 0.046 | 0.404 | 0.0000 | 0.0000 |

CSI and HSS are **0.0000 at every epoch**. The "best" val L1 (0.0387) is
low only because predicting ≈0 on a zero-mean field is itself low-L1.
SSIM peaks at epoch 4 then declines → overfitting (effective train set is
small: ~14 cubes).

### Test set (S1) — the model is beaten by persistence

| Metric | S1 model | Persistence (copy last frame) |
|--------|---------:|------------------------------:|
| CSI | **0.0000** | **0.2917** |
| HSS | −0.0000 | 0.4471 |
| SSIM | 0.518 | — |
| temporal variation ratio | 0.517 | 1.0 = matches truth |

Simply repeating the last observed frame scores **CSI 0.29 / HSS 0.45**;
the trained model scores **0**. It also reproduces only ~52 % of the real
frame-to-frame variation (under-dynamic / over-smoothed).

---

## Direct evidence of the collapse

Predicted vs ground-truth statistics for one window (normalised z-score space):

| | mean | **std** | range |
|--|----:|------:|------:|
| input (the 10 frames given) | 0.30 | 6.29 | [−39, 677] |
| ground truth (t+1…t+4) | 0.31 | **10.38** | [−21, 1214] |
| **prediction** | 0.011 | **0.019** | [−0.08, 0.08] |

The prediction's spatial standard deviation is **~500× smaller** than the
truth and ~330× smaller than the structured input it was given. Per-frame
prediction std stays ~0.01 across all four horizons — i.e. an almost flat,
constant ≈0 field. This is visible directly in the figures
(`outputs/s0_viz/*.png`): predicted crops are near-uniform while the
ground truth shows sharp signed flux structure.

---

## Why this happens

1. **L1 loss + extreme sparsity.** After per-cube normalisation the field
   is zero-mean and heavy-tailed: the vast majority of pixels are near 0,
   genuine flux extremes are rare. The L1-minimising prediction is
   therefore the trivial constant ≈0 — missing the rare spikes costs almost
   nothing in the average. The model learned exactly this.
2. **Non-residual decoding.** The model predicts each frame *from scratch*
   through the conv head, so nothing anchors it to the (non-zero) last
   frame; it is free to output ≈0. A residual/persistence formulation
   (`frame = previous + Δ`) cannot fall below the persistence baseline.
3. **No extreme weighting / single channel** — nothing in the objective
   forces attention to the rare high-flux pixels.

This is the well-documented "blurry forecast" failure of L1/L2 models on
sparse-extreme fields (precipitation-nowcasting literature). The heavier
6-layer attention model hits the same wall (CSI ≈ 0.01).

**Augmentation is not the bottleneck:** S0 (aug) and S1 (no-aug) give the
same CSI 0 and plateau by epoch 2.

---

## What the literature does about sparse extremes

- **Balanced / weighted loss (B-MSE, B-MAE):** weight each pixel by its
  intensity so rare strong-flux pixels dominate the gradient (HKO-7 standard).
- **Plotting-position / inverse-frequency weighting.**
- **Residual / persistence anchoring:** predict the change, baseline = last frame.
- **Generative models (GAN / diffusion, e.g. DGMR):** adversarial loss
  avoids the mean-regression blur of L1/L2.
- **Two-stage detect-then-regress; intensity-aware curricula.**

---

## Next steps (in progress)

- **S2 — residual decode** (`frame = previous + Δ`): cannot score below
  persistence (CSI ≈ 0.29). *Currently training.*
- **S3 — weighted / extreme-pixel loss** on top of S2.

**Takeaway for discussion:** at this scale the limiting factor is the
**objective and output formulation**, not network depth. A plain-L1,
non-residual model — simple or deep — degenerates to the conditional mean
and is outperformed by trivial persistence on the flare metric.
