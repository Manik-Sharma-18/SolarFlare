# S2 — Residual SimpleConvLSTM: Error Explosion

**Change vs S0:** residual decoding — each frame is `previous + Δ` instead
of an absolute prediction. Intent: anchor to persistence (Δ→0 = copy last
frame) so the model cannot collapse to ≈0 as S0/S1 did. Same otherwise
(single-channel flux, L1, no augmentation, batch 8, 15 epochs, patience 3).
Decoding is **cumulative + fully autoregressive** (no teacher forcing):
`frame_t = frame_{t-1} + Δ_t`, with the model's own output fed back.

## Result: it diverges instead of collapsing

| Epoch | Val L1 | CSI | SSIM | Persistence skill | Temporal var ratio |
|------:|-------:|----:|-----:|------------------:|-------------------:|
| 1 | 0.448 | 0.0038 | 0.013 | −765 % | 18.3 |
| 2 | **22.99** | 0.0029 | −0.0002 | −44 125 % | **764** |

- Val loss **blows up** (0.45 → 23) within one epoch.
- **Temporal-variation ratio 18× → 764×**: predictions oscillate far more
  wildly than the ground truth (ratio 1.0 = match). The output is
  exploding, not smoothing.
- Persistence skill deeply negative — far **worse** than copying the last
  frame.

## Diagnosis

The residual connection removed S0's collapse-to-zero but introduced the
opposite failure: **autoregressive error accumulation**. With cumulative
residuals and no teacher forcing, each step adds a Δ computed from the
model's own (already wrong) previous output. Over the 4-frame rollout the
errors compound multiplicatively and the prediction diverges. The model
never sees ground-truth context during training, so it is never taught to
correct its own drift — classic exposure bias, amplified by the residual sum.

## Contrast

| Model | Failure mode | Temporal var ratio | CSI |
|-------|--------------|-------------------:|----:|
| S0 (absolute, L1) | collapse to ≈0 (too smooth) | 0.52 | 0 |
| S2 (residual, no TF) | explosion (too wild) | 764 | ≈0 |

Two opposite degeneracies — both give ≈0 flare skill. The residual model
needs its rollout *stabilised*, not just anchored.

## Next step (running)

**S4 — residual + teacher forcing.** Feed ground-truth frames with a
decaying probability during training (`tf_ratio = tf_start·(1−epoch/E)`),
so the model learns to correct from realistic context and the rollout is
anchored to truth rather than its own compounding error. Test: does the
explosion disappear and does CSI clear the persistence floor (0.29)?
