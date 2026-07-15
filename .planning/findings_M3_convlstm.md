# M3 — ConvLSTM winding-map encoder + JEPA baseline

**Date:** 2026-06-26 · leave-AR-out · `scripts/flare_conv/`, `models/winding_encoder.py`

## Method
- `WindingEncoder`: reuse `ConvLSTMCell`, drop the seq2seq decoder/head. Digest a
  6-frame (72-min) sequence of coarse winding maps → top hidden state → mean+max
  pool → MLP → ≥M/24h logit. 118k params. Cancellation-aware input C=3
  [+w, −w, |w|] asinh, sign-separated at full res before pooling.
- Coarse cache 64², model adaptive-pools to 32² (MPS needs divisible sizes).
- 5-fold leave-AR-out (flaring ARs round-robin'd), OOF window probs, BCE
  pos_weight, 6 epochs/fold, MPS.
- JEPA baseline: archived V5 per-frame 192-d pooled embeddings → XGBoost, same protocol.

## Result ladder (≥M/24h, leave-AR-out)

| Approach | TSS | AUC | unit |
|---|---|---|---|
| JEPA global embed | 0.219 | 0.600 | frame |
| SCALAR mean | 0.250 | 0.623 | frame |
| SPATIAL hand-stats | **0.307** | 0.671 | frame |
| ConvLSTM map-encoder | 0.288 | **0.694** | window |

## Reading (honest)
- **ConvLSTM has the best AUC (0.694)** — best discrimination/ranking of any
  method, beating hand spatial-stats (0.671) and far above the JEPA global
  embedding (0.600). The learned spatiotemporal map encoder extracts more
  rankable signal than hand features. Supports the thesis on the AUC axis.
- **But ConvLSTM TSS (0.288) < hand-stats (0.307).** Not a win on TSS. Likely
  causes, in order: (a) **under-trained** — 6 epochs, 118k params, no HPO, no
  per-fold threshold, 18-min budget vs a mature XGBoost; (b) **eval-unit
  mismatch** — ConvLSTM scored on 7084 windows (stride 3, T_in 6), XGBoost on
  21355 frames; TSS/AUC not on identical units, so the table is indicative not
  apples-to-apples; (c) high-recall operating point (tpr 0.41 fpr 0.12) — the
  max-TSS threshold may sit poorly with so few flaring ARs.
- **JEPA last, as predicted:** a learned *global* (pooled) embedding loses to
  even hand spatial-stats — independent confirmation that spatial structure,
  not representation learning per se, is what helps.

## ALIGNED RE-EVAL (2026-06-26, `aligned_eval.py`) — supersedes the table above

All methods on IDENTICAL folds + IDENTICAL per-frame rows (n=21180, t>=T_IN-1,
coverage 1.0). ConvLSTM tuned: hidden 32→48, T_IN 6→8, 6→12 epochs, cosine LR.

| Method | TSS | AUC |
|---|---|---|
| scalar mean | 0.267 | 0.637 |
| **spatial hand-stats** | **0.361** | **0.690** |
| ConvLSTM (tuned) | 0.290 | 0.664 |

**Verdict (clean, honest):**
1. **Thesis CONFIRMED and stronger:** spatial > scalar decisively — +0.094 TSS
   (0.361 vs 0.267), +0.053 AUC, leave-AR-out, aligned eval. The 2D winding
   structure beats the spatial-mean the literature integrates to.
2. **But hand-stats WIN, not the deep net.** ConvLSTM (0.290/0.664) beats scalar
   yet loses to hand spatial-stats on both metrics.
3. **ConvLSTM overfits.** Tuning UP (more capacity+epochs) made it worse vs the
   earlier light model — train loss collapsed to 0.02–0.17 while OOF didn't
   improve. 11 flaring ARs cannot support a deep spatiotemporal net; capacity is
   wasted on memorising. Lesson: regularise harder / smaller, not bigger — or
   get more data.

**Reframed story for the paper:** spatial winding structure carries flare signal
the scalar mean discards (decisive). Cheap interpretable spatial stats already
capture most of it; the deep encoder needs scale (Track B 232 ARs, or δL_c) to
pay off — here it overfits. The contribution stands on the spatial-vs-scalar gap,
not on deep learning.

### CNN + hybrid (2026-06-26, full aligned run, n=21180)

| Method | TSS | AUC |
|---|---|---|
| scalar | 0.267 | 0.637 |
| **spatial hand-stats** | **0.361** | 0.690 |
| ConvLSTM | 0.290 | 0.664 |
| CNN (time-as-channels) | 0.207 | 0.588 |
| **hybrid (spatial + conv + cnn OOF)** | **0.361** | **0.703** |

- **Recurrence matters:** ConvLSTM (0.290) >> plain CNN (0.207, worst of all,
  below scalar). The static time-as-channels CNN throws away temporal ordering
  and loses badly → winding's temporal dynamics carry signal a static conv can't
  reach. (Caveat: time-as-channels is a weak temporal model; a per-frame-shared
  CNN + temporal pool might close some gap — untested.)
- **Hybrid = best AUC (0.703) but no TSS gain** (0.361, identical to spatial).
  Late-fusing the deep OOF probs onto hand-stats adds a little *ranking* signal
  (+0.013 AUC) but does not move the operating point. Deep features are only
  weakly complementary at this data scale.
- **Conclusion unchanged & reinforced:** hand-stats win; deep nets don't beat
  them; the lever is DATA (more flaring ARs / δL_c), not architecture. Deep pays
  off only at scale. `oof.npz` saved for further fusion experiments.

---
### (earlier per-window verdict, kept for history)

## Next (to make it decisive)
1. **Align eval unit** — score all methods per-frame (or all per-window) so
   TSS/AUC are comparable.
2. **Tune the ConvLSTM** — more epochs + early-stop on val, larger hidden,
   per-fold threshold, light HPO. AUC 0.694 with zero tuning suggests headroom.
3. **Biggest lever stays data**: δL_c decomposition (re-export windCur/windPot)
   and/or more flaring ARs (Track B). 11 flaring ARs caps statistical power.
