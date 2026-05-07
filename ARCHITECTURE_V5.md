# SolarFlare V5 — Spatiotemporal Context Encoder

**Status:** Draft for discussion (2026-04-25). Replaces V4 ConvLSTM forecasting model.

**Premise.** V4 collapsed to persistence because (a) supervised forecasting on 14 cubes is data-starved, (b) loss has no penalty for "predict the mean" mode collapse, (c) ConvLSTM bottleneck cannot exploit non-local spacetime context. V5 adopts Pathak et al. (2016) Context Encoder mechanics, lifted from 2D images to 3D spacetime cubes, with masked self-supervised pretraining as the primary training signal.

---

## 1. Design Goals

| Goal | V4 status | V5 mechanism |
|---|---|---|
| Beat persistence baseline on t+4 | fails | tail-mask finetune + temporal-diff loss + adversarial mode-picking |
| Use 14 cubes efficiently | sliding window only | masked self-supervised pretext multiplies effective examples ≥ 10⁴× |
| Handle quiet-sun dominance | weighted MAE (disabled) | adversarial loss avoids mean-pixel collapse; mask-only L2 focuses gradient on dropped region |
| Cross-AR generalization | center-crop loses 30% area | per-AR FOV patch extraction at fixed Mm scale |
| Transfer to magnetograms | finetune-only path coded | shared self-supervised pretext task — magnetogram pretrain plugs in directly |
| Uncertainty | MC Dropout (off) | adversarial discriminator confidence + ensemble masks at inference |

---

## 2. Two-Phase Training

```
Phase 1: Self-supervised pretrain (masked spacetime inpainting)
   └─ all 14 cubes, no labels, random masks
       └─ produces general spatiotemporal feature encoder

Phase 2: Forecasting finetune (tail-mask)
   └─ last k frames masked, predict from leading frames
       └─ produces operational flare forecaster
```

Phase 1 is the Context Encoder mechanic, generalized to 3D. Phase 2 specializes it.

Optional **Phase 0** (when magnetograms acquired): same Phase 1 recipe on `data_magnetogram/` first, then continue Phase 1 on winding flux.

---

## 3. Architecture

### 3.1 Input tensor

```
x ∈ ℝ^(B, C, T, H, W)
B = 1 (constrained by spatial dims)
C = 2 (asinh-normalized flux + sigmoid extreme indicator) — same as V4
T = 16 (input window — slightly longer than V4's 10 to give context room for temporal masking)
H = 437, W = 877 (per-AR cropped, fp32)
```

Mask `M ∈ {0,1}^(B, 1, T, H, W)` — 1 where dropped, 0 where kept (matches Pathak convention).

Masked input: `x_in = (1 − M) ⊙ x + mean_fill ⊙ M`. Mean fill = 0 in zero-centered asinh space.

### 3.2 Encoder (3D)

Mirror DCGAN-style stack but in 3D:

| Layer | Kernel | Stride (T,H,W) | Out channels | Out shape (T,H,W) |
|---|---|---|---|---|
| conv3d_1 | 4×4×4 | (1, 2, 2) | 32 | 16, 218, 438 |
| conv3d_2 | 4×4×4 | (1, 2, 2) | 64 | 16, 109, 219 |
| conv3d_3 | 4×4×4 | (2, 2, 2) | 128 | 8, 54, 109 |
| conv3d_4 | 4×4×4 | (2, 2, 2) | 256 | 4, 27, 54 |
| conv3d_5 | 4×4×4 | (2, 2, 2) | 512 | 2, 13, 27 |

LeakyReLU(0.2) + BatchNorm3d. Spatial downsamples 2× per layer; temporal downsamples only later (preserve t-resolution early so temporal masks have room).

Final encoder feature volume: `(B, 512, 2, 13, 27) ≈ 360k` activations.

### 3.3 Channel-wise Fully-Connected — Spacetime variant

Pathak's CFC operates per feature map across `n×n`. We extend across **`t × h × w`** per channel.

For each of 512 channels:
```
flatten (2, 13, 27) → 702-dim vector
linear(702 → 702)   per-channel weights, no cross-channel mixing
reshape back to (2, 13, 27)
```

Param cost: `512 × 702² ≈ 252M` (large but tractable). Full FC equivalent would be `512² × 702² ≈ 130G` — not feasible.

Then `1×1×1` conv `512 → 512` mixes channels (cheap, ~262k params).

**Why CFC.** Conv stack has limited receptive field — flux propagation across full 318 Mm domain in 192 min cannot be inferred from local stencils alone. CFC propagates information edge-to-edge per channel without parameter explosion.

**Alternative under consideration.** Replace CFC with a single transformer block over flattened spacetime tokens (~702 tokens × 512 dim). Comparable param count, better long-range mixing. Decision pending discussion.

### 3.4 Decoder (3D)

Mirror of encoder, transposed conv 3D:

| Layer | Kernel | Stride | Out channels | Out shape |
|---|---|---|---|---|
| upconv_1 | 4×4×4 | (2, 2, 2) | 256 | 4, 27, 54 |
| upconv_2 | 4×4×4 | (2, 2, 2) | 128 | 8, 54, 109 |
| upconv_3 | 4×4×4 | (2, 2, 2) | 64 | 16, 109, 219 |
| upconv_4 | 4×4×4 | (1, 2, 2) | 32 | 16, 218, 438 |
| upconv_5 | 4×4×4 | (1, 2, 2) | 16 | 16, 437, 877 |
| conv_out | 1×1×1 | 1 | 1 | 16, 437, 877 |

ReLU + BatchNorm3d on intermediate; tanh on final (output range [-1, 1] matches asinh-normalized space).

**No skip connections from encoder.** Skip connections would let network bypass the bottleneck and trivially copy unmasked input — defeats Context Encoder pretext. (V4 had skip; remove.)

### 3.5 Discriminator (Phase 1 only)

3D conv stack mirroring encoder but smaller. Outputs single scalar real/fake per cube.

| Layer | Kernel | Stride | Out channels |
|---|---|---|---|
| conv3d | 4×4×4 | (1,2,2) | 32 |
| conv3d | 4×4×4 | (2,2,2) | 64 |
| conv3d | 4×4×4 | (2,2,2) | 128 |
| conv3d | 4×4×4 | (2,2,2) | 256 |
| conv3d | (T_final, H_final, W_final) | 1 | 1 |

LeakyReLU(0.2) + BatchNorm3d (no BN on first layer per DCGAN guideline). Sigmoid on output.

**Discriminator does NOT see mask** — same as Pathak. Sees full predicted cube vs full real cube. Avoids trivial seam-detection.

### 3.6 Parameter budget

| Component | Params (estimate) |
|---|---|
| Encoder | ~5 M |
| CFC bottleneck | ~252 M |
| Decoder | ~5 M |
| Discriminator | ~3 M |
| **Total trainable** | **~265 M** |

CFC dominates. If memory/compute limits hit, reduce to 256 final channels (CFC drops to ~63M) or switch to transformer-block alternative.

---

## 4. Mask Strategies

Pathak's mask catalog adapted to spacetime. Each batch samples one strategy uniformly:

| Strategy | Geometry | Volume fraction | Purpose |
|---|---|---|---|
| **Spatial blob** | random deformed 2D shape, replicated across all `t` | 15–25% per frame | spatial flux structure (CE-style) |
| **Temporal frame drop** | drop `k ∈ {1,…,4}` random frames whole | k/T | frame interpolation dynamics |
| **Spacetime tube** | random `(Δt, Δh, Δw) = (3-6, 64-128, 64-128)` cuboid | 5–15% | local AR evolution |
| **Tail mask** | drop last `k=4` frames entirely | 4/16 = 25% | exact forecasting objective |
| **Random region** | arbitrary deformed contour sampled from solar disk masks | ~25% per frame | boundary randomization |

**Phase 1:** sample from {blob, frame-drop, tube, random-region} with equal probability. Tail mask reserved for Phase 2.

**Phase 2:** tail mask only.

**Mask fill.** Mean value (0 in normalized space). Same as Pathak.

**Boundary overlap trick.** Predict 7-pixel band around mask in addition to interior. Reconstruction loss on overlap weighted **10×**. Forces seam continuity. Generalizes to: in temporal frame drops, predict ±1 frame neighbors in overlap; in tubes, predict 7px shell.

**Random region details.** Pathak used PASCAL VOC segmentation masks. We need solar analogue: deformed elliptical blobs at random positions, optionally sampled from segmented sunspot/AR shapes if catalog available. Cheap fallback: random Bezier-closed contours.

---

## 5. Loss Functions

### 5.1 Phase 1 (pretrain)

Pathak joint loss, lifted to 3D:

```
L_rec = || M ⊙ (x − F((1 − M) ⊙ x)) ||²₂      (mean over masked voxels only)

L_adv = max_D  E[ log D(x) + log(1 − D(F((1 − M) ⊙ x))) ]

L = λ_rec · L_rec + λ_adv · L_adv
   λ_rec = 0.999
   λ_adv = 0.001
```

Optional addon (decision pending): replace L2 with **L1 on masked region + L2 on overlap band** — Pathak found L1 ≈ L2, but L1 is more robust on heavy-tailed flux distribution.

### 5.2 Phase 2 (forecast finetune)

Tail-mask only. Switch reconstruction term to flux-aware composite:

```
L = w_l1 · L1(M ⊙ pred, M ⊙ target)
  + w_ssim · (1 − SSIM(pred[t_in:], target[t_in:]))
  + w_ext · ExtremeWeighted(pred, target)        # absolute threshold @ p99.5
  + w_tdiff · L1(Δpred, Δtarget)                  # temporal difference loss
  + w_adv · L_adv                                  # keep adversary
```

Initial weights: `[1.0, 0.5, 1.0, 0.5, 0.001]`. Tunable via grid.

**Why keep adversarial in Phase 2.** V4 dropped all adversarial-like terms and went to pure L1+SSIM → mode collapse to persistence. Adversarial term picks one mode from the predictive distribution rather than averaging.

**Why temporal-diff this time.** V4 had it disabled. Now combined with adversarial term, less risk of confused optimization — adversary handles mode selection, tdiff handles dynamics fidelity.

---

## 6. Training Procedure

### 6.1 Phase 1

| Setting | Value |
|---|---|
| Optimizer | Adam, β1=0.5, β2=0.999 (DCGAN defaults) |
| LR generator | 1e-3 |
| LR discriminator | 1e-4 (10× slower than G — Pathak) |
| Batch | 1 (memory-bound) |
| Grad accum | 4 (effective batch 4) |
| Epochs | 100 (~ until reconstruction PSNR plateaus) |
| Mask sampling | 10 fresh masks per cube per epoch |
| AMP | fp16 mixed-precision |
| Grad clip | 1.0 (looser than V4's 0.5 — adversarial dynamics need room) |
| Augmentation | h/v flips, 90° rotations (16 orientations) — applied before masking |
| Checkpoint | every epoch + best by val reconstruction loss |

Effective examples per epoch: `~1356 windows × 10 masks × 16 orientations ≈ 217k`. Truly distinct since mask is fresh each epoch.

### 6.2 Phase 2

| Setting | Value |
|---|---|
| Init | best Phase 1 checkpoint (encoder + CFC + decoder) |
| Optimizer | Adam |
| LR encoder + CFC | 1e-5 (frozen-ish for first 5 epochs, then unfreeze) |
| LR decoder + new head | 1e-3 |
| LR discriminator | 1e-4 |
| Batch | 1 |
| Epochs | 30 |
| Mask | tail only (last 4 frames of 16-frame window — predict t+1..t+4 from t-11..t) |
| Loss | composite (Section 5.2) |
| Augmentation | h/v flips only (no rotations — solar B-angle has physical meaning at fine scale) |
| Early stopping | patience 8 on val L1 (not adv) |

### 6.3 Data split

Whole-cube split: train = 10 cubes, val = 2, test = 2.
**Cross-campaign isolation.** May 2024 and Oct 2024 may share AR — confirm before split. If shared, group by AR identity not date.

---

## 7. Inference

### 7.1 Forecasting

Standard. Feed 12-frame history, get 4-frame prediction. No autoregression — single shot since model predicts all 4 frames jointly via 3D decoder.

### 7.2 Uncertainty

**Mask-ensemble inference.** Run model N=20 times with N different random masks on past frames (small mask, doesn't kill forecasting context). Spread of predictions = epistemic uncertainty. Replaces V4 MC Dropout.

**Discriminator score** as secondary calibration: D(prediction) = how realistic. Low D-score → high uncertainty.

---

## 8. Evaluation

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| Reconstruction L2 (masked) | primary | — |
| PSNR (masked) | primary | — |
| SSIM (masked) | primary | secondary |
| Discriminator accuracy | monitor (target ~0.5) | monitor |
| MAE per timestep | — | primary |
| RMSE per timestep | — | primary |
| Persistence-relative skill | — | primary |
| CSI @ p99.5 threshold | — | primary |
| HSS @ p99.5 threshold | — | primary |
| Pearson correlation | — | secondary |

Persistence baseline mandatory. Every result reported as `% improvement over persistence`.

---

## 9. Data Pipeline Changes vs V4

| Change | Reason |
|---|---|
| fp64 → fp32 | values fit fp32 trivially, halves storage/RAM |
| center-crop → per-AR FOV crop | V4 loses 30% of Nov 2025 spatial extent |
| flare oversampling 5× | retain (still useful in Phase 2) |
| augmentation: balanced → full 16 orientations Phase 1, balanced Phase 2 | Phase 1 needs max diversity, Phase 2 respects physical orientation |
| sliding window stride 4 → stride 1 Phase 1, stride 4 Phase 2 | Phase 1 is data-starved, Phase 2 benefits from less overlap |
| mask augmentation N=10 fresh per epoch | new — Context Encoder core mechanic |

---

## 10. Open Design Questions (for discussion)

1. **CFC vs transformer block** at bottleneck. CFC is faithful to Pathak; transformer has better long-range mixing. Param count comparable. Compute differs (transformer is `O(n²)` over 702 tokens — fine).
2. **Skip connections.** Pathak omits them for inpainting (correct). For Phase 2 forecasting, skip connections from encoder might help fine-detail. Try ablation.
3. **3D conv vs (2D conv + temporal attention).** True 3D is parameter-heavy. Factored (2D space + 1D time) is cheaper. Solar dynamics may have separable structure. Ablation candidate.
4. **Mask boundary "overlap trick"** — Pathak's 7px is for 64×64 patches in 128×128 images (≈ 11% boundary). Equivalent for our spacetime tubes? Need to scale appropriately.
5. **Discriminator scope.** Whole-cube D? Or per-frame D + temporal-coherence D? Two-discriminator GANs are common in video.
6. **Phase 2 reset of discriminator** vs continuing from Phase 1? Pretext-trained D may be too good for forecasting task.
7. **Magnetogram pretrain** — Phase 0. Does it go *before* or *interleave* with Phase 1 on flux? Can magnetogram features actually transfer to flux given different physical units?
8. **Memory budget.** 16-frame × 437×877 × 2-channel fp32 = 49 MB per sample. Encoder feature maps add 2-3× more. Discriminator another 1×. Single-batch should fit on 24GB GPU; verify.
9. **Random-region mask source.** Pathak uses PASCAL. We need synthetic blobs or sunspot catalog masks. Plain Bezier blobs adequate?
10. **Loss-weight schedule.** Pathak uses fixed `λ_rec = 0.999, λ_adv = 0.001`. Ramp adversarial up over Phase 1 (curriculum)?
11. **Teacher-forcing analogue in Phase 2.** With single-shot 4-frame prediction, no autoregression → no teacher forcing concept. But could mix Phase 2 mask with random tail-prefix masking (predict any 1-4 future frames, not always all 4).
12. **Test-set definition.** Need explicit AR-grouped test set held out *before* any tuning.

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Adversarial training instability (mode collapse, D dominance) | high | DCGAN best practices: BN, LeakyReLU, label smoothing, G:D LR ratio, spectral norm if needed |
| CFC param explosion OOM | medium | reduce final channels to 256; switch to transformer block |
| Phase 1 features don't transfer to Phase 2 (forecasting ≠ inpainting) | medium | run Phase 2 from scratch as ablation; compare |
| 14 cubes still insufficient even with mask augmentation | medium | mandatory: acquire magnetogram pretrain data |
| Adversarial loss hallucinates flares not in data | medium | strong reconstruction weight (λ_rec=0.999); CSI/HSS catches false positives |
| Per-AR FOV crop loses cross-AR generalization (each model sees one AR shape) | low | force at least 4-AR variety in train set |

---

## 12. Implementation Order

1. Data pipeline rewrite: fp32, per-AR FOV crop, mask augmentation classes
2. 3D Encoder + CFC + Decoder skeleton (no adversarial yet)
3. Phase 1 with reconstruction-only loss — verify training converges, masked-region PSNR improves
4. Add discriminator + adversarial loss → retrain Phase 1
5. Phase 2 finetune with composite loss
6. Evaluation harness: persistence baseline + CSI + HSS + per-timestep metrics
7. Mask-ensemble inference + uncertainty
8. (When magnetograms acquired) Phase 0 pretrain + chain to Phase 1

Each step gates the next. Don't skip ablations.

---

## 13. References

- Pathak, Krähenbühl, Donahue, Darrell, Efros. "Context Encoders: Feature Learning by Inpainting." CVPR 2016. arXiv:1604.07379
- Radford, Metz, Chintala. "Unsupervised Representation Learning with Deep Convolutional GANs." ICLR 2016. (DCGAN architecture guidelines)
- Goodfellow et al. "Generative Adversarial Nets." NIPS 2014.
- Tong et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS 2022. (3D mask augmentation prior)
- He et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. (MAE — modern descendant of Context Encoder)
