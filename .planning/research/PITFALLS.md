# Pitfalls: v3.0 Temporal Dynamics & Flare Detection

**Project:** SolarFlare v3.0
**Researched:** 2026-03-07
**Confidence:** HIGH (based on diagnostic evidence and codebase inspection)

## Critical Pitfalls

### Pitfall 1: Loss Balancing Trap (Multiple New Loss Terms)

**What goes wrong:** Adding temporal_diff_weight, temporal_var_weight, asymmetric_alpha, and increased extreme_weight on top of existing L1+SSIM+WeightedMAE creates a 6-7 term composite loss. If weights aren't carefully balanced, one term dominates and others are ignored. Worse: the model may optimize the easiest term (L1 on quiet regions) and ignore the hard terms (temporal dynamics, extreme detection).

**Prevention:**
- Log each loss component separately during training (not just total)
- Start with conservative weights, adjust based on component magnitudes
- Ensure temporal_diff and extreme terms are same order of magnitude as L1
- Monitor the ratio of each component to total loss per epoch

**Phase:** Loss function overhaul (Phase 8)

### Pitfall 2: Overfitting with 4x Model Size on 568 Samples

**What goes wrong:** Going from [16,32,64] (~150K params) to [32,64,128] (~600K+ params) with only 568 training samples. Params-to-sample ratio goes from ~260 to ~1050. Model memorizes training data, val loss diverges.

**Prevention:**
- MC Dropout (0.15) provides regularization
- Balanced augmentation gives effective 3x dataset (1704 samples)
- Class-imbalanced sampling doesn't add data but rebalances exposure
- Monitor train-val gap closely -- if >15%, reduce capacity
- **Fallback:** If overfitting, try [24,48,96] as intermediate scaling

**Detection:** Train loss continues decreasing while val loss increases for 3+ epochs.

**Phase:** Architecture scaling (Phase 10)

### Pitfall 3: Temporal Variation Penalty Instability

**What goes wrong:** The temporal variation penalty `L_var = -lambda * mean(|pred[t+1]-pred[t]|)` is a *negative* loss -- it rewards the model for predicting change. If lambda is too large, the model learns to produce random large changes (noisy predictions) to minimize this penalty, destroying spatial coherence.

**Prevention:**
- Start with lambda=0.05, max 0.3
- Monitor spatial SSIM alongside temporal variation -- if SSIM drops sharply, lambda is too high
- The temporal difference loss (positive term) should be the primary temporal signal; the variation penalty is a gentle nudge

**Phase:** Loss function overhaul (Phase 8)

### Pitfall 4: Asymmetric Loss Breaks Gradient Balance

**What goes wrong:** Asymmetric penalty `alpha * max(0, target-pred)` for underestimation means the model always has a bias toward overestimation. With alpha=3, the model may predict exaggerated flux everywhere to avoid the asymmetric penalty, increasing FP dramatically while reducing FN.

**Prevention:**
- Only apply asymmetric penalty above extreme threshold (not globally)
- Start with alpha=1.5, not 3.0
- Monitor FP rate alongside FN rate -- if FP explodes, alpha is too aggressive
- CSI metric naturally balances TP/FP/FN -- watch it

**Phase:** Loss function overhaul (Phase 8)

### Pitfall 5: WeightedRandomSampler + batch_size=1 Interaction

**What goes wrong:** With batch_size=1, each "batch" is one sample. WeightedRandomSampler oversamples flare sequences 3-5x. But with so few flare sequences in 568 samples, the model sees the same flare sequences many times per epoch, overfitting to those specific flare patterns.

**Prevention:**
- Use moderate oversampling factor (3x, not 5x)
- Combined with balanced augmentation, each flare sequence appears in 3 orientations x 3 oversample = ~9x, which provides some variety
- Track per-file performance to detect if model only learns flares in training files

**Phase:** Training policy (Phase 9)

## Moderate Pitfalls

### Pitfall 6: Spatial Attention Creates Dead Zones

**What goes wrong:** Attention gate learns to always mask out quiet-sun regions. Since most of the image is quiet-sun, the model stops learning background flux dynamics entirely. Skip connections carry zero information for 80%+ of the spatial domain.

**Prevention:**
- Initialize attention gate bias to ~2.0 (sigmoid(2.0) = 0.88), so default is "pass everything through"
- The model must learn to reduce attention, not increase it
- Monitor attention map statistics -- if >50% of pixels get attention < 0.1, investigate

**Phase:** Architecture scaling (Phase 10)

### Pitfall 7: Temporal Attention Collapses to Last Frame

**What goes wrong:** Temporal attention over 10 encoder outputs learns to put all weight on the last frame (t=10), effectively becoming persistence. This is the easiest solution for the attention mechanism since the last frame is most similar to the target.

**Prevention:**
- Use temperature scaling in softmax (temperature > 1.0) to prevent sharp peaking
- Monitor attention weight distribution -- entropy should remain moderate
- Consider additive attention (not pure multiplicative) to retain information from all frames

**Phase:** Architecture scaling (Phase 10)

### Pitfall 8: Cosine LR + New Loss Terms = Unstable Early Training

**What goes wrong:** Cosine LR starts at base LR (1e-4) and decays. But with new loss terms, the effective gradient magnitude changes. Early epochs may be unstable because the model is simultaneously learning new loss landscapes and the LR is at its highest.

**Prevention:**
- Consider warmup: start at 1e-5 for 2-3 epochs, then cosine from 1e-4
- Alternatively, train 5 epochs with old loss weights to stabilize, then introduce new terms
- Monitor gradient norms in first 5 epochs -- should be stable, not oscillating

**Phase:** Training policy (Phase 9)

### Pitfall 9: Kernel Size 5 + Downsampled Input = Receptive Field Overshoot

**What goes wrong:** With downsample_input=True (2x), the effective spatial resolution is ~220x442. Kernel size 5 on this gives significant receptive field per step. After 3 ConvLSTM layers, the receptive field may exceed meaningful spatial scales, causing the model to mix information from physically unrelated regions.

**Prevention:**
- This is unlikely to be a problem in practice (large-scale flux patterns are spatially correlated)
- If test metrics don't improve with k=5, revert to k=3
- Could try k=5 for first layer only, k=3 for deeper layers

**Phase:** Architecture scaling (Phase 10)

### Pitfall 10: Delta Head Scale Parameter Explodes

**What goes wrong:** The learnable scale parameter for delta normalization could grow unbounded, producing exaggerated predictions. No constraint prevents it from reaching 100x or 0.001x.

**Prevention:**
- Initialize to match typical delta magnitude (~0.01-0.1 in normalized space)
- Apply soft clamping: `scale = softplus(raw_scale) + eps`
- Or use `torch.clamp(scale, 0.01, 10.0)` to bound the range
- Monitor scale value per epoch

**Phase:** Architecture scaling (Phase 10)

### Pitfall 11: Checkpoint Incompatibility with Architecture Changes

**What goes wrong:** v3.0 adds spatial attention, temporal attention, and delta scale to the model. Loading a v2.0 checkpoint into a v3.0 model fails with "unexpected key" or "missing key" errors.

**Prevention:**
- Use `strict=False` in load_state_dict for warm-starting from v2.0
- New modules (attention, scale) initialize randomly -- only existing layers transfer
- Document that v3.0 training starts fresh (not from v2.0 checkpoint)

**Phase:** Architecture scaling (Phase 10)

## Information Gaps

- **Optimal temporal_diff_weight relative to L1:** Unknown until empirical testing. Start at 1.0 (equal to L1).
- **Attention mechanism memory cost:** Spatial attention is negligible. Temporal attention over 10 steps with [32,64,128] channels should be manageable with batch_size=1.
- **Training time impact of k=5 + wider channels:** Estimated ~2-3x slower per epoch. Acceptable if val metrics improve.

---
*Research completed: 2026-03-07*
