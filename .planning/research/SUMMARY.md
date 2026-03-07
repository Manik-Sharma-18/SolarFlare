# Project Research Summary

**Project:** SolarFlare v3.0 -- Temporal Dynamics & Flare Detection
**Domain:** Spatiotemporal solar flux forecasting (ConvLSTM encoder-decoder)
**Researched:** 2026-03-07
**Confidence:** HIGH

## Executive Summary

SolarFlare v3.0 addresses two critical deficiencies in the current model: near-zero temporal dynamics (variation ratio of 0.056 -- the model predicts only 6% of real frame-to-frame change) and failed flare detection (CSI of 0.05). The model barely beats a naive persistence baseline (+3-9% skill), indicating it has learned to approximate a time-averaged output rather than genuine forecasting. The root causes are a loss function that does not penalize temporal stationarity, teacher forcing that masks autoregressive weakness, and extreme class imbalance (flare pixels are vastly outnumbered by quiet-sun pixels).

The recommended approach is a four-phase incremental improvement: first instrument evaluation metrics so every subsequent change is measurable, then overhaul the loss function with temporal and asymmetric terms, then adjust training policy (LR schedule, sampling, augmentation), and finally scale the architecture with attention mechanisms and wider channels. This order is critical -- architecture changes without loss and training fixes will simply produce a larger model with the same broken dynamics. All features are implementable with existing PyTorch built-ins; no new dependencies are required.

The primary risks center on loss balancing (6-7 composite terms competing for gradient budget), overfitting (4x parameter increase on only 568 samples), and the temporal variation penalty potentially destabilizing training if weighted too aggressively. Each risk has concrete mitigation strategies: per-component loss logging, dropout + augmentation for regularization, and conservative initial weights with monitoring. The phased approach itself is a risk mitigation strategy -- each phase delivers measurable improvements before the next phase adds complexity.

## Key Findings

### Recommended Stack

No new runtime dependencies are required. All v3.0 features -- temporal difference loss, attention mechanisms, cosine LR scheduling, MC Dropout, class-imbalanced sampling -- use PyTorch built-in operations and modules. This is a significant advantage: zero dependency risk, zero compatibility risk, and all features are MPS-compatible.

**Core technologies (unchanged from v2.0):**
- **PyTorch (existing):** All new features use nn.Conv2d, nn.Linear, F.softmax, tensor ops -- fully MPS-compatible
- **NumPy (existing):** Data pipeline and preprocessing -- no changes needed
- **YAML config (existing):** Extended with new loss, model, and training parameters

**Explicitly deferred:**
- Transformer architectures (overkill for 10-frame sequences)
- torchmetrics (CSI/HSS are ~20 lines to implement)
- Optuna/Ray Tune (defer hyperparameter search to v4.0)
- torch.compile (poor MPS support, risk of subtle bugs)

### Expected Features

**Must have (table stakes -- fix broken dynamics):**
- Temporal difference loss -- directly attacks the 0.056 variation ratio
- Eliminate teacher forcing (tf=0) -- forces honest autoregressive predictions
- Temporal weighting on later timesteps -- allocates gradient budget to harder predictions
- Fix WeightedMAE with absolute threshold -- consistent extreme region penalty
- CSI/HSS metrics + persistence baseline -- standard space weather evaluation

**Should have (differentiators -- improve prediction quality):**
- Spatial attention gate -- learned focus on active regions via skip connections
- Temporal attention over encoder -- weight input frames by relevance
- Asymmetric loss for underestimation penalty -- operationally correct bias
- Wider channels [32,64,128] + kernel size 5 -- more capacity and spatial context
- Class-imbalanced sampling -- rebalances flare vs quiet-sun exposure
- MC Dropout (0.15) + cosine LR scheduler -- regularization and convergence

**Defer to v4.0+:**
- Progressive temporal curriculum (t_out 1->2->4) -- try simpler temporal fixes first
- Temporal difference input channels -- let attention learn what matters
- Multi-scale decoder -- high complexity, uncertain benefit over attention

### Architecture Approach

The v3.0 architecture extends the existing ConvLSTM encoder-decoder with two attention mechanisms and a normalized delta head. The data flow changes from a simple encode-decode-skip pipeline to one where temporal attention weights encoder outputs before the decoder, spatial attention gates filter skip connections, and a learned scale parameter normalizes the delta output. All changes are additive -- no existing modules are removed or restructured.

**Major components modified/added:**
1. **CompositeLoss (training/losses.py)** -- Extended with temporal difference, temporal weighting, temporal variation penalty, and asymmetric extreme terms
2. **SpatialAttentionGate (models/predictor.py)** -- Conv2d + Sigmoid applied before each skip connection to learn active-region focus
3. **TemporalAttention (models/predictor.py)** -- Linear + Softmax over encoder hidden states to weight input frame relevance
4. **Evaluation pipeline (utils/metrics.py + trainer.py)** -- CSI, HSS, persistence baseline, peak flux error, temporal variation ratio wired into validation loop
5. **Data pipeline (solarflare_data/loader.py)** -- WeightedRandomSampler for class-imbalanced training

### Critical Pitfalls

1. **Loss balancing trap** -- 6-7 loss terms will compete for gradient budget. One term (likely L1 on quiet regions) dominates, others ignored. **Avoid by:** logging each component separately, starting with conservative weights, ensuring temporal and extreme terms match L1 magnitude.

2. **Overfitting with 4x model size** -- Going from ~150K to ~600K+ params on 568 samples pushes params-to-sample ratio from 260 to 1050. **Avoid by:** MC Dropout (0.15), balanced augmentation (3x effective dataset), monitoring train-val gap, fallback to [24,48,96] channels.

3. **Temporal variation penalty instability** -- Negative loss term rewards predicting change; if lambda too high, model produces noisy predictions. **Avoid by:** starting lambda=0.05 (max 0.3), monitoring SSIM alongside variation ratio.

4. **Asymmetric loss breaks gradient balance** -- Overestimation bias if alpha too aggressive. **Avoid by:** applying only above extreme threshold, starting alpha=1.5, monitoring FP rate alongside FN.

5. **Spatial attention dead zones** -- Attention learns to mask 80%+ quiet-sun pixels, zeroing skip connections. **Avoid by:** initializing gate bias to ~2.0 (sigmoid(2.0) = 0.88 = pass-most-through default).

## Implications for Roadmap

Based on research, suggested phase structure (continuing from v2.0's 6 phases):

### Phase 7: Evaluation Metrics & Persistence Baseline
**Rationale:** You cannot improve what you cannot measure. Every subsequent phase depends on proper evaluation to validate its impact. This phase has zero risk and unblocks all others.
**Delivers:** CSI, HSS, persistence baseline, peak flux error, temporal variation ratio -- all wired into the training/validation loop with per-timestep logging.
**Addresses:** All evaluation metrics from FEATURES.md table stakes.
**Avoids:** No pitfall risk -- this is pure measurement infrastructure.

### Phase 8: Loss Function Overhaul
**Rationale:** The loss function is the primary lever for fixing temporal dynamics (the core v3.0 objective). Architecture changes without loss fixes produce larger models with the same broken behavior. This must precede architecture scaling.
**Delivers:** Temporal difference loss, temporal weighting, temporal variation penalty, fixed WeightedMAE with absolute threshold, asymmetric extreme penalty. Per-component loss logging for all terms.
**Addresses:** Temporal dynamics features (top 3 priority) and extreme region focus from FEATURES.md.
**Avoids:** Pitfalls 1, 3, 4 -- loss balancing, variation instability, asymmetric gradient imbalance. Each new term needs per-component logging and conservative initial weights.

### Phase 9: Training Policy Changes
**Rationale:** With proper loss functions in place, training policy adjustments amplify their effect. Cosine LR improves convergence, class-imbalanced sampling ensures the model actually sees flare sequences, and eliminating teacher forcing is necessary before architecture changes.
**Delivers:** Cosine LR scheduler (with warmup), balanced augmentation, tf=0, WeightedRandomSampler for flare oversampling.
**Addresses:** Training policy features from FEATURES.md differentiators.
**Avoids:** Pitfalls 5, 8 -- sampler + batch_size=1 overfitting (use moderate 3x factor), cosine LR instability (add warmup).

### Phase 10: Architecture Scaling
**Rationale:** Only after loss and training are stable should architecture capacity increase. Wider channels, attention mechanisms, and delta head normalization add representational power that the improved loss function can now exploit. Doing this first would be wasted computation.
**Delivers:** Spatial attention gates, temporal attention, wider channels [32,64,128], kernel size 5, delta head scale parameter, MC Dropout.
**Addresses:** Architecture scaling features from FEATURES.md differentiators.
**Avoids:** Pitfalls 2, 6, 7, 9, 10, 11 -- overfitting (dropout + augmentation), attention dead zones (bias init), temporal attention collapse (temperature scaling), delta scale explosion (soft clamping), checkpoint incompatibility (strict=False or fresh start).

### Phase 11: Integration Testing & Validation
**Rationale:** Full end-to-end validation against v2.0 baseline. All changes interact in ways that cannot be predicted from per-phase testing alone. This phase exists to catch emergent issues.
**Delivers:** Full training run with all v3.0 features, diagnostic comparison against v2.0, performance report.
**Addresses:** Final validation of all features.
**Avoids:** Compound pitfalls from interaction of all changes.

### Phase Ordering Rationale

- **Metrics first (Phase 7)** because every subsequent change needs measurement infrastructure. Without CSI/HSS and persistence baseline, there is no way to know if Phases 8-10 actually improved anything.
- **Loss before architecture (Phase 8 before 10)** because the diagnostic clearly shows the model's dynamics are broken at the optimization level. A bigger model optimizing the wrong objective produces the same bad predictions faster.
- **Training policy between loss and architecture (Phase 9)** because cosine LR and class-imbalanced sampling interact directly with the new loss terms and should be stable before adding model complexity.
- **Architecture last (Phase 10)** because capacity increase is only useful when the optimization target (loss) and training regime are correct. This also isolates overfitting risk to one phase.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 8 (Loss Overhaul):** Multi-term loss balancing is empirical. Plan for iterative weight tuning with diagnostic runs between adjustments. The interaction between temporal difference loss and temporal variation penalty is the key unknown.
- **Phase 10 (Architecture Scaling):** Spatial and temporal attention behavior on solar flux data is novel. Attention weight distributions should be monitored. Dead zone and collapse risks need active prevention.

Phases with standard patterns (skip research-phase):
- **Phase 7 (Metrics):** CSI, HSS, persistence baseline are textbook implementations. No research needed.
- **Phase 9 (Training Policy):** Cosine LR, WeightedRandomSampler, augmentation are standard PyTorch patterns.
- **Phase 11 (Integration):** Validation protocol is domain-specific but straightforward.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All features use PyTorch built-ins; no new dependencies; MPS compatibility verified |
| Features | HIGH | Priorities derived from diagnostic evidence (variation ratio 0.056, CSI 0.05) |
| Architecture | HIGH | Based on direct codebase inspection; integration points clearly identified |
| Pitfalls | HIGH | Grounded in diagnostic data and known ML failure modes; mitigations are concrete |

**Overall confidence:** HIGH

### Gaps to Address

- **Optimal loss term weights:** The relative weights between temporal_diff, temporal_var, asymmetric_alpha, and extreme_weight are unknown. Must be determined empirically during Phase 8 through iterative diagnostic runs. Start with conservative values documented in PITFALLS.md.
- **Attention memory cost at scale:** Spatial attention is negligible, but temporal attention over 10 frames with [32,64,128] channels has not been profiled. Should be fine with batch_size=1 but verify during Phase 10.
- **Training time impact:** Kernel size 5 + wider channels estimated at ~2-3x slower per epoch. Acceptable if metrics improve, but should be measured early in Phase 10.
- **Intermediate channel scaling:** If [32,64,128] overfits, [24,48,96] is the fallback. Decision point is during Phase 10 based on train-val gap.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection of models/predictor.py, training/losses.py, training/trainer.py, utils/metrics.py, solarflare_data/dataset.py, solarflare_data/loader.py
- Diagnostic run output (2026-03-07) -- temporal variation ratio, CSI, persistence skill metrics
- PyTorch documentation for nn.Conv2d, nn.Linear, F.softmax, WeightedRandomSampler, CosineAnnealingLR

### Secondary (MEDIUM confidence)
- Attention U-Net pattern (spatial attention gates) -- well-established in medical imaging, adapted for solar flux
- ConvLSTM temporal attention -- standard pattern in video prediction literature

### Tertiary (LOW confidence)
- Optimal loss weight ranges (temporal_diff_weight=1.0, temporal_var_weight=0.05-0.3, asymmetric_alpha=1.5-2.0) -- educated estimates, need empirical validation

---
*Research completed: 2026-03-07*
*Ready for roadmap: yes*
