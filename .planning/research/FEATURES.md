# Feature Landscape: v3.0 Temporal Dynamics & Flare Detection

**Domain:** Spatiotemporal solar flux forecasting (ConvLSTM encoder-decoder)
**Researched:** 2026-03-07
**Confidence:** HIGH (diagnostic-validated priorities)

## Diagnostic Context

Current model performance (from diagnostic run 2026-03-07):
- **Temporal variation ratio: 0.056** -- model predicts only 6% of real frame-to-frame change
- **CSI: 0.05** -- almost no flare detection (massive FN count)
- **Skill vs persistence: +3-9%** -- barely beats naive "copy last frame"
- **No overfitting** -- train/val/test metrics similar, headroom for capacity

## Table Stakes

Features required for a credible spatiotemporal forecaster.

### Temporal Dynamics
| Feature | Complexity | Impact | Dependencies |
|---------|-----------|--------|-------------|
| Temporal difference loss | Low | HIGH -- forces model to match rate of change | None |
| Eliminate teacher forcing (tf=0) | Config only | HIGH -- forces robust autoregressive dynamics | None |
| Temporal weighting (penalize later steps more) | Low | MEDIUM -- allocates capacity to harder predictions | None |

### Evaluation Metrics
| Feature | Complexity | Impact | Dependencies |
|---------|-----------|--------|-------------|
| CSI (Critical Success Index) | Low | HIGH -- standard space weather metric | Extreme threshold |
| HSS (Heidke Skill Score) | Low | HIGH -- measures improvement over chance | CSI infrastructure |
| Persistence baseline comparison | Low | HIGH -- null hypothesis for temporal models | None |
| Wire existing metrics into training loop | Low | MEDIUM -- currently computed but not logged | None |
| SSIM as standalone validation metric | Low | MEDIUM -- structural similarity tracking | Existing SSIM code |

### Extreme Region Focus
| Feature | Complexity | Impact | Dependencies |
|---------|-----------|--------|-------------|
| Fix WeightedMAE (absolute threshold) | Low | HIGH -- consistent penalty regardless of frame content | Existing WeightedMAE |
| Increase extreme_weight to 3.0+ | Config only | MEDIUM -- shifts optimization focus to flare regions | None |

## Differentiators

Features that would meaningfully improve prediction quality.

### Architecture Scaling
| Feature | Complexity | Impact | Dependencies |
|---------|-----------|--------|-------------|
| Wider channels [32, 64, 128] | Config only | MEDIUM -- more representational capacity | Dropout for regularization |
| Kernel size 5 | Config only | MEDIUM -- broader spatial context per step | None |
| Spatial attention gate | Medium | HIGH -- learned focus on active regions | Skip connections |
| Temporal attention over encoder | Medium | HIGH -- weight input frames by relevance | Encoder output access |
| MC Dropout (0.15) | Config only | MEDIUM -- regularization + uncertainty | None |

### Loss Function Enhancements
| Feature | Complexity | Impact | Dependencies |
|---------|-----------|--------|-------------|
| Asymmetric loss (penalize missed flares) | Low | HIGH -- operationally correct bias | Extreme threshold |
| Temporal variation penalty | Low | MEDIUM -- rewards predicting change | None |
| Delta head normalization | Low | MEDIUM -- better numerical range for learning | Output head access |

### Training Policy
| Feature | Complexity | Impact | Dependencies |
|---------|-----------|--------|-------------|
| Cosine LR scheduler | Config only | MEDIUM -- better convergence in later epochs | None |
| Balanced augmentation | Config only | MEDIUM -- 3x effective dataset | None |
| Class-imbalanced sampling | Medium | HIGH -- rebalances flare vs quiet-sun exposure | Data pipeline access |
| Peak flux error metric | Low | MEDIUM -- interpretable flare prediction quality | None |

## Anti-Features (Defer to v4.0+)

| Feature | Reason to Defer |
|---------|----------------|
| Progressive temporal curriculum (t_out 1->2->4) | Adds multi-stage complexity; try simpler temporal fixes first |
| Temporal difference input channels | Adds input complexity; let attention learn what matters |
| Multi-scale decoder | High complexity, uncertain benefit over attention |
| Feed frame-to-frame diffs as input | Let temporal difference loss + attention handle this first |

## Priority by Impact

**Highest impact (fix the broken temporal dynamics):**
1. Temporal difference loss -- directly attacks the 0.056 variation ratio
2. Eliminate teacher forcing -- forces honest autoregressive predictions
3. Temporal weighting -- allocates gradient budget to harder timesteps

**Second priority (fix flare detection):**
4. Fix WeightedMAE + increase extreme_weight
5. Asymmetric loss
6. Class-imbalanced sampling

**Third priority (scale model):**
7. Spatial attention + temporal attention
8. Wider channels + kernel size 5
9. MC Dropout

**Fourth priority (measure properly):**
10. CSI/HSS/persistence baseline in training loop

---
*Research completed: 2026-03-07*
