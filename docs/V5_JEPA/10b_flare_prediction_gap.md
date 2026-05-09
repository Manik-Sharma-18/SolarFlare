# 10b — How Close to Actual Flare Prediction?

**Date:** 2026-05-09 | **Branch:** `v5-jepa-lora`
**See also:** `10_architecture_explainer.md`

**Current state: working SSL pretraining pipeline. No flare forecasting capability yet.**

---

## What's Validated

| Component | Status |
|---|---|
| End-to-end gradient flow | ✅ |
| EMA target encoder update | ✅ |
| Mask curriculum (5 policies) | ✅ |
| bf16, grad checkpoint, compile (CUDA) | ✅ |
| MPS NaN bug (SDPA + no_grad) fixed | ✅ |
| 1e8 winding flux clip (correct physics) | ✅ |
| val_loss monotone decreasing | ✅ (0.1950 at ep 4, sanity config) |

---

## Gap 1: Scale (Biggest Problem)

10 AR cubes is critically small. Literature minimum for VideoMAE-style SSL: ~3,000 videos.

> "14 cubes ≪ smallest validated MAE regime (~3k videos, VideoMAE). Pretraining from scratch on 14 cubes will not produce a useful encoder." — `00_overview.md`

50-epoch full run on `v5_path_a.yaml` (384-dim, all 10 cubes) hasn't run. Unknown whether loss converges to useful representations. **Critical next experiment.**

---

## Gap 2: No Downstream Head

Model outputs `[B, T, hp, wp, 384]` embedding tensors. To forecast anything, need:

- **Pixel reconstruction**: DPT/SegFormer pixel decoder → winding flux map
- **Flare classification**: attentive probe → binary M+ flare logit

Neither implemented. Listed pending in `09_progress.md`.

---

## Gap 3: No Forecast Skill Metrics

Current metric: smooth-L1 in latent space. Confirms predictor learns *something*, not whether it correlates with flare occurrence.

Real forecast skill requires:
- **CSI / HSS** — standard solar forecast metrics
- **Persistence baseline** — "tomorrow = today"; hard to beat for magnetic field data
- **TSS** — literature benchmark (pixel-only cap ~0.7; SHARP hybrid ~0.85)

---

## Gap 4: Known Ceiling on Pixel-Only Approach

Even with perfect downstream head, pixel-only winding flux has known TSS ceiling (~0.7).

> "If goal is operational flare forecasting, multimodal (cube + SHARP scalars) is mandatory." — `00_overview.md`

SHARP scalar integration = V5.2, separate pipeline, not yet designed.

---

## Path Forward

```
DONE:    SSL pretraining pipeline ✅
NEXT:    50-epoch full GPU run (v5_path_a.yaml, 384-dim, all 10 cubes)
THEN:    Pixel decoder → reconstruct winding flux → CSI/HSS vs persistence
THEN:    Flare event head (binary probe on pooled embeddings)
LATER:   SHARP scalar integration (V5.2) — breaks past TSS ~0.7 ceiling
```

**Bottom line:** Pipeline is mechanically sound and numerically stable. But SSL pretext success ≠ downstream flare forecasting. V5 is at the foundation layer — the forecasting layer (decoder + event head + skill metrics) hasn't started.
