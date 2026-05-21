# 07 — Verification (drop-Pathak) + V4/V5.0/V5.1 summary

## 12. Verification — does dropping Pathak lose feature extraction or augmentation?

**Concern:** "If we drop Pathak entirely, will we not do feature extraction and increase data size to match model size?"

**Answer: NO. Two separate things were conflated.**

**A. Feature extraction is preserved (and improved).**
- Pathak: encoder learns features via pixel reconstruction + adversarial loss.
- V-JEPA: encoder learns features via embedding-space prediction + EMA target.
- JEPA's *whole purpose* = learning latent representations. More principled than Pathak. No GAN instability.
- Frozen-encoder + decoder paradigm requires good features → JEPA is the modern winner.

**B. Mask augmentation is preserved (and stronger).**
- Pathak: 25% random region masks.
- V-JEPA: 75-90% multi-block 3D tubes — much more aggressive masking.
- Same `N_masks × orientations × sliding_windows` multiplication math.
- V-JEPA's high mask ratio + tube structure forces motion semantics, not interpolation.

**C. "Match data size to model size" — Pathak never solved this.**
- Mask augmentation creates *variants of same physics*, not new physics.
- 21 cubes × N masked views ≠ N independent samples — they share underlying dynamics.
- Same statistical bottleneck regardless of Pathak vs V-JEPA.

**How V5 actually solves data/model mismatch:**

| Path | Mechanism | Effective pretrain corpus |
|---|---|---|
| A (Surya LoRA) | Encoder pretrained externally by NASA on 9 yr × 12-min × 13-ch SDO. We train ~95 M predictor head on our cubes only. | 218 TB SDO data (transferred via Surya weights) |
| B (V-JEPA on SuryaBench) | We pretrain V-JEPA on millions of SuryaBench samples, then transfer to our cubes. | SuryaBench (full solar cycle, 2010-2024) |
| JEPA-from-scratch on 21 cubes (V5.0 actual) | small-scale validation only | 21 cubes only — sanity ceiling 0.00831 (E09); thesis-scale ceiling open |

**Conclusion:** Dropping Pathak loses nothing on feature extraction or augmentation. V-JEPA does both better. Real solution to data/model mismatch = external pretrain corpus (Surya weights OR SuryaBench), enabled by JEPA path, never solved by Pathak.

---

## 13. Summary table — V4 vs V5.0 vs V5.1

> **2026-05-08 update:** V5.0 = Path B (JEPA-from-scratch). Path A (LoRA on Surya) ABANDONED — HelioSpectFormer hard-locked to img_size=4096 / 60-min / 13ch; AR cubes variable HxW / 12-min / 1ch. No adapter bridges. See `01_path_a.md` + finding F10 in `12_experiments_findings.md`.

| Aspect | V4 (current) | V5.0 (Path B: JEPA-from-scratch, **ACTIVE**) | V5.0-A (Path A: LoRA Surya, **ABANDONED**) |
|---|---|---|---|
| Encoder | SA-ConvLSTM ~few M, trained from scratch | ViT context + EMA target (ViT-Small at sanity scale), trainable | Surya 366 M, frozen |
| Pretrain | None | Self-supervised on 21 AR cubes (E09 et al.) | None (would use Surya weights) |
| Forecast head | Decoder ConvLSTM + delta head | Block-causal predictor in embedding space (downstream decoder TBD) | V-JEPA-2-AC predictor + SegFormer decoder |
| Loss | L1 + SSIM in pixel space | smooth-L1 in embedding space + masked-loss (INTERSECTION valid+mask) | smooth-L1 embedding + pixel ×0.1 |
| Trainable params | ~all | ~33 M (context + predictor); ~22 M frozen EMA target | ~95 M (would be) |
| Failure mode addressed | persistence collapse | embedding-space loss avoids; mask catalog forces motion semantics | — (architectural lock prevented test) |
| Compute | ~hours | sanity ~5 h MPS / ~3 h CUDA per 100ep | — |
| Risk | already failed | small data (21 cubes) — F8 floor open at thesis scale | architectural lock — KILLED |
| Time to train | days | hours–days (sanity); path_a scale unknown | — |
