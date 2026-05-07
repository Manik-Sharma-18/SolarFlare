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
- 14 cubes × 217 k masked views ≠ 217 k independent samples — they share underlying dynamics.
- Same statistical bottleneck regardless of Pathak vs V-JEPA.

**How V5 actually solves data/model mismatch:**

| Path | Mechanism | Effective pretrain corpus |
|---|---|---|
| A (Surya LoRA) | Encoder pretrained externally by NASA on 9 yr × 12-min × 13-ch SDO. We train ~95 M predictor head on our cubes only. | 218 TB SDO data (transferred via Surya weights) |
| B (V-JEPA on SuryaBench) | We pretrain V-JEPA on millions of SuryaBench samples, then transfer to our cubes. | SuryaBench (full solar cycle, 2010-2024) |
| Pathak from scratch on 14 cubes | (would have failed) | 14 cubes only — too small for any FM |

**Conclusion:** Dropping Pathak loses nothing on feature extraction or augmentation. V-JEPA does both better. Real solution to data/model mismatch = external pretrain corpus (Surya weights OR SuryaBench), enabled by JEPA path, never solved by Pathak.

---

## 13. Summary table — V4 vs V5.0 vs V5.1

| Aspect | V4 (current) | V5.0 (Path A: LoRA Surya) | V5.1 (Path B: pretrain) |
|---|---|---|---|
| Encoder | SA-ConvLSTM ~few M, trained from scratch | Surya 366 M, frozen | V-JEPA ViT-L 300 M, pretrained on SuryaBench |
| Pretrain | None | None (use Surya) | V-JEPA on SuryaBench |
| Forecast head | Decoder ConvLSTM + delta head | V-JEPA-2-AC predictor (~85M) + SegFormer decoder | Same |
| Loss | L1 + SSIM in pixel space | smooth-L1 in embedding + L1+SSIM pixel ×0.1 | Same |
| Trainable params | ~all | ~95 M (predictor + LoRA + decoder + adapter) | Same + encoder pretrain |
| Failure mode addressed | persistence collapse | embedding-space loss avoids by construction | embedding-space loss avoids by construction |
| Compute | ~hours | ~hundreds GPU-hr | ~10 k+ GPU-hr |
| Risk | already failed | domain gap Surya↔our cubes | smallest validated MAE on ~3 k clips, we have 14 cubes (use SuryaBench) |
| Time to train | days | days | weeks |
