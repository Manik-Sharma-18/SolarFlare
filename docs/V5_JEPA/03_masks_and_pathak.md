# 03 — Mask catalog + why-not-Pathak

## 5. Mask catalog (Phase 1 / pretrain)

Adapted from V-JEPA + Brain-JEPA cross-time/cross-ROI ideas. Drop Pathak random-region as primary; keep as ablation.

| Mask | Shape | Scale | Purpose |
|---|---|---|---|
| Short tube | spatial block, all T | 15% area | local active-region structure |
| Long tube | spatial block, all T | 40% area | global flux-sheet motion |
| Future block | full spatial, last K frames | 100%×K/T | force real future prediction (V-JEPA 2-AC causal) |
| Cross-time (Brain-JEPA) | full spatial, random T frames | 30% time | non-causal temporal infill |
| Tail (downstream) | full spatial, last `t_out` frames | 100%×t_out/T | aligns with deployment use case |
| Random-region (Pathak) | irregular blob | 25% area | ablation only |

Mix during pretrain: 50% short+long tube, 30% future block, 20% cross-time. Total token mask ratio: ~75-85%.

---

## 6. Why we are not building Pathak Context Encoder

Original `ARCHITECTURE_V5.md` (Pathak port) had:
- 3D conv encoder/decoder (252 M CFC bottleneck dominated)
- Joint L2 + adversarial loss (λ_rec=0.999, λ_adv=0.001)
- 3D DCGAN discriminator
- Mean-fill masks with 7 px overlap × 10 weight

Reasons to drop:
| Pathak choice | Why obsolete |
|---|---|
| Adversarial loss | Modern field (MAE, JEPA, all FM) drops GAN. Training instability + minimal benefit at scale. |
| Channel-wise FC | 252 M params for fully-connected over spatial × temporal. ViT's self-attention does the same job at a fraction of params. |
| Pixel-space L2 | Causes blur / persistence collapse — exactly V4's failure. JEPA latent loss avoids by construction. |
| 3D conv | Tubelet-patch + ViT is more parameter-efficient and benefits from RoPE. |
| Random region masks | V-JEPA's multi-block 3D tubes outperform on motion tasks (+10-21 pts SSv2). |

Pathak (2016) was the right architecture for 2016. **2026 best practice is V-JEPA 2-AC + frozen-encoder + LoRA.**
