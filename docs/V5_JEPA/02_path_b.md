# 02 — Path B: pretrain V-JEPA from scratch (NOW PRIMARY — V5.0)

> **STATUS UPDATE (2026-05-07):** Path A abandoned (Surya hard-locked to 4096/60min/13ch).
> Path B promoted from "V5.1 fallback" to **V5.0 primary**. Implemented at small scale
> (ViT-Small encoder + EMA target + block-causal predictor, ~55M params total).
> Sanity green on MPS + 5-epoch sanity green on RTX 5060 Ti CUDA.
> Current state: `09_progress.md`. Pretrain corpus / SuryaBench section below is
> the long-term spec; current scaffold trains directly on AR cubes.

Originally scoped as fallback if Path A fails OR if sovereign weights required —
both conditions now met (Path A unviable, no encoder pretraining done yet).

## 4.1 Encoder pretrain corpus
- **SuryaBench** (arXiv 2508.14107). SDO/HMI+AIA, 2010-2024, ML-ready, full solar cycle. Free.
- ~218 TB raw; subset to magnetogram-equivalent channels for our forecasting use case.

## 4.2 V-JEPA pretrain spec
| Spec | Value | Source |
|---|---|---|
| Backbone | ViT-L/16 (~300 M) | V-JEPA |
| Patch | 16×16, tubelet 2 | V-JEPA |
| Pos enc | 3D RoPE | V-JEPA 2 |
| Predictor | 12 layers × 384 dim (~22 M) | V-JEPA |
| EMA target | 0.998 → 1.0 cosine | V-JEPA |
| Mask ratio | 75-85% (back off from V-JEPA's 90% — solar has higher spatial autocorr) | adapted |
| Mask type | multi-block 3D tubes (short scale 0.15 + long scale 0.4) + future-block (full-spatial × last `t_out` frames) | V-JEPA + future-mask |
| Loss | smooth-L1 in feature space + dist-weighted L1 on context tokens (V-JEPA 2.1 trick) | V-JEPA 2.1 |
| Optimizer | AdamW, peak lr 6.25e-4, warmup-constant-cooldown | V-JEPA 2 |
| Resolution schedule | progressive: 128² crops → native-AR cooldown (cube-bucketed) | V-JEPA 2 (~8× compute savings) |
| Total | ~252 k iters | V-JEPA 2 |

## 4.3 Optional: replace EMA with LeJEPA
- LeJEPA (arXiv 2511.08544, Nov 2025) proves isotropic Gaussian is optimal target distribution. SIGReg regularizer (~50 LoC) replaces EMA + stop-grad + teacher schedule. Single hyperparameter. ViT-H/14 reaches 79% IN-1k linear probe.
- **Eliminates EMA tuning headaches.** Worth using for a clean training loop on small data.

## 4.4 Optional: SAR-JEPA gradient-feature target
- SAR-JEPA predicts multi-scale gradient features instead of raw pixels — addresses speckle noise. Magnetogram noise is structurally similar.
- Replace target encoder output with gradient-magnitude features at multiple scales.

## 4.5 After pretrain → freeze → forecast head
Same as `01_path_a.md` §3.2-3.5. Encoder is now ours instead of Surya.
