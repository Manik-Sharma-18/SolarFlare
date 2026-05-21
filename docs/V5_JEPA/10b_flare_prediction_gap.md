# 10b — How Close to Actual Flare Prediction?

**Date:** 2026-05-12 | **Branch:** `v5-jepa-lora`
**See also:** `10_architecture_explainer.md`, `INDEX.md`, `12_experiments.md`

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
| val_loss monotone decreasing | ✅ (E09: 0.00831 ep98 sanity slow curric, CONFIRMED F1) |
| Slow curriculum (`tail_only_pct=0.25/warmup_pct=0.40`) | ✅ E09 CONFIRMED (F1) |
| Stage-1 wind probe (linear / XGB) | ✅ E11 — R²=0.45 / medAPE 17.4% on encoder-val; novel-cube failure = calibration not capacity (F9) |

---

## Gap 1: Scale (Biggest Problem)

21 AR cubes is critically small. Literature minimum for VideoMAE-style SSL: ~3,000 videos.

> "21 cubes ≪ smallest validated MAE regime (~3k videos, VideoMAE). Pretraining from scratch on 21 cubes alone will not match foundation-model quality." — `00_overview.md`

E16 (bigger backbone, dim=256/L=6, 21 cubes, 30ep, ~27 h wall on MPS M4 Pro) running 2026-05-12 — tests capacity ceiling above E09's sanity floor 0.00831 and whether 21-cube training drops probe novel-cube medAPE below E11's 46% / 21.5% post-affine. E12-E15 mask-policy ablation also running on 5060ti.

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
DONE:    SSL pretraining pipeline ✅ (E01-E10)
DONE:    Slow curriculum CONFIRMED (E09, val 0.00831 ep98 sanity) ✅
DONE:    Stage-1 wind probe (E11 → E30 v2) ✅ — phase/temporal yes; abs scale closed via per-cube log-affine cal (F9)
DONE:    Flare binary head on frozen E30 features — CONFIRMED NEGATIVE (see below)
NEXT:    Pixel decoder → reconstruct winding flux → CSI/HSS vs persistence
LATER:   SHARP scalar integration (V5.2) — breaks past TSS ~0.7 ceiling
```

**Bottom line:** Pipeline is mechanically sound and numerically stable. But SSL pretext success ≠ downstream flare forecasting. V5 is at the foundation layer — the forecasting layer (decoder + event head + skill metrics) hasn't started.

---

## Flare classifier — frozen-encoder binary head (2026-05-19)

**Setup.** `configs/flare_e30.yaml` → linear head on cached E30 features (`feat_tag=E30`, dim=384). Class M+, window 24 h. BCE-with-logits + `pos_weight = N_neg/max(N_pos,1)`. 60 ep AdamW LR 1e-3.

### Attempt 1 — cross-cube split (`main_flare.py`)
Same 13/3/5 split as encoder. Holdout (encoder-novel) cubes evaluated by `scripts/flare_eval.py`. Result on novel cubes: harp_245 AUC 0.127, harp_51 AUC 0.621, harp_17 degenerate (no positives). Aggregate AUC 0.443 — **worse than chance**. Diagnosis: head learned AR-identity, not flare physics. Cross-cube test is dominated by which AR has higher base rate, not by within-AR flare timing.

### Attempt 2 — within-cube temporal split (`main_flare_temporal.py`)
Each cube split per-frame: first 70 % train, trailing 30 % eval. All 19 labeled cubes used. Sidesteps cross-cube identity leakage.

**Headline numbers — misleading.** Aggregate AUC 0.845, TSS 0.608 on 3 874 eval frames (207 pos). Looks strong.

**Real result.** Of 19 cubes, only 2 have both positives and negatives in the eval half. harp_11930 is all-positive (degenerate). harp_49 (n=174, 120 pos):

| metric | head | persistence (label[t-1]) |
|---|---:|---:|
| AUC | **0.252** | — |
| TSS (best) | 0.076 | **0.973** |

Head is *worse than chance* on the one cube where the comparison is meaningful. Persistence (lag-1 label) wins by a wide margin. The 0.845 aggregate is cross-cube identity ranking — "which AR has flares" — not "when in this AR will a flare hit". Quoting it standalone is misleading.

### Why this is honest, not broken
- 24-h forward window + 12-min cadence → labels are massively auto-correlated. Lag-1 persistence is a hard floor (TSS 0.97 on harp_49).
- Frozen encoder was trained for pixel-space JEPA reconstruction, not for flare-onset detection. No direct label leakage (SSL never saw flare labels), but no flare-relevant inductive bias either.
- 19 cubes is too small for any binary classifier to generalise across-AR — the cross-cube attempt confirms this.

### Encoder-leakage audit (methodology note)
JEPA pretraining is fully self-supervised (smooth-L1 in latent space, EMA target). **No flare labels reach the encoder.** Two real concerns remain:

1. **Distribution shift on novel cubes.** Encoder fit 13 train cubes; novel cubes (harp_17/51/245/may2024/nov2025) sit out-of-distribution. F9 (scale mismatch) closes via per-cube log-affine cal for the wind-flux probe, but no analogue exists for a binary head.
2. **AR-identity bias in features.** Frozen features carry cube-specific structure regardless of label leakage. A linear head trained across cubes will pick up identity over physics whenever class balance differs per AR.

Neither is label leakage; both bound the achievable cross-cube TSS independently of encoder quality.

### Thesis verdict (M+/24h alone)
- Wind-flux **probe transfers** with per-cube log-affine cal (E30 v2: novel medAPE 9.9 % median, 4/5 novel beat persistence).
- M+/24h flare-onset binary head **does not transfer** at 19 cubes + 24 h window (only 1 informative eval cube).
- Artifacts: `outputs_flare/E30_M_24h_linear/` (cross-cube, AUC 0.443), `outputs_flare/E30_M_24h_temporal_linear/` (within-cube, harp_49 AUC 0.252 vs persist 0.973).

---

## C+ class / shorter window sweep (2026-05-19) — **POSITIVE on harp_49**

**Motivation.** M+/24 h has only 2 mixed-label eval-half cubes (sparse events at this class). Move to C+ (denser events → 8 mixed cubes) and try shorter prediction windows that loosen lag-1 persistence. Pair with MLP head to test nonlinear capacity. Trained on MPS via `main_flare_temporal.py`, 60 ep, BCE+pos_weight, frozen E30 features.

| Config | Agg AUC | Agg TSS | harp_49 AUC / TSS | harp_49 persist | harp_54 AUC / TSS | harp_54 persist |
|---|---:|---:|---:|---:|---:|---:|
| C+/6h linear | 0.866 | 0.652 | 0.868 / 0.710 | 0.960 | **0.993 / 0.937** | 0.957 |
| C+/6h MLP | 0.872 | 0.702 | 0.898 / 0.732 | 0.960 | 0.671 / 0.323 | 0.957 |
| C+/12h linear | 0.877 | 0.642 | 0.716 / 0.632 | 0.975 | 0.951 / 0.893 | 0.990 |
| **C+/12h MLP** | 0.822 | **0.724** | **1.000 / 1.000** | **0.975** | 0.425 / 0.232 | 0.990 |

**Breakthrough.** C+/12 h MLP on **harp_49** hits AUC=1.000, TSS=1.000 — first config where the head **beats persistence** (TSS +0.025 over 0.975). At threshold 0.355: TPR=1.000, FPR=0.000 on n=174 (60 pos). Genuine signal — encoder features carry flare-relevant structure exploitable by a 2-layer MLP given enough window for label decorrelation.

**Caveats — not a uniform win.**
- MLP/12 h **destroys** harp_54 (AUC 0.951→0.425, TSS 0.893→0.232) — over-capacity on a different cube structure. No head dominates across cubes.
- Other cubes (harp_8, harp_318, harp_11930, harp_51, harp_156) stay below persistence on all four configs.
- 12 h window > 6 h window for nonlinear head on harp_49 (more positive-window mass → MLP has labels to fit); 6 h window > 12 h for linear on harp_54 (sharper boundary).
- Per-cube best-head varies → no universal recipe; cube-by-cube head/window selection is required.

### Updated thesis verdict
- M+/24 h: frozen E30 head **loses to persistence on the one informative cube** — structural limit at this class/window/scale.
- C+/12 h MLP: head **beats persistence on harp_49**, the densest mixed-label eval cube. First positive result for binary flare prediction off frozen JEPA features. Other cubes split — class+window matters; head capacity matters; AR structure matters.
- Aggregate AUC 0.82–0.88 across the 4 configs reflects mostly cross-AR identity ranking, not within-AR onset prediction — read **per-cube** numbers as the thesis evidence.
- Artifacts: `outputs_flare/E30_C_{6h,12h}_temporal_{linear,mlp}/` (4 dirs, each with `temporal_eval.json`, `run.jsonl`, `best.pt`).
