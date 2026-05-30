# Dual-Head Classifier — S12 / S13 / S14 / S15 / S16 findings

Snapshot 2026-05-29. Output of the **structural pivot** flagged in the S-series verdict (`04_experiments_and_state.md` §S-series, 2026-05-26): loss-reweighting and representation levers exhausted, output-parametrization was the only remaining axis.

## Hypothesis

L1 + extreme_pixel_weight (S6/S11 working composite) **regresses extremes to a smoothed conditional mean** that never crosses the meteorological threshold (TP=0 → CSI=0). Add a **per-pixel classifier head** trained with BCE-pos-weight on the extreme mask: explicit binary output for "extreme this frame" rather than relying on the regression to overshoot.

Model: `SimpleConvLSTM(enable_classifier_head=True)` returns `(regression_pred, extreme_logits)`. Loss: `α·L1(pred, target) + β·BCE-pos-weight(logits, extreme_mask)`. Metric: `val_csi_classifier = CSI(sigmoid(logits) > 0.5)`, printed per-epoch in `training/trainer/reporting.py:40` (added 2026-05-29).

## Arms

| Arm | α/β | pos_w | classification | extreme_pixel_weight | Result |
|---|---|---|---|---|---|
| S12 | 1/1 | 60 | BCE | — | original (cosine LR) killed mid-run; rerun id 81 queued on mini_mps |
| **S13** | 1/1 | 100 | BCE | — | **TEST CLS-CSI 0.0434, HSS 0.0749 — matches persistence_csi 0.043 (FIRST S-arm)** |
| S14 | 1/1 | 60 | **focal** γ=2 α=0.25 | — | DEAD — **CLS-CSI=0 every epoch**, focal over-suppresses positives. Best val 0.0072 ep10 (numeric scale only, no learning). SSIM 0.79–0.91, pers skill -2..+35%, var ratio 0.21–0.88 unstable |
| S15 | 0.1/1 | 60 | BCE | — | queued (failed 2× to nvidia-smi blip; awaiting CUDA after S16) |
| **S16** | 1/1 | 60 | BCE | **25** | **running ep5+/15** — hybrid of S13 (classifier) + S11 (extreme_pixel_weight). Joint-train of post-hoc S17 combo. Early: val 0.124 ep5 ✓ (18% drop ep1→5), SSIM 0.89, pers skill +31%, CLS-CSI 0.0008–0.002 (weaker than S13 — regression eating loss budget) |

All use **constant LR (1e-3, no cosine)**, `tf_decay_epochs=5`, `patience=8`, 15 epochs, fixed test cubes [245, 274, 49].

## S13 (id 73) — BREAKTHROUGH

- 15 ep on 5060ti, ~6.5h.
- Val_loss 0.1641 ep1 → **0.1491 ep13 best** (monotone-ish, small bumps ep3/10/14).
- Val CLS-CSI 0.003 ep2 → 0.0066 ep12. SSIM 0.62 → 0.91. Persistence skill +30%.
- **Test CSI (classifier): 0.0434, HSS 0.0749 — matches persistence_csi 0.043 (FIRST EVER)**.
- Test CSI (regression): 0.0090 — regression dead (pos_w 100 crushes L1 budget).
- Test persistence skill: 35.7%.

## S14 (id 77) — DEAD ARM

Focal BCE `−α(1-p)^γ log p` with γ=2, α=0.25 over-downweights the few positives — classifier never learns. **CLS-CSI=0 across all 15 epochs.** Best val 0.0072 ep10 (focal numeric scale, not a learning signal). Regression head SSIM 0.79–0.91 = baseline S0–S11 behaviour. **Verdict:** focal is the wrong instrument for this imbalance level (~0.1% positive). Use BCE-pos-weight or skip.

## S17 inference combo (no retraining, 2026-05-29)

Post-hoc fusion of S11 regression × S13 classifier on harp_11930, 3 active windows, full-frame tiled.

| Variant | Mean MAE (int flux) | vs S11 |
|---|---|---|
| **B_soft: S11 × (1 + 2·σ(S13_logits))** | **206M** | **−8%** ✅ |
| S11_only | 223M | baseline |
| A_gate: S11 × (σ>0.5) hard mask | 238M | +7% ❌ |
| S13_only (regression head) | 300M | +34% (S13 reg dead) |
| C_blend: S13 where mask else S11 | 301M | +35% ❌ |
| Persistence (pred=0) | 484M | +117% |

Mask coverage 5.5–5.9% per set — classifier fires on small extreme regions only.

**Lessons:** soft boost > hard gate (zeroing background hurts); S13 regression useless alone; 8% lift is small → joint-train (S16) should compound by letting representations co-evolve.

Script: `scripts/s0_viz/_s17_combo.py`.

## Compare on harp_11930 (single-step pred, NOT autoregressive)

| # | Arm | Mean MAE | vs pers (484M) |
|---|-----|----------|-----|
| 1 | S3 composite | 199M | 41% |
| 2 | S11 fasttf+extreme | 223M | 46% |
| 6 | S13 dual_pw100 | 300M | 62% |
| 10 | S4 residual+TF | 806M | **167% (worse)** |
| 11 | S9 BatchNorm | 1311M | **271% (worse)** |

`harp_11930` is **TRAIN split** — S3's win includes overfit; trust persistence and test_csi for generalization.

## Staircase autoregressive (S3, S11 on harp_11930, 20 steps stride 2)

Both arms pathological:

- **S3 mode-collapses** within 4 frames: field |max| drops 13M → 100K (100×). Frame-to-frame diff <0.1% post-collapse. **var ratio 0.028**, MAE 1.5e8.
- **S11 explodes** after step 9: behaves like S3 for first 9 steps, then variance explodes — frame |max| climbs 80K → 2M+ by frame 39. **var ratio 15.8**, MAE 7.6e8 (5× worse).

Neither model is *predicting* under chained autoregression — converges to stable point (S3) or runaway (S11). Even the best current arms can't sustain multi-step rollouts. Scripts: `scripts/s0_viz/_staircase_harp11930.py`, `scripts/s0_viz/_staircase_viz.py`. Outputs: `outputs/staircase_<arm>_harp_11930/{integrated_flux.png, staircase_grid_<N>steps.png, series.npz}`.

## Next levers (post-S16)

If S16 stalls at S13 level (~0.04 test CLS-CSI) or below:
1. **pos_weight sweep** with extreme_pixel_weight=25 fixed → Pareto frontier of regression vs classifier.
2. **Quantile regression head** (pinball τ=0.5/0.9/0.99) → predict 99th-pct amplitude directly, skip classifier detour.
3. **Event-onset reframe** → per-frame `P(any extreme in next K frames)` using GOES-aligned labels (independent of per-pixel rate). Escape route already flagged.
4. **GradNorm balancing** → auto-balance α/β so per-batch grad norms equal.

## Code touched (2026-05-29 wave)

- `training/losses/dual_head.py` — added `extreme_pixel_weight` kwarg → uses `WeightedMAELoss` when >1.0 (default 1.0 = back-compat plain L1).
- `training/losses/factory.py:54` — pass `extreme_pixel_weight` from config.
- `training/trainer/reporting.py:40` — print `CLS-CSI` per epoch.
- `scripts/s0_viz/infer.py`, `scripts/s0_viz/fullframe.py` — unwrap `(pred, logits)` tuple for dual-head viz.
- `configs/ablations/S16_*.yaml` — new arm.
- `configs/finetune_winding_flux.yaml` — `scheduler.type: constant` (was cosine).
- New scripts: `scripts/s0_viz/{_compare_harp11930,_s17_combo,_staircase_harp11930,_staircase_viz}.py`.

## Related

- Parent: [`04_experiments_and_state.md`](04_experiments_and_state.md) §S-series.
- Prior: [`findings_simple_convlstm_S0_S1.md`](findings_simple_convlstm_S0_S1.md), [`findings_S2_residual.md`](findings_S2_residual.md).
