# V4 ConvLSTM — Experiments & State History

Snapshot of the `Version_4` branch (planning-phase docs frozen 2026-03-09). Predates the V5 JEPA pivot. All numbers below come from `.planning/` artefacts; run-level raw logs in `outputs/` are listed only.

## Current Milestone

- **Active milestone:** v3.0 *Temporal Dynamics & Flare Detection* — Phases 7-11.
- **Status:** all 5 phases / 12 plans complete (`progress.percent: 100`, `.planning/STATE.md`).
- **Last activity:** 2026-03-09 — quick-task `quick-03` swapped bilinear resize for center-crop to 437×877 (commit `dca00b3`); preceded by `quick-02` extreme-threshold fix (`08cf9ea`).
- **Headline result:** v3.0 vs v2.0 comparison shipped with **MIXED** verdict (`COMPARISON.md`, 2026-03-09 08:30 UTC). Temporal variation ratio +258%, CSI −74%. Phase 11-02 was the milestone deliverable.
- **No active follow-on milestone.** v3.0 closed; v3.1+ items in `REQUIREMENTS.md` Future are unplanned.

## Phase History (one-liner per phase)

v2.0 *Stabilization & Cross-Platform* (shipped 2026-02-04, git `0585351` → `df2234f`, 6 phases / 16 plans):

| Phase | One-liner |
|---|---|
| 01 Device Foundation | Cross-platform device API (CUDA/MPS/CPU), seeding, ConvLSTM.py prototype deleted (−698 LOC). |
| 02 Config & Errors | ConfigValidationError accumulates all errors; NaN/Inf gradient guards; emergency Ctrl-C checkpoint. |
| 03 Checkpoints | Atomic temp+rename I/O; cross-device resume; optimizer / scheduler / epoch state included. |
| 04 Data Pipeline | mmap rewrite; deterministic flip aug; macOS spawn / Linux fork; loud failure on missing files. |
| 05 MPS Compatibility | `safe_outer` / `safe_quantile`, channel-loop SSIM, kernel caching/tiling, Welford O(1) MC-Dropout. |
| 06 Test Coverage | First `tests/` dir; 88 tests across device, config, losses, model shapes, checkpoint, data pipeline. |

v3.0 *Temporal Dynamics & Flare Detection* (Phases 7-11, completed 2026-03-09):

| Phase | Plans | Summary |
|---|---|---|
| 07 Evaluation Metrics | 07-01, 07-02 | TDD 10 metric fns (CSI, HSS, persistence, SSIM, peak flux, var ratio, per-t RMSE+corr) + 29 tests; `validate()` returns dict, contingency accumulated across batches. |
| 08 Loss Overhaul | 08-01..03 | Fixed WeightedMAE to absolute thresh; AsymmetricExtremeLoss; temporal-diff, temporal-var, per-t weighting; CompositeLoss restructured to 6 components with per-component logging. |
| 09 Training Policy | 09-01, 09-02 | Cosine LR (eta_min=1e-6); WeightedRandomSampler over flare sequences (spatial density >2%); tf_start=0.0. |
| 10 Architecture Scaling | 10-01..03 | SA-ConvLSTM (memory projection, manual bmm+softmax for MPS); TemporalAttention + AttentionGate; channels [32,64,128]; kernel 5; learned `delta_scale`; MC Dropout 0.15. |
| 11 Integration & Validation | 11-01, 11-02 | 25-epoch full run; `generate_comparison.py` (743 LOC) emits `COMPARISON.md` + 3 PNGs against hardcoded v2.0 baseline. |

## Key Decisions

- **Roadmap ordering:** metrics → loss → training → architecture → integration.
- **No new deps** in v3.0; all PyTorch built-ins.
- **Loss/eval threshold:** 0.3456 (matched between loss and evaluation; cross-check warns on drift). Quick-02 later corrected the operational threshold to **0.277** (99th pct in normalized space); the 0.3456 value was 99.5th pct and miscalibrated. Quick-02 also replaced `np.any()` flare detection (flagging 100% of windows) with spatial density >2%.
- **Center-crop over resize:** quick-03 → center-crop **437×877**, preserves 0.36 Mm/px scale across cube groups.
- **Delta head:** learnable `nn.Parameter` `delta_scale` (init 100.0), excluded by name from weight decay.
- **MPS attention:** all attention paths use manual `bmm + softmax`; no `F.scaled_dot_product_attention`.
- **v2.0 baselines hardcoded** in `generate_comparison.py` (V2_BASELINE dict) per explicit user decision.
- **25 epochs, not 50,** for the v3.0 integration run — wall-clock constraint, documented in `COMPARISON.md` Methodology.
- **ConvLSTM.py deleted, not archived** in Phase 01-03 — git history is the archive.

## Metrics Tracked (`utils/metrics.py`)

Pure stateless functions; lazy-imports `ssim` from `training.losses` to dodge cycles.

- `compute_metrics` — MAE total + MAE per timestep (mean over B, C, H, W).
- `compute_rmse`, `compute_correlation` — global.
- **`compute_csi(tp, fp, fn)`** = TP / (TP+FP+FN); returns 0.0 on zero denom.
- **`compute_hss(tp, fp, fn, tn)`** = 2(TP·TN − FP·FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN)); range [−1, 1], 0.0 on zero denom.
- **`accumulate_contingency(pred, target, threshold)`** — `abs(value) > threshold` (standard meteorological), returns per-t `(tp, fp, fn, tn)` as Python ints.
- `compute_persistence_prediction` — last input frame, expanded; `compute_persistence_skill` = (1 − model_mae / persistence_mae) · 100.
- `compute_ssim_per_timestep` — `data_range=2.0` default (asinh-normalized).
- `compute_peak_flux_error` — per-t `|max(pred) − max(target)|`, batch-mean.
- `compute_temporal_variation_ratio` — `mean(|Δ-pred|) / mean(|Δ-target|)`; T≤1 → 1.0, zero-target-variation → 0.0.
- `compute_rmse_per_timestep`, `compute_correlation_per_timestep` — per-t; correlation → 0.0 on zero variance.

Threshold convention: **0.3456 in normalized asinh space** was used for v3.0 training and the comparison; quick-02 swapped it to **0.277** for downstream work.

## Comparison Artefacts (`generate_comparison.py`)

Standalone, 743 LOC, `--output-dir DIR`. Verdict logic: both var-ratio + CSI improve → PASS; one → MIXED; neither → REGRESSION.

Outputs: `COMPARISON.md`, `comparison_metrics.png` (2×2 per-t MAE/RMSE/corr/persistence-skill), `comparison_temporal.png` (1×3 var ratio + CSI + HSS), `comparison_samples.png`.

**v3.0 vs v2.0, single-run seed 42, test split, 25 epochs MPS:**

| Metric | v2.0 | v3.0 | Δ % |
|---|---|---|---|
| Temporal variation ratio | 0.060 | 0.215 | +258% |
| CSI (thresh 0.3456) | 0.0510 | 0.0135 | −74% |
| HSS | 0.0920 | 0.0235 | −75% |
| MAE avg t+1..t+4 | 0.1092 | 0.0971 | −11% |
| RMSE avg | 0.1527 | 0.1593 | +4% |
| Correlation avg | 0.5058 | 0.3960 | −22% |
| Persistence skill % avg | 4.48 | 3.81 | −15% |
| SSIM (v3.0 only) | N/A | 0.2856 | — |

Persistence baseline itself: CSI 0.0549, HSS 0.1020 — v3.0 CSI **below** persistence. v2.0 SSIM not captured (metric only landed in Phase 7). Interpretation: persistence trap broken (variation 3.6×) but variation not flare-calibrated.

## Outstanding Work (`IMPROVEMENT_NOTES.md`)

- **More winding-flux cubes** — flagged as single highest-impact improvement (Phase G, item 4.5/7); only 7 train files / 568 samples.
- **Progressive temporal curriculum** (t_out 1→2→4, item 7.5; Phase F) — explicit Out-of-Scope.
- **Temporal-difference input channels** (7.2) — never implemented; queued as DATA-03.
- **Multi-scale decoder** (3.5) — never started.
- **Magnetogram multi-quantity input** (DATA-01) — data not yet available.
- **Optuna/Ray sweep**, **inference API refactor** — explicit Out-of-Scope.
- **CSI recovery work** flagged in 11-02 SUMMARY: threshold tuning, oversample-weight up, full 50-epoch run.

## Known Concerns (`.planning/codebase/CONCERNS.md`, 2026-02-02)

Most pre-v2.0 concerns closed in Phases 1-6. Surviving:

- **Predictor double-preprocess** when `downsample_input=False` — known quirk; production uses True.
- **MPS runtime validation deferred** — no CI hardware.
- **Optimal loss weights** (`temporal_diff`, `temporal_var`, `asymmetric_alpha`) unknown; tuned ad hoc.
- **Overfitting risk** at 4× param increase vs 568 samples; fallback `[24,48,96]` never invoked.
- ~~**Skip-connection dim mismatch on odd spatial sizes**~~ — resolved 2026-05-21 (`predictor.py:_match_spatial`): center-crop / zero-pad for ≤ 2 px drift, raises `RuntimeError` otherwise.
- **Batch size 1** in legacy config; scaling never profiled.
- **SSIM cost on ≥256-sq tensors** despite kernel caching.

Also resolved 2026-05-21 (post-V5-pivot cleanup):
- `models/uncertainty.py` arbitrary-`confidence_level` fallback rewritten to use `torch.erfinv` correctly.
- `models/predictor.py` vestigial `input_up` `ConvTranspose2d` removed.
- `models/attention.py` `TemporalAttention` now raises clear error when `T > t_max`.
- `solarflare_data/harp_loader.py` `_compute_per_cube_norm` now filters artefact-zero pixels from σ.
- `solarflare_data/dataset.py` `_apply_augmentation` honours `is_pseudoscalar` for D4 chirality sign-flip.
- `main.py` reads `data.loader: harp_zarr` and routes through `load_harp_zarr_data`.

## Ablation matrix (queued)

21-arm matrix lives under `configs/ablations/`. Each YAML is standalone (no runtime merge). All arms train from scratch (no `transfer_learning`) so each result reflects the architecture / policy alone. Two families: **A/B** (deep 6-layer `SolarFluxPredictor` flag ablations) and **S** (minimal 2-layer `SimpleConvLSTM` baseline series).

### A/B family — deep model (`SolarFluxPredictor`)

| Arm | Δ vs baseline | Question |
|---|---|---|
| `A0_baseline` | none (production) | reference |
| `A1_win64` | window 64×64 (vs 128) | smaller spatial scale → more samples; better? |
| `A2_win192` | window 192×192 | larger field; drops small cubes |
| `A3_signed_asinh` | per-cube `sign·asinh(\|x\|/s)` (vs linear zscore) | heavy-tail compression helps? |
| `A4_no_sa_convlstm` | vanilla ConvLSTM (no SAM) | SA memory net positive? |
| `A5_no_temporal_attn` | no `TemporalAttention` | tempattn earns its params? |
| `A6_no_attn_gate` | no `AttentionGate` | attngate earns its params? |
| `A7_channels_small` | `[16,32,64]` (~4× fewer params) | underfit or close the gap? |
| `A8_kernel3` | `kernel_size=3` (vs 5) | smaller field suffices? |
| `A9_tstride1` | temporal stride 1 (vs 4) | denser sampling → better val? |
| `A10_aug_aggressive` | full D4 incl rotations (chirality-aware) | rotations help? |
| `B0_convlstm_pure` | all 3 attention add-ons off (SA + tempattn + attngate) | does pure ConvLSTM beat the full attention stack? |

### S family — minimal `SimpleConvLSTM` (2-layer, hidden 64, k3, 1ch flux)

Built incrementally: each S-arm adds one change to chase the **flare-CSI≈0** failure. Root cause (confirmed S0/S1): L1 on a zero-mean heavy-tailed field makes "predict the smooth mean" optimal, so `|pred|` never crosses the extreme threshold (0.528 z) → TP=0 → CSI=0.

| Arm | Δ vs prior S-arm | Question / result |
|---|---|---|
| `S0_simple_convlstm` | minimal baseline, L1, D4 aug | match the deep model? → **collapse to ≈0 field, CSI 0** |
| `S1_simple_convlstm_noaug` | S0 minus augmentation | is aug the bottleneck? → **no, same collapse; persistence CSI 0.29 wins** |
| `S2_simple_convlstm_residual` | S0 + residual decode (frame = prev + Δ) | anchor to persistence vs collapse? → **explodes (var ratio 18→764, no TF)** |
| `S4_simple_convlstm_residual_tf` | S2 + teacher forcing (tf 1.0→0) | curb autoregressive explosion? → **cured (var ratio ~1); TEST CSI 0.099 HSS 0.178 SSIM 0.438** |
| `S5_simple_convlstm_alldata` | S4 + GroupNorm + 90/5/5 split (~all data) | more data lift CSI? → **SSIM 0.70, persSkill 30%, but CSI still ~0 (smooths harder)** |
| `S3_simple_convlstm_composite` | S5 + composite loss (extreme_pixel_weight 25 + asymmetric α5 + SSIM) | does tail-reweighting break L1 mean-collapse → lift CSI without var-ratio explosion? → **best val CSI 0.0236 (ep13, full 15ep) ≈ S6 extreme-only ⇒ SSIM/asym/temporal add ~0. var_ratio 0.037 (still smoothed). old-split test CSI 0.** |
| `S6_simple_convlstm_extremeonly` | S5 + composite with ONLY extreme_pixel_weight 25 (SSIM/asym/temporal off) | which composite term drives S3's lift? → **val CSI 0.0215 ≈ S3 ⇒ extreme_pixel_weight IS the active ingredient (SSIM/asym/temporal add ~0). old-split test CSI 0 (flareless cube).** |
| `S8_simple_convlstm_zscore_fixedtest` | S5 (L1+zscore) on the **fixed informative test set** [245,274,49] | matched control for S7 → **TEST CSI 0.0033 vs persistence 0.043 — model LOSES to persistence ~13×. var_ratio 0.19, SSIM 0.71.** |
| `S7_simple_convlstm_signed_asinh` | S8 + `signed_asinh` norm (heavy-tail compression), eval thr 0.0346 | does tail-compression keep field sharp vs zscore+L1? → **TEST CSI 0.0042 ≈ S8 0.0033, both ≪ persistence 0.04. Hypothesis REJECTED. SSIM 0.98 is asinh-space artifact (not comparable), var_ratio early-exploded 8.8→settled 0.30.** |
| `S9_simple_convlstm_s4_fixedtest` | S8 but `norm_type=batch` (= "S4 on fixed test") | is BatchNorm the sharpness driver? → **YES, via OVER-variance: var_ratio 3.7 (val hit 19) vs GroupNorm 0.19. TEST CSI 0.0118 — best model CSI on fixed test (3.5× S8) but still ≪ persistence 0.043. Unstable: SSIM collapsed 0.51→0.09, early-stop ep5. S4's visual "sharpness" = uncontrolled variance, not forecast skill.** |
| `S10_simple_convlstm_fast_tf` | S8 + `tf_decay_epochs=5` + patience 8 (complete TF curriculum) | does free-run training cure t+3/t+4 drift + lift CSI? → **var_ratio bumped 0.75 @ep7 then L1 dragged back to 0.028 @ep15. TEST CSI 0.0027 ≈ S8. TF fix necessary-but-insufficient; L1 defeats it long-run.** |
| `S11_simple_convlstm_fasttf_extreme` | S10 fast-TF + S6 extreme_pixel_weight (both levers) | do stacked levers beat persistence? → **TEST CSI 0.0176 — BEST of whole S-series (5× S8), additive, but still 2.4× below persistence 0.043. var_ratio re-collapsed 0.036.** |
| `S12_simple_convlstm_dual_head` | S11 + **classifier head** (`enable_classifier_head=True`), α=β=1 BCE pos_weight 60 | does explicit per-pixel extreme classifier escape L1 smoothing? → first structural pivot. Original (cosine LR) killed mid-run for LR study; rerun on mini (id 76) in progress. |
| `S13_simple_convlstm_dual_posweight100` | S12 + pos_weight 100 + **constant LR** (no cosine) | crank class-imbalance weight → cross persistence? → **TEST CSI (classifier) 0.0434, HSS 0.0749 — MATCHES persistence_csi 0.043 for the first time.** Regression head still dead (test CSI 0.0090). Pers skill 35.7%. Val_loss 0.151→0.149 monotone-ish, CLS-CSI 0.003→0.007. |
| `S14_simple_convlstm_dual_focal` | S12 + focal loss (γ=2, α=0.25) over BCE | does focal-loss shape lift the classifier past S13? → **DEAD: CLS-CSI=0 every epoch. Focal over-suppresses positives at this imbalance (~0.1%).** |
| `S15_simple_convlstm_dual_classifier_dominant` | S12 + α=0.1 (regression weight 10× lower) | does freeing the regression budget for the classifier help? → queued (id 78 retry, awaiting CUDA). |
| `S16_simple_convlstm_dual_extreme_weighted` | S13 + **extreme_pixel_weight 25** (S11 lever) + pos_w 60 | joint-train S11×S13 hybrid → **test CLS-CSI 0.0562 (+31% over persistence), staircase var_ratio 0.61, MAE 2.13e8 — staircase champion.** |
| `S18_simple_convlstm_dual_extreme_pw100` | S16 + pos_w 60→100 | classifier weight crank → reg head dead; staircase unstable. REJECTED. |
| `S20p5_dilated_gates` | S16 + dilated conv gates (d=2) | larger RF → CLS-CSI 0.0548 (-2% vs S16), staircase var_ratio 0.54. NEUTRAL. |
| `S22_orthogonal_init` | S16 + orthogonal recurrent init | did not converge; no test_results. INCOMPLETE. |
| `S30_stratified_sampler` | S16 + density-proportional window sampler | CLS-CSI 0.0545 (-3%), reg CSI 0.044 > persistence, staircase var_ratio 3.2. TIES. See [[s30_density_sampler_result]]. |
| `S30b_sampler_plus_fixes` | S30 + 6-hyperparam bundle | CLS-CSI 0.027 (-52%). REJECTED — don't bundle >2 knobs. See [[s30b_six_fix_bundle_toxic]]. |
| `S31_masked_loss` | S16 + masked loss (`base_pixel_weight=0.01`) | single-step CLS-CSI 0.0582 (+3.6%) but staircase var_ratio 30, MAE 4.5e9. REJECTED — single-step illusion. See [[s31_masked_loss_new_best]]. |
| `S33_S16_val_fix` | **S16 + val_cubes=[harp_8, harp_54]** (silent harp_17→flare-bearing); **S32_histogram_match** NOT RUN | val now tracks test CSI. ep4: CLS-CSI 0.0802, var_ratio 30 (AR-broken). **ep12 (promoted): CLS-CSI 0.0714, var_ratio 2.46, no drift.** Val still single-step biased. See [`findings_val_cube_fix.md`](findings_val_cube_fix.md). |
| `S20_depthwise_widen192` | S16 + depthwise-sep gates + hidden 64→192 (matched params) + val_cubes pin | ep2: CLS-CSI 0.0803, staircase var_ratio 11.5 (AR-broken). **ep10 (promoted)**: staircase MAE 2.52e8, var_ratio **1.24** (closest-to-ideal of any arm), drift_r -0.29. Single-step CSI for ep10 pending reeval. |
| `S34_S33_tin20` | S33 + t_in 10→20 (4hr lookback) | CLS-CSI 0.0822, persistence_csi rose to 0.0647 (more info for trivial baseline too) → +27% lift. **Both ep2 and ep10 AR-broken** (var_ratio 0.47, 22.78). **Longer input window HURT staircase.** |
| `S35_S33_tin30` | S33 + t_in 10→30 (6hr lookback) | CANCELLED mid-ep4 after S34 negative result. |
| `S36_physics_smoothness` | S33 + temporal-grad-matching loss (λ=0.5) + `early_stop_metric: ar_composite` | CLS-CSI 0.0739 (+23% over persistence), reg CSI 0.033, SSIM 0.535. tgrad weight 0.5 too weak — train_tgrad 3.5→4.3 actually INCREASED. AR gate selected ep7 (var 0.45). Solid but trails S16/S31. |
| `S37_event_detection` | S33 + α=0 (regression silent) + ar_composite | CLS-CSI 0.0865 NEW HIGH (+45% lift) BUT regression head runaway (SSIM 0.018, persistence skill -541%). α=0 toxic for joint task — classifier learns, decoder loses gradient. Senior ML #1 reframe disproven. |
| `S42_S16_new_data` | S16 + **expanded dataset (21→28 cubes, +7 new HARPs: 1028, 10769, 11149-Gannon, 2137, 2748, 833, 892, +33 X events)**, val=[harp_10769] | KILLED ep5 — val cube harp_10769 has 0 X-events, silent-val bug repeated. CLS-CSI plateaued 0.04-0.05, train loss decreased monotonically (decoupled signal). |
| `S42b_S16_new_data_xval` | S42 + val_cubes=[harp_2748] (6 X + 6 M + 36 C) + grad_clip 0.5→1.0 | 15 epochs early-stopped. **Test (ep6): CLS-CSI 0.0555 (+40% lift over new-data persistence 0.0395), SSIM 0.713, reg CSI 0.019 (HALF persistence — mode collapse).** var_ratio 0.55 test, 0.24 val. Val_loss gate selected mode-collapsed ckpt (ep6 var 0.24); ep5 (var 0.44, CLS-CSI 0.0497 val) was better AR but val_loss missed it — single-step illusion pattern again. Grad max climbed 2521→26260 across epochs (X-event amplitudes). |
| `S43_S20_arch_new_data` | S42b + S20 arch (depthwise + hidden 192) | DONE 12 ep (early-stop pat=8). Best val_loss=0.4058 ep4. CLS-CSI peak 0.0481 ep9, ep11/12 lost (logging-bug, fixed in reporting.py). var_ratio oscillates 0.14-0.60 (TF=0 shock ep5-6). **Single-step illusion: val-gate selected ep4 but ep8/9 had higher cls_csi.** 0.90M params. |
| `S44_S43_k5_hidden256` | S43 + kernel 3→5 + hidden 192→256 + AMP + batch 8→6 | DONE 11 ep (early-stop pat=8). **Broke val_loss=0.40 plateau: ep3=0.3987** (first on new data). CLS-CSI peak **0.0540 ep5** but val_loss gate saved ep3 (var_ratio=0.27, mode-collapsed) — single-step illusion. ep11 test: CLS-CSI 0.0470, HSS 0.0832, reg-CSI 0.0170, SSIM 0.651. Saved: best_model.pt (ep3) + ep11 final. 1.96M params. |
| `S45_S44_window128_aug` | S44 + window 64→128 + D4 balanced aug + batch 6→2 | queued (id=128). Uses new `any_best_of` early-stop gate (tracks val_loss, ar_composite, cls_csi, persistence_skill simultaneously; patience=5) — first arm to use this. |
| `S46_lowpass_asinh1e3` | S16 + signed_asinh softening 1e3 (EDA-proven) + low-pass loss (avg_pool2d L1 on block means) | CLS-CSI 0.0473, staircase var_ratio 0.001 (flat). Norm+lowpass bought nothing over tiny S16. |
| `S47_pool4_tin25` | 4× spatial avg-pool (pixel 0.36→1.44 Mm, covers >56 Mm predictable band), t_in 25 | **CLS-CSI 0.0567 — only axis that beat S16 (+8%)**, staircase var 2.29, AR drift +0.65 bounded. Pooling, NOT capacity, is the lever. |
| `S48_pool4_tin25_tfdecay12` | S47 + slow TF decay (tf_decay_epochs 5→12) | CLS-CSI 0.0573 (best single-shot) **BUT EXPLODES in AR rollout** (flux −1.7e5→−5e11). Slow TF decay = exposure bias. REJECTED. Don't raise tf_decay_epochs. |
| `S49_pool4_ss_perstep` | S47 + per-step scheduled sampling (TF ramps down across decode rollout) | curve corr +0.059 ± 0.373 (6-window), vr 0.73 — within noise floor. |
| `S50_pool4_s16arch` | S47 data + S16 tiny arch (h64 k3, ~0.4M) | curve +0.073 ± 0.371, vr 1.34. Pooled-tiny ties pooled-big → **capacity is dead weight**. |
| `S51_s50_trajloss` | S50 + `spatial_mean_weight=10` trajectory loss (L1 on per-frame spatial-mean \|field\|) | **BEST single-shot model: CLS-CSI 0.0656, reg 0.0389 (both records).** Curve corr still in noise floor — trajectory loss did not clear the ceiling. |
| `S52_longroll_tout16` | t_out 4→16 (train at the eval horizon) | CLS-CSI 0.0467, **test MAE flat across all 16 frames (no divergence)**. Robust 38-window curve corr: 16f +0.008 CI[−0.119,+0.144], 40f +0.048 CI[−0.049,+0.140] — **not distinguishable from 0 even at trained horizon**. Stabilises amplitude, not curve. |
| `S54_S16_maxdata` | S16 UNCHANGED + **maximal data** (27 cubes after archiving harp_11149; harp_245/274/49 folded into train → 24 train / 2 val / 1 random test; harp_11149 Gannon Storm archived as noisiest). `train_run.py` guarded to skip test on empty test set. | **DIRECT DATA-SCALING TEST: more data does NOT help forecasting.** 15 ep, best val ep7 (val_loss 2.998) — only 0.4% below ep2, then 8 stale epochs; val CSI noise-banded 0.16–0.22, CLS-CSI 0.21–0.24 whole run (no trend). Test (1 hyper-active random cube, NOT comparable to pinned [245,274,49]): reg-CSI 0.095 / cls-CSI 0.131, **per-step corr 0.05–0.06 (dead, = every prior arm)**, pers-skill 24.3%, var_ratio 0.205. The 0.13 cls-CSI is base-rate inflation (89k-flare-window cube), not a gain. Aleatoric ceiling confirmed a 4th way (after S5, S42b, pretrained FMs). S53 (test-pinned 22-cube precursor) cancelled before completion. |

Findings: [`findings_simple_convlstm_S0_S1.md`](findings_simple_convlstm_S0_S1.md), [`findings_S2_residual.md`](findings_S2_residual.md), [`findings_dual_head.md`](findings_dual_head.md), [`findings_architecture_research.md`](findings_architecture_research.md), [`findings_val_cube_fix.md`](findings_val_cube_fix.md), [`findings_new_data_expansion.md`](findings_new_data_expansion.md), [`findings_curve_ceiling.md`](findings_curve_ceiling.md).

**S-series verdict (updated 2026-06-06).** S0–S11 lose to persistence (best S11 0.0176). S13 MATCH (0.0434). S14 focal DEAD. **S16 = staircase champion** (CLS-CSI 0.0562 +31%, var_ratio 0.61, MAE 2.13e8). S20-ep10 = closest-to-ideal var_ratio (1.24); S20-ep2/S31/S33-ep4/S34 = single-step illusion (CLS-CSI ↑ but AR-broken). S36 physics smoothness +23% (tgrad weight too weak). S37 α=0 +45% lift but reg dead (toxic for joint task). **Bootstrap (1000×, n=74000):** S16 +30.6% [+8.7,+63.1], S33-ep12 +19.3% [+6.4,+35.8], S31 +35.3% [+12.9,+66.8]. All P(lift>0)≥0.998 — **all significant; CIs overlap → not distinguishable on single-step alone**. **S34 t_in=20 negative result:** longer input window HURTS staircase. **New 2026-06-04: dataset expansion — +7 HARPs (incl. harp_11149 = May 2024 Gannon Storm, 15 X events), +33 X-class events total (2→35, 17×). val cube switched to harp_2748 (X-rich) in S42b+; grad_clip relaxed 0.5→1.0 (X-event amplitudes were being throttled).** S42b test CLS-CSI 0.0555 (+40% lift but raw absolute lower than S16's 0.077 — single-step illusion + mode collapse). **S43 (S20 arch + new data) DONE 12 ep**: best val_loss 0.4058 ep4, CLS-CSI peak 0.0481 ep9 — matches S42b's classifier ceiling without breaking val_loss plateau. **S44 (k=5+hidden 256+AMP) DONE 11 ep**: BROKE PLATEAU val_loss=0.3987 ep3 (first <0.40 on new data). CLS-CSI peak 0.0540 ep5 (val_loss gate missed it — single-step illusion, saved ep3 var_ratio=0.27). Test (ep11 final): CLS-CSI 0.0470, reg-CSI 0.0170, SSIM 0.651. Two-strike reconcile patch confirmed working — controller marked done correctly on actual finish. **S45 pending requeue** (DB shows `failed` from pre-patch reconcile cascade). Logging bug fix (cls_csi now persisted to ckpt history) in reporting.py. CUDA-only policy.

**Curve-objective verdict (updated 2026-06-15).** Objective reframed to autoregressive staircase + spatial-mean winding-flux **curve** prediction (NOT single-step CLS-CSI), on **harp_11930 only** (245/274/49 have limb effect). EDA verdicts applied: signed_asinh softening 1e3 ≫ 1e6 (kurtosis 3 vs 9k); noise is spatial not temporal (12-min cadence fine); registration drift refuted; only the low-k (>14 Mm) envelope is forecastable. Arms S46–S52: **pooling (S47, +8%) is the only lever that beat tiny S16; capacity (S43/S44/S46) is dead weight** (S50 pooled-tiny ties pooled-big). S48 slow-TF-decay EXPLODES in rollout (exposure bias). **S51 = best single-shot (CLS-CSI 0.0656, reg 0.0389).** S52 longer-rollout (t_out16) flattens AR divergence (flat 16-frame MAE) = amplitude stability. **CEILING RESULT:** robust 38-window bootstrap shows EVERY arm at curve corr ≈ +0.045 with 95% CI spanning 0 — including S52 at its trained horizon. **Genuine predictability ceiling** (not model/loss/horizon): GT spatial-mean curve autocorr only 0.11–0.17 (near-white); spatial averaging integrates away the predictable low-k spatial pattern (pixel lag-1h corr 0.69 w/ blur). Stop iterating loss/arch/horizon for curve corr; reframe to spatial-pattern fidelity at short horizon OR a more autocorrelated scalar. **Data-scaling also ruled out (S54, 2026-06-26):** S16 arch on maximal data (24 train cubes vs 21) plateaus identically — val best ep7 = 0.4% over ep2, per-step corr still 0.05. More data is not the lever for forecasting (it only ever helped classification). Full writeup: [`findings_curve_ceiling.md`](findings_curve_ceiling.md). Report chapter: `docs/reports/s_series/curve_objective.tex`.

**Pretrained foundation-model check (2026-06-23).** Ran zero-shot TS foundation models on the harp_11930 spatial-mean curve (univariate, ctx 64, ~30-window bootstrap): TimesFM-2.5 +0.118 CI[+0.023,+0.221] vr 0.15 (marginally clears 0, ~15% amplitude — faint drift only); Chronos-bolt −0.109 (no skill, flat). TimesFM (curve-only) ≥ our full-field ConvLSTM → spatial field adds ~nothing; the +0.12 is just the 0.11–0.17 autocorr floor. **SOTA pretrained also hits the ceiling → data, not models.** Surya (NASA/IBM solar FM) not applicable (full-disk 4096² 13-ch SDO + flare-class head only). Probe `scripts/pretrained/curve_tsfm.py`, isolated `~/tsfm_venv` on 5060ti. Figure `outputs/ceiling/ceiling_with_tsfm.png`.

**CNN+LSTM hybrid + sharpened mechanism (2026-06-23).** Frozen INAF SigLIP2 flare-CNN → 768-d per-frame embedding → residual-LSTM (residual MLP block + persistence output skip) forecasts curve; trained 25 cubes, eval harp_11930. Ridge probe embedding→*instantaneous* curve R²=**0.748** (encoder captures curve — spatial rep NOT the bottleneck) but forecast corr +0.049 CI[−0.039,+0.137] H=16 (neg at H=4/8) ≈ ConvLSTM, < TimesFM. **Mechanism sharpened: curve is MEAN-REVERTING** (lag-1 autocorr 0.53 BUT climatology beats persistence +8%, differenced autocorr −0.40) → optimal forecast ≈ climatological mean (flat, = what Chronos/TimesFM do); mean-removed curve-corr objective measures residual deviations = near-white. Ceiling triply confirmed (our arms + pretrained TS FMs + CNN+LSTM hybrid). Infra `scripts/pretrained/cnn_features.py` + `cnn_lstm.py`, features `outputs/cnnfeat/`.

**Eval robustness.** Test cubes pinned to `[harp_245, harp_274, harp_49]` since 2026-05-26 (`data.test_cubes`). **Val cubes pinned 2026-06-02** to `[harp_8, harp_54]` via new `data.val_cubes` field (silent harp_17 had 0 GOES events; val_loss didn't track test CSI). Discriminator = per-cube extreme-pixel rate (|z|>0.528), orthogonal to GOES. Split impl: `solarflare_data/loader.py:assign_files_with_fixed_test`. Pre-fix S13–S32 selected by silent val; S33+ use informative val.

Generator: `configs/ablations/build_configs.py`. Per-arm output dir: `outputs/ablations/<arm_id>/`. Recommended minimum-viable subset if compute-constrained: **A0 + A4 + A5 + A6** (covers the architecture-flag story).

## V5 outputs archive

V5 run artefacts moved to `archive/v5_jepa/outputs/` (2.6 GB) on 2026-05-21:
- `outputs_v5_*/` — encoder checkpoints + `run.jsonl`
- `outputs_probe/` — wind-flux probe heads + per-cube calibration
- `outputs_flare/` — binary flare classifier heads + `temporal_eval.json`

Working tree clean of V5 outputs. Source code preserved on branch `v5-jepa-lora`.

## Testing (`tests/`)

13 test files: `test_attention.py`, `test_center_crop.py`, `test_checkpoint.py`, `test_config.py`, `test_data_pipeline.py`, `test_device.py`, `test_losses.py`, `test_metrics.py`, `test_model.py`, `test_sa_convlstm.py`, `test_transfer_learning.py` (+ `__init__.py`, `conftest.py`). Counts per planning docs: **88 tests at v2.0 ship**, **117 after Phase 7-01** (+29 metric tests). No coverage number tracked. `CONCERNS.md` and `TESTING.md` (both 2026-02-02) claimed `tests/` did not exist — those audits are STALE relative to the current tree.

## Output dirs / Files reference

- Per-arm: `outputs/ablations/<arm_id>/` (checkpoints, metadata, test_results, training_history).
- Remote (5060ti): `/home/indra/solarflare/outputs/ablations/<arm_id>/`. Local `find outputs/` won't see remote runs.
- Planning: `.planning/{STATE,PROJECT,MILESTONES,ROADMAP}.md`, `.planning/codebase/{ARCHITECTURE,CONCERNS,CONVENTIONS,STACK}.md`, `.planning/phases/*-SUMMARY.md`.
- Tests: 13 files under `tests/` (88 at v2.0 ship, 117 after Phase 7-01).
- Comparison: `COMPARISON.md`, `generate_comparison.py`, `utils/metrics.py`.
