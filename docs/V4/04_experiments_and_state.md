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
| `S16_simple_convlstm_dual_extreme_weighted` | S13 + **extreme_pixel_weight 25** (S11 lever) + pos_w 60 (not 100) | joint-train S11×S13 hybrid (validated by S17 inference combo +8%) — does extreme-weighted L1 recover regression amplitude alongside classifier? → **running 5060ti ep5/15: val 0.124 ✓ (18% drop ep1→5), SSIM 0.89, pers skill +31%, CLS-CSI 0.001–0.002 (lower than S13).** |

Findings: [`findings_simple_convlstm_S0_S1.md`](findings_simple_convlstm_S0_S1.md), [`findings_S2_residual.md`](findings_S2_residual.md), [`findings_dual_head.md`](findings_dual_head.md).

**S-series verdict (updated 2026-05-29).** S0–S11 (regression-only) all lose to persistence 0.043 (best S11 test CSI 0.0176). **S13 dual-head (pos_w 100, constant LR) MATCHES persistence (test CSI 0.0434 = 0.043) — first S-arm to do so.** Output-parametrization pivot validated. S14 (focal) DEAD. **S16** (S13+extreme_pixel_weight 25, joint S11×S13 hybrid) running — early signal SSIM 0.89, pers skill +31% by ep5, classifier weaker. **S17 inference combo** (post-hoc `S11×(1+2σ(S13_logits))`) gave 8% MAE lift on harp_11930 — confirms hybrid hypothesis. **Staircase autoregressive** on harp_11930 (`scripts/s0_viz/_staircase_harp11930.py`): S3 mode-collapses to 100K within 4 frames; S11 explodes after step 9 to 2M+ — neither sustains chained rollouts. **Constant LR is default** (`configs/finetune_winding_flux.yaml`); cosine washed late-epoch gradient. **val_csi_classifier** printed per-epoch (`trainer/reporting.py:40`) — val_loss misleading for dual_head (dominated by L1). **Fixed-test note:** val CSI=0 across S7–S15 because val is 1 random (often flareless) cube; trust test CSI. **CUDA-only policy 2026-05-29:** new training arms default `--slot-pref 5060ti_cuda`; MPS slots only for low-priority confirmation runs (e.g. S12 rerun id 81).

**Eval robustness (2026-05-26).** The default random split ([train,test,val], seed 42) put a **near-flareless cube** in test (pers_csi 0.001) → flare CSI uninformative. Fix: `data.test_cubes` config field pins a fixed, informative test set (`harp_245, harp_274, harp_49` — top extreme-pixel rate, see `data/_extreme_rate_per_cube.json`; `harp_8` excluded as pathological). Discriminator is **per-cube extreme-pixel rate** (|z|>0.528), which is **orthogonal to GOES events** (e.g. `harp_11930` has 14 M/X flares but low pixel-rate; `harp_17` high pixel-rate, 0 GOES) — GOES is reserved for the separate event-based-CSI reframe. Split impl: `solarflare_data/loader.py:assign_files_with_fixed_test`. S0–S6 ran on the old random split; re-run on `test_cubes` for clean cross-arm CSI.

Generator: `configs/ablations/build_configs.py`. Per-arm output dir: `outputs/ablations/<arm_id>/`. Recommended minimum-viable subset if compute-constrained: **A0 + A4 + A5 + A6** (covers the architecture-flag story).

## V5 outputs archive

V5 run artefacts moved to `archive/v5_jepa/outputs/` (2.6 GB) on 2026-05-21:
- `outputs_v5_*/` — encoder checkpoints + `run.jsonl`
- `outputs_probe/` — wind-flux probe heads + per-cube calibration
- `outputs_flare/` — binary flare classifier heads + `temporal_eval.json`

Working tree clean of V5 outputs. Source code preserved on branch `v5-jepa-lora`.

## Testing (`tests/`)

13 test files: `test_attention.py`, `test_center_crop.py`, `test_checkpoint.py`, `test_config.py`, `test_data_pipeline.py`, `test_device.py`, `test_losses.py`, `test_metrics.py`, `test_model.py`, `test_sa_convlstm.py`, `test_transfer_learning.py` (+ `__init__.py`, `conftest.py`). Counts per planning docs: **88 tests at v2.0 ship**, **117 after Phase 7-01** (+29 metric tests). No coverage number tracked. `CONCERNS.md` and `TESTING.md` (both 2026-02-02) claimed `tests/` did not exist — those audits are STALE relative to the current tree.

## Output Directories (listing only)

- `outputs/` — `best_model.pt`, `checkpoints/`, `metadata.json`, `predictions.png`, `test_results.json`, `training_history.{json,png}`, `training_log.txt`, historic logs `training_30epoch_fullres.log`, `training_30epoch_old.log`, `training_5epoch_l1ssim.log`.
- `solarflare_results/` — earlier run: `best_model.pt`, `history.json`, `predictions.png`, `training_history.png`.
- `logs/` — `controller.log`, `experiment_controller.log`; stray `main_v5_*.log` are V5 contamination.
- Quick-task dirs in `.planning/quick/`: `001-model-architecture-doc`, `2-fix-broken-extreme-threshold-correct-val`, `3-fix-spatial-dimension-mismatch-by-center`.

## Files Reference

- `.planning/STATE.md`, `PROJECT.md`, `MILESTONES.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `IMPROVEMENT_NOTES.md`
- `.planning/codebase/{ARCHITECTURE,CONCERNS,CONVENTIONS,STACK,STRUCTURE,TESTING}.md`
- `.planning/phases/{01..11}-*/*-SUMMARY.md`
- `.planning/milestones/{v2-MILESTONE-AUDIT,v2.0-REQUIREMENTS,v2.0-ROADMAP}.md`
- `COMPARISON.md`, `generate_comparison.py`, `utils/metrics.py`
- `improvements.md` (516 LOC, overlaps `IMPROVEMENT_NOTES.md`), `architecture.md`, `README.md`
- `tests/` (13 `test_*.py`)
- `outputs/`, `logs/`, `solarflare_results/`
