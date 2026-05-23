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

11-arm matrix lives under `configs/ablations/`. Each YAML is standalone (no runtime merge). All arms train from scratch (no `transfer_learning`) so each result reflects the architecture / policy alone.

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
