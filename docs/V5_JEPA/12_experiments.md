# 12 — Experiment Log

All V5 training runs. Source of truth for val curves, configs, and per-run findings.
Config scale: **sanity** = dim 192, 4 cubes, t_in=4/t_out=2. **path_a** = dim 384, 10 cubes, t_in=10/t_out=4.

- Hard truths (slow-curriculum CONFIRMED, 1e8 clip, etc.) → `12_experiments_findings.md`.
- Older / stale detail (E01–E08, E10) → `12_experiments_archive.md`. Summary rows below stay authoritative.
- Entry-point hub: `INDEX.md`.

---

## Summary table

| ID | Date | Config | Device | Epochs | Best val | Converged? | Notes |
|---|---|---|---|---|---|---|---|
| E01 | 2026-05-08 | sanity, no-mask | MPS | 1 | 0.2018 | — | First green. Pipeline + EMA valid. |
| E02 | 2026-05-08 | sanity, no-mask | CUDA 5060ti | 5 | 0.1848 (ep0) | No | Plateau ep≥1. Overfit tiny config. |
| E03 | 2026-05-08 | sanity, no-mask, 1e8 clip | MPS | 5 | **0.0407** (ep4) | Partial | 5× better than E02. Clip fix. |
| E04 | 2026-05-08 | sanity, mask-ON | MPS | 5 | 0.1950 (ep4) | No | Mask catalog first run. Still falling. |
| E05 | 2026-05-10 | sanity, mask-ON, 50ep | MPS | 50 | **0.0689** (ep49) | No | 18-ep plateau. Fast curriculum. → archive. |
| E06 | 2026-05-10 | sanity, mask-OFF, 50ep | MPS | 50 | **0.0125** (ep38) | Yes | Saturated ep24+. Tiny dataset limit. → archive. |
| E07 | 2026-05-10 | path_a, mask-ON, 20ep | CUDA | 6 | 0.1272 (ep5) | No | Tmux died ep6. Resumed as E08. → archive. |
| E08 | 2026-05-11 | path_a, mask-ON, resume | CUDA | +14 | — | Stale | Blocked by 5060ti contention. → archive. |
| E09 | 2026-05-11 | sanity, mask-ON, 100ep slow | MPS | 98 | **0.00831** (ep98) | **Yes** | Slow curriculum CONFIRMED. See F1 in findings. |
| E10 | 2026-05-11 | sanity, tube-only, 100ep slow | MPS | 49 | 0.0414 (ep34) | No | Killed mid-warmup. Superseded by E12. → archive. |
| E11 | 2026-05-12 | wind-flux probe on E09 | — | — | linear R²=0.45 (val); novel medAPE 46% | — | Stage-1 probe. See section. |
| E12 | 2026-05-12 | sanity, tube-only, 100ep slow | CUDA 5060ti | 99 | **0.04017** (ep41) | No (diverged) | 5× worse than E09. Tube collapse F3. |
| E13 | 2026-05-12 | sanity, tube+future, 100ep slow | CUDA 5060ti | 100 | **0.01812** (ep92) | Partial | Tube+future best of ablation. ~2.2× E12. F3 follow-up. |
| E14 | 2026-05-12 | sanity, tube+cross, 100ep slow | CUDA 5060ti | 100 (resumed) | **0.01172** (ep91) | Yes | Resumed post host-reboot; full 100ep finished. Cross_time alone < uniform mix. |
| E15 | 2026-05-13 | sanity, uniform mix, 100ep slow | CUDA 5060ti | 100 | **0.00530** (ep99) | **Yes** | **New SOTA.** 1.57× E09, 3.4× E13. Tube+future+cross uniform mix wins. F11. |
| E16 | 2026-05-12 | mini-mps, dim=256/L=6, 21 cubes, 30ep | MPS M4 Pro | 5 | 0.237 (stalled) | No | **STALE** — last log 2026-05-12 21:29, ep5 only. Slot preempted by E17-E24 sweep. Relaunch decision pending. |
| E17 | 2026-05-13 | sanity, EMA τ=0.990, 100ep slow | mini_mps | 100 | **0.00761** (ep99) | **Yes** | Lower τ beats E09 anchor (0.00831). EMA sweep arm. |
| E18 | 2026-05-13 | sanity, EMA τ=0.994, 100ep slow | mini_mps | 100 | **0.00938** (ep99) | **Yes** | τ=0.994 worse than E09 anchor (0.00831). τ↓ trend confirmed. |
| E19 | 2026-05-12 | sanity, EMA τ=0.998, 100ep slow | mini_mps | 46 (running) | 0.03740 (ep46) | Running | EMA-decay sweep. Still descending. |
| E20 | 2026-05-14 | sanity, EMA τ=0.9995, 100ep slow | CUDA 5060ti | 100 | **0.03010** (ep99) | **Yes** | τ=0.9995 worst of sweep. Monotonic τ↓-better confirmed. |
| E21 | 2026-05-12 | sanity, mask ratio=0.60, 100ep slow | mini_mps | — | — | SUPERSEDED | Anchor E09 mix; supplanted by E25 on uniform mix. Cancel from MPS queue. |
| E22 | 2026-05-12 | sanity, mask ratio=0.75, 100ep slow | mini_mps | — | — | SUPERSEDED | → E26. |
| E23 | 2026-05-12 | sanity, mask ratio=0.85, 100ep slow | mini_mps | — | — | SUPERSEDED | → E27. |
| E24 | 2026-05-12 | sanity, mask ratio=0.90, 100ep slow | mini_mps | — | — | SUPERSEDED | → E28. |
| E25 | 2026-05-14 | sanity, ratio=0.60, uniform mix, 100ep | CUDA 5060ti | 100 | **0.00960** (ep99) | **Yes** | Mask ratio sweep — starved end of range. |
| E26 | 2026-05-14 | sanity, ratio=0.75, uniform mix, 100ep | CUDA 5060ti | 100 | **0.00876** (ep99) | **Yes** | **Ratio winner.** Marginal vs E15 (r=0.80 0.00530); slightly better than E25. |
| E27 | 2026-05-14 | sanity, ratio=0.85, uniform mix, 100ep | CUDA 5060ti | 100 | **0.01597** (ep99) | **Yes** | Past peak — ratio too high. |
| E28 | 2026-05-14 | sanity, ratio=0.90, uniform mix, 100ep | CUDA 5060ti | 36 | 0.05798 (ep36) | TERMINATED | Killed ep36 — not needed. Trajectory worse than E27 at every epoch; concave shape confirmed at r=0.75. Saved ~2h to launch thesis run. |
| E29 | 2026-05-14 | **THESIS attempt 80ep**: path_b full (dim=384, 12L, 21 cubes), 80ep, τ=0.990, ratio=0.75, uniform mix, slow curric | CUDA 5060ti | 1 | 0.4093 (ep0) | KILLED 21:54 | Actual rate 4.9s/step (vs 2s estimate) — total ETA 11.5d, misses May 19 by 7d. CUDA per-epoch flat (E15 confirms no warmup spike). Replaced by E29b. Cfg: `v5_thesis.yaml`. |
| E29b | 2026-05-14 | **THESIS 3-day**: same winners, compressed: 30ep × 1000 steps/ep (was 80×2000). Total opt steps 30K. | CUDA 5060ti | 17 | 0.00977 (ep5 tail-mask fluke); plateau 0.034 ep12-17 | KILLED 2026-05-16 | Plateau confirmed mid-mask broadening — 21 cubes / 30K steps = 1430 steps/cube, 3.5× thinner than E15 sanity. Replaced by curated E30. |
| E30 | 2026-05-16 | **THESIS curated** (scaled v2): 13 train / 3 val / 5 holdout, 100ep × 2000 steps. Same winners (τ=0.990, r=0.75, uniform mix, slow curric). 15.4K steps/cube (3× E15 sanity 5K). | CUDA 5060ti | 100 | **0.002680** (ep99) | **Yes** | **SOTA — CONFIRMED 2026-05-19 21:26 IST.** 49.4% below E15 floor (0.00530), 67.7% below E09 anchor (0.00831). Monotonic descent ep57→99 (no overfit). s/step 0.842 locked all 100ep. Holdout: harp_17, harp_51, harp_may2024, harp_nov2025, harp_245. Cfg: `v5_thesis_curated.yaml`. |
| E31 | 2026-05-16 | **LOW-DIM long**: dim=192/L=6 enc, L=4/d=192 pred, 19 train cubes, 80ep×1000 steps. Cross-attn predictor toggle (Option A). Same winners (τ=0.990, r=0.75, uniform mix, slow curric). | mini_mps | 20 (resumed) | 0.0647 (ep20) | Running | Cross-attn fix at ep20 — see F12. Cfg: `v5_e31_lowdim_long.yaml`. ETA ~3d. |
| E30-probe | 2026-05-19 | wind-flux probe on E30 v2 features (linear + MLP, dim=384, splits mirror encoder). | mac mini CPU | 60 (linear ep17 best, MLP ep2 best) | val R²=**0.700/0.729** r=**0.84/0.87**; novel R²=+**0.17/+0.23** r=**0.43/0.50** | — | Stage-2 probe. **Novel transfer flipped from random (E11 r=0.06) to meaningful (r=0.50).** Detail: [`12_E30_thesis.md`](12_E30_thesis.md#stage-2-wind-probe-on-e30-2026-05-19--done). |

---

## E09 — 100-ep mask-ON slow curriculum (MPS, sanity) — **COMPLETED**

**Config:** `configs/v5_mini_mask_on_100ep_slow_curric.yaml` (4 cubes: harp_17/45/51/83, dim=192).
**Device:** mini_mps. **Date:** 2026-05-11. **Duration:** ~10.1 h. **Status:** ep98/100 (effectively converged).

Key change vs E05:

| Param | E05 | E09 |
|---|---|---|
| `tail_only_pct` | 0.10 → ep5 | **0.25 → ep25** |
| `warmup_pct` | 0.20 → ep15 | **0.40 → ep65** |
| Total epochs | 50 | **100** |

| ep | val | ep | val |
|---|---|---|---|
| 0 | 0.210 | 80 | 0.0149 |
| 25 (tail→blend) | ~0.05 | 90 | 0.0100 |
| 65 (full mix) | ~0.025 | 95 | 0.00875 |
| 79 | 0.0152 | 98 | **0.00831** ← best |

**Best: 0.00831 ep98.** Monotonic descent ep79→98 (no plateau).

**Curves:** `figures/E09_mask_on_100ep_slow_loss.png`; cross-experiment compare in `figures/all_val_compare.png`.

> **Regenerate all loss-curve PNGs:** `PYTHONPATH=. python3 scripts/plot_loss_curves.py` → writes to `docs/V5_JEPA/figures/`.

**Result.** Slow curriculum fix CONFIRMED (F1 in findings). No 18-ep plateau like E05; transition through full mix smooth. 8× better than E05 (0.0689), 1.5× better than E06 mask-OFF (0.0125) on harder mask-ON task.

**Limitation for thesis:** mini arch (dim=192, 4 layers, 4 cubes). Backbone ceiling for downstream probe unknown. E16 (dim=256/L=6, 21 cubes) tests this.

**Artifacts:** `outputs_v5_mini_mask_on_slow/{best,last}.pt` ep98 val 0.00831; `run.jsonl`.

---

## E11 — Wind-flux probe on E09 (Stage-1 feasibility, professor-meeting demo)

**Date:** 2026-05-12. **Backbone:** E09 mini ckpt (val 0.00831 ep98). **Target:** ⟨|wind|⟩(t), abs, clip 1e7. **Splits:** train `{harp_17,45,83}`, encoder-val `{harp_51}`, novel 17 cubes (encoder never saw). **Pool:** spatial mean of valid+non-pad tokens → `[T,192]` cached per cube. **Norm:** log10(y+1) z-scored on train cubes only. **Heads:** linear (193 params), MLP (D→256→1).

**Headline (linear):**
| Eval set | persist R² / medAPE | linear R² / medAPE |
|---|---|---|
| harp_51 (encoder val) | −0.20 / 22.6% | **+0.45 / 17.4%** |
| 17 novel | −0.93 / 12.7% | −0.00 / 46% |
| 3 train (sanity) | −0.02 / 23.2% | +0.20 / 24.5% |

**Stage-1b ceiling (XGBoost):** harp_51 R²=0.528 medAPE=16.1% (lifts linear). Novel aggregate flat (R²≈0). Train R² 0.20→0.61 — confirms **linear-capacity** was bottleneck on in-distribution; **off-distribution failure is calibration, not capacity**.

**Stage-1c rate-of-change (Δy = y(t)−y(t−1)):** **negative result**. Probe MAE > persist-zero MAE on every novel cube; harp_51 XGB R²=−0.11. Spatial-mean pool collapses inter-frame variance — Δ-targets need richer pooling.

**Result.** Frozen E09 features carry phase/temporal structure (median per-cube Pearson r ≈ 0.73 on novel cubes). Absolute scale not transferable from 3 train cubes. Per-cube affine calibration + richer pooling are the obvious Stage-2 levers. See finding F9.

**Artifacts:** `outputs_probe/{linear,mlp}/best.pt`, `outputs_probe/eval/{README.md,probe_metrics.md,probe_xgb.md,probe_delta.md,e09_jepa_curve.png,overlay_*.png}`.

---

## E12-E15 — Mask-policy ablation (CUDA, sanity, 100ep slow)

4-arm ablation isolating which mask policies drive E09's full-mix win. All mini (dim=192, 4 layers, 4 cubes), 100 epochs, slow curriculum (`tail_only_pct=0.25`, `warmup_pct=0.40`), bf16, CUDA 5060ti.

| ID | policy_mix | Config | Out dir |
|---|---|---|---|
| E12 | `{tube:1.0}` | `v5_mini_mask_tube_only_cuda.yaml` | `outputs_v5_e12_tube_only_cuda/` |
| E13 | `{tube:0.7, future:0.3}` | `v5_mini_mask_tube_future_cuda.yaml` | `outputs_v5_e13_tube_future_cuda/` |
| E14 | `{tube:0.7, cross_time:0.3}` | `v5_mini_mask_tube_cross_cuda.yaml` | `outputs_v5_e14_tube_cross_cuda/` |
| E15 | `{tube:0.34, future:0.33, cross_time:0.33}` | `v5_mini_mask_uniform_cuda.yaml` | `outputs_v5_e15_uniform_cuda/` |

### E12 — tube-only — **COMPLETED 2026-05-12**

~3h 25m, 100ep. Out: `outputs_v5_e12_tube_only_cuda/`. Best **0.04017 @ ep41**, then val rose 4× to 0.172 by ep99 (train kept falling 0.026). Full divergence past ep65 full-mix transition. Canonical conclusions → F3. Curve: `figures/E12_tube_only_cuda_loss.png`.

**E13/E14/E15 outcome:** all 3 pairs/trios stable past ep65. E15 trio best (0.00530), E14 cross_time pair stable (0.01172), E13 future pair stable (0.01812). Tube-only is the only divergent arm — confirms tube *requires* mix-companions, not just curriculum pacing.

### E13 — tube+future — **COMPLETED 2026-05-12**

~3h 30m. **Best val 0.01812 @ ep92.** Monotonic, no late divergence. Future-policy prevents tube-collapse. 2.2× better than E12; 2.2× worse than E09. Curve: `figures/E13_tube_future_cuda_loss.png`.

### E14 — tube+cross — **COMPLETED 2026-05-12**

ep0–40 first attempt, host reboot 21:34, resumed via `--resume last.pt`. **Best val 0.01172 @ ep91** (n_val=99). Monotonic. Cross_time alone: 2.7× better than E12, 1.5× better than E13, 1.4× worse than E09, 2.2× worse than E15 — pair < trio.

### E15 — uniform mix — **COMPLETED 2026-05-13 — NEW SOTA**

Auto-launched by watcher on E14 run_end. Wallclock ~3h 25m, 100 ep, CUDA 5060ti. Out: `outputs_v5_e15_uniform_cuda/` (remote).

Final: **best val 0.00530 @ ep99**, monotonic descent through ep99. n_val=100, no missed epochs.

| ep | val | ep | val | ep | val |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.210 | 50 | ~0.013 | 92 | ~0.0061 |
| 25 (tail→blend) | ~0.045 | 65 (full mix) | ~0.0098 | 98 | 0.00535 |
| 41 | ~0.018 | 80 | ~0.0072 | **99** | **0.00530** ← best |

**Headline.** Uniform {tube:0.34, future:0.33, cross_time:0.33} beats E09 {0.5/0.3/0.2} by 1.57×; E13 by 3.4×; E14 by 2.2×; E12 by 7.6×. Tube over-weighted in E09; cross_time + future under-weighted. CONFIRMED → F11.

**Curve:** `figures/E15_uniform_cuda_loss.png`. **Implication:** Arm B (E21–E24 mask-ratio) anchor superseded — re-run with uniform mix. Wind probe should re-eval on E15 ckpt.

---

## E16 — Bigger-backbone arm (MPS M4 Pro, 21 cubes, 30ep) — **running**

**Config:** `configs/v5_mini_path_a_mps.yaml` (encoder dim=256/L=6, predictor L=4/hidden=256; 8.8M trainable / 14.3M total).
**Device:** mini_mps (M4 Pro, 64 GB unified). **Date:** 2026-05-12. **PID:** 77660. **Out:** `outputs_v5_mini_path_a_mps/`.

**Why this run:** 5060ti unavailable (trading-sim contention). Tests capacity ceiling on the *real* 21-cube data — E09 backbone (dim=192/L=4) on 4 cubes left two open questions: (a) does wider/deeper backbone beat E09's val 0.00831 ceiling, (b) does training on 21 cubes drop probe novel-cube medAPE below E11's 46% / 21.5%-post-affine.

**Setup:** 17 train / 4 val cubes (`split_seed=0`, by_harp_id), t_in=6 / t_out=2, slow curriculum, bf16, grad_checkpoint=on, bucket_by_shape=on, batch=1 grad_accum=4, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.

**Wall-clock plan:** `max_steps_per_epoch=300`, ~2.7 s/batch steady on MPS. 300 × 4 (accum) × ~2.7 s ≈ 54 min/ep × 30 ep ≈ **27 h ≈ 1.14 d**.

**Pending:** probe re-eval on novel cubes after `best.pt` saves — direct compare with E11 medAPE.

---

## E17–E24 — EMA decay × mask ratio ablation (MPS mini, sanity, 100ep slow)

Two-arm sweep anchored on E09 (val 0.00831 ep98, τ=0.996, ratio=0.80, mix {0.5/0.3/0.2}). All sanity (dim=192, 4 cubes, slow curriculum, bf16). Submitted via controller queue 2026-05-12 23:20.

**Arm A — EMA decay** (ratio=0.80 fixed):

| ID | τ | Config | Out dir |
|---|---|---|---|
| E17 | 0.990  | `v5_e17_ema_0990.yaml`  | `outputs_v5_e17_ema_0990/` |
| E18 | 0.994  | `v5_e18_ema_0994.yaml`  | `outputs_v5_e18_ema_0994/` |
| E19 | 0.998  | `v5_e19_ema_0998.yaml`  | `outputs_v5_e19_ema_0998/` |
| E20 | 0.9995 | `v5_e20_ema_09995.yaml` | `outputs_v5_e20_ema_09995/` |

Prediction (per config header): faster τ → target collapses to context → val artificially low, probe R² drops. Slower τ → underfit target.

**Arm B — mask `target_ratio`** (τ=0.996, mix fixed):

| ID | ratio | Config | Out dir |
|---|---|---|---|
| E21 | 0.60 | `v5_e21_ratio_060.yaml` | `outputs_v5_e21_ratio_060/` |
| E22 | 0.75 | `v5_e22_ratio_075.yaml` | `outputs_v5_e22_ratio_075/` |
| E23 | 0.85 | `v5_e23_ratio_085.yaml` | `outputs_v5_e23_ratio_085/` |
| E24 | 0.90 | `v5_e24_ratio_090.yaml` | `outputs_v5_e24_ratio_090/` |

Prediction (per config header): starved mask (0.60) → cheap train, val degrades, probe worst. Excess mask (0.90) → over-regularized.

**Caveat**: B uses E09's mix; if E12-E15 unseats E09's mix, B may need re-run.

**E17/E18 DONE** — see summary table + F11/F12. Monotonic τ↓-better.

**Launch context**: First queue-dispatched runs to succeed. Prior 13 controller launches (May 8 onward, ids 21–35) died ~35s due to `$TMUX` leak (controller daemon ran inside its own tmux, subprocess spawned `tmux new-session` nested). Fixed `4db3259` (`env -u TMUX`). E17/E18 ran on mini_mps (~10.5h). E20/E25–E28 on 5060ti CUDA (~3.4h).

E30 v2 full detail → [`12_E30_thesis.md`](12_E30_thesis.md). E30-probe (Stage-2) results in same file.
