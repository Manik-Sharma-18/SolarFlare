# 12 — Experiment Log

All V5 training runs. Source of truth for val curves, configs, findings.
Config scale: **sanity** = dim 192, 4 cubes, t_in=4/t_out=2. **path_a** = dim 384, 10 cubes, t_in=10/t_out=4.

---

## Summary table

| ID | Date | Config | Device | Epochs | Best val | Converged? | Notes |
|---|---|---|---|---|---|---|---|
| E01 | 2026-05-08 | sanity, no-mask | MPS | 1 | 0.2018 | — | First green. Pipeline + EMA valid. |
| E02 | 2026-05-08 | sanity, no-mask | CUDA 5060ti | 5 | 0.1848 (ep0) | No | Plateau ep≥1. Overfit tiny config. |
| E03 | 2026-05-08 | sanity, no-mask, 1e8 clip | MPS | 5 | **0.0407** (ep4) | Partial | 5× better than E02. Clip fix. |
| E04 | 2026-05-08 | sanity, mask-ON | MPS | 5 | 0.1950 (ep4) | No | Mask catalog first run. Still falling. |
| E05 | 2026-05-10 | sanity, mask-ON, 50ep | MPS | 50 | **0.0689** (ep49) | No | Still falling at end. 18-ep plateau. |
| E06 | 2026-05-10 | sanity, mask-OFF, 50ep | MPS | 50 | **0.0125** (ep38) | Yes | Saturated ep24+. Tiny dataset limit. |
| E07 | 2026-05-10 | path_a, mask-ON, 20ep | CUDA | 6 | 0.1272 (ep5) | No | Tmux died ep6. Resumed as E08. |
| E08 | 2026-05-11 | path_a, mask-ON, resume | CUDA | +14 | — | In progress | Resume from E07 ep5 checkpoint. |
| E09 | 2026-05-11 | sanity, mask-ON, 100ep slow | MPS | — | — | Queued | Slow curriculum fix for E05 plateau. |

---

## E01 — First green (MPS, 1 ep, no-mask)

**Config:** `v5_sanity.yaml` (no mask). **Device:** MPS. **Date:** 2026-05-08.

train=0.187, val=0.202. Both finite. Validated: pipeline, EMA update, curriculum, bf16, MPS attn fallback.

---

## E02 — CUDA 5-ep sanity (no-mask)

**Config:** `v5_sanity.yaml`. **Device:** CUDA 5060ti. **Date:** 2026-05-08.

| ep | val |
|---|---|
| 0 | 0.1848 |
| 1–4 | plateau ~0.185 |

Overfits 3-cube training set by ep1. Expected at sanity scale.

---

## E03 — 1e8 clip rerun (MPS, 5 ep, no-mask)

**Config:** `v5_sanity.yaml`, `WIND_FLUX_CLIP=1e8`. **Device:** MPS. **Date:** 2026-05-08.

| ep | val |
|---|---|
| 0 | 0.2018 |
| 2 | 0.1050 |
| 4 | **0.0407** |

**Key finding:** winding-flux clip 1e5 → 1e8 gives 5× better val. Prior BZ_CLIP_GAUSS=1e5 was destroying real signal (per-pixel max ~1e7, clip 100× too tight). Baseline for no-mask path.

---

## E04 — Mask catalog first run (MPS, 5 ep, mask-ON)

**Config:** `v5_sanity.yaml`, `masking.enabled=true`, default mix `{tube:0.5, future:0.3, cross_time:0.2}`.
**Device:** MPS. **Date:** 2026-05-08.

| ep | train | val |
|---|---|---|
| 0 | 0.2345 | 0.2789 |
| 2 | 0.2237 | 0.2182 |
| 4 | 0.1988 | **0.1950** |

Monotone decreasing; finite throughout. Validated: mask path end-to-end, curriculum transition (tail→full mix), INTERSECTION loss-mask, MPS attn fallback under mask path. Val higher than E03 (expected — mask task harder + smaller loss denominator: only masked tokens graded).

---

## E05 — 50-ep mask-ON (MPS, sanity)

**Config:** `configs/v5_mini_mask_on_50ep.yaml` (dim=192, 4 cubes, mask-ON).
**Device:** MPS. **Date:** 2026-05-10. **Duration:** ~5.3 h.

| ep | val | ep | val |
|---|---|---|---|
| 0 | 0.258 | 25 | 0.113 |
| 6 | 0.108 | 31 | 0.100 |
| 7–31 | **plateau ~0.105–0.127** | 38 | 0.082 |
| 32 | 0.096 | 49 | **0.069** |

Best: **0.0689** ep49. Still falling at run end (~0.0008/ep).

**Finding — curriculum plateau:** 18-epoch stall ep7–31. Caused by mask curriculum transition too aggressive: `tail_only_pct=0.10` (tail-only ends ep5), `warmup_pct=0.20` (full mix from ep10). Model stalled adapting to tube+future+cross_time after ep10. Recovery ep32+ = eventually adapted.

**Not converged.** Would need ~100+ epochs to asymptote.

---

## E06 — 50-ep mask-OFF (MPS, sanity, baseline)

**Config:** `configs/v5_mini_mask_off_50ep.yaml` (dim=192, 4 cubes, `masking.enabled=false`).
**Device:** MPS. **Date:** 2026-05-10. **Duration:** ~5.3 h (sequential after E05).

| ep | val | ep | val |
|---|---|---|---|
| 0 | 0.071 | 15 | 0.023 |
| 4 | 0.043 | 24 | 0.015 |
| 6 | 0.031 | 38 | **0.0125** |
| 13 | 0.026 | 49 | 0.0128 |

Best: **0.01253** ep38. Plateau ep24–49.

**Finding — tiny-dataset saturation:** 3 training cubes memorized by ep24. No further gain. Mask-OFF at sanity scale informative only up to ~ep25.

**Losses NOT comparable to E05:** mask-OFF graded on all tokens; mask-ON graded on masked tokens (~80% ratio). Higher absolute val in mask-ON expected.

---

## E07 — path_a first run (CUDA 5060ti, 6 ep, mask-ON)

**Config:** `configs/v5_path_a.yaml` (dim=384, 10 cubes, t_in=10/t_out=4, 33M trainable).
**Device:** CUDA 5060ti. **Date:** 2026-05-10. **max_steps_per_epoch=2000**, 20ep target.

| ep | val |
|---|---|
| 0 | 0.231 |
| 3 | 0.179 |
| 5 | **0.127** |

Stopped at ep6 step=12030 (t=63s). Tmux session died (idle timeout / venv warning triggered CRASHED state). Not a training failure. Checkpoint saved: `outputs_v5/last.pt` (ep5).

---

## E08 — path_a resume (CUDA 5060ti, ep6→19)

**Config:** `configs/v5_path_a.yaml --resume outputs_v5/last.pt`. Resumes from ep6.
**Device:** CUDA 5060ti. **Date:** 2026-05-11. **Status:** running.

LR schedule resumes correctly: `global_step=12000/40000`, warmup done at step 4000, cosine position inherited.

---

## E09 — 100-ep mask-ON slow curriculum (MPS, sanity)

**Config:** `configs/v5_mini_mask_on_100ep_slow_curric.yaml`.
**Device:** MPS. **Date:** 2026-05-11 (queued, runs after E08 slot frees — wrong, mini_mps free).
**Status:** running on mini_mps.

Key change vs E05:

| Param | E05 | E09 |
|---|---|---|
| `tail_only_pct` | 0.10 → ep5 | **0.25 → ep25** |
| `warmup_pct` | 0.20 → ep15 | **0.40 → ep65** |
| Total epochs | 50 | **100** |

Hypothesis: slow transition prevents 18-ep stall seen in E05. Full mix starts ep65, giving 35 epochs in hardest regime.

---

## Key findings so far

1. **Clip 1e8 essential.** Winding flux ≠ B-field. Clip 1e5 destroyed real signal. Fixed in E03.
2. **Mask-OFF saturates fast** at sanity scale (~ep24). Not informative beyond that.
3. **Mask-ON curriculum needs care.** Fast transition = 18-ep plateau. Slow warmup hypothesis tested in E09.
4. **path_a scale (dim=384, 10 cubes) is the real signal.** Sanity runs (E05/E06) only test relative ablations. E07/E08 is where convergence matters.
5. **Losses across mask-ON/OFF not comparable** — different denominator (masked tokens only vs all).
