# 12-archive — Experiment Log Archive

Detailed sections for early / stale runs moved out of `12_experiments.md` to keep
that file under the 200-line cap. Summary rows remain in the main file's table.

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

## E05 — 50-ep mask-ON (MPS, sanity) — fast curriculum

**Config:** `configs/v5_mini_mask_on_50ep.yaml` (dim=192, 4 cubes, mask-ON). **Device:** MPS. **Date:** 2026-05-10. **Duration:** ~5.3 h.

| ep | val | ep | val |
|---|---|---|---|
| 0 | 0.258 | 25 | 0.113 |
| 6 | 0.108 | 31 | 0.100 |
| 7–31 | **plateau ~0.105–0.127** | 38 | 0.082 |
| 32 | 0.096 | 49 | **0.069** |

Best: 0.0689 ep49. Still falling at run end (~0.0008/ep). Not converged.

**Finding — curriculum plateau (superseded by E09 slow curriculum):** 18-epoch stall ep7–31. Caused by `tail_only_pct=0.10 / warmup_pct=0.20` transitioning too fast. Recovery ep32+. See finding F7 in `12_experiments_findings.md`.

**Curve:** `figures/E05_mask_on_50ep_loss.png`.

---

## E06 — 50-ep mask-OFF (MPS, sanity, baseline)

**Config:** `configs/v5_mini_mask_off_50ep.yaml` (dim=192, 4 cubes, `masking.enabled=false`). **Device:** MPS. **Date:** 2026-05-10. **Duration:** ~5.3 h.

| ep | val | ep | val |
|---|---|---|---|
| 0 | 0.071 | 15 | 0.023 |
| 4 | 0.043 | 24 | 0.015 |
| 6 | 0.031 | 38 | **0.0125** |
| 13 | 0.026 | 49 | 0.0128 |

Best: 0.01253 ep38. Plateau ep24–49. Tiny-dataset saturation: 3 training cubes memorized by ep24. See finding F4.

**Losses NOT comparable to E05:** mask-OFF grades all tokens; mask-ON grades only masked tokens (~80% ratio). See F5.

**Curve:** `figures/E06_mask_off_50ep_loss.png`.

---

## E10 — 100-ep mask-tube-only (MPS, sanity, ablation) — killed mid-warmup

**Config:** `configs/v5_mini_mask_tube_only_100ep.yaml` (dim=192, 4 cubes, `policy_mix={tube:1.0}`, slow curriculum). **Device:** MPS. **Date:** 2026-05-11. **Status:** killed ep49 (mid-warmup, before full-mix transition).

| ep | val | ep | val |
|---|---|---|---|
| 0 | 0.165 | 30 | 0.0468 |
| 6 | 0.0905 | 34 | **0.04141** ← best |
| 15 | 0.0644 | 49 | 0.0487 |

Trailed E09 by ~5× at matched epoch. Killed before full-mix transition (ep65), so ambiguous whether tube-only would have diverged. **Superseded by E12** (same arm, completed 99 ep on CUDA, confirmed divergence — see F3).

**Curve:** `figures/E10_mask_tube_only_100ep_loss.png`.

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
**Device:** CUDA 5060ti. **Date:** 2026-05-11. **Status:** stale — blocked by trading-sim GPU contention; no remote jsonl tailed since.

LR schedule resumes correctly: `global_step=12000/40000`, warmup done at step 4000, cosine position inherited.
