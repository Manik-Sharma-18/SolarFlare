# 14 — Unswept Ablations: τ, mask ratio, patch size, t_in/t_out

**Status:** PLAN. Queued after E13–E16 settle.
**Created:** 2026-05-12.
**Anchor:** E09 recipe (sanity scale, 4 cubes, 100 ep slow curric, mask-ON full mix). val=0.00831 ep98 CONFIRMED (F1).

---

## TL;DR

Four hparams currently DEFAULTED from V-JEPA-2-AC paper with no in-project ablation:

1. **EMA decay τ** — locked at 0.996, never tested.
2. **Mask target ratio** — locked at 0.80, Q5 in `05_open_questions.md` flags 0.75 vs 0.90 open.
3. **Patch size** — locked at 16, never compared 8 / 32.
4. **t_in / t_out** — sanity uses 4/2, path_a uses 10/4; never isolated. Q2 in open questions.

Sweep order = expected effect-size × cost:
**τ first** (cheap, high prior effect) → **mask ratio** (after policy-mix winner from E13/E14/E15) → **t_in/t_out** (medium cost, capacity-relevant) → **patch size** (highest cost, biggest blast radius).

---

## Common protocol

All arms inherit E09 anchor unless noted:

```yaml
# Anchor (configs/v5_mini_mask_on_100ep_slow_curric.yaml)
encoder:    {dim: 192, layers: 4, heads: 4, mlp_ratio: 4}
predictor:  {layers: 3, hidden: 192, heads: 4, mlp_ratio: 4, block_causal: true}
patch_size: 16
mask:       {target_ratio: 0.80, tube_short_area: 0.15, tube_long_area: 0.40,
             policy_mix: {tube: 0.5, future: 0.3, cross_time: 0.2}}
curriculum: {tail_only_pct: 0.25, warmup_pct: 0.40}
training:   {lr: 3e-4, schedule: cosine_warmup, warmup_pct: 0.10, epochs: 100,
             batch: 1, grad_accum: 4, precision: bf16, target_ema_decay: 0.996}
data:       {t_in: 4, t_out: 2, cubes: [harp_17, harp_83, harp_45, harp_51]}
```

**Metrics (in order of importance):**

1. Best val smooth-L1-embedding loss + epoch of best.
2. Divergence behavior past curriculum transition (ep65+; F3 signature).
3. Train/val gap at best (overfit proxy).
4. Stage-1 wind-flux probe R² + medAPE on E11 protocol (after pretraining done; ~30 min add-on).

**Slot policy:** CUDA 5060ti preferred (~5 h/run; E12 = 3.5 h, E13 = 5 h). MPS Mac Studio as overflow (~10 h/run). Avoid Mac Mini MPS while E16 occupies it. CPU slots only if 5060ti queue is >12 h deep.

**Anchor reproduction first.** Every batch of arms must include a re-run of E09 anchor on the same slot/device to control for drift (host reboot, library version). Tag as `E_anchor_<batch>`. Pass = val within ±10% of 0.00831.

---

## Ablation A — EMA decay τ (4 arms)

**Why this matters:** target update speed gates whether the predictor sees a stable or a chasing target. Too-fast (τ low) → target collapses to context, trivial loss; too-slow (τ high) → underconverged in 100 ep. F1 single-point validation at 0.996.

**Predictions:**

- 0.990 — target moves fast; expect loss artificially low (predictor matches a moving but easy target). Watch probe R² — if pretraining loss looks great but probe collapses, target collapse confirmed.
- 0.994 — likely close to anchor.
- 0.998 — slightly stiffer target; predict modest val improvement, +5–15 ep to converge.
- 0.9995 — very stiff target; predict underconverged at 100 ep; loss curve still falling at ep99.

**Arms:**

| ID | τ | Config (new) | Slot |
|---|---|---|---|
| E17 | 0.990 | `v5_e17_ema_0990.yaml` | 5060ti_cuda |
| E18 | 0.994 | `v5_e18_ema_0994.yaml` | 5060ti_cuda |
| E19 | 0.998 | `v5_e19_ema_0998.yaml` | 5060ti_cuda |
| E20 | 0.9995 | `v5_e20_ema_09995.yaml` | 5060ti_cuda |

Anchor (0.996) = E09 reuse. Total cost ≈ 20 h CUDA.

**Decision rule:** lowest val that is monotone (no divergence) AND highest probe R²/lowest medAPE. If 0.990 wins val but loses probe, target collapse — keep 0.996. If 0.998 wins both, switch and re-anchor.

---

## Ablation B — Mask target ratio (4 arms)

**Why this matters:** sample efficiency vs task difficulty. 0.75 = MAE/Brain-JEPA default; 0.85–0.90 = V-JEPA-2 vision default. AR cubes are sparser than natural video — higher ratio may starve context.

**DEPENDS-ON E13/E14/E15:** lock policy mix to the winning combination before launching. If E13 (tube+future) wins, use `{tube:0.7, future:0.3, cross_time:0.0}`; if E14 wins, `{tube:0.7, cross:0.3}`; if E15 (uniform) wins, `{0.34, 0.33, 0.33}`. Do NOT run with E09's `{0.5, 0.3, 0.2}` if a cleaner winner exists.

**Arms:**

| ID | target_ratio | Notes | Slot |
|---|---|---|---|
| E21 | 0.60 | starve mask; predict train loss low, val high (data leakage from context) | 5060ti_cuda |
| E22 | 0.75 | Brain-JEPA default | 5060ti_cuda |
| E23 | 0.85 | midway V-JEPA-2 | 5060ti_cuda |
| E24 | 0.90 | V-JEPA-2 vision high end | 5060ti_cuda |

Anchor (0.80 + winning policy mix) = re-run as E_anchor_B.

**Predictions:**

- 0.60 → cheap loss, overfits context, probe likely worst.
- 0.85–0.90 → harder reconstruction; may need ep > 100 to converge but probe should improve if features generalize.

**Decision rule:** same as A — lowest val + highest probe R². If trade-off, weight probe (downstream task) 2× pretraining loss.

---

## Ablation C — Patch size (3 arms)

**Why this matters:** token count = (H/p)·(W/p)·t scales as 1/p². Patch 8 ≈ 4× tokens (16× attention FLOPs) vs patch 16; patch 32 ≈ 4× fewer. Smaller patches → finer spatial detail but quadratic attention cost; larger → cheap but may erase small-scale wind-flux structure (granule scale ~1–2 Mm; pixel = 0.364 Mm → patch 32 = 11.6 Mm, may smear sub-granule chirality).

**Compute risk:** sanity cubes vary ~120–400 px per side. Patch 8 on a 400×400 cube × t=6 frames = 6·50·50 = 15,000 tokens. May OOM at sanity scale on MPS; CUDA 5060 Ti has 16 GB and should fit.

**Arms:**

| ID | patch_size | Notes | Slot |
|---|---|---|---|
| E25 | 8 | 4× tokens; quadratic attn; pre-flight VRAM check required | 5060ti_cuda |
| E26 | 16 | anchor (E09 reuse) | n/a |
| E27 | 32 | 4× fewer tokens; cheap; may smear chirality | 5060ti_cuda or mini_mps |

**Pre-flight (required for E25):** dry-run 1 step with `--max-steps 1` on smallest cube (harp_45). Confirm peak VRAM < 14 GB. If OOM, drop t_in to 3 OR enable `grad_checkpoint: true` and note the deviation.

**Predictions:**

- p=8 → val better by ≥20% but 4–6× wall-clock. Check probe — finer patches may help wind-flux probe.
- p=32 → val worse; probe may collapse on small AR cubes.

**Decision rule:** keep p=16 unless p=8 wins by >20% on val AND probe R². Patch change has highest blast radius (changes token count, RoPE units, val tile policy) — adopt only with strong evidence.

---

## Ablation D — t_in / t_out (6 arms)

**Why this matters:** context length governs how much history the predictor sees; horizon governs forecast difficulty. **Floor: t_out ≥ 4 (48 min)** per project goal (forecast). Q2 in `05_open_questions.md` flags both unswept.

Time-token count = (t_in + t_out). Memory scales linearly here.

**Arms:**

| ID | t_in | t_out | Total frames | Notes | Slot |
|---|---|---|---|---|---|
| E28 | 4 | 2 | 6 | E09 anchor (reuse) | n/a |
| E29 | 6 | 2 | 8 | more history, short horizon | 5060ti |
| E30 | 8 | 2 | 10 | matches E16 capacity arm context | 5060ti |
| E31 | 4 | 4 | 8 | satisfies 48-min forecast floor; same context | 5060ti |
| E32 | 6 | 4 | 10 | balanced; predict best forecast | 5060ti |
| E33 | 10 | 4 | 14 | matches path_a; longest; cost ceiling | 5060ti |

**Predictions:**

- More history (E29/E30) → better val if AR dynamics are temporal not just spatial. Marginal if frame-to-frame change small.
- t_out=4 (E31/E32/E33) → val WORSE than t_out=2 in absolute terms (harder task) but more useful for downstream forecast probe.

**Decision rule:** compare on **per-frame normalized loss** (val / t_out) — raw val unfair to longer horizons. Also compare probe on last predicted frame. Winner = lowest per-frame loss with t_out ≥ 4.

---

## Total compute estimate

| Ablation | Arms | Cost (CUDA) | Cost (MPS overflow) |
|---|---|---|---|
| A — τ | 4 + 1 anchor | 25 h | 50 h |
| B — mask ratio | 4 + 1 anchor | 25 h | 50 h |
| C — patch | 2 + 1 anchor | ~30 h (p=8 slow) | OOM risk |
| D — t_in/t_out | 5 + 1 anchor | 30 h (E33 longest) | 60 h |
| **Total** | **15 new + 4 anchors** | **~110 h CUDA** | **~210 h MPS** |

CUDA-only: ~5 days continuous on 5060ti. Realistic plan: interleave A + B on 5060ti, run D on Mac Studio MPS in parallel (once E16 frees Mac Mini), defer C until A/B winners locked.

---

## Sequencing

1. **Wait** for E13/E14/E15 to finish → lock mask policy mix.
2. **Re-run E09 anchor** on 5060ti as control (E_anchor_A) — verifies host reboot / lib drift didn't shift baseline.
3. **Ablation A (τ)** — 4 arms, sequential on 5060ti. Settles fast (cheap, large effect prior).
4. **Ablation B (mask ratio)** — 4 arms with locked policy mix + winning τ.
5. **Ablation D (t_in/t_out)** — 5 arms; once Mac Studio free, run in parallel batches.
6. **Ablation C (patch size)** — last. Adopt only if E25 clearly wins.

Each ablation: append rows to `12_experiments.md`. Promote any CONFIRMED winner to `12_experiments_findings.md`. Update `05_open_questions.md` Q2 / Q5 when answered.

---

## Open risks

- **Sanity scale may not generalize.** Winners at dim=192 / 4 cubes may flip at dim=384 / 21 cubes (E16 path). Re-validate top-2 arms per ablation at E16 scale before locking.
- **Confounded by curriculum.** τ and mask ratio interact with curriculum schedule. Hold curriculum fixed at E09 slow (0.25 / 0.40); revisit only if all ablations agree.
- **Probe head not yet stabilized.** Stage-2 probe pending (F9). Use Stage-1 R²/medAPE as soft signal, not gating metric, until Stage-2 lands.
- **EMA τ × LR interaction unknown.** If τ ablation flat, suspect LR dominates; consider joint (τ, LR) follow-up.

---

## Pointers

- Anchor config: `configs/v5_mini_mask_on_100ep_slow_curric.yaml`
- E13 pattern (mask sweep): `configs/v5_mini_mask_tube_future_cuda.yaml`
- Open questions: `docs/V5_JEPA/05_open_questions.md` Q2, Q5
- Findings: F1 (curric), F2 (clip), F3 (mask policy), F9 (probe)
- Probe protocol: `docs/V5_JEPA/11_winding_flux_probe_head.md`
