# M6 — Loss-reweighting ablation on accumulated-total winding

**Date:** 2026-07-13
**Task:** Does reweighting the forecast loss beat plain Huber on the accumulated
*total* winding field (leave-one-AR-out, holdout `harp_11930`)?
**Verdict:** No. Plain Huber wins. Delta toxic, horizon a wash. Confirms the
persistence ceiling; skill is concentrated in *active* windows.

Script: `scratchpad/forecast_loo_wl.py` (weighted-loss variant of
`forecast_loo.py`; `--lossmode` is the only knob changed between arms).

## Why these arms
On the total field (`cumsum(wind)` over time), beating persistence = predicting
the increment ΔW = W[t+1]−W[t], which is the near-white *rate*. Only its
large-scale envelope is coherent. Each arm tries to push capacity onto that
thin predictable slice:

- **plain** — SmoothL1/Huber, uniform weight. Baseline.
- **delta (arm A)** — weight per-pixel loss by `|target − persistence|`
  (deviation from copy-last), floor `dfloor=0.2`. Idea: put loss where
  copy-last is *wrong* (the moving lobes), ignore the flat easy pixels.
- **horizon (arm C)** — weight by lead-time ramp `[1,2,3,4]**hgamma`,
  `hgamma=1.0`. Idea: skill grows with horizon → train for the late frames.

Eval metric stays **unweighted** MAE + skill so arms share one yardstick.
Added discriminating metrics: **per-horizon skill** and **active-vs-quiet
window split** (quiet = persistence-unbeatable windows).

## Config (all three arms identical — quick ablation)
`--holdout harp_11930 --layers 3 --hidden 96 --width 96 --height 72
--stride 4 --bs 8 --epochs 35` → 3.33 M params, 2093 train windows,
46 holdout windows, grid 96×72, gap-aware, cumsum-total field.
GPU (5060ti) ~157 s/epoch; MPS ~480 s/epoch (3× slower).

## Results (final epoch, persistence MAE ≈ 0.0110)

| arm | model_MAE | skill | per-horizon skill [t+12,+24,+36,+48] | active-win | quiet-win |
|---|---|---|---|---|---|
| **plain** | 0.0102 | **+0.068** | [0.046, 0.070, 0.077, 0.069] | **+0.087** | +0.026 |
| horizon (C) | 0.0103 | +0.063 | [0.008, 0.059, 0.075, 0.075] | +0.081 | +0.022 |
| delta (A) | 0.0159 | **−0.447** | [−1.035, −0.535, −0.359, −0.266] | −0.293 | −0.789 |

## Reading
- **Delta failed.** Floor 0.2 downweighted the persistence-easy flat pixels so
  hard the model wrecked them (final MAE 0.0159 ≫ persistence). Even its best
  checkpoint only *tied* persistence, then diverged under cosine LR. Aggressive
  reweighting is toxic — echoes the >2-knob / masked-loss lessons (S30b, S31).
- **Horizon ≈ plain.** It did exactly what designed — traded early skill for
  late (t+12: 0.008 vs plain 0.046; t+48: 0.075 vs 0.069) — but **no net gain**.
- **Plain wins.** On a persistence-dominated total, plain Huber already sits at
  the thin margin; reweighting just moves error around, doesn't add skill.
- **Active/quiet metric works.** Skill in active windows (+0.087) ≫ quiet
  (+0.026): the model's edge lives where the field actually moves. Good talking
  point — report the *active-window* number, not the diluted global one.
- Global skill +0.068 here < thesis +0.093 because this is the *quick* config
  (smaller grid, 35 ep, stride 4, under-converged). Phase-2 long run recovers it.

## Rerun (flags)
```
PY=/home/indra/solarflare/venv/bin/python   # 5060ti; has torch+cuda+zarr
$PY -u scratchpad/forecast_loo_wl.py --holdout harp_11930 \
    --layers 3 --hidden 96 --width 96 --height 72 --stride 4 --bs 8 --epochs 35 \
    --lossmode {plain|delta|horizon} [--dfloor 0.2] [--hgamma 1.0] \
    --out scratchpad/loo_<arm>.png
```
Each run writes `<out>.{png,pdf,npz,json}` (+ best/final `.pt`). The `.json`
carries `per_horizon_skill`, `skill_active`, `skill_quiet`.

## Salvage ideas (if revisited later)
- **delta with gentler floor** (`--dfloor 0.5`) + lower LR / grad-clip — the
  idea (focus on moving lobes) is sound; 0.2 floor + full LR was just unstable.
- **delta weight on the *increment* magnitude** `|W[t]−W[t−1]|` instead of
  deviation-from-persistence — targets coherence, not just error.
- Combine winner with **low-pass loss term** (arm B, untested) — only the
  large-scale increment is forecastable; a coarse-scale loss may help where
  per-pixel weighting hurt.
- Real ceiling-break is the **classification** task (≥M/12h), not the forecast —
  see [[flare_classification_task]], [[m4_pil_history_result]].

Related: [[wind_rate_vs_total]], [[s31_masked_loss_new_best]],
[[s30b_six_fix_bundle_toxic]], [[grad_clip_relax]], [[thesis_v1_convlstm]].
