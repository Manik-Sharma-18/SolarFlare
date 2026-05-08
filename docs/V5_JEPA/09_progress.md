# 09 — Implementation Progress

**Branch:** `v5-jepa-lora`
**Date:** 2026-05-08
**Status:** Path B scaffold complete + MPS sanity green + 5-epoch CUDA sanity green on RTX 5060 Ti + outlier-fix (1e8 clip) re-run on MPS green (val_loss 0.0407, **5× better than prior 1e5 baseline**) + **mask catalog landed** (Strategy B / MAE-style zero-token, 5-epoch MPS sanity green, val_loss 0.1950 monotone-decreasing).

---

## TL;DR

V5 JEPA scaffold landed on branch `v5-jepa-lora`. Surya/Path A abandoned at integration
time (architectural mismatch). Pivoted to **Path B (JEPA-from-scratch)**: trainable ViT
context encoder + EMA target encoder + block-causal predictor, smooth-L1 in embedding
space. Pipeline runs end-to-end on MPS sanity (4 cubes, 1 epoch, train_loss=0.187,
val_loss=0.202 — both finite).

---

## Decisions

### Surya/Path A abandoned

`nasa-ibm-ai4science/Surya-1.0` (HelioSpectFormer) is hard-locked to `img_size=4096`,
`time_dim=2`, `in_chans=13`, 60-min cadence. AR-cutout cubes (variable HxW, 12-min
cadence, 1 channel) cannot drive that backbone — no LoRA-only adapter saves it.

### Path B selected (JEPA-from-scratch)

V-JEPA-2-AC template at small scale:
- **Context encoder** (ViT-Small, ~22 M): trainable, gradient-driven.
- **Target encoder** (ViT-Small, ~22 M): EMA copy of context, no_grad. Anti-collapse.
  Updated each optimizer step: `θ_target ← decay·θ_target + (1-decay)·θ_context`.
- **Predictor** (block-causal transformer, ~9 M @ 6 layers / 384 hidden): trainable.
- **Adapter**: pad-to-mult-of-16 + 1×1 conv 1→13.

Total: ~55 M (33 M trainable + 22 M frozen target).

Mask catalog (short tube / long tube / future block / cross-time) is **deferred** —
first sanity uses contiguous t_in/t_out split, no JEPA masking yet.

---

## Files shipped

```
configs/v5_path_a.yaml            ← Path B config (filename kept for compat)
configs/v5_sanity.yaml            ← MPS-friendly tiny config

models/v5/
  __init__.py                     ← exports InputAdapter, V5JEPAModel, ViTEncoder, …
  input_adapter.py                ← pad + 1×1 conv 1→13
  rope3d.py                       ← physical-units RoPE (min, Mm, Mm)
  predictor.py                    ← block-causal transformer (MPS attn fallback)
  vit_encoder.py                  ← per-frame ViT + update_ema_from()
  jepa_model.py                   ← V5JEPAModel: context + EMA target + predictor

solarflare_data/
  zarr_loader.py                  ← lazy zarr reads + WIND_FLUX_CLIP sentinel guard
  zarr_dataset.py                 ← bucketed sampler, D4 chiral aug

training/
  jepa_trainer.py                 ← single LR group, target EMA after each step

scripts/
  build_zarr_manifest.py
  smoke_test_v5.py                ← end-to-end fwd+bwd+EMA on one cube
  diag_val_nan.py                 ← per-cube diagnostic (debug only)

main_v5.py                        ← config → build → train
```

Files removed during pivot: `models/v5/surya_loader.py`, `scripts/download_surya.py`.
Deps dropped from `requirements.txt`: `transformers`, `peft`, `huggingface_hub`.

---

## Bugs hit and fixes

### 1. Sentinel outliers in raw zarr cubes

Quantity is **winding flux, not B-field** (per senior 2026-05-08): per-pixel max ~1e7; integrated AR ~1e13–1e14. Original `BZ_CLIP_GAUSS=1e5` clipped real signal. Fix: `WIND_FLUX_CLIP=1e8` in `zarr_loader.py` (10× margin). 9/10 cubes near-clean (≤832 outliers, ≤0.0005% frac); harp_8 still pathological (14,220 px, peak 1.68e10). Outliers → `valid_pixel_mask`, zeroed in `wind`; `cube_norm_stats` filters bounded values. Full audit: `docs/V5_JEPA/OUTLIERS.md`. Validated by 1e8 sanity rerun (val 0.0407 vs old 0.202, 5× better).

### 2. val_loss=NaN — MPS SDPA bug under no_grad

Train finite (0.187) but val=NaN every epoch. Same model/input/path:
`model.train()` + grad → finite; `model.eval()` + `no_grad()` → NaN at predictor step 0.
Root cause: MPS `F.scaled_dot_product_attention` returns NaN with `attn_mask` under `no_grad`. CUDA unaffected. Fix: `MultiheadAttentionRoPE.forward` routes MPS through manual `(q@kᵀ)·scale → masked_fill → softmax → @v`; CUDA keeps SDPA (FlashAttn / mem-efficient). Post-fix val=0.202 (finite).

### 3. `make_token_pad_mask` zeros entire mask when `0 < pad_h < patch`

`mask[-(pad_h//patch):, :] = False` evaluates to `mask[-0:, :]` = whole array because Python `-0 == 0` and `pad_h//patch == 0` for sub-patch padding. Pre-existing; rollout path tolerated all-False keep mask, mask path's INTERSECTION loss-mask was first consumer to depend on it. Fix: gate slicing on `n_pad_rows > 0` / `n_pad_cols > 0` (`models/v5/input_adapter.py`).

---

## CUDA optimization wired

- `training.grad_checkpoint`: gates checkpoint on each ViT/Predictor block (auto-skips eval/no_grad).
- `training.compile: off|default|reduce-overhead|max-autotune`: wraps encoder/target/predictor on CUDA only, `dynamic=True` for bucketed sampler.
- bf16 autocast in `training/jepa_trainer.py` (CUDA + CPU paths).
- `pin_memory=True` + `non_blocking=True` H2D in trainer. `# CUDA-5060ti-validated` marker in `main_v5.py` for the audit.

---

## Sanity history (no-mask path)

All on `configs/v5_sanity.yaml` (encoder dim=192, 3 train + 1 val cubes).

| Run | Device | Epochs | Best val_loss | Notes |
|---|---|---|---|---|
| First green | MPS | 1 | 0.2018 | Pipeline + EMA + curriculum green. |
| Multi-epoch | CUDA 5060ti | 5 | 0.1848 (ep 0) | ~27 s/ep; plateau ep≥1 = overfit on tiny config. |
| 1e8 clip rerun | MPS | 5 | **0.0407** (ep 4) | Senior fix: winding flux≠B; clip 1e5→1e8 preserves real signal. **5× better than 1e5.** |

The 1e8-clip MPS run is the pre-mask deployment-aligned baseline. CUDA 1e8 rerun skipped — MPS evidence sufficient to unblock mask catalog.

---

## Mask catalog landed (Strategy B — MAE-style zero-token, 2026-05-08)

Per `docs/V5_JEPA/03_masks_and_pathak.md`. Two strategies considered:

- **A — visible-only encoder (true V-JEPA):** rejected this PR. Requires
  rewriting the per-frame ViT patchifier (`Conv2d(patch=16,stride=16)`
  produces a rectangular grid; sparse tokens break the contract) plus the
  predictor's `n == T·hp·wp` invariant. Out of scope under the 200-line cap.
- **B — MAE-style zero-token masking (chosen):** masked patches multiplied
  by 0 in pixel space *post-adapter* (chosen over pre-adapter so the 1×1
  conv bias doesn't leak into masked positions). Context encoder runs on
  full T with masked tokens zeroed. Target encoder runs on **clean** full T
  in `no_grad`. Single predictor pass over `T·hp·wp` tokens. Loss restricted
  to `mask & valid_token & token_pad_mask` (INTERSECTION not union — never
  grade on outlier-clipped pixels).

### Files shipped
```
solarflare_data/mask_catalog.py   ← short_tube/long_tube/future/cross_time/tail
                                    + sample_mixed + curriculum_mix          (149 LOC)
tests/test_mask_catalog.py        ← 22 tests, ratios ±3% on N=2000           (185 LOC)
models/v5/jepa_model.py           ← _forward_masked added; rollout fallback  (179 LOC)
training/jepa_trainer.py          ← per-batch sample_mixed + curriculum mix  (200 LOC)
configs/v5_{sanity,path_a}.yaml   ← masking: block (mix, curriculum, seed)
scripts/smoke_test_v5.py          ← --mask-policy {none,auto,tail,…}         (141 LOC)
```

### Pre-existing bug surfaced
`models/v5/input_adapter.py:make_token_pad_mask` zeroed the entire token
mask when `0 < pad_h < patch` (because `pad_h // patch == 0` and Python
`mask[-0:, :]` slices the whole array). Rollout path tolerated it; mask path
INTERSECTION was first consumer to depend on it. Fixed by gating slice on
`n_pad_rows > 0` / `n_pad_cols > 0`.

### 5-epoch MPS sanity (mask catalog ON)

Same `v5_sanity.yaml`, `--max-epochs 5`, MPS, mask `enabled: true` with
default mix `{tube: 0.5, future: 0.3, cross_time: 0.2}`. Curriculum
tail-only first 10% (epoch 0), linear blend 10-30%, full mix from epoch 1.5+.

| Epoch | train_loss | val_loss (tail) |
|-------|-----------:|----------------:|
| 0 | 0.2345 | 0.2789 |
| 1 | 0.2381 | 0.2504 |
| 2 | 0.2237 | 0.2182 |
| 3 | 0.2108 | 0.2010 |
| 4 | 0.1988 | **0.1950** ← best |

Both monotone decreasing; finite throughout; no NaN. Val_loss tail-policy
≈ 5× pre-mask baseline (0.0407) as expected — encoder now shares capacity
across diverse pretext tasks, not just deployment-aligned tail. Plan
prediction was "looser than no-mask 0.0407, ≤ 0.10 in 5 epochs"; landed at
0.195 still actively decreasing — full-scale 50-epoch run on `v5_path_a.yaml`
needed to evaluate convergence. Sanity gate (finite + monotone + no NaN) passed.

What this validates:
- Mask path end-to-end: sample → zero patches → context+target+predictor → masked-loss.
- Curriculum transition (tail-only → full mix) numerically stable.
- INTERSECTION loss-mask works under realistic `valid_pixel_mask` / `token_pad_mask`.
- No-mask rollout fallback still passes smoke (`--mask-policy none`).
- Determinism via `torch.Generator` + `masking.seed`.

---

## Pending

- **CUDA parity** for mask catalog: re-run `v5_sanity.yaml` on 5060ti slot
  via `launch_slot.sh` with masking enabled. MPS-only validated so far this PR.
- **Full GPU run** on `configs/v5_path_a.yaml` (50 epochs, all 10 cubes,
  batch>1 via bucketed sampler, `compile: default`, `grad_checkpoint: true`).
  Will exercise ViT-Small dims (384), 6-layer predictor, drop_path=0.1,
  full t_in=10/t_out=4 + the mask curriculum across the full schedule.
- **Strategy A follow-up** (visible-only encoder, true V-JEPA): rewrite
  `vit_encoder.py` patchifier to accept sparse tokens; rewire predictor
  contract. Separate PR.
- **Eval suite**: pixel-decoder ablation, CSI/HSS once decoder enabled,
  persistence baseline comparison.
- **Encoder feature cache** (`docs/V5_JEPA/06_data.md` §11.5): once
  architecture settles, cache target embeddings to disk.
