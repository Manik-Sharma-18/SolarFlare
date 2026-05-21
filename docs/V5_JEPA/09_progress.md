# 09 — Implementation Progress

**Branch:** `v5-jepa-lora`
**Date:** 2026-05-13
**Index:** see [`INDEX.md`](INDEX.md) for entry-point hub (best-results table, concept pages, active research).
**Status:** Path B scaffold complete. **E15 uniform mask mix new sanity SOTA — val 0.00530 ep99** (1.57× E09). F11 CONFIRMED. E12–E14 ablation arm done (tube-only diverges; pair < trio). E17 (τ=0.990) done, val 0.00761 — lower τ helps. E18 (τ=0.994) in flight on mini_mps. 5060ti slot free — relaunch capacity arm there. E16 capacity arm STALE (preempted at ep5).

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

Quantity is **winding flux, not B-field** (per senior 2026-05-08): per-pixel max ~1e7; integrated AR ~1e13–1e14. Original `BZ_CLIP_GAUSS=1e5` clipped real signal. Fix: `WIND_FLUX_CLIP=1e8` in `zarr_loader.py` (10× margin). 9/10 cubes near-clean (≤832 outliers, ≤0.0005% frac); harp_8 still pathological (14,220 px, peak 1.68e10). Outliers → `valid_pixel_mask`, zeroed in `wind`; `cube_norm_stats` filters bounded values. Full audit: [`concepts/wind_flux_clipping.md`](concepts/wind_flux_clipping.md). Validated by 1e8 sanity rerun (val 0.0407 vs old 0.202, 5× better).

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

Catalog + curriculum + why-not-Pathak now in [`concepts/mask_strategies.md`](concepts/mask_strategies.md). Code: `solarflare_data/mask_catalog.py` (149 LOC, 22 tests). Strategy A (true V-JEPA sparse-token encoder) deferred — requires patchifier rewrite. E04 5-ep MPS sanity validated mask path end-to-end (val 0.195 ep4, finite + monotone). E09 100-ep slow curriculum converged val 0.00831 ep98 (F1 CONFIRMED).

Pre-existing bug surfaced + fixed: `models/v5/input_adapter.py:make_token_pad_mask` zeroed the entire token mask when `0 < pad_h < patch` (Python `mask[-0:, :]` slices whole array). Rollout path tolerated it; mask path INTERSECTION was first consumer. Fix: gate slice on `n_pad_rows > 0` / `n_pad_cols > 0`.

---

## E30 v2 thesis run landed (2026-05-19 21:26 IST)

Full ViT-Small (dim=384/L=12) on 13 curated train cubes, 100ep × 2000 steps, all sanity-sweep winners locked in (τ=0.990, ratio=0.75, uniform mix, slow curriculum). **Final val 0.002680 ep99 — SOTA.**

- 49.4% below E15 sanity floor (0.00530), 67.7% below E09 anchor (0.00831)
- Monotonic descent ep57→99 (no overfit, no plateau)
- s/step 0.842 locked all 100ep (~49.7 min/ep), wallclock ~52h
- Survived 7h45m tailscale outage 2026-05-17 23:59→05-18 07:45 IST via `--resume`
- BucketedShapeSampler with XL cubes excluded → 5.8× speedup vs E29b

Detail: [`12_E30_thesis.md`](12_E30_thesis.md). Curves: `figures/E30_v2_thesis_curated_loss.png`, `figures/E30_v2_vs_sanity.png`.

---

## Flare binary head on E30 — MIXED (2026-05-19)

**M+/24h NEGATIVE.** Cross-cube novel agg AUC **0.443**; within-cube harp_49 AUC **0.252** vs persist **0.973**. Lag-1 floor too high at this class/window.

**C+/12h MLP POSITIVE on harp_49.** Move to C+ (8 mixed eval cubes), 12h window (decorrelates persistence), 2-layer MLP on MPS. **harp_49 AUC 1.000, TSS 1.000 vs persist 0.975** — head beats persistence. Other cubes split (MLP destroys harp_54 0.951→0.425). C+/{6h,12h}×{linear,MLP} sweep documented; per-cube head/window selection required.

JEPA SSL never sees flare labels (no direct leakage). Wind-flux regression transfers cleanly; flare classification transfers conditionally — frozen E30 features carry flare-relevant structure exploitable by nonlinear heads at the right class+window. Detail: [`10b_flare_prediction_gap.md`](10b_flare_prediction_gap.md).

---

## Pending

- **Stage-2 wind probe** (queued): linear + MLP heads on E30 v2 `best.pt` features × 5 holdout cubes (harp_17, harp_51, harp_may2024, harp_nov2025, harp_245). Add per-cube affine calibration. Compare medAPE vs E11 baseline (46% novel, 17.4% encoder-val).
- **E16 bigger-backbone MPS arm STALE** — last log 2026-05-12 21:29 ep5. Slot preempted; superseded by E30 v2.
- **Strategy A follow-up** (visible-only encoder, true V-JEPA): rewrite `vit_encoder.py` patchifier to accept sparse tokens; rewire predictor contract. Separate PR.
- **Eval suite**: pixel-decoder ablation, CSI/HSS once decoder enabled, persistence baseline comparison.
- **Encoder feature cache** (`docs/V5_JEPA/06_data.md` §11.5): once architecture settles, cache target embeddings to disk.
