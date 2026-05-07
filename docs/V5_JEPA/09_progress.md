# 09 — Implementation Progress

**Branch:** `v5-jepa-lora`
**Date:** 2026-05-08
**Status:** Path B scaffold complete + MPS sanity green + 5-epoch CUDA sanity green on RTX 5060 Ti.

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
  zarr_loader.py                  ← lazy zarr reads + BZ_CLIP_GAUSS sentinel guard
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

10 cubes contain non-NaN values up to **10⁷–10¹⁰ Gauss** in `wind`. Physical max in
real ARs is ~5,000 G — these are corruption/sentinel codes, not flagged as NaN. Per-cube
z-score statistics absorbed them, σ blew up to 41,000 on harp_51, downstream max|z|
hit thousands.

**Fix:** `BZ_CLIP_GAUSS = 1.0e5` in `solarflare_data/zarr_loader.py`. Values above
threshold treated as invalid (added to `valid_pixel_mask`, set to 0 in `wind`).
`cube_norm_stats` filters bounded values before computing μ/σ. Verified: harp_51
σ went from 41,452 → 2,968, max|z| from thousands → 32.

### 2. val_loss=NaN — MPS SDPA bug under no_grad

After Path B refactor, train_loss finite (0.187) but val_loss=NaN every epoch.
Bisected: same model + same input + same forward path,
- `model.train()` + grad enabled → finite, all cubes
- `model.eval()` + `torch.no_grad()` → NaN at predictor step 0, all cubes

Root cause: PyTorch MPS `F.scaled_dot_product_attention` returns NaN with
`attn_mask` under `torch.no_grad`. CUDA path unaffected.

**Fix:** in `MultiheadAttentionRoPE.forward`, MPS device routes to a manual
`(q@kᵀ)·scale → masked_fill → softmax → @v` path. CUDA keeps `F.scaled_dot_product_attention`
(FlashAttention / mem-efficient kernel). Verified post-fix: val_loss=0.202 (finite).

---

## CUDA optimization wired

- `training.grad_checkpoint: bool` — gates `torch.utils.checkpoint.checkpoint(use_reentrant=False)`
  on each `ViTEncoder` block + each `BlockCausalPredictor` block. Auto-skips in eval/no_grad.
- `training.compile: off|default|reduce-overhead|max-autotune` — wraps `encoder`,
  `target_encoder`, `predictor` with `torch.compile(dynamic=True, mode=…)` on CUDA only.
  `dynamic=True` handles variable HxW from bucketed sampler.
- bf16 autocast already wired in `training/jepa_trainer.py` (CUDA path).
- `pin_memory=True` + `non_blocking=True` H2D copies in trainer.

---

## First sanity result (MPS, 1 epoch)

```
config:  configs/v5_sanity.yaml
device:  mps
cubes:   harp_17, harp_45, harp_83 (train)  /  harp_51 (val)
windows: 10,648 train  /  402 val
epochs:  1
[epoch 0] train_loss=0.1869  val_loss=0.2018
```

Both finite. Pipeline + EMA + curriculum = green on MPS.

## Multi-epoch sanity (CUDA, 5 epochs, RTX 5060 Ti)

Launched via `solarflare-training` skill → `scripts/launch_slot.sh 5060ti_cuda`.
Code rsynced to `/home/indra/solarflare`. Same `v5_sanity.yaml`, `--max-epochs 5`,
`--device cuda` (injected by launcher). Wall ~135 s total ≈ 27 s/epoch.

| Epoch | train_loss | val_loss |
|-------|------------|----------|
| 0     | 0.1660     | **0.1848** ← best |
| 1     | 0.1370     | 0.2624   |
| 2     | 0.1230     | 0.2472   |
| 3     | 0.0978     | 0.2397   |
| 4     | 0.0873     | 0.2470   |

Train ↓47% over 5 epochs. Val plateaued ~0.24 after epoch 0 — overfit on tiny
config (encoder dim=192, 3 train cubes, single val cube `harp_51`). Expected with
this scale; not a bug.

What this validates:
- CUDA path end-to-end: pipeline + EMA target update + curriculum + bf16 autocast.
- No NaN / no crash on real CUDA hardware (vs the MPS-only SDPA bug fixed earlier).
- `pin_memory=True` + `non_blocking=True` audit passing in `launch_slot.sh`.
- launcher integration: `--device <device>` flag injection works.

Required to enter `main_v5.py` to satisfy CUDA audit:
```
# CUDA-5060ti-validated
```
plus comment pointing at `non_blocking=True` lines in `training/jepa_trainer.py`.

---

## Pending

- **Mask catalog** (`solarflare_data/masking.py`): short tube, long tube, future block,
  cross-time, tail. Required for proper JEPA masking — current code splits by t_in/t_out
  only, no in-window masking yet.
- **Full GPU run** on `configs/v5_path_a.yaml` (50 epochs, all cubes, batch>1 via bucketed
  sampler, `compile: default`, `grad_checkpoint: true`). Will exercise the pieces sanity
  skipped: ViT-Small dims (384), 6-layer predictor, drop_path=0.1, full t_in=10/t_out=4.
- **Eval suite**: pixel-decoder ablation, CSI/HSS once decoder enabled, persistence
  baseline comparison.
- **Encoder feature cache** (`docs/V5_JEPA/06_data.md` §11.5): once architecture settles,
  cache target embeddings to disk to avoid recomputing each epoch.
