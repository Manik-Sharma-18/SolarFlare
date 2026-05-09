# 10 — Architecture Explainer: JEPA + V5 Model

**Date:** 2026-05-09 | **Branch:** `v5-jepa-lora`
**See also:** `10b_flare_prediction_gap.md`

---

## How JEPA Works (General)

**JEPA = Joint Embedding Predictive Architecture.** Predict in *embedding space*, not pixel space.

Problem with pixel-space prediction (V4's L1/SSIM): model wastes capacity on unpredictable noise; safe bet = collapse to mean ("persistence" failure). JEPA fix: predict the *representation* of the future. Representation discards noise by construction.

```
Input: t_in visible frames + t_out masked/target frames
         │                         │
         ▼                         ▼ (no gradient)
 Context Encoder              Target Encoder
 (trained online)             (EMA copy of context)
         │                         │
         └─────────► Predictor ────┘
                     Loss = smooth_L1(z_pred, z_target)
```

**Why EMA target encoder?** Without it both encoders collapse to constant (loss → 0, model learns nothing). EMA creates a "slow teacher" that lags context encoder — asymmetry prevents collapse without contrastive loss or negatives.

```
θ_target ← 0.996·θ_target + 0.004·θ_context   (each optimizer step)
```

**V-JEPA extension to video**: masking is 3D spatiotemporal tubes. Mask a block across all frames (tube) or all spatial positions at future frames (future block). Forces temporal dynamics learning.

---

## V5 Model Walkthrough

### Data Pipeline

```
zarr: wind[H,W,T] fp32 → clip 1e8 (winding flux, physical max ~1e7)
    → zero 14k pathological px (harp_8) via valid_pixel_mask
    → D4 chiral aug (H/V flip = sign flip, 90°/270° = sign flip)
    → window: t_in=10 + t_out=4 frames → [B, T=14, 1, H, W]
```

### InputAdapter (`models/v5/input_adapter.py`)

```
[B,T,1,H,W] → pad to mult-of-16 → 1×1 Conv 1→13ch → [B,T,13,Hp,Wp]
token_pad_mask: [hp,wp] bool, False at padded positions
```

### Context Encoder — ViT-Small (`models/v5/vit_encoder.py`)

```
[B,T,13,Hp,Wp] → reshape [B*T,13,Hp,Wp]
    → Conv2d(13,384,k=16,s=16) patch embed
    → 12 transformer blocks, bidirectional attention within each frame
    → reshape [B,T,hp,wp,384]
```

Each frame encoded independently (shared weights). No temporal mixing here — that's the predictor's job. 2D RoPE encodes physical position in Megameters (0.364 Mm/px × patch). ~22M params, receives gradients.

### Target Encoder

EMA copy of context encoder. `requires_grad=False`, always `no_grad`. Sees **clean unmasked** input → stable target representations.

### BlockCausalPredictor (`models/v5/predictor.py`)

```
z: [B, T*hp*wp, 384]
    → 3D RoPE coords: (time_minutes, y_Mm, x_Mm) per token
    → block-causal mask: frame t sees only frames 0..t
    → 6 transformer blocks → [B, T*hp*wp, 384]
```

Within-frame: full bidirectional spatial attention. Across-frames: strictly causal. 3D RoPE in physical units — 4-patch offset = 23.3 Mm, cadence encoded as actual minutes.

### Two Forward Paths (`models/v5/jepa_model.py`)

**Rollout (inference / no-mask):** autoregressive in embedding space.
```python
for step in range(t_out):
    z_out = predictor(z_seq)          # predict all positions
    z_last = z_out[:, -tpf:]          # grab last frame
    z_seq = cat([z_seq, z_last])      # extend sequence
loss = smooth_L1(z_pred[t_in:], z_target[t_in:])
```

**Masked pretext (training):** MAE-style zero-token.
```python
x_ctx = x_adapt * (~mask_upsampled)   # zero masked patches post-adapter
z_ctx    = context_encoder(x_ctx)     # gradient path
z_target = target_encoder(x_adapt)   # no_grad, clean input
z_pred   = predictor(z_ctx, T_full)
# loss only at: mask & valid_pixel_token & not_pad_token  (INTERSECTION)
```

### Mask Curriculum (`solarflare_data/mask_catalog.py`)

| Epoch | Policy | Ratio |
|---|---|---|
| 0 | tail-only (last t_out frames) | t_out/T |
| 0–1.5 | linear blend | — |
| 1.5+ | 50% tube + 30% future + 20% cross_time | ~15–40% |

Curriculum starts deployment-aligned (tail = predict next N frames), diversifies to harder pretext tasks.

### Scale Summary

| Component | Params | Grad |
|---|---|---|
| InputAdapter | ~0.2M | ✅ |
| Context Encoder (ViT-Small, 12L) | ~22M | ✅ |
| Target Encoder (EMA) | ~22M | ❌ |
| Predictor (6L, 384-dim) | ~9M | ✅ |
| **Total** | **~55M** | 33M trainable |
