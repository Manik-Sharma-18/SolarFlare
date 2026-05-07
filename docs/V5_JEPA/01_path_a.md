> **STATUS: ABANDONED at integration time (2026-05-07).**
> `nasa-ibm-ai4science/Surya-1.0` (HelioSpectFormer) is hard-locked to
> `img_size=4096` / `time_dim=2` / `in_chans=13` / 60-min cadence — incompatible
> with our variable-HxW, 12-min, single-channel AR cubes. No LoRA-only adapter
> saves it. Pivoted to **Path B (JEPA-from-scratch)**.
> Active state: `09_progress.md`. Active spec: `02_path_b.md`.
> Document below kept for historical context only.

# 01 — Path A: LoRA on Surya (ABANDONED — see 09_progress.md)

## 3.1 Encoder
- **Surya 366 M backbone, frozen.** 2D spatiotemporal transformer: 2 spectral-gating blocks + 8 long-short-range attention blocks + decoder. Patch 16×16. Internal dim 1280. Native input: 4096×4096, 13 channels (8 AIA + 5 HMI).
- **Input adapter** (trainable): `Conv2d(1, 13, 1×1)` learnable channel mixer projecting single-channel winding flux to Surya's 13-channel embedding space. Single-channel input is clean — no semantic alignment risk across HMI/AIA.
- **Spatial alignment for variable AR dims:** pad each cube to nearest multiple-of-16 (patch size). NO fixed crop. ViT tokenization handles arbitrary (H, W). Provide `attention_mask` to ignore padding tokens.
- **Position encoding: physical-units RoPE-3D.** Token coordinate = `(t_idx · 12 min, y_idx · 0.364 Mm, x_idx · 0.364 Mm)`. Frequency basis lives in physical space (min, Mm), not pixel index. Justification: variable cube dims + constant 0.364 Mm/pixel scale → physical-unit RoPE transfers natively across cubes; pixel-index RoPE would conflate "same physical extent, different pixel grid" cases. y_idx = H-axis index, x_idx = W-axis index per locked convention.

## 3.2 Forecast head (V-JEPA 2-AC template)
- **Small block-causal transformer predictor** on top of frozen encoder embeddings.
- Spec:
  - Layers: 12
  - Hidden: 768
  - Heads: 12
  - MLP ratio: 4
  - **Block-causal attention across time** (frame t cannot see frame t+1)
  - **3D RoPE** in physical units (min, Mm, Mm) — see §3.1
  - Total params: ~85 M (small relative to encoder)
- **Pixel decoder (optional, for pixel-space output):** SegFormer-style all-MLP decoder on multi-scale tokens (3 layers tapped from Surya: shallow/mid/deep), bilinear upsample to native cube `(H, W)`. ~3-10 M params.

## 3.3 LoRA injection
- LoRA rank `r=16` on `Q, V` of last 4 attention blocks of Surya encoder.
- Encoder LayerNorms remain frozen.
- LoRA params: ~5 M.

## 3.4 Loss
- **Primary:** smooth-L1 in embedding space between predictor outputs and frozen encoder's embedding of true future frames. Sum over `t+1..t+t_out`. (V-JEPA 2-AC recipe.)
- **Secondary (if pixel decoder enabled):** L1 + (1-SSIM) in pixel space, weight 0.1× primary.
- **Curriculum:** train 1-step for first 30 % of epochs, then `⌈t_out/2⌉`-step, then full `t_out`-step. Reduces error accumulation. (Boundaries open per `05_open_questions.md` Q10 — V-JEPA 2 alternative is parallel teacher-forcing + 2-step rollout summed.)
- **Optional flux conservation regularizer** (MC-JEPA pattern): predict explicit velocity field; warp t→t+1 with predicted flow; photometric + smoothness loss.

## 3.5 Training recipe
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| β | (0.9, 0.999) |
| Weight decay | 0.04 |
| Peak LR (predictor) | 3e-4 |
| Peak LR (LoRA) | 5e-4 |
| Schedule | Linear warmup 10% → cosine decay to 1% peak |
| Epochs | 50 |
| Batch size | as large as fits; cache encoder outputs on disk |
| Grad clip | 1.0 |
| Mixed precision | bf16 |
| Drop-path | 0.1 in predictor |
| Dropout | 0.1 in predictor |
| EMA on head | decay 0.999 for eval |

## 3.6 Total trainable params
- Predictor: ~85 M
- Pixel decoder: ~5 M
- LoRA: ~5 M
- Input adapter: <0.1 M
- **Total trainable: ~95 M** (vs. 366 M frozen Surya).
