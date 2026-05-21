# 00 — Overview, locked assumptions, TL;DR, goal, decision tree

**Status:** Draft synthesis. Supersedes `ARCHITECTURE_V5.md` (Pathak Context Encoder draft). Three deep-research reports (JEPA family, frozen-encoder + decoder patterns, masked SSL for spatiotemporal physics) folded in.

**Date:** 2026-04-25 (revised after professor meeting same day).

> **2026-05-08 update:** Path A (Surya/LoRA) **abandoned** at integration time —
> HelioSpectFormer hard-locked to 4096 / 60-min / 13ch, incompatible with our
> variable-HxW, 12-min, single-channel AR cubes. Path B (JEPA-from-scratch)
> promoted to V5.0 primary and implemented at small scale (ViT-Small + EMA target
> + block-causal predictor). MPS + CUDA sanity both green. Current state:
> `09_progress.md`.

---

## Update log — post-professor meeting (2026-04-25)

Stakeholder clarifications applied:
- **Cadence confirmed: 12 min.**
- **Forecast horizon: not fixed.** 48 min acceptable as floor. `t_in` and `t_out` are tunable hyperparameters, not specs.
- **Data format change: `.zarr` fp32** (was float64 `.npy`). Sample inspected: see `06_data.md`.
- **Variable AR spatial dims per cube.** No fixed (H, W). Architecture must handle ragged spatial extents.
- **Single channel only**: winding flux. No HMI/AIA expansion planned.
- **Temporal data growing**: more cubes incoming over time. Path B (pretrain from scratch) becomes more viable as N grows.
- **Asinh + center-crop preprocessing pipeline = under review.** Flag as future research lever (see `06_data.md`).

### Locked coordinate / scale / sign assumptions (post-senior 2026-05-03 / 2026-05-04)
- **Pixel scale = 0.364 Mm/pixel, constant across ALL data past, present, future.** Physical extent per cube = `(H · 0.364, W · 0.364)` Mm. Locked permanently.
- **No metadata beyond `wind` + `Time` will EVER be provided.** No `.zattrs`, no coord arrays, no lat/lon, no Carrington longitude, no SHARP-style scalars in zarr. Lock permanently. External SHARP fetch (V5.2) is separate pipeline.
- **Cross-cube AR re-identification impossible.** Each cube = isolated sample.
- **Axis convention: `wind[H, W, T]` = `wind[Y, X, T]`.** W = X (horizontal), H = Y (vertical). Matches old `.npy` metadata.
- **Sign convention LOCKED chiral (pseudoscalar).** Winding flux = right-vs-left-handed twist; sign carries chirality. Augmentation must respect: H-flip, V-flip, 90°/270° rotation flip sign — always paired with explicit negation. 180° rotation preserves. See `06_data.md` §11.4.
- Limb-projection effects, differential rotation, line-of-sight noise = absorbed by encoder as unmodelled nuisance.
- **Brain-JEPA gradient-positioning DROPPED** (needed lat/lon).

Architectural consequences:
1. Variable spatial dims → ViT/JEPA token-based path strongly favored over fixed-grid conv. RoPE-3D handles arbitrary extents natively.
2. Drop center-crop (lost ~30% Nov 2025 area). Use full native AR resolution per cube.
3. fp32 .zarr → chunked lazy I/O via zarr/dask. Loader rewrite required.
4. SimVPv2 baseline still viable but only on resampled-to-fixed-grid pipeline; relegate to baseline-only role.
5. Single-channel input → cleaner Surya adapter (1→13 learned projection, no semantic alignment risk across HMI/AIA).

---

## 0. TL;DR — what the field actually does in 2026

1. **Pathak-style adversarial inpainting is obsolete for SSL.** Modern field uses (a) high-mask-ratio MAE *without* GAN, (b) JEPA-style embedding-space prediction, or (c) **forecasting-as-pretext**. Drop the discriminator and the channel-wise FC bottleneck.
2. **Surya (NASA-IMPACT/IBM, arXiv 2508.14112, Aug 2025)** is direct prior art for solar foundation modeling. 366 M params, 9 yr × 12-min × 13 channels SDO. **Pretrains by forecasting t+60 min from two prior frames — not by inpainting.** Achieves flare TSS 0.436 (best pixel-only solar FM published). **Weights public:** `nasa-ibm-ai4science/Surya-1.0` on HuggingFace.
3. **21 cubes ≪ smallest validated MAE regime (~3 k videos, VideoMAE).** Pretraining a foundation encoder from scratch on 21 cubes alone will not match foundation-model quality. Original options:
   - **Path A — adapt Surya** (~~recommended~~ **ABANDONED 2026-05-08**): img_size=4096 / 60-min / 13ch hard-locked; no adapter bridges AR cubes. See `01_path_a.md` + F10.
   - **Path B — pretrain from scratch** (**ACTIVE**): JEPA-from-scratch at small scale on our 21 cubes, validated sanity ceiling val 0.00831 (E09, F1). Pretrain on SuryaBench remains future option.
4. **Pixel-only flare models cap around TSS 0.7.** SHARP-parameter + temporal hybrids (CNN-TCN) hit **TSS 0.85**. Moirai2 on GOES X-ray alone hits **TSS 0.74**. **A pixel-only V5 has known ceiling. If goal is operational flare forecasting, multimodal (cube + SHARP scalars) is mandatory.**
5. **V-JEPA 2-AC** (Meta, June 2025) is the exact architectural template for V5: frozen large JEPA encoder + small block-causal predictor in embedding space + L1 loss + teacher-forcing/rollout curriculum. Designed for the regime V5 sits in: small post-training data, frozen large SSL encoder, multi-step future prediction.
6. **V4 mode collapse to persistence is a known L1+SSIM-in-pixel-space failure.** JEPA's predictor-in-latent-space objective is the published cure. Pixel-MSE forecasters waste capacity modeling unpredictable noise; latent prediction discards nuisance variability.

---

## 1. Goal restated

Forecast next `t_out` frames of magnetic winding flux from `t_in` input frames at 12-min cadence.

- **`t_in`, `t_out` are tunable hyperparameters**, not specs. Sweep candidates: t_in ∈ {6, 10, 14, 20}, t_out ∈ {2, 4, 6, 8}.
- **Floor: t_out=4 (≈48 min ahead).** Acceptable lower bound per professor.
- **Spatial: variable per cube.** No fixed (H, W). Architecture handles ragged extents.
- **Channels: 1 (winding flux).**
- **Signed scalar field.** Positive AND negative values both physical (winding is pseudoscalar; sign = chirality). Confirmed via diverging colormap in `analyze_wind.py`. Augmentation + normalization must preserve sign.
- **Precision: fp32.** NaN present in source data (loader must be NaN-aware).

Optionally extend with downstream flare-event head (binary 24-h M+ classifier).

Available data: **21 AR `.zarr` cubes** (12-min cadence, fp32) — 19 legacy harps + 2026-05 ingest (`harp_may2024`, `harp_nov2025`). Per-cube frame counts in `06_data.md` and root `CLAUDE.md`.

---

## 2. Architectural decision tree (historical — see 2026-05-08 update below)

```
Is goal pixel-accurate winding-flux forecast?    → V-JEPA encoder + DPT/SegFormer pixel decoder
Is goal flare event classification?              → V-JEPA encoder + attentive probe
Is goal both?                                    → Shared encoder + two heads (MC-JEPA pattern)
Compute budget < 1 k GPU-hr?                     → Path A: LoRA on frozen Surya    [ABANDONED 2026-05-08]
Compute budget 10-100 k GPU-hr?                  → Path B: V-JEPA pretrain from scratch
Compute budget < 100 GPU-hr?                     → Skip pretrain entirely. SimVPv2 end-to-end as baseline.
```

~~Default path: **Path A**~~ **Resolved 2026-05-08:** Path A abandoned (HelioSpectFormer architectural lock — see `01_path_a.md` + F10). Active path: **V5.0 Path B — JEPA-from-scratch at small scale on 21 cubes** (E09 sanity floor val 0.00831 CONFIRMED).
