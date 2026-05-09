# 11 — Winding-Flux 1D Forecast Probe: Research + Architecture

**Date:** 2026-05-10 | **Branch:** `v5-jepa-lora`
**See also:** `10_architecture_explainer.md`, `10b_flare_prediction_gap.md`

Replacement for full pixel-decoder. Predict spatially-averaged ⟨wind⟩(t) curve from frozen JEPA latents. Detect rise/peak → flare lead signal.

---

## Encoder/decoder decoupling (key principle)

JEPA encoder = general feature extractor. Train once on SSL pretext. Freeze. Attach any number of small heads:

| Target data | Head | Cost |
|---|---|---|
| ⟨wind⟩(t) curve | 1D MLP probe | ~25k |
| Pixel reconstruction | DPT/SegFormer decoder | ~10M |
| Flare event binary | Attentive pool + linear | ~50k |
| AR centroid | Spatial regression | ~30k |
| SHARP scalars (USFLUX, R_VALUE) | Per-scalar MLP | ~25k each |
| ⟨\|B\|⟩(t) curve | 1D MLP probe | ~25k |

All heads see same frozen `z_emb [B, T, hp, wp, 384]`. Train independently, no interference. Add new head whenever you get new label data — no encoder retrain.

Caveat: probe quality bounded by what encoder learned. If JEPA pretext didn't surface flare-relevant features, no head can recover them. Linear-probe accuracy = SSL quality benchmark.

---

## Research findings

### Confirmed (with sources)

**Winding flux PRECEDES flares by hours, not days.**
- Raphaldini, Prior & MacTaggart 2022, ApJ 927:156: *"sharp inputs of both quantities are found to precede individual flaring events by several hours"* — explicitly **6–7 hr** before AR 11318 events.
- Mechanism: rapid input of *current-carrying* winding marks emergence of topologically twisted flux through photosphere. Precursor signal.

**Winding flux ≠ |B| signal.**
- Prior & MacTaggart 2020, Proc. R. Soc. A: winding *independent of field strength*. Topological entanglement (renormalization of helicity flux).
- MacTaggart & Prior 2019 (arXiv:1909.01071): winding *detects when twisted cores reach photosphere* — info magnetograms alone miss.

**Flux emergence pattern → flare-productive ARs.**
- Kutsenko et al. 2021 (arXiv:2105.03886): flare-productive ARs show *faster emergence rate* in 2–3 day window pre-flare. Emergence rate, not absolute peak, correlates.

### Partial verdict on phase-lag claim

**Claim:** "Winding flux peak LAGS B-field spatial-average peak when flare occurs."

Could not confirm exact phase-lag in primary literature. What lit says instead:
- Winding has *sharp positive inputs* (jumps) hours before flare. Not framed as peak-lag vs |B|.
- |B| / unsigned-flux time series in flaring ARs typically *plateaus or keeps rising* through emergence — peak often AFTER flare. So "winding lags |B|" may be inverted in lit framing.

**Recommendation:** anchor on *winding rise event* as precursor, not specific peak-lag direction. Better-supported.

---

## Architecture — single-head forecast probe

```
Input:  z_emb [B, T, hp, wp, 384]   (frozen, from predictor output)
        valid_pixel_token [B, hp, wp]
        token_pad_mask     [hp, wp]

Step 1 — masked spatial pool per frame:
  m = valid_pixel_token & token_pad_mask          # [hp, wp]
  z_t = (z_emb * m).sum((hp,wp)) / m.sum()        # [B, T, 384]

Step 2 — temporal MLP head:
  h = MLP(384 → 64)                                # [B, T, 64]
  wind_1d_pred = Linear(64 → 1)                    # [B, T]

loss = smooth_L1(wind_1d_pred, ⟨wind⟩_target)
```

~25k params (default). One head. One target. No SHARP dependency.

### Probe-size spectrum

| Probe | Params | Tests | Overfit risk (12 cubes) |
|---|---|---|---|
| Linear (1 layer) | ~400 | Pure encoder quality. Lit benchmark. | Low |
| Small MLP (384→64→1) | ~25k | Light non-linearity. Default. | Low |
| Medium MLP (384→256→64→1) | ~115k | More capacity. | Medium |
| Big MLP (384→512→256→1) | ~330k | Deep non-linear regression. | High |
| Tiny transformer (2L, 192d) | ~900k | Cross-frame attention on pooled latents. | High |

12 cubes ≈ ~6800 windows (T=14, stride 1), ~100s effective independent samples after temporal correlation.

**Strategy: linear → small → medium, scale only on evidence.**
- *Phase 1 linear (~400):* sets floor. Bad RMSE → encoder issue, not head.
- *Phase 2 small (~25k):* default. Refines if linear shows signal.
- *Phase 3 medium (~115k):* only if Phase 2 underfits AND val tracks train.

**Why not just go big:** big MLP masks poor encoder (linear probe = honest SSL eval per V-JEPA/DINOv2). 330k params on ~6.8k correlated windows → memorize, val-fail. ⟨wind⟩(t) smooth/low-frequency — deep approximator unnecessary; if it helps materially, suspect target leak.

**When bigger IS right:** more cubes (V5.2, 50+), multi-task head (⟨wind⟩+⟨|B|⟩+flare-binary shared trunk), cross-frame mixing needed (small temporal conv/attention before pool).

Config knob: `head_dims: [384, 64, 1]` in `configs/probe.yaml`. Run linear + small side-by-side, report both RMSEs.

### Two flavors — pick one

**(a) Reconstruction probe.** Predict ⟨wind⟩(t) at all T frames encoder saw. Tests "do latents preserve global signal?" Cheap sanity.

**(b) Forecast probe (recommended).** Predict ⟨wind⟩ at masked future positions. Use predictor latents at `t >= t_in`. Tests "do latents predict future signal?" Direct flare-lead use.

### Forecast inference

```python
# t_in=10 visible, t_out=4 future (48 min ahead at 12-min cadence)
z_pred = predictor(z_ctx)                # [B, T, hp, wp, 384]
z_future = z_pred[:, t_in:]              # [B, t_out, hp, wp, 384]
z_future_pool = masked_pool(z_future)    # [B, t_out, 384]
wind_future = head(z_future_pool)        # [B, t_out]
```

Stack autoregressive → arbitrary horizon. 6 hr ahead = 30 frames. Predictor already block-causal so this composes cleanly.

### Targets

```python
wind_target = wind[H, W, T].mean(axis=(0, 1))       # spatial mean over valid pixels
```

Mask out harp_8 pathological pixels via `valid_pixel_mask` before mean.

---

## Flare signal extraction

Two options on top of predicted curve:

| Method | What |
|---|---|
| Peak detector | scipy `find_peaks` on predicted curve. Flag if peak in next 30 frames. |
| Rise detector | dwind/dt threshold (Raphaldini "sharp input"). Truer to lit. |

Rise detector beats peak detector — Raphaldini found *jump* precedes flare, not absolute max. Use rolling-window slope > τ.

---

## Why this beats pixel decode

| Pixel decode | 1D forecast |
|---|---|
| ~10M params | ~25k |
| Target shape [B, T, H, W] ~440×884 | [B, T] |
| TSS ceiling ~0.7 | Peak/rise detection on known precursor |
| Loss in pixel space (high noise) | Loss on spatial mean (low noise) |
| Eval pixel L1 (uninterpretable) | RMSE + lead-time vs lit 6–7 hr |
| Reconstructs noise | Compress to 1D — discards noise by construction |
| Fits poorly on M2 MPS | Trains on M2 MPS |

---

## Validation protocol

1. Compute ⟨wind⟩(t) from each train cube. Plot vs flare-event timestamps (need GOES catalog merge).
2. Confirm rise-event detector recovers Raphaldini "several hours before" pattern in our 12 cubes.
3. Train probe on 10 cubes, hold out 2 (cube-level split, no AR leakage).
4. Metric: forecast RMSE + peak-detection F1 + lead-time distribution. Compare to persistence (peak now = peak in 6 hr).

---

## Tradeoff

Lose: spatial localization (where in AR will flare). 1D curve says when, not where.
Gain: simplicity, signal-to-noise, direct lit comparison, trains on M2 MPS.

If "where" matters later → add coarse 2D head (e.g., 4×4 grid pool) without losing 1D channel. Decoupling principle: heads compose freely.

---

## Risks

- **12 cubes still tiny.** Probe small (~25k), feasible — but flare-positive event count per cube is binding constraint, not pixel count.
- **GOES merge required.** Flare timestamps aligned to cube `Time` array. Not in repo yet.
- **Probe quality ceilinged by JEPA quality.** Need 50-epoch full GPU run convergence first.

---

## Implementation plan

1. New module `models/v5/wind_probe_head.py` (~25k params, under 200 lines).
2. New trainer pass `training/probe_trainer.py` — frozen encoder, head-only optimizer.
3. Target loader: extend `solarflare_data/zarr_loader.py` with `compute_wind_mean_1d(cube)` cached as `wind_1d.npy`.
4. GOES catalog merge script `scripts/merge_goes_to_cube.py` → produces flare-timestamp index per cube.
5. Eval script `scripts/eval_wind_probe.py` — RMSE, F1, lead-time distribution vs persistence.

Ordered: (3) → (1) → (2) → train → (4) → (5).

---

## Sources

- [Raphaldini, Prior & MacTaggart 2022, ApJ 927:156 (IOP)](https://iopscience.iop.org/article/10.3847/1538-4357/ac4df9)
- [Prior & MacTaggart 2020, Proc. R. Soc. A — Magnetic winding: what is it good for](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0483)
- [MacTaggart & Prior 2019, arXiv:1909.01071 — Helicity and winding fluxes as indicators of twisted flux emergence](https://arxiv.org/abs/1909.01071)
- [MacTaggart & Prior 2020, arXiv:2009.11712 — Magnetic winding key to topological complexity](https://arxiv.org/abs/2009.11712)
- [Kutsenko et al. 2021, arXiv:2105.03886 — Flux emergence prior to strongest flares](https://arxiv.org/abs/2105.03886)
- [Rice & Yeates 2023, Sol. Phys. — Winding-based helicity SHARP magnetograms](https://link.springer.com/article/10.1007/s11207-023-02211-9)
- [Raphaldini NAM poster](https://ras.ac.uk/sites/default/files/NAM/P84%20-%20Raphaldini.pdf)
