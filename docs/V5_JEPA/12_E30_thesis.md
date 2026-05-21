# E30 v2 — THESIS curated (CUDA, dim=384, 13 train cubes, 100ep×2000 steps) — **COMPLETED 2026-05-19 21:26 IST**

**Config:** `configs/v5_thesis_curated.yaml`. **Device:** 5060ti CUDA. **Out:** `outputs_v5_thesis_curated/`.
Summary row: see `12_experiments.md` table.

## Headline

**Final val 0.002680 ep99 — SOTA.**
- 49.4% below E15 sanity floor (0.00530)
- 67.7% below E09 sanity anchor (0.00831)
- Monotonic descent ep57→99 (no overfit, no plateau)
- s/step 0.842 locked all 100ep (train 1683.7s ± 0.2s, val 1298.3s ± 0.2s, ~49.7 min/ep)
- Wallclock: ~52h end-to-end (start 2026-05-16, finish 2026-05-19 21:26 IST)

## Setup

- **Splits:** 13 train / 3 val / 5 holdout. Train: harp_8/26/43/45/49/54/83/116/156/219/274/316/318. Val: harp_86/221/11930. Holdout: harp_17/51/may2024/nov2025/245.
- **Arch:** ViT-Small (dim=384, 12L enc / 6L pred, heads=6, patch=16), t_in=10, t_out=4. 55.79M total / 33.22M trainable.
- **Winners (locked from sanity sweeps):** EMA τ=0.990 (E17), mask ratio=0.75 (E26), uniform mix {tube:0.34, future:0.33, cross_time:0.33} (E15), slow curriculum (tail_only_pct=0.25, warmup_pct=0.40), prediction curriculum 0.30/0.30/0.40.
- **Optim:** AdamW LR=3e-4, warmup_pct=0.10, cosine, min_lr_ratio=0.01, wd=0.04, bf16, grad_accum=4, grad_checkpoint, SDPA on CUDA. BucketedShapeSampler (XL cubes excluded shrinks bucketed batches → 5.8× speedup vs E29b).
- **Steps/cube:** 200K opt steps / 13 cubes = 15.4K — 3× E15 sanity, 11× E29b.

## Trajectory

| ep | val | Note |
|---:|---:|---|
| 0 | 0.4093 | first launch baseline (same as killed E29) |
| 44 | ~0.0078 | crosses E09 anchor (0.00831) |
| 57 | 0.005645 | descent confirmed after resume |
| 61 | 0.005221 | breaches E15 floor (0.00530) — 1.4% below |
| 74 | < 0.004 | sub-0.004 |
| 89 | < 0.003 | sub-0.003 |
| 94 | 0.002860 | |
| 95 | 0.002776 | |
| 96 | 0.002728 | |
| 97 | 0.002702 | |
| 98 | 0.002693 | |
| **99** | **0.002680** | **FINAL BEST** |

Monotonic last 6 ep. No regression.

## Operational notes

- **Mid-flight rescale (2026-05-16):** first launch 50ep×1000 hit 0.84s/step (5.8× faster than E29b due to XL-cube exclusion). Rescaled to 100ep×2000 to use freed budget. Saved budget per ep enabled +50ep and 2× steps/ep.
- **Tailscale outage (2026-05-17 23:59 → 2026-05-18 07:45 IST, ~7h45m):** 5060ti offline, training died. Resumed via `launch_slot.sh --resume outputs_v5_thesis_curated/last.pt`. Required `._orig_mod.` key remap (torch.compile wrap strip) added to `main_v5.py`. Resumed cleanly at ep57.
- **User-requested mid-flight restart (ep64):** killed tmux, relaunched. last.pt had actually advanced to ep68; lost ~30min re-running ep68 train (same loss ≈0.00394).
- **Monitor cadence:** /loop hourly cron (`2d4f0ee5` at :13), 24+ fires across run; Monitor not used (file unreachable during outage).

## Artifacts

- `outputs_v5_thesis_curated/best.pt`, `last.pt` (both 489MB, ep99 best)
- `outputs_v5_thesis_curated/run.jsonl` (280 lines, 100 epochs)
- `docs/V5_JEPA/figures/E30_v2_thesis_curated_loss.png` — single-run curve
- `docs/V5_JEPA/figures/E30_v2_vs_sanity.png` — E30 v2 vs E09/E15/E17 anchors
- `docs/V5_JEPA/figures/E30_v2_curves.png`, `E30_v2_progress_ep65.png` — mid-flight snapshots

## Stage-2 wind probe on E30 (2026-05-19) — **DONE**

Frozen E30 encoder → spatially-pooled features [T,384] cached per cube. Linear + MLP heads, log10 z-norm target, 60 ep AdamW. Splits mirror encoder (13 train / 3 val / 5 holdout). Eval: `outputs_probe/E30_eval/probe_metrics.md`.

| Set | n | persist R² / r | linear R² / r | MLP R² / r | best medAPE |
|---|---|---|---|---|---|
| encoder-val (held-out, in-allowlist) | 3 | +0.632 / +0.82 | +0.700 / +0.84 | **+0.729 / +0.87** | 6.1% (persist) / 7.3% (MLP) |
| novel cubes (encoder never saw) | 5 | -0.681 / +0.16 | +0.170 / +0.43 | **+0.228 / +0.50** | 8.9% (persist) / 22.3% (MLP) |
| encoder-train (sanity) | 13 | -0.927 / +0.04 | +0.008 / +0.14 | +0.023 / +0.21 | dragged by harp_8 outlier |

**Verdict: GOOD.**

- vs E11 baseline (E09 features): val R² 0.45→**0.700** (+57%), val r 0.70→**0.87** (+24%). Novel R² ~−0.0→**+0.17** (was random, now positive). Novel r 0.06→**+0.43** (+610%).
- Generalisation flipped: holdout cubes the encoder never saw now score positive R² with linear probe. Was zero before.
- Persistence beats MLP on novel medAPE (8.9% vs 22.3%) — F9 calibration gap still bounds absolute % error on unseen ARs; relative ranking (R², r) is what the encoder now provides.
- harp_8 still pathological (MAE 1.8e4 vs ~5e2 typical) → drags train aggregate. Per-cube reads cleanly otherwise.
- Artifacts: `outputs_probe/E30_linear/best.pt`, `outputs_probe/E30_mlp/best.pt`, `outputs_probe/E30_eval/{probe_metrics.md,probe_metrics.json,overlay_*.png}`.

### Per-cube affine calibration (linear probe, 30/70 split per cube)

Fit y = a·ŷ + b on leading 30% of each cube, eval remaining 70%. Tests F9 (scale-mismatch hypothesis).

| Cube | raw R² / MAPE | **cal R² / MAPE** | a | b |
|---|---|---|---|---|
| harp_11930 (val) | +0.529 / 12.5% | **+0.803 / 7.8%** | +0.55 | +1.3e3 |
| harp_221 (val) | +0.099 / 49.2% | **+0.636 / 18.8%** | +1.05 | -2.4e2 |
| harp_86 (val) | +0.648 / 6.5% | **+0.669 / 8.9%** | +0.85 | +3.8e2 |
| harp_17 (novel) | -10.78 / 161% | **+0.657 / 16.1%** | +0.69 | -3.3e2 |
| harp_51 (novel) | -30.19 / 133% | **+0.185 / 10.3%** | +0.27 | +6.7e2 |
| harp_may2024 (novel) | -2.69 / 28.8% | **+0.735 / 5.0%** | +0.59 | +2.6e3 |
| harp_nov2025 (novel) | -1.36 / 19.3% | **+0.365 / 7.8%** | +0.82 | +9.6e2 |
| harp_245 (novel) | +0.342 / 29.1% | **-0.148 / 220%** | +3.71 | -6.6e3 |

**Median calibrated medAPE across 8 cubes: 9.6% (linear) / 11.8% (MLP).** Was 22% raw MLP novel. Per-cube R² jumps from negative on most novel cubes to +0.18 to +0.74 (harp_245 outlier excluded). **F9 CONFIRMED:** encoder features encode correct temporal ranking; absolute scale drifts per cube. One a,b pair per cube closes the gap.

- 4/5 novel cubes reach <17% medAPE after calibration (matches val-cube performance).
- harp_245 fails linear calibration: target distribution heavy-tailed (y std/mean=3.0 vs ~0.2 elsewhere; max 2.3e+05 ≈ 100× median 1.8e+03). Cal-split (leading 30%) absorbed the spike; LS polyfit pulled to a=3.71, b=−6.6e+03 extrapolating wildly. Persistence wins on harp_245 because target is flat between rare spikes (acf1=0.13 but median APE 5.5%).

### Log-space calibration (robust fix for harp_245)

`log(y) = a·log(pred) + b → y = exp(b)·pred^a`. Multiplicative form matches wind-flux structure (spans 3 OOM).

| Cube | linear cal R²/MAPE | **log cal R²/MAPE** |
|---|---|---|
| harp_245 (outlier) | −0.148 / **220%** | **+0.446 / 38.7%** |
| harp_17 | +0.657 / 16.1% | **+0.754 / 13.3%** |
| harp_221 | +0.636 / 18.8% | **+0.682 / 15.7%** |
| harp_86 | +0.669 / 8.9% | **+0.685 / 5.3%** |
| harp_11930 | +0.803 / 7.8% | **+0.818 / 7.0%** |
| harp_may2024 | +0.735 / 5.0% | **+0.760 / 4.6%** |
| harp_51 | +0.185 / 10.3% | +0.307 / 11.8% |
| harp_nov2025 | +0.365 / 7.8% | +0.343 / 8.0% |

**Median (linear probe): linear-cal 9.6% → log-cal 9.9%** (tied at median; mean drops from 52% to 13% — harp_245 no longer destroys aggregate). 7/8 cubes improve or tie under log-cal. harp_245 R² flips negative→positive.

**vs persistence on novel cubes (log-cal):** 4/5 wins — harp_17 13.3 vs 27.9, harp_51 11.8 vs 22.6, harp_may2024 4.6 vs 10.0, harp_nov2025 8.0 vs 10.7. harp_245 loses (38.7 vs 5.5) — structurally harder, AR with sparse spikes + flat tail.

- Diagnostic: `scripts/diagnose_harp245.py` + `scripts/test_log_calibration.py`. Eval: `outputs_probe/E30_eval/probe_calibration.md`.

## Stage-2 flare binary head on E30 (2026-05-19) — **MIXED, see `10b_flare_prediction_gap.md`**

Frozen E30 features → BCE+pos_weight head. Two sweeps: M+/24h (linear) and C+/{6h,12h}×{linear,MLP} on MPS.

**M+/24h — NEGATIVE.**
- Cross-cube novel-agg AUC **0.443** (worse than chance — AR identity).
- Within-cube temporal: only 1 informative eval-half cube (harp_49, n=174, 120 pos). Head AUC **0.252** vs persist TSS **0.973**. Lag-1 wins.

**C+/{6h,12h}×{linear,MLP} — POSITIVE on harp_49.** Denser class (8 mixed-label eval cubes); 12h window decorrelates labels from persistence.

| Config | Agg AUC / TSS | harp_49 AUC / TSS (persist) | harp_54 AUC / TSS (persist) |
|---|---|---|---|
| C+/6h linear | 0.866 / 0.652 | 0.868 / 0.710 (0.960) | **0.993** / 0.937 (0.957) |
| C+/6h MLP | 0.872 / 0.702 | 0.898 / 0.732 (0.960) | 0.671 / 0.323 (0.957) |
| C+/12h linear | 0.877 / 0.642 | 0.716 / 0.632 (0.975) | 0.951 / 0.893 (0.990) |
| **C+/12h MLP** | 0.822 / **0.724** | **1.000 / 1.000** (0.975) | 0.425 / 0.232 (0.990) |

harp_49 C+/12h MLP TPR=1.000 FPR=0.000 at thr=0.355 — **first config beating persistence** on an informative cube. MLP hurts harp_54 (over-capacity on different cube structure); no head dominates across cubes.

**Verdict.** Wind-flux regression transfers (above). Flare classification: M+/24h fails (lag-1 floor too high, 1 informative cube); C+/12h MLP delivers a genuine positive on harp_49. Per-cube head/window selection is required — frozen E30 features carry flare-relevant structure exploitable by nonlinear heads given enough window mass.

Artifacts: `outputs_flare/E30_M_24h_linear/`, `outputs_flare/E30_M_24h_temporal_linear/`, `outputs_flare/E30_C_{6h,12h}_temporal_{linear,mlp}/`.
