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
