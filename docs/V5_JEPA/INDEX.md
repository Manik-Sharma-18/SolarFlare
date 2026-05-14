# V5 JEPA — Index

Entry-point hub for V5 JEPA docs. No content lives here — pure TOC + links.
Date: 2026-05-13. Branch: `v5-jepa-lora`.

---

## Best results (mask-ON unless noted)

| Arch / scale | Config | Best val | Epoch | Run | Status |
|---|---|---|---|---|---|
| Sanity, mask-OFF | `v5_mini_mask_off_50ep.yaml` | 0.0125 | 38/50 | E06 | saturated (F4) |
| Sanity, mask-ON fast curric | `v5_mini_mask_on_50ep.yaml` | 0.0689 | 49/50 | E05 | unconverged (F7) |
| Sanity, mask-ON slow curric (E09 anchor) | `v5_mini_mask_on_100ep_slow_curric.yaml` | 0.00831 | 98/100 | E09 | CONFIRMED (F1) |
| Sanity, tube-only (ablation) | `v5_mini_mask_tube_only_cuda.yaml` | 0.04017 | 41/100 | E12 | diverged ep65+ (F3) |
| Sanity, tube+future (ablation) | `v5_mini_mask_tube_future_cuda.yaml` | 0.01812 | 92/100 | E13 | done |
| Sanity, tube+cross (ablation) | `v5_mini_mask_tube_cross_cuda.yaml` | 0.01172 | 91/100 | E14 | done |
| **Sanity, uniform mix (ablation)** | **`v5_mini_mask_uniform_cuda.yaml`** | **0.00530** | **99/100** | **E15** | **NEW SOTA — CONFIRMED (F11)** |
| Sanity, EMA τ=0.990 | `v5_e17_ema_0990.yaml` | **0.00761** | 99/100 | E17 | **τ winner** — beats E09 anchor |
| Sanity, EMA τ=0.994 | `v5_e18_ema_0994.yaml` | 0.00938 | 99/100 | E18 | done |
| Sanity, EMA τ=0.998 | `v5_e19_ema_0998.yaml` | 0.01107 | 99/100 | E19 | done |
| Sanity, EMA τ=0.9995 | `v5_e20_ema_09995.yaml` | 0.03010 | 99/100 | E20 | done — worst, monotone τ↓-better |
| Sanity, ratio=0.60 (uniform mix) | `v5_e25_*.yaml` | 0.00960 | 99/100 | E25 | done |
| Sanity, ratio=0.75 (uniform mix) | `v5_e26_*.yaml` | **0.00876** | 99/100 | E26 | **ratio winner** |
| Sanity, ratio=0.85 (uniform mix) | `v5_e27_*.yaml` | 0.01597 | 99/100 | E27 | done — past peak |
| Sanity, ratio=0.90 (uniform mix) | `v5_e28_*.yaml` | 0.05798 | 36 | E28 | TERMINATED — not needed |
| Bigger MPS (dim=256/L=6, 21 cubes) | `v5_mini_path_a_mps.yaml` | 0.237 | 5 | E16 | STALE — slot preempted |
| Stage-1 wind probe (linear) | on E09 features | R²=0.45 / medAPE 17.4% (harp_51) | — | E11 | calibration not capacity (F9) |

Full run log: [`12_experiments.md`](12_experiments.md). Hard truths: [`12_experiments_findings.md`](12_experiments_findings.md). Archive: [`12_experiments_archive.md`](12_experiments_archive.md).

---

## Key entry points

| File | Purpose |
|---|---|
| [`09_progress.md`](09_progress.md) | Narrative source of truth (decisions + bugs + sanity history). |
| [`12_experiments.md`](12_experiments.md) | Live run log (E12–E16 + summary table). |
| [`12_experiments_findings.md`](12_experiments_findings.md) | CONFIRMED / HYPOTHESIS findings F1–F10. |
| [`12_experiments_archive.md`](12_experiments_archive.md) | Stale / superseded run detail (E01–E08, E10). |
| [`00_overview.md`](00_overview.md) | Project mission + locked priors. |
| [`02_path_b.md`](02_path_b.md) | Path B architecture spec (chosen path). |
| [`01_path_a.md`](01_path_a.md) | Path A archived spec (ABANDONED 2026-05-08). |
| [`06_data.md`](06_data.md) | Zarr layout, 21 cubes, splits, D4 chiral aug. |
| [`05_open_questions.md`](05_open_questions.md) | Live open questions (Q1, Q10 closed). |
| [`10_architecture_explainer.md`](10_architecture_explainer.md) | Plain-language arch walkthrough. |
| [`10b_flare_prediction_gap.md`](10b_flare_prediction_gap.md) | Why pretraining ≠ flare prediction yet. |
| [`11_winding_flux_probe_head.md`](11_winding_flux_probe_head.md) | Stage-1 probe spec (E11). |
| [`13_sgnnet_mlp_swap_backlog.md`](13_sgnnet_mlp_swap_backlog.md) | SGNNet MLP-swap backlog. |
| [`14_unswept_ablations.md`](14_unswept_ablations.md) | Planned τ / mask-ratio / patch / t_in,t_out sweeps (E17–E33). |
| [`07_verify_summary.md`](07_verify_summary.md) | Verification cards. |
| [`08_sources.md`](08_sources.md) | Source paper list. |
| [`04_multimodal_baselines.md`](04_multimodal_baselines.md) | Multimodal baselines reference. |

---

## Concept pages

| Concept | File |
|---|---|
| Wind flux clip 1e8 — physics, harp_8 pathology, guard code | [`concepts/wind_flux_clipping.md`](concepts/wind_flux_clipping.md) |
| Mask catalog + curriculum (fast vs slow) + why-not-Pathak | [`concepts/mask_strategies.md`](concepts/mask_strategies.md) |

---

## Active research

- **E12–E15 mask-policy ablation DONE.** E15 uniform mix new SOTA (0.00530, 1.57× E09). F11. Re-anchor downstream sweeps on uniform mix.
- **E17–E20 EMA τ sweep DONE.** τ=0.990 winner (0.00761). Monotonic — lower τ better up to 0.990. Higher τ → worse: 0.994=0.00938, 0.998=0.01107, 0.9995=0.03010.
- **E25–E28 mask-ratio sweep DONE.** Uniform mix anchored on E15. **r=0.75 winner (E26 0.00876).** Concave: E25 (0.60)=0.00960, E26 (0.75)=0.00876, E27 (0.85)=0.01597. E28 (0.90) TERMINATED ep36 — already worse than E27 at every epoch, concave shape proven, saved 2h to launch thesis.
- **E29 THESIS RUN** (2026-05-14): full ViT-Small dim=384, 21 cubes, 80 epochs. Applies τ=0.990 + ratio=0.75 + uniform mix + slow curriculum. ETA ~89h → done ~May 18 10am. Cfg: `configs/v5_thesis.yaml`.
- **E16 capacity arm STALE.** Last log 2026-05-12 21:29 ep5. Slot preempted by EMA sweep. Superseded by E29 thesis.
- **Stage-2 probe** (queued): per-cube affine + richer pooling on E29 thesis ckpt.

---

## Dead ends / abandoned

- **Path A (LoRA on Surya)** — HelioSpectFormer hard-locked to img_size=4096, 60-min, 13ch. AR cubes are variable HxW, 12-min, 1ch. No adapter bridges. Killed 2026-05-08. Don't reintroduce `transformers` / `peft` / `huggingface_hub`. See F10 + [`01_path_a.md`](01_path_a.md).
- **`BZ_CLIP_GAUSS=1e5`** — destroyed legitimate 10⁵–10⁷ winding-flux peaks. Replaced by `WIND_FLUX_CLIP=1e8`. See F2 + [`concepts/wind_flux_clipping.md`](concepts/wind_flux_clipping.md).
- **Fast curriculum (`tail_only_pct=0.10/warmup_pct=0.20`)** — caused 18-ep plateau ep7–31 in E05. Replaced by slow curriculum 0.25/0.40. See F7.
- **Tube-only mask policy** — diverges past full-mix transition on 4 cubes (E12). Future + cross_time policies do real regularization work. See F3.
- **Δy spatial-mean probe** — spatial pool collapses inter-frame variance; MAE > persist-zero MAE on every novel cube. Richer pooling needed for Δ-targets. See F9 + E11 Stage-1c.

---

## Backlog (deferred)

- **Strategy A (true V-JEPA visible-only encoder):** rewrite `vit_encoder.py` patchifier to accept sparse tokens; rewire predictor contract. Separate PR.
- **Pixel-decoder ablation** + CSI/HSS + persistence baseline. Blocked on path_a convergence.
- **Encoder feature cache:** target embeddings to disk once arch settles. ~2× speedup.
- **harp_8 outlier spatial clustering investigation.** Defer until pretraining settles.

---

## Evidence tags

CONFIRMED / HYPOTHESIS / STALE. See [`CLAUDE.md`](../../CLAUDE.md#evidence-tags) for rules.
