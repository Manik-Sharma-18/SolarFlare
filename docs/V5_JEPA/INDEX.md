# V5 JEPA — Index

Entry-point hub for V5 JEPA docs. No content lives here — pure TOC + links.
Date: 2026-05-19. Branch: `v5-jepa-lora`.

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
| **THESIS curated (dim=384/L=12, 13 train cubes, 100ep×2000 steps)** | **`v5_thesis_curated.yaml`** | **0.00268** | **99/100** | **E30 v2** | **SOTA — 49% below E15 floor, 67% below E09 anchor** |
| Stage-1 wind probe (linear) | on E09 features | R²=0.45 / medAPE 17.4% (harp_51) | — | E11 | calibration not capacity (F9) |
| **Stage-2 wind probe** (linear/MLP) | on E30 v2 features | val R²=**0.700/0.729** r=**0.84/0.87**; novel R²=**+0.17/+0.23** r=**0.43/0.50** | — | E30-probe | encoder generalises to unseen ARs (E11 novel was random) |
| **+ per-cube affine cal** (linear) | leading 30% / eval 70% | **median medAPE 9.6%** across 8 cubes; 4/5 novel <17% | — | E30-probe-cal | **F9 CONFIRMED** — scale mismatch, not capacity. harp_245 outlier (linear cal explodes). |
| **+ log-space cal** (linear probe) | log(y)=a·log(pred)+b | **median medAPE 9.9%; mean 13%** (was 52%); 7/8 cubes improve; harp_245 R² −0.15→+0.45 | — | E30-probe-cal | log-cal robust to heavy-tail novel cubes. 4/5 novel still beat persistence. |

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
- **E29 KILLED** (2026-05-14 21:54): full ViT-Small 80ep run terminated after ep0 val=0.4093. Actual rate 4.9s/step × curric 1.5× → 11.5d ETA, missed May 19 deadline by 7d. CUDA per-epoch flat (no warmup speedup; E15 sanity confirmed). Cfg: `configs/v5_thesis.yaml`.
- **E29b KILLED** (2026-05-16): ep17 plateau val=0.0341 (ep5 0.00977 was tail-only-mask fluke). 21 cubes / 30K steps = 1430 steps/cube too thin. Replaced by E30 curated.
- **E30 THESIS curated v2 DONE** (2026-05-19 21:26 IST): 13 train / 3 val / 5 holdout, 100ep × 2000 steps = 200K opt steps. **Final val 0.002680 ep99 — SOTA. 49.4% below E15 floor (0.00530), 67.7% below E09 anchor (0.00831).** Monotonic descent ep57→99 (no overfit). s/step 0.842 locked all 100ep. Holdout: harp_17, harp_51, harp_may2024, harp_nov2025, harp_245. Cfg: `configs/v5_thesis_curated.yaml`. Curves: `figures/E30_v2_thesis_curated_loss.png`, `figures/E30_v2_vs_sanity.png`.
- **E16 capacity arm STALE.** Last log 2026-05-12 21:29 ep5. Slot preempted by EMA sweep. Superseded by E30 v2 thesis.
- **Stage-2 probe DONE** (2026-05-19): linear + MLP heads on frozen E30 v2 features (spatial pool, dim=384). Splits mirror encoder. **Val R² 0.700→0.729 (linear→MLP), r 0.84→0.87.** **Novel cubes R² +0.17/+0.23, r 0.43/0.50** — was random (r=0.06) on E09. Detail: [`12_E30_thesis.md`](12_E30_thesis.md#stage-2-wind-probe-on-e30-2026-05-19--done).
- **Per-cube affine calibration DONE** (2026-05-19): fit y=a·ŷ+b on leading 30% of each cube. **Median medAPE collapses 22%→9.6%.** 4/5 novel cubes reach <17% medAPE (matches val cubes). Per-cube R² flips from −10 to −30 (raw) to +0.18 to +0.74 (calibrated). **F9 CONFIRMED**: scale-mismatch, not capacity. harp_245 outlier (linear cal explodes to a=3.7, b=−6.6e3; target has 100× spikes in cal split).
- **Log-space calibration DONE** (2026-05-19): `log(y)=a·log(pred)+b → y=exp(b)·pred^a`. Matches multiplicative wind-flux structure (spans 3 OOM). **harp_245 R² −0.15→+0.45, MAPE 220%→38.7%** — outlier no longer broken. 7/8 cubes improve under log-cal. Median tied with linear-cal (9.9% vs 9.6%) but **mean novel medAPE 52% → 13%** — robust default. 4/5 novel cubes still beat persistence (harp_17/51/may2024/nov2025).

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
