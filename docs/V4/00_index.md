# V4 ConvLSTM — Documentation Index

Snapshot of the `Version_4` branch as of 2026-05-21. Generated after the V5 JEPA pivot was reverted. Source code lives on this branch; V5 work archived under `/archive/v5_jepa/` (docs + thesis) and `v5-jepa-lora` branch (full source).

## Contents

| File | Scope |
|---|---|
| [`01_models.md`](01_models.md) | Module-level reference for `models/` — ConvLSTM, SA-ConvLSTM, attention, predictor, MC Dropout uncertainty. API + invariants + composition. |
| [`02_training.md`](02_training.md) | `main.py` entry, `Trainer` loop, composite loss (6 terms), checkpoint atomicity, transfer-learning two-phase flow, config diff (pretrain vs fine-tune), `inference.py`. |
| [`03_data.md`](03_data.md) | Raw structured-array `.npy` → `cube_*.npz` preprocess **and** HARP `*.zarr` cubes via `harp_loader.py`, mmap dataset, deterministic augmentation, AR-identity splits, device + MPS-op handling, visualisation & animation utilities. |
| [`04_experiments_and_state.md`](04_experiments_and_state.md) | v2.0 → v3.0 phase history, v3.0 vs v2.0 MIXED verdict numbers, metrics tracked, decisions, **29-arm ablation matrix (A/B deep + S SimpleConvLSTM S0–S16, configs/ablations/)**, dual-head classifier breakthrough (S13 matches persistence_csi 0.043), S16 hybrid in progress, **V5 outputs archive pointer**, known concerns. |
| [`findings_dual_head.md`](findings_dual_head.md) | Dual-head classifier pivot (S12–S16): S13 matched persistence, S14 focal DEAD, S16 hybrid (extreme_pixel_weight + classifier) running. S17 inference combo +8% MAE lift validates hybrid. Staircase autoregressive analysis: S3 collapses, S11 explodes. |
| [`findings_val_cube_fix.md`](findings_val_cube_fix.md) | S30–S33 structural arms + val cube fix (silent harp_17 → harp_8/54). S33-ep4 vs ep12 lesson: val signal still single-step biased; ep12 = first AR-stable arm. S16 still staircase champion. |
| [`findings_new_data_expansion.md`](findings_new_data_expansion.md) | 2026-06-04 dataset expansion: +7 HARPs (incl. harp_11149 = May 2024 Gannon Storm, 15 X) lifting X-event count 2→35 (17×). S42 silent val (harp_10769, 0 X) → killed. S42b val=harp_2748 (6 X) + grad_clip 1.0 → 13 ep, test CLS-CSI 0.0555 (+40% lift but -28% absolute vs S16). Val_loss gate picked mode-collapsed ckpt; ep5 (var 0.44) was better AR but missed. Loader OOM fix + HARP→NOAA mapping via JSOC drms. |
| [`findings_architecture_research.md`](findings_architecture_research.md) | 2017–2025 spatio-temporal lit sweep — depthwise-separable gates, E3D-LSTM, TrajGRU, MIM, MS-RNN multi-scale. Ranked recipes for capacity-without-param-bloat; quick-win = depthwise-separable + widened hidden (proposed S20). |
| [`findings_flare_classification.md`](findings_flare_classification.md) | 2026-06-23 pivot to a CLASSIFICATION task: per-frame ≥M-flare-in-12h from FastTOP physics CSVs + HEK GOES labels. Leave-AR-out (GroupKFold) baseline ROC-AUC 0.75, TSS 0.38 — real signal, unlike the curve regression. `scripts/flare_clf/`. |
| [`findings_curve_ceiling.md`](findings_curve_ceiling.md) | 2026-06-11→15 curve-objective pivot (S46–S52) on harp_11930. EDA verdicts (asinh softening 1e3, spatial-not-temporal noise, drift refuted). Pooling (S47) beats capacity; S48 exposure-bias explosion; S51 best single-shot (CLS-CSI 0.0656). **Predictability ceiling**: robust 38-window bootstrap — every arm's spatial-mean curve corr CI straddles 0; GT curve is near-white (autocorr 0.11–0.17). |
| [`glossary.md`](glossary.md) | Every term in `docs/reports/s_series/main.pdf` defined — physics, model, training, losses, metrics (incl CSI compute), eval modes, arm references. Use for explaining the report to seniors. |

## Read order

1. `README.md` at repo root — high-level project goal + ConvLSTM intuition.
2. `architecture.md` at repo root — supplementary diagrams.
3. `04_experiments_and_state.md` — what's been tried, what's current state, what's next.
4. `01_models.md` + `02_training.md` + `03_data.md` — module-level mechanics.
5. `docs/MODEL_ARCHITECTURE.md` (pre-existing) — narrative end-to-end walkthrough.

## V5 reference

If you need V5 JEPA content:
- Docs: `archive/v5_jepa/docs/V5_JEPA/`
- Thesis: `archive/v5_jepa/thesis/`
- Run artefacts (checkpoints, probe + flare-classifier outputs, ~2.6 GB): `archive/v5_jepa/outputs/`
- Source: `git checkout v5-jepa-lora` (or selective `git checkout v5-jepa-lora -- <path>`).

V5 abandoned 2026-05-21 — JEPA pretraining gave conditional-only probe results across 21-cube corpus; cross-AR generalisation gap not closed at this scale. Project reverted to V4 ConvLSTM (this branch).
