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
