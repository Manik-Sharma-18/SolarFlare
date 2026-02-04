---
status: passed
verified: 2026-02-03
phase: 01-device-foundation-tech-debt
---

# Phase 1: Device Foundation & Tech Debt — Verification Report

## Goal
Pipeline auto-detects the best available device and runs identically on CUDA, MPS, and CPU -- with legacy dead code removed and reproducible seeding in place.

## Must-Have Verification

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `resolve_device()` auto-detects CUDA > MPS > CPU | ✓ | `utils/device.py:11` — checks `torch.cuda.is_available()` then `torch.backends.mps.is_available()` |
| 2 | `config.yaml` has `device: "auto"` (not nested) | ✓ | Top-level string field, not `device.use_cuda` object |
| 3 | DummyGradScaler for MPS/CPU, real only for CUDA | ✓ | `utils/device.py:get_grad_scaler` — `if use_amp and device.type == "cuda"` returns real, else dummy |
| 4 | Forced unavailable device raises RuntimeError | ✓ | `utils/device.py:46-51` — explicit RuntimeError for cuda/mps if unavailable |
| 5 | ConvLSTM.py deleted from project root | ✓ | `ls ConvLSTM.py` → "No such file or directory" |
| 6 | `inference.py` uses `resolve_device()`, no `'cuda'` | ✓ | `inference.py:23,38,176` — imports and uses resolve_device; `grep "'cuda'"` returns nothing |
| 7 | `main.py` uses `resolve_device(config['device'])` | ✓ | `main.py:28,56,251` — imports and calls resolve_device; no `get_device` anywhere |
| 8 | `seed_everything()` called before training | ✓ | `main.py:33` (definition), `main.py:53` (called before resolve_device and data loading) |
| 9 | `clear_device_cache()` in training loop | ✓ | `training/trainer.py:275` — called after validation, before early stopping |
| 10 | `torch.rand` replaces `np.random.rand` in predictor | ✓ | `models/predictor.py:287` — `torch.rand(1).item()`; no `np.random.rand` remains |
| 11 | `torch.rand` replaces `np.random.rand` in dataset | ✓ | `solarflare_data/dataset.py:74,78` — both augmentation flips use `torch.rand`; no `np.random.rand` |

## Success Criteria Check

| Criterion | Verified |
|-----------|----------|
| `python main.py` on Apple Silicon selects MPS | ✓ resolve_device('auto') checks MPS after CUDA |
| `device: mps` forces MPS or errors clearly | ✓ RuntimeError with clear message |
| AMP on MPS uses DummyGradScaler | ✓ Only CUDA+AMP gets real GradScaler |
| ConvLSTM.py gone, inference.py uses new API | ✓ Both confirmed |
| Identical seed → identical loss sequences | ✓ seed_everything() seeds torch+numpy+random; torch.rand for all stochastic decisions |

## Result

**Status: PASSED** — All 11 must-haves verified against actual code. Phase goal achieved.
