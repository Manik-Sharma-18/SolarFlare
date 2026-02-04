# SolarFlare — Solar Flare Prediction Pipeline

## What This Is

A solar flare prediction system using a ConvLSTM encoder-decoder architecture that forecasts spatiotemporal solar flux maps. The pipeline runs cross-platform on CUDA, MPS, and CPU with robust error handling, crash-safe checkpoints, memory-efficient data loading, and automated test coverage.

## Core Value

The prediction pipeline must run reliably on CUDA, MPS, and CPU without silent failures, memory blowups, or data corruption — producing identical results regardless of device.

## Requirements

### Validated

- ✓ ConvLSTM encoder-decoder model with multi-scale processing — existing
- ✓ Autoregressive sequence-to-sequence prediction with teacher forcing — existing
- ✓ Composite loss function (L1 + MS-SSIM + weighted MAE) — existing
- ✓ Asinh/robust/fixed normalization strategies — existing
- ✓ Dual-channel mode (flux + extreme indicator) — existing
- ✓ Data augmentation (random flips) — existing
- ✓ Gradient checkpointing for memory efficiency — existing
- ✓ MC Dropout uncertainty estimation — existing
- ✓ Mixed precision training (AMP) on CUDA — existing
- ✓ Training visualization and animation tools — existing
- ✓ Seamless device auto-detection (CUDA > MPS > CPU) with MPS-specific op alternatives — v2.0
- ✓ Incremental uncertainty estimation (Welford's algorithm) to prevent OOM — v2.0
- ✓ Lazy/memory-mapped data loading for medium datasets (10-50GB) — v2.0
- ✓ Legacy ConvLSTM.py removed and data pipelines unified — v2.0
- ✓ Config validation with clear error messages before training starts — v2.0
- ✓ NaN/Inf gradient detection and handling — v2.0
- ✓ Silent failure prevention in data loading (fail if >N% files missing) — v2.0
- ✓ Normalization parameters embedded in checkpoint (self-contained) — v2.0
- ✓ Checkpoint resume support (optimizer, scheduler, epoch state) — v2.0
- ✓ Memory cleanup between epochs (device-aware cache management) — v2.0
- ✓ SSIM loss optimization for large tensors (kernel caching, tiling) — v2.0
- ✓ Atomic checkpoint writes with cross-device resume — v2.0
- ✓ Graceful shutdown with emergency checkpoint on Ctrl+C — v2.0
- ✓ Test coverage for all modified modules (88 tests) — v2.0

### Active

(None — next milestone requirements to be defined)

### Out of Scope

- Hyperparameter tuning framework (Optuna/Ray Tune) — defer to future milestone
- Additional validation metrics (ROC-AUC, spatial MSE, gradient health) — defer to future milestone
- Inference API refactor (class-based, batch/streaming) — defer to future milestone
- Mobile/edge deployment — not relevant yet
- Multi-GPU/distributed training — single device focus for now
- torch.compile — poor MPS support, can introduce subtle bugs

## Context

Shipped v2.0 with 7,034 LOC Python across 82 files.
Tech stack: Python 3.9+ / PyTorch >= 2.0.0, numpy, yaml.
88 automated tests passing (1 skipped for MPS availability).
Pipeline trains on CUDA, MPS, and CPU with identical device API.
Lazy mmap data loading handles 10-50GB datasets without OOM.
Known quirk: predictor forward() applies preprocess twice when downsample_input=False (tests work around; production uses downsample_input=True).

## Constraints

- **Tech stack**: Python 3.9+ / PyTorch >= 2.0.0 — existing stack, no framework changes
- **Compatibility**: Must maintain identical model output on CUDA and CPU (MPS may have minor floating-point variance, which is acceptable)
- **Data format**: Must remain compatible with existing .npy data files and config.yaml structure
- **Dependencies**: Minimize new dependencies — prefer stdlib and PyTorch builtins

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Delete ConvLSTM.py rather than archive | Organized modules are authoritative; git history preserves old code | ✓ Good — cleaned 698 lines, no issues |
| MPS-specific op alternatives over CPU fallback | User wants equivalent functionality on Mac GPU, not degraded perf | ✓ Good — safe_outer, safe_quantile, channel-loop SSIM |
| Welford's algorithm for uncertainty | Eliminates OOM from stacking all MC samples in memory | ✓ Good — O(1) memory, Gaussian CI approximation |
| Lazy loading for data pipeline | Dataset is 10-50GB, current full-memory approach won't scale | ✓ Good — mmap with lazy open, worker-safe |
| Test what we touch | Validates fixes without scope-creeping into full test suite | ✓ Good — 88 tests, all passing |
| Four public functions replace single get_device() | Cleaner API, device resolved once at startup | ✓ Good |
| ConfigValidationError accumulates all errors | Users fix everything in one pass | ✓ Good |
| NaN check before backward pass | Prevents wasted computation and gradient corruption | ✓ Good |
| Atomic checkpoint writes (temp + rename) | Prevents corruption on crash | ✓ Good |
| macOS spawn, Linux fork for multiprocessing | Prevents mmap handle issues on macOS | ✓ Good |
| Index multiplication for augmentation | Deterministic, no random state issues across workers | ✓ Good |

---
*Last updated: 2026-02-04 after v2.0 milestone*
