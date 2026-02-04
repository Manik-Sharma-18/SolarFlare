# Feature Landscape: PyTorch Pipeline Stabilization

**Domain:** Production-quality PyTorch training pipeline (ConvLSTM encoder-decoder)
**Researched:** 2026-02-02
**Confidence:** MEDIUM

## Table Stakes

Features expected from any production ML training pipeline.

### Device Management

| Feature | Complexity | Current State |
|---------|------------|---------------|
| Auto-detect best device (CUDA > MPS > CPU) | Low | Only detects CUDA. MPS absent. Config uses boolean use_cuda. |
| MPS fallback for unsupported ops | Medium | No MPS awareness at all. |
| Device-aware AMP handling | Medium | get_amp_context handles CUDA/CPU. MPS not handled. |
| Consistent tensor placement | Low | Looks correct but untested on MPS. |

### Checkpoint and Resume

| Feature | Complexity | Current State |
|---------|------------|---------------|
| Full checkpoint resume (model + optimizer + scheduler + epoch + best_loss) | Medium | Saves state but train_model() has no resume path. |
| Normalization params in checkpoint | Low | Only in metadata.json, not checkpoint. |
| Atomic checkpoint writes (write-to-temp-then-rename) | Low | Uses torch.save() directly. No atomic write. |

### Error Handling and Validation

| Feature | Complexity | Current State |
|---------|------------|---------------|
| Config validation before training | Medium | Zero validation. config.get() with defaults masks missing keys. |
| NaN/Inf detection in loss | Low | No NaN checking. Training continues with NaN loss. |
| Data loading failure threshold | Low | Catches all exceptions silently. Could train on 1 file out of 100. |
| Gradient health monitoring | Low | clip_grad_norm_ called but return value discarded. |

### Memory Management

| Feature | Complexity | Current State |
|---------|------------|---------------|
| Epoch-boundary cache cleanup | Low | No torch.cuda.empty_cache() or torch.mps.empty_cache(). |
| Incremental uncertainty (Welford) | Medium | Stacks all N predictions in memory. O(N) VRAM. |
| DataLoader pin_memory awareness | Low | Hardcoded pin_memory=True. Should be CUDA-only. |

### Data Loading

| Feature | Complexity | Current State |
|---------|------------|---------------|
| Lazy/memory-mapped data loading | High | All cubes in RAM for entire training. |
| Reproducible data splits | Low | Augmentation uses unseeded np.random. |
| Worker process error handling | Low | num_workers=2 with no worker_init_fn. |

### Test Coverage

| Feature | Complexity | Current State |
|---------|------------|---------------|
| Model forward pass shape tests | Low | Zero tests. No tests/ directory. |
| Loss function unit tests | Low | Zero tests. |
| Checkpoint save/load roundtrip | Low | Zero tests. |
| Data pipeline integration test | Medium | Zero tests. |
| Device compatibility smoke test | Low | Zero tests. |

## Differentiators

Beyond basic reliability — valuable for cross-platform robustness.

| Feature | Complexity | Notes |
|---------|------------|-------|
| MPS-specific op alternatives | High | Actual MPS implementations, not CPU fallback |
| Training run reproducibility (full seeding) | Low | torch/numpy/python seeds + deterministic mode |
| Graceful interrupt handling (SIGINT) | Low | Save checkpoint on Ctrl+C |
| Per-epoch memory profiling | Low | Log peak memory per device type |
| Config schema with types and ranges | Medium | Dataclasses or explicit validation |

## Anti-Features

Things to NOT build during stabilization.

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Multi-GPU / DistributedDataParallel | Massive complexity. Stabilize single-device first. |
| Hyperparameter tuning (Optuna/Ray) | Adds dependencies, complicates entry point. |
| TensorBoard / W&B integration | Adds dependency, not needed for stabilization. |
| torch.compile | Poor MPS support, can introduce subtle bugs. |
| Model architecture changes | Stabilization means model stays identical. |
| Inference API refactor | Separate concern from stabilization. |
| Data format migration (HDF5/Zarr) | np.load(mmap_mode='r') is sufficient for 10-50GB. |
| Additional validation metrics | Scope creep — defer to future milestone. |

## Feature Dependencies

1. Config validation first — catches errors early
2. Device detection before MPS work — must know device before device-specific decisions
3. Atomic writes before resume — don't add resume that loads corrupted checkpoints
4. NaN detection before gradient monitoring — basic health check before diagnostics

## MVP Priority Order

**Must-have (Foundation):**
1. Config validation
2. Device auto-detection (CUDA > MPS > CPU)
3. NaN/Inf detection in loss
4. Data loading failure threshold
5. Atomic checkpoint writes
6. Checkpoint resume support
7. Basic test suite

**Should-have (Hardening):**
8. MPS op audit and alternatives
9. Normalization params in checkpoint
10. Welford's uncertainty estimation
11. Epoch-boundary cache cleanup
12. pin_memory device awareness
13. Lazy data loading (mmap)
14. Graceful interrupt handling

**Nice-to-have (Polish):**
15. Reproducible seeding
16. Gradient health monitoring
17. Per-epoch memory profiling

---
*Feature landscape analysis: 2026-02-02*
