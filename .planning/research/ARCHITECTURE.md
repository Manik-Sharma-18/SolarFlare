# Architecture Patterns

**Project:** SolarFlare — PyTorch ConvLSTM Solar Flux Predictor
**Researched:** 2026-02-02
**Focus:** Stabilization and cross-platform support for existing codebase
**Confidence:** HIGH

## Current Architecture

```
main.py                          # Orchestrator: config -> data -> model -> train -> eval
  ├── config.yaml                # All configuration (flat YAML, no validation)
  ├── utils/device.py            # Device selection (CUDA-only + CPU fallback)
  ├── solarflare_data/
  │     loader.py                # Data loading, normalization, splitting
  │     dataset.py               # SolarFluxDataset (eager, holds all cubes in RAM)
  ├── models/
  │     predictor.py             # SolarFluxPredictor (encoder-decoder ConvLSTM)
  │     convlstm.py              # ConvLSTM cell and layer
  │     uncertainty.py           # MC Dropout uncertainty (predict_with_uncertainty)
  ├── training/
  │     trainer.py               # train_epoch, validate, train_model, load_checkpoint
  │     losses.py                # CompositeLoss, SSIM, WeightedMAE
  ├── utils/
  │     metrics.py               # MAE, RMSE, correlation
  │     visualization.py         # Plotting utilities
  │     animation.py             # Animation utilities
  └── inference.py               # Standalone inference (hardcoded device string)
```

## Component Change Map

| Component | Change Type | Scope | Risk |
|-----------|------------|-------|------|
| `utils/device.py` | **Extend** | Add MPS detection, unify AMP/scaler logic | Low |
| `solarflare_data/dataset.py` | **Replace internals** | Lazy loading via mmap in `__getitem__` | Medium |
| `solarflare_data/loader.py` | **Modify** | Pass file paths to dataset instead of loaded arrays | Medium |
| `training/trainer.py` | **Extend** | Add checkpoint resume, save more state | Low |
| `main.py` | **Extend** | Add config validation before pipeline runs | Low |
| `models/uncertainty.py` | **Replace algorithm** | Welford's online algorithm instead of stacking | Low |
| `inference.py` | **Fix** | Use device.py instead of hardcoded strings | Low |
| `config.yaml` | **Extend** | Add device: auto, resume_from field | Low |

No new modules needed. All changes fit within existing boundaries.

## Pattern 1: Unified Device Management

```
utils/device.py
  get_device(config)          -> torch.device  (CUDA > MPS > CPU)
  get_amp_context(use_amp, device) -> context manager
  get_grad_scaler(use_amp, device) -> GradScaler or _DummyGradScaler
```

Key decisions:
- Config changes from `use_cuda: bool` to `device: auto|cuda|mps|cpu`
- MPS uses DummyGradScaler (existing pattern handles this)
- MPS autocast via `torch.amp.autocast(device_type='mps')`
- Device-aware cache cleanup: `torch.cuda.empty_cache()` vs `torch.mps.empty_cache()`

Propagation: device.py → main.py:40 → inference.py:171 → config.yaml

## Pattern 2: Lazy Data Loading

```
Current:  loader.py loads all .npy → passes arrays to Dataset
Proposed: loader.py finds files, computes norm params → passes paths to Dataset
          Dataset.__getitem__: np.load(path, mmap_mode='r')[start:end]
```

Key decisions:
- `np.load(path, mmap_mode='r')` — OS manages memory via virtual memory
- Normalization must happen at preprocessing time, not load time
- mmap works with num_workers > 0 IF each worker opens its own file descriptor
- pin_memory conditional: `pin_memory = (device.type == 'cuda')`

Propagation: dataset.py → loader.py only. No changes to trainer.py or main.py.

## Pattern 3: Checkpoint Resume

Extended checkpoint dict:
```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'scaler_state_dict': ...,      # NEW
    'best_val_loss': float,         # NEW
    'patience_counter': int,        # NEW
    'history': dict,                # NEW
    'normalization_params': dict,   # NEW
    'rng_state': {...},             # NEW
    'config': dict
}
```

Key decisions:
- Save `latest_checkpoint.pt` every epoch + `best_model.pt` on improvement
- Cross-device resume: manually move optimizer state tensors to target device
- Teacher forcing schedule naturally correct from resumed epoch counter

Propagation: trainer.py → config.yaml (add resume_from) → main.py

## Pattern 4: Config Validation

- Validate at entry (main.py), not in every module
- Cross-field checks most valuable: dual_channel + input_channels, AMP + device type
- Warn vs error: missing data dir = error, AMP on CPU = warning
- Simple function with explicit checks, no external validation library

## Pattern 5: Welford's Algorithm for Uncertainty

```python
# O(1) memory instead of O(N)
mean = torch.zeros_like(first_pred)
M2 = torch.zeros_like(first_pred)
for n in range(1, N+1):
    pred = model(x)
    delta = pred - mean
    mean += delta / n
    delta2 = pred - mean
    M2 += delta * delta2
variance = M2 / (N - 1)
std = torch.sqrt(variance)
```

Same function signature — callers unaffected.

## Dependency Order for Implementation

### Layer 0: No Dependencies (Parallel)
1. `utils/device.py` MPS support
2. `models/uncertainty.py` Welford's
3. Config validation function

### Layer 1: Depends on Layer 0
4. `config.yaml` schema update
5. `main.py` wiring (validate_config, new config keys)
6. `inference.py` device fix

### Layer 2: Depends on Layer 1
7. `solarflare_data/dataset.py` lazy loading
8. `solarflare_data/loader.py` changes

### Layer 3: Depends on Layer 2
9. `training/trainer.py` checkpoint resume
10. Error handling in main.py (wraps final integrated pipeline)

## Practical Phase Grouping

| Phase | Modules | Rationale |
|-------|---------|-----------|
| 1 | device.py, inference.py, config.yaml device section | Cross-platform device — unblocks Apple Silicon testing |
| 2 | main.py validation, config schema | Catch errors early before expensive operations |
| 3 | dataset.py, loader.py | Lazy loading — isolated data pipeline change |
| 4 | trainer.py, checkpoint resume | Depends on stable training loop |
| 5 | uncertainty.py (Welford's), memory cleanup | Independent improvements |
| 6 | Tests for all modified modules | Validates everything |

## Integration Points

1. **main.py:40** — device.py change + config schema change both affect this
2. **trainer.py:222** — device.py change affects scaler returned for MPS
3. **loader.py:163-174** — lazy loading changes Dataset constructor signature
4. **trainer.py:275-287** — resume adds fields; must be backward-compatible with existing checkpoints
5. **config.yaml** — multiple phases modify; define target schema upfront

---
*Architecture analysis: 2026-02-02*
