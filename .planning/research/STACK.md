# Technology Stack Research: MPS Support & Cross-Platform GPU

**Project:** SolarFlare v2 — Stabilization & Cross-Platform
**Researched:** 2026-02-02
**Confidence:** MEDIUM (MPS specifics need runtime verification)

## MPS Op Compatibility Matrix

All core ops in this codebase are MPS-compatible as of PyTorch >= 2.2:

| Operation | Used In | MPS Support | Notes |
|-----------|---------|-------------|-------|
| Conv2d | convlstm.py, predictor.py | YES | Fully supported |
| ConvTranspose2d | predictor.py | YES | Fully supported |
| F.interpolate(mode='nearest') | predictor.py:263,276 | YES | Supported; verify spatial alignment |
| F.avg_pool2d | losses.py (MS-SSIM) | YES | Supported |
| F.conv2d (grouped) | losses.py:54 (SSIM kernel) | YES | Depthwise pattern — test correctness |
| sigmoid, tanh | convlstm.py | YES | Element-wise ops fully supported |
| torch.cat, torch.stack | Throughout | YES | Fully supported |
| torch.quantile | uncertainty.py:129 | NO | Not implemented on MPS — needs alternative |
| torch.outer | losses.py (Gaussian kernel) | LOW confidence | Use broadcast multiply as safer alternative |

## Key Findings

### AMP on MPS
- `torch.amp.autocast(device_type='mps')` — YES (PyTorch 2.3+)
- `GradScaler` — NO (CUDA-only, Apple GPUs handle float16 differently)
- Existing `_DummyGradScaler` pattern already handles MPS correctly
- Recommendation: Route MPS through DummyGradScaler, enable autocast conditionally

### Device Configuration Change
- Current: `use_cuda: bool` — forces Mac users to get CPU silently
- Recommended: `device: "auto"|"cuda"|"mps"|"cpu"` with auto-detection priority CUDA > MPS > CPU

### Checkpoint Resume Fields Needed
Current checkpoint saves: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, val_loss, config

Additional fields needed:
- `best_val_loss` — for early stopping continuity
- `patience_counter` — for early stopping continuity
- `history` — accumulated training metrics
- `scaler_state_dict` — if using real GradScaler (CUDA only)
- `rng_state` — torch/numpy/python random states for reproducibility
- `normalization_params` — embed in checkpoint, not just metadata.json

### Memory-Mapped Data Loading
- `np.load(path, mmap_mode='r')` — zero new dependencies needed
- Works with existing .npy format
- OS manages memory paging automatically
- Sliding window access pattern is mmap-friendly
- CAVEAT: mmap + DataLoader num_workers > 0 requires worker-local file descriptor opens

### No New Runtime Dependencies
- MPS detection: `torch.backends.mps` (bundled)
- Memory mapping: `numpy.lib.format.open_memmap` (bundled)
- Checkpoint resume: `torch.save/load` (bundled)
- Only new dev dependency: pytest

## MPS-Specific Alternatives Needed

| Unsupported Op | Alternative | Location |
|---------------|-------------|----------|
| `torch.quantile` | `torch.sort` + index selection, or move to CPU for quantile computation | uncertainty.py:129 |
| `torch.outer` (low confidence) | `g.unsqueeze(1) * g.unsqueeze(0)` broadcast multiply | losses.py SSIM kernel |

## Recommendations

1. **Device detection refactor comes first** — gates all other MPS work
2. **AMP refactor is low-risk** — existing DummyGradScaler pattern handles MPS
3. **Run MPS verification checklist on target Mac** before implementing MPS-specific code paths
4. **Memory-mapped data loading is independent of MPS** — can be parallel
5. **Checkpoint resume is independent of MPS** — can be parallel
6. **Pin PyTorch version** after confirming MPS compatibility

## Open Questions

- Exact PyTorch version installed on target Mac
- Whether preprocessed data format (multiple .npy files) needs multi-file mmap wrapper
- Whether `torch.outer` works on MPS in installed PyTorch version

---
*Stack research: 2026-02-02*
