# Pitfalls: PyTorch MPS Support & Pipeline Stabilization

**Project:** SolarFlare v2 — Stabilization & Cross-Platform
**Researched:** 2026-02-02
**Confidence:** HIGH for codebase-specific issues, MEDIUM for MPS-specific (needs runtime verification)

## Critical Pitfalls

### Pitfall 1: MPS AMP/GradScaler Incompatibility

**What goes wrong:** Enabling AMP on MPS with CUDA's GradScaler errors or silently produces wrong gradients. GradScaler is CUDA-only.

**Prevention:**
- Route MPS through `_DummyGradScaler` (existing pattern)
- Disable AMP on MPS unless bfloat16 confirmed on target hardware
- Add explicit device-type branching in get_amp_context

**Detection:** NaN in loss within first 5-10 batches on MPS with AMP enabled.

**Phase:** MPS device support (first phase).

**Codebase location:** `utils/device.py:26-38` — no MPS path exists.

### Pitfall 2: SSIM Grouped Conv May Fail on MPS

**What goes wrong:** `losses.py:54` uses `F.conv2d` with `groups=pred.size(1)` (depthwise pattern). MPS grouped convolution shaders have had correctness bugs.

**Prevention:**
- Test: `ssim(x, x)` must return exactly 1.0 on MPS
- If broken: provide loop-over-channels alternative for MPS
- Consider making MS-SSIM optional per-device

**Detection:** `ssim(x, x) != 1.0` on MPS.

**Phase:** MPS support phase.

### Pitfall 3: DummyGradScaler Has No NaN Guard

**What goes wrong:** CUDA's GradScaler auto-skips optimizer.step() on NaN gradients. `_DummyGradScaler` in `device.py:54-67` always calls `optimizer.step()`, propagating NaN into weights on MPS/CPU.

**Prevention:**
- Add NaN-gradient check to DummyGradScaler.step()
- Or add explicit check after scaler.unscale_() in training loop
- Guard checkpoint save: never save if loss is NaN

**Phase:** Gradient handling phase.

### Pitfall 4: Memory-Mapped Arrays + DataLoader Workers = Corruption

**What goes wrong:** `np.load(file, mmap_mode='r')` with `num_workers > 0` — forked workers share file descriptors. On macOS (fork without exec), this causes silent data corruption or segfaults.

**Prevention:**
- Open mmap inside each worker's `__getitem__`, not at Dataset construction
- Or use `worker_init_fn` to re-open mmaps per worker
- Test with num_workers=2 on macOS: compare batch stats vs num_workers=0

**Detection:** Compare batch statistics (mean/std) between num_workers=0 and num_workers=2.

**Phase:** Data loading refactor phase.

### Pitfall 5: Checkpoint Resume Doesn't Restore Scheduler State

**What goes wrong:** `load_checkpoint` (trainer.py:311) restores model and optimizer but NOT scheduler. CosineAnnealingLR resets, LR jumps from 1e-5 back to 1e-3, destabilizing training.

**Prevention:**
- Save and restore scheduler state dict
- Verify LR after resume matches pre-crash value
- Resume epoch counter from checkpoint, not from 0
- Also save: best_val_loss, patience_counter, history

**Detection:** Print LR after resume — if it doesn't match pre-crash, scheduler restore is broken.

**Phase:** Checkpoint resume phase.

**Codebase location:** `trainer.py:202-219` creates scheduler fresh, `311-333` has no scheduler restore.

## Moderate Pitfalls

### Pitfall 6: Optimizer State Device Mismatch on Cross-Device Resume

**What goes wrong:** `torch.load(path, map_location=device)` remaps model tensors but NOT optimizer state tensors (momentum buffers, Adam exp_avg). Loading CUDA checkpoint on MPS crashes at first optimizer.step().

**Prevention:**
```python
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)
```

**Phase:** Checkpoint resume phase.

### Pitfall 7: torch.quantile Not Implemented on MPS

**What goes wrong:** `uncertainty.py:129` uses `torch.quantile()` for confidence intervals. Not implemented on MPS.

**Prevention:**
- Move tensor to CPU for quantile, move result back
- Or implement via `torch.sort` + index selection

**Phase:** MPS support phase.

### Pitfall 8: Lazy Loading Breaks Normalization Stats Computation

**What goes wrong:** `loader.py:70-93` computes normalization by loading all data and sampling values. Lazy loading means data isn't in memory for stats.

**Prevention:**
- Compute norm stats in preprocessing step, save to metadata.json
- Require metadata.json exists for lazy loading mode
- Never compute global statistics lazily

**Phase:** Data loading refactor phase.

### Pitfall 9: pin_memory=True on Non-CUDA Devices

**What goes wrong:** `loader.py:447,454,461` hardcode `pin_memory=True`. Wastes memory on MPS/CPU.

**Prevention:** `pin_memory = (device.type == 'cuda')`

**Phase:** Data loading or MPS support phase.

### Pitfall 10: Non-Contiguous Tensors on MPS

**What goes wrong:** `permute`, `view`, `reshape` create non-contiguous tensors. MPS Metal shaders may assume contiguous layout, producing wrong conv2d outputs.

**Key locations:**
- `losses.py:213` — `pred.permute(0,2,1,3,4).reshape(B*T,C,H,W)` — add `.contiguous()`
- `predictor.py:199-202` — `x.view(B*T_in, ...)` chains

**Prevention:** Add `.contiguous()` after permute/reshape that feeds into convolution.

**Detection:** Compare per-element output on MPS vs CPU. Diff > 1e-4 indicates contiguity issue.

**Phase:** MPS support phase.

### Pitfall 11: np.random in Dataset Not Fork-Safe

**What goes wrong:** `dataset.py:74-81` uses `np.random.rand()` for augmentation. Forked workers inherit same random state, producing identical augmentation.

**Prevention:**
- Use `worker_init_fn` to reseed numpy per worker
- Or switch to `torch.rand()` which handles worker seeding automatically

**Phase:** Data loading refactor phase.

### Pitfall 12: NaN Detection After backward() Is Too Late

**What goes wrong:** Checking `loss.isnan()` after backward means NaN already propagated into gradients.

**Prevention:**
- Check loss BEFORE backward for forward-pass NaN
- Check gradients after unscale_() before optimizer.step()
- If NaN detected N times consecutively: reload last good checkpoint, reduce LR

**Phase:** Gradient handling phase.

**Codebase location:** `trainer.py:80-84` — no NaN check in the backward/step sequence.

## Minor Pitfalls

### Pitfall 13: MPS Memory Not Freed by torch.cuda.empty_cache()

**Prevention:** Device-aware cleanup:
```python
if device.type == 'cuda': torch.cuda.empty_cache()
elif device.type == 'mps': torch.mps.empty_cache()
```

### Pitfall 14: teacher_forcing_ratio Uses np.random Breaking Reproducibility

`predictor.py:289` — use `torch.rand(1).item()` instead of `np.random.rand()`.

### Pitfall 15: F.interpolate on MPS May Have Spatial Alignment Bugs

Test by comparing pixel-level output on MPS vs CPU for exact tensor shapes used in model.

## Phase-Specific Summary

| Phase | Key Pitfalls | Priority Actions |
|-------|-------------|-----------------|
| MPS device support | AMP/GradScaler (#1), SSIM grouped conv (#2), torch.quantile (#7), contiguity (#10), F.interpolate (#15) | Device-aware branching, unit tests comparing MPS vs CPU |
| Data loading refactor | mmap + workers (#4), normalization stats (#8), pin_memory (#9), np.random fork (#11) | Worker-local mmap opens, pre-computed stats |
| Checkpoint resume | Scheduler not restored (#5), optimizer device mismatch (#6) | Save/restore all state, manual device remapping |
| NaN/gradient handling | DummyScaler no NaN guard (#3), late NaN detection (#12) | Add NaN checks before optimizer.step() |
| Config validation | Incompatible combinations (#14 from FEATURES) | Validate all config before data loading |

---
*Pitfalls analysis: 2026-02-02*
