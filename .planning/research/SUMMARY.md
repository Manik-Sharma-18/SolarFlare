# Project Research Summary

**Project:** SolarFlare v2 — Stabilization & Cross-Platform
**Domain:** Production ML Training Pipeline (PyTorch ConvLSTM)
**Researched:** 2026-02-02
**Confidence:** HIGH (codebase-specific), MEDIUM (runtime MPS verification needed)

## Executive Summary

SolarFlare is a PyTorch-based ConvLSTM encoder-decoder for solar flux prediction. The stabilization milestone focuses on making the existing research prototype production-ready and cross-platform (CUDA + Apple Silicon MPS). Research reveals that the core model architecture is sound, but the pipeline lacks critical production safeguards: no config validation, no checkpoint resume, no NaN detection, silent data loading failures, and Mac users currently fall back to CPU because MPS detection is absent.

The recommended approach is a layered stabilization strategy that prioritizes risk mitigation and cross-platform compatibility. Start with device detection (unblocks Apple Silicon testing), add config validation and error handling (catch issues early), then tackle data loading improvements (memory-mapped loading for large datasets), and finally add checkpoint resume capability. All core PyTorch operations used in the model are MPS-compatible, but three edge cases need alternatives: `torch.quantile` (uncertainty quantiles), grouped convolution correctness (SSIM loss), and non-contiguous tensor handling.

The main risk is MPS-specific correctness bugs that may not surface until runtime on target hardware. Mitigation strategy: unit tests that compare MPS vs CPU outputs element-wise, explicit testing of SSIM grouped convolution, and graceful fallback to CPU for unsupported operations. The existing `_DummyGradScaler` pattern already handles MPS correctly (no GradScaler needed), which reduces AMP integration risk significantly.

## Key Findings

### Recommended Stack

The existing stack requires no new runtime dependencies. All enhancements use PyTorch and NumPy built-in capabilities. Device detection uses `torch.backends.mps` (bundled), memory-mapped loading uses `np.load(mmap_mode='r')` (bundled), and checkpoint resume extends existing `torch.save/load` (bundled). The only new dependency is `pytest` for test coverage (dev dependency).

**Core technologies:**
- **PyTorch 2.3+**: Required for MPS autocast support — verify exact version on target Mac before implementing MPS-specific code paths
- **NumPy memory mapping**: `np.load(mmap_mode='r')` enables lazy loading without format migration — OS manages virtual memory automatically
- **Existing _DummyGradScaler**: Already handles non-CUDA devices correctly — route MPS through this, no new scaler logic needed

**Configuration change:**
- Replace `use_cuda: bool` with `device: "auto"|"cuda"|"mps"|"cpu"` where auto-detection follows priority CUDA > MPS > CPU

**Critical version requirement:**
- PyTorch 2.3+ for `torch.amp.autocast(device_type='mps')` — this gates MPS AMP support

### Expected Features

Research identified 17 table-stakes features expected in production ML pipelines. The codebase has partial implementations for many but lacks critical production safeguards. Most concerning gaps: zero config validation (bad configs silently fail or train incorrectly), no checkpoint resume (can't recover from crashes), no NaN detection (training continues with NaN loss corrupting weights), and silent data loading failures (could train on 1 file out of 100).

**Must have (table stakes):**
- **Config validation** — catches incompatible settings before expensive operations (e.g., dual_channel + wrong input_channels, AMP on CPU)
- **Device auto-detection** — CUDA > MPS > CPU priority with explicit config override
- **NaN/Inf detection** — check loss before backward, check gradients before optimizer.step()
- **Atomic checkpoint writes** — write-to-temp-then-rename prevents corrupted checkpoints
- **Checkpoint resume** — restore model + optimizer + scheduler + epoch + best_loss + patience_counter + history
- **Data loading failure threshold** — error if X% of files fail to load (currently catches all exceptions silently)
- **Basic test suite** — shape tests, loss function correctness, checkpoint roundtrip

**Should have (hardening):**
- **MPS op alternatives** — `torch.quantile` fallback, grouped conv validation, contiguity guarantees
- **Normalization params in checkpoint** — embed preprocessing metadata, not separate metadata.json
- **Welford's uncertainty** — O(1) memory instead of O(N) for MC Dropout predictions
- **Epoch-boundary cache cleanup** — device-aware: `torch.cuda.empty_cache()` vs `torch.mps.empty_cache()`
- **pin_memory device awareness** — only enable for CUDA (wastes memory on MPS/CPU)
- **Memory-mapped data loading** — lazy loading for large datasets without RAM bottleneck
- **Graceful interrupt handling** — save checkpoint on Ctrl+C

**Nice-to-have (polish):**
- **Full reproducibility** — torch/numpy/python seeds + deterministic mode + RNG state in checkpoint
- **Gradient health monitoring** — log gradient norms, detect vanishing/exploding
- **Per-epoch memory profiling** — track peak memory per device type

**Defer to v2+ (anti-features for stabilization):**
- Multi-GPU / DistributedDataParallel (massive complexity)
- Hyperparameter tuning frameworks (adds dependencies)
- TensorBoard / W&B integration (not needed for stabilization)
- torch.compile (poor MPS support, can introduce subtle bugs)
- Model architecture changes (stabilization means model stays identical)
- Data format migration to HDF5/Zarr (np.load mmap is sufficient)

### Architecture Approach

The codebase has clean separation of concerns with no architectural debt. All changes fit within existing module boundaries — no new modules needed. The pipeline follows standard PyTorch patterns: config -> data loading -> model creation -> training loop -> evaluation. Components communicate through well-defined interfaces (device detection returns torch.device, Dataset.__getitem__ returns tensors, trainer functions take model/optimizer/data).

**Major components and changes:**
1. **utils/device.py** — Extend with MPS detection, unify AMP context and scaler logic (LOW risk)
2. **solarflare_data/dataset.py + loader.py** — Replace eager loading with lazy mmap-based loading (MEDIUM risk — changes data pipeline internals)
3. **training/trainer.py** — Extend checkpoint save/load with additional state fields (LOW risk — backward compatible with existing checkpoints)
4. **main.py** — Add config validation at entry point before pipeline runs (LOW risk)
5. **models/uncertainty.py** — Replace stacking algorithm with Welford's online algorithm (LOW risk — same function signature)
6. **inference.py** — Fix hardcoded device strings to use device.py (LOW risk)
7. **config.yaml** — Extend schema with device field and resume_from field (LOW risk)

**Key architectural decisions:**
- **Unified device management**: Single module (device.py) handles detection, AMP context, scaler creation — all other modules consume its outputs
- **Lazy loading pattern**: loader.py passes file paths to Dataset instead of loaded arrays; Dataset.__getitem__ opens mmap handles per-access
- **Extended checkpoint format**: Backward-compatible dict extension with new optional fields for resume capability
- **Entry-point validation**: Validate all config at main.py entry, not distributed across modules

### Critical Pitfalls

Research identified 15 pitfalls, 5 critical. Most dangerous: MPS AMP/GradScaler incompatibility (NaN in first 10 batches), memory-mapped arrays + DataLoader workers causing silent corruption on macOS, and checkpoint resume not restoring scheduler (LR jumps from 1e-5 back to 1e-3 destabilizing training).

1. **MPS AMP/GradScaler incompatibility** — GradScaler is CUDA-only; enabling on MPS errors or produces wrong gradients. Prevention: Route MPS through existing `_DummyGradScaler`, disable AMP unless bfloat16 confirmed. Detection: NaN in loss within first 5-10 batches on MPS with AMP.

2. **SSIM grouped convolution may fail on MPS** — `losses.py:54` uses depthwise convolution pattern (`groups=pred.size(1)`). MPS Metal shaders have had correctness bugs with grouped conv. Prevention: Test `ssim(x, x) == 1.0` on MPS; provide loop-over-channels fallback if broken.

3. **DummyGradScaler has no NaN guard** — CUDA GradScaler auto-skips optimizer.step() on NaN gradients; `_DummyGradScaler` doesn't. On MPS/CPU, NaN propagates into weights. Prevention: Add NaN-gradient check to DummyGradScaler.step() or explicit check after scaler.unscale_().

4. **Memory-mapped arrays + DataLoader workers = corruption** — `np.load(mmap_mode='r')` with `num_workers > 0`: forked workers share file descriptors. On macOS, this causes silent data corruption or segfaults. Prevention: Open mmap inside each worker's `__getitem__`, not at Dataset construction. Detection: Compare batch statistics (mean/std) between num_workers=0 and num_workers=2.

5. **Checkpoint resume doesn't restore scheduler state** — `load_checkpoint` (trainer.py:311) restores model and optimizer but NOT scheduler. CosineAnnealingLR resets, LR jumps from 1e-5 back to 1e-3, destabilizing training. Prevention: Save and restore scheduler state dict, verify LR after resume matches pre-crash value.

**Additional moderate pitfalls:**
- Optimizer state device mismatch on cross-device resume (CUDA checkpoint -> MPS loads model but optimizer tensors crash at first step)
- `torch.quantile` not implemented on MPS (uncertainty.py:129 needs CPU fallback or torch.sort alternative)
- Lazy loading breaks normalization stats computation (must compute in preprocessing, save to metadata)
- `pin_memory=True` hardcoded for all devices (wastes memory on MPS/CPU)
- Non-contiguous tensors on MPS (permute/reshape chains need `.contiguous()` before convolutions)

## Implications for Roadmap

Based on dependency analysis and risk assessment, suggested phase structure follows a layered approach: foundations first (device detection + validation), then independent improvements (data loading, checkpoint resume), then polish (memory optimization, tests). This ordering unblocks Apple Silicon testing early, catches errors before expensive operations, and isolates risky changes (data loading) from stable changes (config validation).

### Phase 1: Cross-Platform Device Support
**Rationale:** Unblocks Apple Silicon testing and gates all other MPS work. Device detection is a prerequisite for device-specific code paths (cache cleanup, pin_memory, AMP handling). Low-risk changes to single module (device.py) plus config schema update.

**Delivers:**
- Auto-detection with priority CUDA > MPS > CPU
- Device-aware AMP context and scaler creation
- MPS routed through existing DummyGradScaler
- Config changes: `use_cuda: bool` -> `device: auto|cuda|mps|cpu`
- Fixes inference.py hardcoded device strings

**Addresses features:**
- Device auto-detection (table stakes)
- Device-aware AMP handling (table stakes)

**Avoids pitfalls:**
- Pitfall #1: MPS AMP/GradScaler incompatibility
- Pitfall #13: Device-specific memory cleanup

**Research flag:** SKIP — standard pattern, well-documented in PyTorch docs.

### Phase 2: Configuration Validation & Error Handling
**Rationale:** Catch errors early before expensive operations (data loading, model instantiation). Config validation is independent of device detection but benefits from knowing device type for cross-field checks (AMP + device, dual_channel + input_channels). Adds critical production safeguards with minimal code change.

**Delivers:**
- Config validation function with explicit checks
- Required field validation (data_dir, input_channels)
- Cross-field validation (dual_channel + input_channels, AMP + device type)
- NaN/Inf detection in training loop (before backward, after unscale_)
- Data loading failure threshold (error if >X% files fail)
- Atomic checkpoint writes (write-to-temp-then-rename)

**Addresses features:**
- Config validation (table stakes)
- NaN/Inf detection (table stakes)
- Data loading failure threshold (table stakes)
- Atomic checkpoint writes (table stakes)

**Avoids pitfalls:**
- Pitfall #3: DummyGradScaler has no NaN guard
- Pitfall #12: Late NaN detection (after backward too late)
- Improved error messages reduce debugging time

**Research flag:** SKIP — validation patterns are straightforward, no external research needed.

### Phase 3: Memory-Mapped Data Loading
**Rationale:** Isolated change to data pipeline that doesn't affect training loop or model. Can develop and test independently. Addresses memory bottleneck for large datasets without changing data format. Medium risk due to mmap + DataLoader worker interaction on macOS.

**Delivers:**
- Lazy loading via `np.load(mmap_mode='r')` in Dataset.__getitem__
- Worker-safe mmap handling (open per-access, not at construction)
- Device-aware pin_memory (CUDA-only)
- Pre-computed normalization stats (preprocessing step)
- Reproducible augmentation (worker_init_fn for numpy seeding)

**Addresses features:**
- Memory-mapped data loading (should-have)
- pin_memory device awareness (should-have)
- Reproducible data splits (nice-to-have)

**Avoids pitfalls:**
- Pitfall #4: mmap + DataLoader workers corruption
- Pitfall #8: Lazy loading breaks normalization stats
- Pitfall #9: pin_memory on non-CUDA devices
- Pitfall #11: np.random not fork-safe

**Research flag:** NEEDS RESEARCH for "Safe mmap + DataLoader multiprocessing patterns on macOS" — this is a sharp edge that needs validation.

### Phase 4: Checkpoint Resume
**Rationale:** Depends on atomic checkpoint writes from Phase 2. Requires stable training loop (no NaN crashes that would test resume incorrectly). Extends existing checkpoint save/load with additional state fields.

**Delivers:**
- Extended checkpoint format (backward-compatible)
- Save: epoch, model, optimizer, scheduler, scaler, best_val_loss, patience_counter, history, norm_params, rng_state
- Resume path in train_model() with epoch continuation
- Cross-device resume (manual optimizer state remapping to target device)
- Config field: resume_from (path to checkpoint)

**Addresses features:**
- Checkpoint resume (table stakes)
- Normalization params in checkpoint (should-have)
- Full reproducibility (nice-to-have — RNG state)

**Avoids pitfalls:**
- Pitfall #5: Scheduler not restored
- Pitfall #6: Optimizer state device mismatch on cross-device resume

**Research flag:** SKIP — standard PyTorch checkpoint patterns, well-documented.

### Phase 5: MPS Operation Compatibility
**Rationale:** Requires Phase 1 (device detection) complete. This phase addresses MPS-specific edge cases discovered in op audit. Includes runtime verification tests that compare MPS vs CPU outputs.

**Delivers:**
- `torch.quantile` alternative for uncertainty.py (CPU fallback or torch.sort)
- SSIM grouped convolution validation test (ssim(x,x) == 1.0)
- Contiguity guarantees (add .contiguous() after permute/reshape before conv)
- MPS vs CPU correctness tests (element-wise comparison)
- Fallback mechanism for broken ops (graceful degradation to CPU)

**Addresses features:**
- MPS op alternatives (should-have)
- Device compatibility smoke tests (table stakes from test coverage)

**Avoids pitfalls:**
- Pitfall #2: SSIM grouped conv may fail on MPS
- Pitfall #7: torch.quantile not implemented on MPS
- Pitfall #10: Non-contiguous tensors on MPS
- Pitfall #15: F.interpolate spatial alignment bugs

**Research flag:** NEEDS RESEARCH for "MPS Metal shader correctness verification" and "MPS-safe alternatives for unsupported ops" — this is hardware-specific and needs runtime validation on target Mac.

### Phase 6: Memory Optimization & Polish
**Rationale:** Independent improvements that don't affect correctness. Can be developed in parallel after Phases 1-4 are stable. Low risk, high value for production deployments.

**Delivers:**
- Welford's algorithm for uncertainty (O(1) memory instead of O(N))
- Epoch-boundary cache cleanup (device-aware)
- Gradient health monitoring (log norms, detect vanishing/exploding)
- Graceful interrupt handling (save checkpoint on SIGINT)
- Per-epoch memory profiling (log peak memory per device)

**Addresses features:**
- Welford's uncertainty (should-have)
- Epoch-boundary cache cleanup (should-have)
- Gradient health monitoring (nice-to-have)
- Graceful interrupt handling (should-have)
- Per-epoch memory profiling (nice-to-have)

**Avoids pitfalls:**
- Memory accumulation over long training runs
- Silent gradient issues (vanishing/exploding)

**Research flag:** SKIP — all standard patterns, no novel research needed.

### Phase 7: Test Coverage
**Rationale:** Tests validate all previous phases. Should be written incrementally during each phase, but a final sweep ensures comprehensive coverage. Tests are the contract that stabilization is complete.

**Delivers:**
- Model forward pass shape tests (various input shapes, batch sizes)
- Loss function unit tests (SSIM correctness, composite loss components)
- Checkpoint save/load roundtrip tests (all state preserved)
- Data pipeline integration tests (mmap correctness, worker safety)
- Device compatibility smoke tests (forward pass on CUDA/MPS/CPU produces same output)
- Config validation tests (all validation rules covered)
- NaN detection tests (loss NaN, gradient NaN, optimizer.step() skipped)

**Addresses features:**
- Test coverage (table stakes)

**Validates:**
- All pitfall mitigations are effective
- Cross-platform compatibility is real, not aspirational

**Research flag:** SKIP — testing patterns are standard, pytest documentation is sufficient.

### Phase Ordering Rationale

**Layer 0 (Phases 1-2): Foundations**
- Device detection and config validation have no dependencies and unblock all subsequent work
- Device detection gates MPS-specific code paths (Phase 5)
- Config validation catches errors before expensive operations (all phases benefit)
- Low risk, high impact — stabilize the foundation first

**Layer 1 (Phases 3-4): Core Improvements**
- Data loading (Phase 3) and checkpoint resume (Phase 4) are independent and can be parallel
- Both are medium complexity, benefit from stable foundation
- Checkpoint resume depends on atomic writes from Phase 2

**Layer 2 (Phases 5-6): Platform-Specific & Optimization**
- MPS compatibility (Phase 5) depends on device detection from Phase 1
- Memory optimization (Phase 6) is independent, low risk
- Both can be parallel

**Layer 3 (Phase 7): Validation**
- Tests written incrementally but validated comprehensively at end
- Final gate before declaring stabilization complete

**Critical path:** Phase 1 -> Phase 5 (device detection gates MPS work)
**Parallel opportunities:** Phases 3 & 4 (data loading + checkpoint resume), Phases 5 & 6 (MPS compatibility + memory optimization)

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 3 (Data Loading):** "Safe mmap + DataLoader multiprocessing on macOS" — forked workers + file descriptors is a sharp edge, needs validation of worker-local mmap open pattern
- **Phase 5 (MPS Compatibility):** "MPS Metal shader correctness" and "MPS-safe op alternatives" — hardware-specific, needs runtime verification on target Mac (exact PyTorch version, exact macOS version)

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Device Detection):** PyTorch device management is well-documented
- **Phase 2 (Config Validation):** Standard validation patterns, no novel research needed
- **Phase 4 (Checkpoint Resume):** Standard PyTorch checkpoint patterns
- **Phase 6 (Memory Optimization):** Welford's algorithm is textbook, cache cleanup is PyTorch API
- **Phase 7 (Test Coverage):** pytest patterns are standard

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new dependencies needed; all capabilities are PyTorch/NumPy built-ins. PyTorch version requirement (2.3+) needs verification on target Mac. |
| Features | HIGH | Feature landscape based on codebase audit + production ML pipeline standards. Prioritization is opinionated but defensible. |
| Architecture | HIGH | Codebase structure is clean, changes fit existing boundaries. Dependency analysis is based on actual code inspection. |
| Pitfalls | MEDIUM | Codebase-specific pitfalls are HIGH confidence (directly observed). MPS-specific pitfalls are MEDIUM (need runtime verification on target hardware). |

**Overall confidence:** HIGH for codebase analysis and phase planning, MEDIUM for MPS runtime behavior.

### Gaps to Address

**MPS runtime verification needed:**
- Exact PyTorch version on target Mac (gates MPS autocast support)
- `torch.outer` support on MPS (used in SSIM Gaussian kernel)
- Grouped convolution correctness on MPS (critical for SSIM loss)
- F.interpolate spatial alignment on MPS (used in predictor upsampling)

**Validation during Phase 5:**
- Run MPS vs CPU element-wise output comparison on actual hardware
- Test SSIM loss: `ssim(x, x)` must return exactly 1.0 on MPS
- Test uncertainty quantiles with CPU fallback
- Measure MPS performance vs CPU (if MPS is slower, document trade-offs)

**Data format assumption:**
- Research assumes preprocessed data is multiple .npy files (as seen in loader.py)
- If data format changes, mmap strategy may need adjustment

**Resolution strategy:**
- Phase 1 can proceed immediately (device detection is PyTorch API)
- Phase 5 requires access to target Mac for verification
- If MPS verification reveals blockers: graceful fallback to CPU is acceptable (still better than current state where Mac users get CPU silently)

## Sources

### Primary (HIGH confidence)
- **Codebase inspection:** All research based on direct analysis of SolarFlare codebase (commit 0585351)
- **PyTorch documentation:** Device management, AMP API, checkpoint format (official docs, version 2.2+)
- **NumPy documentation:** Memory-mapped file I/O (official docs)

### Secondary (MEDIUM confidence)
- **PyTorch MPS backend limitations:** Known unsupported operations (community reports, GitHub issues)
- **Production ML pipeline standards:** Common patterns for error handling, validation, checkpoint resume (industry practice)

### Tertiary (LOW confidence, needs validation)
- **MPS Metal shader correctness:** Grouped convolution bugs reported in PyTorch 2.0-2.2 (may be fixed in 2.3+, needs runtime verification)
- **macOS fork + mmap behavior:** Forked workers sharing file descriptors (needs testing on target macOS version)

---
*Research completed: 2026-02-02*
*Ready for roadmap: YES*
