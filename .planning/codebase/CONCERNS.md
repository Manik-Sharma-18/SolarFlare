# Codebase Concerns

**Analysis Date:** 2026-02-02

## Tech Debt

**Legacy Prototype File Still in Repository:**
- Issue: `ConvLSTM.py` (698 lines) appears to be a prototype/backup version that duplicates functionality from the organized module structure in `models/`, `solarflare_data/`, and `training/`
- Files: `/Users/manik/Solar/SolarFlare/ConvLSTM.py`
- Impact: Code maintainability confusion. Future changes may need to be made in multiple places. Unclear which version is authoritative. Increases merge/update complexity.
- Fix approach: Verify that `ConvLSTM.py` is not used by any scripts. Archive to a separate `deprecated/` folder or remove entirely. Update any references to use the organized module structure instead.

**Duplicate Dataset Implementation:**
- Issue: `ConvLSTM.py` contains its own `SolarFluxDataset` class that mirrors the one in `solarflare_data/dataset.py`, but with different features (no dual-channel, no extreme_threshold support)
- Files: `ConvLSTM.py:19-62` vs `solarflare_data/dataset.py:13-124`
- Impact: Maintenance burden. Newer dual-channel features are not available in the prototype. Risk of regression if someone inadvertently uses outdated dataset class.
- Fix approach: Delete the duplicate in `ConvLSTM.py` after confirming no active code depends on it.

**Preprocessing Step Not Integrated:**
- Issue: `preprocess_data.py` exists to convert raw `.npy` files to preprocessed cubes, but the main pipeline can operate in two modes (raw vs preprocessed) with different code paths
- Files: `preprocess_data.py`, `solarflare_data/loader.py:187-314` (load_preprocessed_data), and main.py:47-72
- Impact: Two separate data loading pipelines need to stay in sync. If bugs are fixed in one path, they may persist in the other. Configuration complexity.
- Fix approach: Migrate all users to preprocessed data path. Consider removing raw data loading after deprecation period. Or: Create unified loader that handles both transparently.

## Memory & Performance Concerns

**Potential Memory Issues During Inference:**
- Issue: `predict_with_uncertainty()` in `models/uncertainty.py:67-72` stacks all n_samples predictions in memory before computing statistics. For large spatial sizes (H, W) and many samples (n_samples=20+), this creates a tensor of shape (n_samples, B, C, T_out, H, W) which can exceed VRAM
- Files: `models/uncertainty.py:18-85`
- Impact: OOM (Out of Memory) errors when running uncertainty estimation on high-resolution data
- Improvement path: Accumulate statistics incrementally (running mean/std) instead of storing all samples. Use Welford's algorithm to update mean/variance one sample at a time.

**Inefficient Data Loading for Large Sequences:**
- Issue: `load_and_prepare_data()` in `solarflare_data/loader.py:14-184` loads all files into memory as dense cubes and keeps them resident for the entire training session
- Files: `solarflare_data/loader.py:61-98`
- Impact: For datasets with many timesteps or large spatial dimensions (e.g., 1000+ frames, 512x512), RAM usage can be substantial (multiple GB). Limits scalability.
- Improvement path: Lazy-load or memory-map `.npy` files. Load only the specific frames needed by the DataLoader batch. Consider using HDF5 with chunked storage.

**No Memory Cleanup Between Epochs:**
- Issue: Training loop accumulates computation graphs and cached tensors without explicit cleanup
- Files: `training/trainer.py:240-299` (train_model main loop)
- Impact: Long training runs (25+ epochs) may experience memory fragmentation or gradual memory growth due to PyTorch's caching behavior
- Improvement path: Add `torch.cuda.empty_cache()` after validation phase if using GPU. Profile memory during training to identify leaks.

## Error Handling & Robustness

**Insufficient Error Handling in Data Loading:**
- Issue: Exception handler in `solarflare_data/loader.py:74-97` catches and silently continues on file load errors, only printing a message
- Files: `solarflare_data/loader.py:74-97`
- Impact: If most data files fail to load, training may proceed with insufficient data without alarming the user. Silent failures make debugging difficult.
- Fix approach: Raise an error if more than N% of files fail to load. Log which specific files failed with detailed error messages. Require user acknowledgment before proceeding with partial data.

**Hardcoded Extreme Threshold Default:**
- Issue: In `solarflare_data/loader.py:289`, if dual-channel mode is enabled but no `extreme_threshold` in metadata, code defaults to hardcoded 0.8
- Files: `solarflare_data/loader.py:281-289`
- Impact: Silently uses incorrect threshold without warning. May produce inconsistent results between old/new preprocessed data
- Fix approach: Raise error if extreme_threshold missing when dual_channel=True. Require it to be explicitly set in config or computed during data loading.

**Missing Validation of Config Parameters:**
- Issue: `main.py` and `training/trainer.py` read config values without comprehensive validation or defaults for critical parameters
- Files: `main.py:29-33`, `training/trainer.py:171-219`
- Impact: Typos in config.yaml silently use None or cause cryptic errors later in training
- Fix approach: Add config validation function that checks all required keys exist and are correct types before training starts.

**Unbounded Gradient Accumulation:**
- Issue: `training/trainer.py:70-84` uses gradient clipping, but only after scaled backward. If very large gradients cause overflow before clipping, loss can become NaN
- Files: `training/trainer.py:66-84`
- Impact: Training can fail silently with NaN losses without clear indication of root cause
- Improvement path: Add checks for NaN/Inf losses after backward pass. Log gradient statistics. Consider gradient clipping before backward instead of after.

## Known Issues & Workarounds

**Model Output Channel Handling in Dual-Channel Mode:**
- Issue: When `dual_channel=True`, model accepts 2-channel input (flux + extreme) but outputs 1 channel (flux only). The residual prediction logic in `models/predictor.py:280-308` reconstructs multi-channel output by concatenating predicted flux with other input channels
- Files: `models/predictor.py:280-308`, `solarflare_data/dataset.py:83-92`
- Symptoms: Confusing tensor shape conversions. If output_channels != 1, the reconstruction logic may fail
- Workaround: Keep output_channels=1 when using dual_channel mode. Extreme indicator is not predicted, only used as input context.

**Teacher Forcing with Multi-Channel Data:**
- Issue: `models/predictor.py:291-308` handles teacher forcing by mixing ground-truth flux with predicted/input extreme channels. This assumes extreme channel can be reused from input without recomputation
- Files: `models/predictor.py:291-308`
- Symptoms: If extreme channel should be recomputed based on predicted flux values, it won't be during teacher forcing
- Current behavior: Extreme indicator stays constant throughout prediction sequence during teacher forcing
- Safe modification: Only use teacher_forcing_ratio=0.0 with dual-channel mode. If teacher forcing needed, disable dual-channel.

**Inconsistent Normalization Between Training and Inference:**
- Issue: Normalization parameters are computed during data loading and stored in metadata, but `inference.py:68-75` assumes the same normalization scheme. If metadata.json is missing or uses different parameters, inference results will be incorrect
- Files: `inference.py:61-75`, `solarflare_data/loader.py:102-122`
- Impact: Model reloaded in different environment (e.g., production) may apply wrong normalization, producing nonsensical predictions
- Fix approach: Embed normalization parameters in the checkpoint file alongside model weights, not just metadata.json.

## Security Considerations

**Unsafe YAML Loading:**
- Issue: `main.py:31` uses `yaml.safe_load()` which is safe, but if config file ever contains arbitrary Python objects, could be exploited
- Files: `main.py:29-33`
- Current mitigation: Using `safe_load()` prevents pickle-based attacks
- Recommendations: Validate all config values against expected schema after loading. Use typed config class (e.g., dataclasses, Pydantic) instead of raw dict access.

**Path Traversal in File Loading:**
- Issue: Data directory paths are read from config without validation. User could specify `../../../etc/passwd` as data_dir
- Files: `solarflare_data/loader.py:42-45`, `main.py:64`
- Current mitigation: Code uses `.glob()` pattern matching which implicitly limits scope
- Recommendations: Validate that resolved data path is within expected directory (e.g., project root). Use `Path.resolve()` and check that it starts with allowed base path.

## Performance Bottlenecks

**SSIM Loss Computation on Large Tensors:**
- Issue: `training/losses.py:24-72` computes SSIM using 2D convolutions on full-resolution feature maps. With 5D tensors (B, C, T, H, W), this is expensive
- Files: `training/losses.py:75-120` (ms_ssim)
- Problem: Using SSIM loss when spatial size >= 32 (line 220 in trainer.py). For 256x256 or larger, convolution overhead dominates
- Cause: F.conv2d is called multiple times (2-5 scales). No batching of multiple scales.
- Improvement path: Use pre-computed SSIM lookup tables or approximations for fast iteration. Cache Gaussian kernels. Skip SSIM for first N epochs when training is unstable.

**Inefficient Teacher Forcing Probability Check:**
- Issue: `models/predictor.py:285-289` calls `np.random.rand()` for each timestep in a loop. This is slower than vectorized random sampling
- Files: `models/predictor.py:239-309`
- Impact: Inference slows down when teacher_forcing_ratio > 0. Not critical but suboptimal.
- Improvement path: Sample all timesteps' decisions at once before the loop: `use_teacher_mask = np.random.rand(self.t_out) < teacher_forcing_ratio`

**Checkpoint File Includes Full Config:**
- Issue: Checkpoint saved at `training/trainer.py:278-287` includes entire config dict in state, which can bloat the file size
- Files: `training/trainer.py:278-287`
- Impact: For large configs or multiple checkpoints, disk space accumulates. Slower save/load operations.
- Improvement path: Save only essential parameters (t_in, t_out, input_channels, output_channels). Store full config separately or inline critical values as model attributes.

## Fragile Areas

**ConvLSTM State Initialization:**
- Files: `models/convlstm.py:174-188`
- Why fragile: Hidden state initialization assumes zero tensor initialization. If model is used in streaming/online mode where states need to persist across batches, initialization must happen once not every forward pass. Currently happens every call in the forward method.
- Safe modification: Add optional `hidden_state` parameter to forward pass (already present in code). Document that repeated forward calls without passing hidden_state will reinitialize.
- Test coverage: No tests for state persistence across multiple forward passes

**Encoder-Decoder Skip Connection Dimension Matching:**
- Files: `models/predictor.py:261-268`
- Why fragile: Skip connection assumes matching spatial dimensions after upsampling. If spatial size is odd, F.interpolate with 'nearest' mode can produce mismatched sizes
- Safe modification: Always ensure spatial dims are powers of 2. Add assertion before concatenation: `assert dec_up.shape[2:] == h1_skip.shape[2:]`
- Test coverage: No tests for odd spatial dimensions (e.g., 321x321)

**Data Augmentation Consistency:**
- Files: `solarflare_data/dataset.py:73-81`
- Why fragile: Augmentation applies horizontal and vertical flips independently to same data. If both flips happen, equivalent to 180° rotation. Not a bug but unintuitive.
- Safe modification: Document behavior. Consider using mutually exclusive flips if 180° rotation not desired.
- Test coverage: No tests verify augmentation produces expected transformations

## Scaling Limits

**Fixed Batch Size of 1:**
- Config: `config.yaml:37` sets `batch_size: 1`
- Current capacity: Single sample per batch (B=1)
- Limit: Cannot efficiently use GPU parallelism. Training is I/O bound, not compute bound
- Scaling path: Increase batch size if GPU memory allows. May require smaller spatial resolution or reduced t_in/t_out. Profile memory usage at batch_size=2, 4 to find limit.

**Preprocessing Requires Full Data in Memory:**
- Issue: `solarflare_data/loader.py:115-122` normalizes all datasets at once
- Current capacity: Can handle ~100GB of .npy files on typical server with 256GB RAM
- Limit: OOM for petabyte-scale datasets or embedded systems
- Scaling path: Implement streaming normalization (read chunks, update running statistics, write normalized chunks back)

**Sequential Validation/Test Loop:**
- Files: `training/trainer.py:255-258`
- Current capacity: Full dataset must fit in GPU memory during validation
- Limit: Validation becomes bottleneck for very large test sets
- Scaling path: Implement streaming validation with fixed-size accumulator for metrics

## Dependencies at Risk

**PyTorch Version Pinning:**
- Risk: `requirements.txt` specifies `torch>=2.0.0` which is a wide range
- Impact: Code may break with PyTorch 2.1+ if API changes occur (e.g., torch.amp.autocast signature)
- Files: `requirements.txt:4`, `training/trainer.py:14`, `utils/device.py:33-36`
- Recommendation: Pin to specific minor version `torch==2.1.2` after testing compatibility. Document PyTorch version requirements.

**NumPy Compatibility:**
- Risk: `requirements.txt` allows `numpy>=1.21.0`. Code uses `.ndarray.copy()` which is stable, but structured array handling could break in future versions
- Files: `solarflare_data/loader.py:128-130` (unique, structured arrays)
- Recommendation: Test with latest NumPy version quarterly. Pin to `numpy<2.0` for now (NumPy 2.0 has breaking changes).

## Missing Features & Coverage Gaps

**No Test Coverage:**
- Issue: No `tests/` directory. No unit tests for any modules.
- Files: Entire project lacks `.test.py` or `_test.py` files
- Risk: Refactoring has high regression risk. Data loading logic untested. Loss functions untested.
- Priority: HIGH
- Fix approach: Create `tests/` directory with pytest. Add tests for: data loading, normalization, dataset splitting, loss functions, model forward pass shapes.

**No Checkpoint Resume Logic:**
- Issue: `training/trainer.py` saves best model but doesn't support resuming from checkpoint mid-training
- Files: `training/trainer.py:311-333` (only loads model, not optimizer/scheduler state)
- Impact: If training crashes after 20 epochs, must restart from epoch 1 (wasting 20 epochs of progress)
- Fix approach: Add `--resume-from-checkpoint` flag to main.py. Load optimizer and scheduler state from checkpoint.

**No Hyperparameter Tuning Framework:**
- Issue: Config values are global. No sweep over learning rates, batch sizes, architectures
- Files: `main.py`, `config.yaml`
- Impact: Finding optimal hyperparameters requires manual config edits
- Recommendation: Integrate with Optuna or Ray Tune for automated hyperparameter search

**No Validation Metrics Beyond MAE:**
- Issue: Only MAE per timestep is tracked. No rank correlation, spatial accuracy metrics, or skill scores for solar flare prediction
- Files: `utils/metrics.py:55` (compute_metrics returns only mae_per_timestep)
- Impact: Cannot assess whether model is learning physically meaningful patterns
- Fix approach: Add spatial MSE, temporal correlation, ROC-AUC for flare detection thresholds.

**Limited Inference API:**
- Issue: `inference.py` is standalone script, not importable module. No batch prediction support.
- Files: `inference.py`
- Impact: Hard to integrate model into production pipelines
- Recommendation: Refactor into `models/inference.py` with class-based API supporting batch predictions, batched uncertainty, streaming predictions.

## Untested Areas

**Multi-Scale SSIM Loss:**
- What's not tested: MS-SSIM implementation in `training/losses.py:75-120`
- Files: `training/losses.py:75-120`
- Risk: Downsampling logic or weight application could be incorrect. No reference implementation comparison.
- Recommendation: Compare output with torchvision's SSIM (if available) or official MS-SSIM paper implementation.

**Gradient Checkpointing:**
- What's not tested: Gradient checkpointing in `models/predictor.py:215-220` produces same gradients as non-checkpointed version
- Files: `models/predictor.py:215-220`
- Risk: Recomputation could introduce numerical differences. Checkpointing not enabled by default (disabled in config.yaml)
- Recommendation: Run test comparing gradients with/without checkpointing enabled. Verify loss matches.

**Dual-Channel Mode End-to-End:**
- What's not tested: Full pipeline with dual_channel=true. Config has it enabled but unclear if all code paths validated.
- Files: Config `data.dual_channel: true`, but no tests for entire pipeline
- Risk: Shape mismatches, incorrect extreme threshold, recomputation logic
- Recommendation: Run training with dual_channel=true through full pipeline. Validate output shapes at each stage.

**Uncertainty Quantification Integration:**
- What's not tested: End-to-end uncertainty estimation during main training
- Files: `main.py:188-224` (uncertainty section) - only runs if enabled
- Risk: Code may be broken due to lack of usage. Dropout integration may not work as expected.
- Recommendation: Enable uncertainty=true in config and test full pipeline. Verify uncertainty estimates are reasonable (not all constant or NaN).

---

*Concerns audit: 2026-02-02*
