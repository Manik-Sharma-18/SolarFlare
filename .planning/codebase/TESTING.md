# Testing Patterns

**Analysis Date:** 2026-02-02

## Test Framework

**Status:** No automated testing framework detected

**Finding:**
- No test files exist in the codebase (search for `*test*.py`, `test_*.py`, `*_test.py` found nothing)
- No test runner configuration (no `pytest.ini`, `setup.cfg`, `tox.ini`, `conftest.py`)
- No testing dependencies in `requirements.txt` (contains only: torch, numpy, matplotlib, pyyaml, tqdm)
- No test directory structure (no `tests/`, `test/`, `spec/` directories)

**Implication:** This is a research/prototype codebase where testing has not been formalized. Validation occurs through manual evaluation in notebooks and visual inspection of outputs.

## Validation Approach

**Current Strategy:** Manual evaluation with visual outputs and metrics logging

**Patterns Observed:**

1. **Output Validation:**
   - Models save checkpoints only when validation loss improves
   - Best model selected via early stopping with patience mechanism
   - Example from `trainer.py`:
   ```python
   if val_loss < best_val_loss:
       best_val_loss = val_loss
       torch.save(checkpoint_dict, checkpoint_path)
       print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
       patience_counter = 0
   ```

2. **Metrics Computation:**
   - File: `utils/metrics.py`
   - Functions available but not integrated into test suite:
     - `compute_metrics()`: Returns MAE total and per-timestep
     - `compute_rmse()`: Root Mean Squared Error
     - `compute_correlation()`: Pearson correlation coefficient
   - Example from validation loop:
   ```python
   metrics = compute_metrics(predictions, Y_target)
   all_mae_per_timestep.append(metrics['mae_per_timestep'])
   ```

3. **Inference Testing:**
   - `inference.py` module for running predictions on new data
   - Manual visual inspection via:
     - `visualize_predictions()`: Saves prediction comparisons
     - `animate_flare_sequence()`: Creates animations from GIFs
     - `animate_prediction_vs_truth()`: Side-by-side prediction comparison

4. **Configuration-Driven Validation:**
   - Model parameters logged at training start
   - Training history saved as JSON for post-hoc analysis
   - Example from `main.py`:
   ```python
   test_results = {
       'test_loss': float(test_loss),
       'test_mae_per_timestep': test_mae_per_timestep.tolist()
   }
   with open(output_dir / 'test_results.json', 'w') as f:
       json.dump(test_results, f, indent=2)
   ```

## Data Validation

**Dataset Validation:**
- File: `solarflare_data/loader.py`
- Checks before dataset creation:
  - Directory existence: `if not data_path.exists(): raise FileNotFoundError`
  - File availability: `if len(npy_files) == 0: raise FileNotFoundError`
  - Successful loading: `if len(datasets) == 0: raise ValueError`

**Normalization Validation:**
- Computes and logs statistics on raw data
- Example from `loader.py`:
  ```python
  print(f"  Shape: T={flux_cube.shape[0]}, H={flux_cube.shape[1]}, W={flux_cube.shape[2]}")
  print(f"  Value range: [{flux_cube.min():.2f}, {flux_cube.max():.2f}]")
  ```

## Model Validation

**Checkpoint-Based Validation:**
- Training loop validates model every epoch
- Function: `validate()` in `trainer.py`
- Computes loss on validation set with no gradient computation
- Example:
```python
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    show_progress: bool = True,
    output_channels: int = 1
) -> tuple:
    """Validate model on a dataset."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_in, Y_out, _ in iterator:
            # ...
            loss = loss_fn(predictions, Y_target)
            total_loss += loss.item()
```

**Loss Computation:**
- Primary loss: Configurable via `loss` config key
- Options:
  - `'l1'`: Simple L1/MAE loss
  - `'composite'`: Combined L1 + MS-SSIM + weighted extreme loss
  - `'weighted'`: Emphasis on extreme flux values
- Example configuration:
```yaml
loss:
  type: "composite"
  l1_weight: 1.0
  ssim_weight: 0.5
  extreme_weight: 1.0
  use_ms_ssim: false
  ssim_data_range: 2.0
```

## Loss Functions

**Available Losses:**
- File: `training/losses.py`

1. **L1 Loss (MAE):**
   - PyTorch built-in `nn.L1Loss()`
   - Used as baseline

2. **MS-SSIM (Multi-Scale Structural Similarity):**
   - Custom implementation for visual quality
   - Computes SSIM at multiple scales
   - Helps preserve spatial structure details
   - Configuration:
     ```python
     ssim_val = ms_ssim(pred, target, data_range=self.ssim_data_range)
     ssim_loss = 1.0 - ssim_val
     ```

3. **WeightedMAELoss:**
   - Emphasizes extreme flux regions (solar flares)
   - Weight increases with absolute target value
   - Implementation:
   ```python
   abs_error = torch.abs(pred - target)
   abs_target = torch.abs(target)
   normalized_magnitude = abs_target / (abs_target.max() + 1e-6)
   weights = self.base_weight + self.extreme_weight * normalized_magnitude
   weighted_error = weights * abs_error
   ```

4. **CompositeLoss:**
   - Combines three components: L1 + MS-SSIM + weighted extreme
   - Total: `l1_weight * L1 + ssim_weight * (1 - MS_SSIM) + extreme_weight * WeightedMAE`
   - Optional component breakdown via `return_components=True`

## Training Validation Loop

**Pattern:**
```python
for epoch in range(1, epochs + 1):
    # Train
    train_loss = train_epoch(
        model, train_loader, optimizer, scaler, device,
        tf_ratio, epoch, loss_fn, use_amp, grad_clip, show_progress, output_channels
    )

    # Validate
    val_loss, val_mae_per_timestep = validate(
        model, val_loader, device, loss_fn, use_amp, show_progress, output_channels
    )

    # Log
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_mae_per_timestep'].append(val_mae_per_timestep.tolist())

    # Early stopping
    if val_loss < best_val_loss:
        # save checkpoint
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

## Uncertainty Quantification (MC Dropout)

**Testing Framework:** Monte Carlo Dropout

**Usage:**
- File: `models/uncertainty.py`
- Requires `dropout_rate > 0.0` in model config
- Runs forward pass multiple times with different dropout masks
- Function: `predict_with_uncertainty()`
- Example from `main.py`:
```python
if uncertainty_config.get('enabled', False):
    from models.uncertainty import predict_with_uncertainty
    n_samples = uncertainty_config.get('n_samples', 20)
    mean_pred, uncertainty = predict_with_uncertainty(
        model, X_in, n_samples=n_samples
    )
```

## Visualization as Validation

**Tools:**
- File: `utils/visualization.py`
- `visualize_predictions()`: Compare predicted vs ground truth frames
- `plot_training_history()`: Visualize loss curves and metrics over epochs

**Supported Formats:**
- File: `utils/animation.py`
- `animate_flare_sequence()`: Create animation from GIF
- `animate_prediction_vs_truth()`: Side-by-side prediction comparison
- `animate_with_uncertainty()`: Overlay uncertainty maps
- `create_difference_animation()`: Show error magnitudes

## Configuration-Based Testing

**Patterns:**
- Test parameters controlled via YAML config (`config.yaml`)
- No hardcoded test paths or parameters
- Example:
```yaml
data:
  use_preprocessed: true
  preprocessed_dir: "./data_processed"
  t_in: 10
  t_out: 4

training:
  batch_size: 1
  epochs: 25

uncertainty:
  enabled: false
  n_samples: 20
```

## Suggested Testing Approaches

**For Future Implementation:**

1. **Unit Tests (unittest or pytest):**
   - Test individual components: `ConvLSTMCell`, normalization functions
   - Location: `tests/` directory

2. **Data Validation Tests:**
   - Verify dataset shape correctness
   - Verify normalization statistics
   - Check for NaN/Inf values in preprocessing

3. **Model Tests:**
   - Forward pass shape tests
   - Gradient flow tests
   - Checkpoint save/load roundtrip

4. **Integration Tests:**
   - Full training pipeline on small subset
   - Data pipeline from raw files to dataloaders
   - Inference on test data

5. **Regression Tests:**
   - Maintain baseline metrics in JSON
   - Compare new runs against baseline
   - Track loss curves across versions

---

*Testing analysis: 2026-02-02*
