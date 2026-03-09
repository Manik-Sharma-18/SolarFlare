---
phase: quick-02
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - config.yaml
  - solarflare_data/dataset.py
  - solarflare_data/loader.py
  - training/losses.py
  - training/trainer.py
  - main.py
  - preprocess_data.py
  - utils/config_validator.py
  - data_processed/metadata.json
  - tests/test_losses.py
  - tests/test_data_pipeline.py
  - tests/test_metrics.py
  - tests/test_config.py
  - generate_comparison.py
autonomous: true
requirements: [QUICK-02]

must_haves:
  truths:
    - "Extreme threshold in config.yaml uses 99th percentile (0.277) instead of incorrect 0.3456"
    - "Dual-channel indicator uses normalized-space threshold, producing nonzero channel 2 values"
    - "Flare detection uses spatial density criterion (>2% pixels above threshold) instead of np.any()"
    - "Loss functions use 0.277 as default extreme threshold"
    - "Metadata stores both raw and normalized extreme thresholds"
  artifacts:
    - path: "config.yaml"
      provides: "Updated extreme_threshold values"
      contains: "extreme_threshold: 0.277"
    - path: "solarflare_data/dataset.py"
      provides: "Spatial density flare detection and working dual-channel"
      contains: "extreme_pixel_fraction"
    - path: "solarflare_data/loader.py"
      provides: "Normalized threshold for dual-channel in preprocessed path"
      contains: "extreme_threshold_normalized"
  key_links:
    - from: "config.yaml"
      to: "training/losses.py"
      via: "extreme_threshold config value"
      pattern: "0\\.277"
    - from: "solarflare_data/loader.py"
      to: "solarflare_data/dataset.py"
      via: "normalized threshold passed to dataset"
      pattern: "extreme_threshold"
    - from: "solarflare_data/dataset.py"
      to: "build_index flare detection"
      via: "spatial density criterion"
      pattern: "extreme_pixel_fraction"
---

<objective>
Fix three fundamental bugs in the extreme threshold system: (1) correct the hardcoded 0.3456 value to 0.277 (99th percentile), (2) fix the dual-channel indicator to use normalized-space threshold instead of raw-space, and (3) replace the useless np.any() flare detection with a spatial density criterion (>2% of pixels above threshold).

Purpose: The model's second input channel is currently all zeros (dead), flare detection flags every window as positive, and the CSI/HSS threshold is 10% too high. Fixing these makes dual-channel mode, flare oversampling, and extreme-event metrics actually functional.

Output: Updated config, dataset, loader, loss, trainer, and test files with correct threshold and density-based flare detection.
</objective>

<execution_context>
@/Users/indra/.claude/get-shit-done/workflows/execute-plan.md
@/Users/indra/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@config.yaml
@solarflare_data/dataset.py
@solarflare_data/loader.py
@training/losses.py
@training/trainer.py
@main.py
@preprocess_data.py
@data_processed/metadata.json

<interfaces>
<!-- Key threshold flow through the codebase -->

config.yaml:
  loss.extreme_threshold: 0.3456       # -> WeightedMAELoss, AsymmetricExtremeLoss, CompositeLoss
  evaluation.extreme_threshold: 0.3456 # -> validate() CSI/HSS, main.py flare_extreme_threshold
  normalization.extreme_threshold_percentile: 99.5  # -> preprocess_data.py percentile calc

data_processed/metadata.json:
  normalization.extreme_threshold: 28070.39  # RAW space, read by load_preprocessed_data()

solarflare_data/loader.py:
  load_preprocessed_data() line 651: extreme_threshold = norm_info.get('extreme_threshold')
  # This raw value (28070) gets passed to SolarFluxDataset.extreme_threshold
  # which _compute_extreme_channel() uses against NORMALIZED data -> sigmoid always ~0

solarflare_data/dataset.py:
  _compute_extreme_channel(flux): uses self.extreme_threshold against normalized flux
  build_index(): is_flare = bool(np.any(output_frames > extreme_threshold))

training/losses.py:
  WeightedMAELoss(threshold=0.3456)
  AsymmetricExtremeLoss(threshold=0.3456)
  CompositeLoss(extreme_threshold=0.3456)
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix config, metadata, and add normalized threshold to preprocessing</name>
  <files>config.yaml, preprocess_data.py, data_processed/metadata.json, utils/config_validator.py</files>
  <action>
1. **config.yaml**: Change ALL occurrences of `extreme_threshold: 0.3456` to `extreme_threshold: 0.277` (the 99th percentile in normalized space). This affects both `loss.extreme_threshold` and `evaluation.extreme_threshold`. Change `extreme_threshold_percentile: 99.5` to `extreme_threshold_percentile: 99` in the normalization section. Add a comment explaining the value: `# 99th percentile in normalized asinh space (~17,400 raw)`.

2. **config.yaml**: Add a new config key under `data:` section: `flare_density_threshold: 0.02` with comment `# Fraction of pixels above extreme threshold to flag as flare (0.02 = 2%)`.

3. **preprocess_data.py**: In `compute_normalization()`, after computing `extreme_threshold` (raw space), also compute and store the normalized-space equivalent:
   ```python
   extreme_threshold_normalized = float(np.arcsinh(extreme_threshold / asinh_softening) / scale)
   ```
   Add `'extreme_threshold_normalized': extreme_threshold_normalized` to the returned dict.

4. **data_processed/metadata.json**: Add `"extreme_threshold_normalized": 0.3143` to the normalization section (alongside the existing raw `extreme_threshold: 28070.39`). This is computed as `arcsinh(28070.39 / 1000.0) / 12.815 = 0.3143`. Keep the existing raw value for backward compatibility.

5. **utils/config_validator.py**: Update the default value comment from 0.3456 to 0.277 in the extreme_threshold validation section. Keep the cross-check warning logic between loss and evaluation thresholds.
  </action>
  <verify>
    <automated>python3 -c "
import yaml, json
config = yaml.safe_load(open('config.yaml'))
assert config['loss']['extreme_threshold'] == 0.277, f'loss threshold: {config[\"loss\"][\"extreme_threshold\"]}'
assert config['evaluation']['extreme_threshold'] == 0.277, f'eval threshold: {config[\"evaluation\"][\"extreme_threshold\"]}'
assert config['data'].get('flare_density_threshold') == 0.02, 'missing flare_density_threshold'
meta = json.load(open('data_processed/metadata.json'))
assert 'extreme_threshold_normalized' in meta['normalization'], 'missing normalized threshold in metadata'
assert abs(meta['normalization']['extreme_threshold_normalized'] - 0.3143) < 0.001
print('All config/metadata checks passed')
"</automated>
  </verify>
  <done>config.yaml uses 0.277 everywhere, metadata.json has normalized threshold, preprocess_data.py computes normalized threshold</done>
</task>

<task type="auto">
  <name>Task 2: Fix dual-channel indicator and replace np.any() with spatial density flare detection</name>
  <files>solarflare_data/dataset.py, solarflare_data/loader.py, main.py</files>
  <action>
1. **solarflare_data/loader.py** `load_preprocessed_data()` (around line 651): The current code reads `extreme_threshold = norm_info.get('extreme_threshold', None)` which gets the RAW value (28070). Change this to prefer the normalized value:
   ```python
   extreme_threshold = norm_info.get('extreme_threshold_normalized', None) if dual_channel else None
   ```
   If `extreme_threshold_normalized` is not in metadata (backward compat), compute it on the fly:
   ```python
   if extreme_threshold is None and dual_channel:
       raw_et = norm_info.get('extreme_threshold')
       if raw_et is not None:
           softening = norm_info.get('asinh_softening', 1000.0)
           scale = norm_info.get('scale', 1.0)
           import math
           extreme_threshold = math.asinh(raw_et / softening) / scale
           logger.info("Computed normalized extreme_threshold: %.4f from raw %.2f", extreme_threshold, raw_et)
   ```
   This ensures `_compute_extreme_channel()` receives a value in [-1,1] space, not raw space.

2. **solarflare_data/loader.py** `load_and_prepare_data()` (around line 498): Same fix. The raw `extreme_threshold` from `norm_params` is passed to the dataset. After normalization is computed, also compute the normalized version:
   ```python
   if dual_channel:
       raw_et = norm_params.get('extreme_threshold')
       if raw_et is not None and norm_method == 'asinh':
           softening = norm_params.get('asinh_softening', 1000.0)
           scale_val = norm_params.get('scale', 1.0)
           import math
           extreme_threshold = math.asinh(raw_et / softening) / scale_val
       else:
           extreme_threshold = norm_params.get('extreme_threshold', None)
   else:
       extreme_threshold = None
   ```

3. **solarflare_data/dataset.py** `build_index()` (line 305-309): Replace the `np.any()` flare detection with spatial density criterion. Change the function signature to accept `flare_density_threshold: float = 0.02` instead of just a binary any-pixel check. The new logic:
   ```python
   if extreme_threshold is not None:
       output_frames = mmap[window_start + t_in : window_start + t_in + t_out]
       # Spatial density: fraction of pixels above threshold per frame
       extreme_pixels = np.abs(output_frames) > extreme_threshold
       extreme_fraction = extreme_pixels.mean()  # across all output pixels
       is_flare = bool(extreme_fraction > flare_density_threshold)
   ```
   Note: use `np.abs()` for consistency with the dual-channel indicator which uses absolute values.

4. **solarflare_data/dataset.py** `build_index()`: Update the docstring to reflect the spatial density approach. Remove "any pixel > threshold" language, replace with "spatial density exceeds flare_density_threshold".

5. **solarflare_data/loader.py**: Pass the new `flare_density_threshold` through both `load_and_prepare_data()` and `load_preprocessed_data()` to `build_index()`. Add the parameter with default 0.02.

6. **main.py** (lines 82-88): Pass the flare_density_threshold from config to the data loading functions:
   ```python
   flare_density_threshold = config['data'].get('flare_density_threshold', 0.02)
   ```
   Pass it to both `load_preprocessed_data()` and `load_and_prepare_data()`.

IMPORTANT: The `_compute_extreme_channel()` in dataset.py does NOT need its formula changed -- it is a sigmoid around the threshold, which will now work correctly since the threshold will be ~0.277 or ~0.314 in normalized space instead of 28070. The sigmoid `(|flux| - 0.277) / (0.277 * 0.5)` will produce meaningful values for normalized flux in [-1, 1].
  </action>
  <verify>
    <automated>python3 -c "
import numpy as np
import sys
sys.path.insert(0, '.')

# Test 1: _compute_extreme_channel with normalized threshold produces nonzero output
from solarflare_data.dataset import SolarFluxDataset
ds = SolarFluxDataset.__new__(SolarFluxDataset)
ds.extreme_threshold = 0.277  # normalized space
flux = np.array([[[0.0, 0.1, 0.3, 0.5, -0.4]]])  # (1,1,5) some above threshold
result = ds._compute_extreme_channel(flux)
assert result.max() > 0.5, f'Channel 2 should have high values for pixels > 0.277, got max={result.max():.4f}'
assert result.min() < 0.5, f'Channel 2 should have low values for pixels < 0.277, got min={result.min():.4f}'
print(f'Dual-channel test passed: min={result.min():.4f}, max={result.max():.4f}')

# Test 2: build_index with density criterion
from solarflare_data.dataset import build_index
import tempfile, os
# Create a cube where only 1% of pixels are extreme (should NOT be flare at 2% threshold)
cube = np.random.uniform(-0.1, 0.1, (20, 10, 10)).astype(np.float32)
# Make exactly 1 out of 100 pixels extreme in output frames
cube[15, 0, 0] = 0.5  # 1 pixel out of 100 per frame
tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
np.save(tmp.name, cube)
tmp.close()
idx, flags = build_index(
    file_paths=[tmp.name],
    file_assignments={'train': [0]},
    t_in=10, t_out=4, stride=1,
    split='train',
    extreme_threshold=0.277,
    flare_density_threshold=0.02,
)
# With only 1/100 pixels extreme per frame, density < 2%, should be False
n_flare = sum(flags)
print(f'Density test: {n_flare}/{len(flags)} flagged as flare (expect 0 or very few)')
os.unlink(tmp.name)

# Test 3: build_index with many extreme pixels (should be flare)
cube2 = np.random.uniform(-0.1, 0.1, (20, 10, 10)).astype(np.float32)
cube2[10:14, :3, :] = 0.5  # 30% of pixels in output frames are extreme
tmp2 = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
np.save(tmp2.name, cube2)
tmp2.close()
idx2, flags2 = build_index(
    file_paths=[tmp2.name],
    file_assignments={'train': [0]},
    t_in=10, t_out=4, stride=1,
    split='train',
    extreme_threshold=0.277,
    flare_density_threshold=0.02,
)
n_flare2 = sum(flags2)
print(f'High-density test: {n_flare2}/{len(flags2)} flagged as flare (expect most/all)')
assert n_flare2 > 0, 'Dense extreme pixels should trigger flare flag'
os.unlink(tmp2.name)

print('All dataset/loader checks passed')
"</automated>
  </verify>
  <done>Dual-channel indicator produces nonzero values for extreme pixels in normalized space; flare detection uses spatial density (>2% pixels) instead of np.any(); both load paths pass normalized threshold</done>
</task>

<task type="auto">
  <name>Task 3: Update loss defaults, trainer defaults, tests, and generate_comparison.py</name>
  <files>training/losses.py, training/trainer.py, tests/test_losses.py, tests/test_metrics.py, tests/test_config.py, tests/test_data_pipeline.py, generate_comparison.py</files>
  <action>
1. **training/losses.py**: Change ALL default `threshold=0.3456` and `extreme_threshold=0.3456` to `0.277` in:
   - `WeightedMAELoss.__init__` (line 276)
   - `AsymmetricExtremeLoss.__init__` (line 315)
   - `CompositeLoss.__init__` (line 459)
   - `get_loss_function()` fallback (line 625)

2. **training/trainer.py**: Change default `extreme_threshold: float = 0.3456` to `0.277` in:
   - `validate()` signature (line 200)
   - `train_model()` where it reads eval config with fallback (line ~567)

3. **main.py**: Change the fallback defaults from 0.3456 to 0.277:
   - Line 85: `config.get('evaluation', {}).get('extreme_threshold', 0.277)`
   - Line 232: `extreme_threshold = eval_config.get('extreme_threshold', 0.277)`

4. **tests/test_losses.py**: Update ALL test instances of `threshold=0.3456` to `threshold=0.277`:
   - Lines 136, 143, 166 (WeightedMAELoss tests)
   - Lines 226, 248, 268 (AsymmetricExtremeLoss tests)
   - Line 578 (CompositeLoss config test)
   - Update the comment at line 147 from "above threshold=0.3456" to "above threshold=0.277"
   - Update line 182 comment similarly

5. **tests/test_metrics.py**: Update lines 362, 400 from `extreme_threshold=0.3456` to `0.277`.

6. **tests/test_config.py**: Update lines 195, 241 from `"extreme_threshold": 0.3456` to `0.277`.

7. **tests/test_data_pipeline.py**: If any references to 0.3456, update to 0.277.

8. **generate_comparison.py**: Update the three references to 0.3456 (lines 494, 644, 676) to 0.277. Change text from "99.5th percentile" to "99th percentile" where applicable.

Run existing test suite to confirm nothing breaks.
  </action>
  <verify>
    <automated>cd /Volumes/T9/IndraAstra/manik/SolarFlare && python3 -m pytest tests/test_losses.py tests/test_config.py tests/test_metrics.py -x -q 2>&1 | tail -20</automated>
  </verify>
  <done>All default thresholds changed from 0.3456 to 0.277 across loss functions, trainer, main.py, and all tests. Test suite passes.</done>
</task>

</tasks>

<verification>
1. `grep -rn "0\.3456" --include="*.py" --include="*.yaml" .` returns zero matches (all replaced with 0.277)
2. `python3 -m pytest tests/ -x -q` passes all tests
3. Dual-channel indicator produces values in (0, 1) range for normalized flux data, not all zeros
4. Flare detection with spatial density flags ~5-15% of windows, not 100%
</verification>

<success_criteria>
- No occurrence of 0.3456 remains anywhere in .py or .yaml files
- metadata.json contains extreme_threshold_normalized field
- _compute_extreme_channel() produces meaningful nonzero output when given normalized flux
- build_index() uses spatial density criterion (fraction > 0.02) instead of np.any()
- All existing tests pass with updated threshold values
- config.yaml has flare_density_threshold: 0.02 under data section
</success_criteria>

<output>
After completion, create `.planning/quick/2-fix-broken-extreme-threshold-correct-val/2-SUMMARY.md`
</output>
