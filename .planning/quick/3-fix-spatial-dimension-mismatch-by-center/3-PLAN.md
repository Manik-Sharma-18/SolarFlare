---
phase: quick-03
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - config.yaml
  - main.py
  - solarflare_data/loader.py
autonomous: true
requirements: [QUICK-03]

must_haves:
  truths:
    - "All 14 cubes are center-cropped to 437x877 at load time, not bilinear-resized"
    - "Physical scale (0.36 Mm/pixel) is preserved identically across all cube groups"
    - "Config uses crop_size key instead of target_size"
    - "Both raw and preprocessed load paths apply center-crop"
    - "Existing test suite still passes (206 tests)"
  artifacts:
    - path: "config.yaml"
      provides: "crop_size: [437, 877] replacing target_size: [448, 896]"
      contains: "crop_size"
    - path: "main.py"
      provides: "Reads crop_size from config and passes to loader"
      contains: "crop_size"
    - path: "solarflare_data/loader.py"
      provides: "Center-crop logic replacing F.interpolate bilinear resize"
      contains: "center_crop"
  key_links:
    - from: "config.yaml"
      to: "main.py"
      via: "config['data']['crop_size']"
      pattern: "crop_size"
    - from: "main.py"
      to: "solarflare_data/loader.py"
      via: "crop_size parameter to load_and_prepare_data and load_preprocessed_data"
      pattern: "crop_size=crop_size"
    - from: "solarflare_data/loader.py"
      to: "loaded cube data"
      via: "center-crop slicing instead of F.interpolate"
      pattern: "center_crop"
---

<objective>
Replace bilinear resize with center-crop for spatial dimension normalization across all 14 data cubes.

Purpose: Three cube groups have different native dimensions (440x884, 627x877, ~520x1044) but share 0.36 Mm/pixel spacing. Bilinear resize to 448x896 distorts physical scale differently per group, creating inconsistent Mm/pixel for ConvLSTM kernels and a train/test domain gap. Center-cropping to 437x877 (largest common rectangle) preserves physical scale perfectly.

Output: Updated config.yaml, main.py, and loader.py with crop-based spatial normalization.
</objective>

<execution_context>
@/Users/indra/.claude/get-shit-done/workflows/execute-plan.md
@/Users/indra/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/3-fix-spatial-dimension-mismatch-by-center/3-CONTEXT.md

<interfaces>
<!-- Key signatures the executor needs from the existing codebase -->

From main.py (lines 94-95, config reading):
```python
target_size_cfg = config['data'].get('target_size')
target_size = tuple(target_size_cfg) if target_size_cfg else None
```

From main.py (lines 110, 128, passed to both load functions):
```python
target_size=target_size,
```

From solarflare_data/loader.py load_and_prepare_data signature (line 356):
```python
target_size: Optional[Tuple[int, int]] = None,
```

From solarflare_data/loader.py load_and_prepare_data resize block (lines 423-430):
```python
if target_size is not None:
    th, tw = target_size
    if flux_cube.shape[1] != th or flux_cube.shape[2] != tw:
        import torch as _torch
        import torch.nn.functional as _F
        t = _torch.from_numpy(flux_cube).float().unsqueeze(1)
        t = _F.interpolate(t, size=(th, tw), mode='bilinear', align_corners=False)
        flux_cube = t.squeeze(1).numpy()
```

From solarflare_data/loader.py load_preprocessed_data signature (line 578):
```python
target_size: Optional[Tuple[int, int]] = None,
```

From solarflare_data/loader.py load_preprocessed_data resize block (lines 641-654):
```python
if target_size is not None:
    th, tw = target_size
    if cube.shape[1] != th or cube.shape[2] != tw:
        import torch as _torch
        import torch.nn.functional as _F
        t = _torch.from_numpy(cube).float().unsqueeze(1)
        t = _F.interpolate(t, size=(th, tw), mode='bilinear', align_corners=False)
        cube = t.squeeze(1).numpy()
if target_size and (orig_shape[1] != cube.shape[1] or orig_shape[2] != cube.shape[2]):
    print(f"    Shape: {orig_shape} -> {cube.shape} (resized)")
```

From config.yaml (line 21):
```yaml
target_size: [448, 896] # Resize all samples to uniform (H, W) for batch_size > 1
```

From config.yaml (line 50):
```yaml
batch_size: 4           # Batch size (target_size enables uniform spatial dims)
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace bilinear resize with center-crop in loader.py and update config pipeline</name>
  <files>solarflare_data/loader.py, main.py, config.yaml</files>
  <action>
**1. Add center-crop helper to loader.py** (near top, after imports, before any function defs):

Create a module-level helper function `_center_crop`:
```python
def _center_crop(arr: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Center-crop a (T, H, W) array to (T, crop_h, crop_w).

    Raises ValueError if crop dimensions exceed array dimensions.
    """
    _, h, w = arr.shape
    if crop_h > h or crop_w > w:
        raise ValueError(
            f"Crop size ({crop_h}, {crop_w}) exceeds array spatial dims ({h}, {w})"
        )
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return arr[:, y0:y0 + crop_h, x0:x0 + crop_w]
```

**2. Update `load_and_prepare_data()` in loader.py:**
- Rename parameter `target_size` to `crop_size` (line 356)
- Replace the resize block (lines 423-430) with center-crop:
  ```python
  if crop_size is not None:
      ch, cw = crop_size
      if flux_cube.shape[1] != ch or flux_cube.shape[2] != cw:
          flux_cube = _center_crop(flux_cube, ch, cw)
  ```
  This removes the `import torch` and `F.interpolate` calls entirely from this block.
- Update docstring to reference crop_size instead of target_size.

**3. Update `load_preprocessed_data()` in loader.py:**
- Rename parameter `target_size` to `crop_size` (line 578)
- Replace the resize block (lines 641-649) with center-crop:
  ```python
  if crop_size is not None:
      ch, cw = crop_size
      if cube.shape[1] != ch or cube.shape[2] != cw:
          cube = _center_crop(cube, ch, cw)
  ```
  This removes the `import torch` and `F.interpolate` calls entirely from this block.
- Update the log message (line 653-654) from "resized" to "cropped":
  ```python
  if crop_size and (orig_shape[1] != cube.shape[1] or orig_shape[2] != cube.shape[2]):
      print(f"    Shape: {orig_shape} -> {cube.shape} (center-cropped)")
  ```
- Update docstring and comment (line 632) from "resize" to "crop".

**4. Update main.py (lines 93-95):**
- Change config key from `target_size` to `crop_size`:
  ```python
  # Center-crop for uniform spatial dims (preserves physical scale)
  crop_size_cfg = config['data'].get('crop_size')
  crop_size = tuple(crop_size_cfg) if crop_size_cfg else None
  ```
- Change both call sites (lines 110, 128) from `target_size=target_size` to `crop_size=crop_size`.

**5. Update config.yaml:**
- Line 21: Replace `target_size: [448, 896]` with `crop_size: [437, 877]`
- Update the comment from "Resize all samples to uniform (H, W) for batch_size > 1" to "Center-crop all cubes to uniform (H, W), preserving 0.36 Mm/pixel scale"
- Line 50: Update batch_size comment from "target_size enables uniform spatial dims" to "crop_size enables uniform spatial dims"
  </action>
  <verify>
    <automated>python3 -m pytest tests/ -x --tb=short -q</automated>
  </verify>
  <done>
- config.yaml has `crop_size: [437, 877]` (no `target_size` key)
- main.py reads `crop_size` from config and passes `crop_size=` to both loader functions
- loader.py has `_center_crop()` helper and both load functions use it instead of F.interpolate
- No torch import inside the crop blocks (pure numpy slicing)
- All 206 existing tests still pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Verify center-crop correctness with a targeted smoke test</name>
  <files>tests/test_center_crop.py</files>
  <action>
Create `tests/test_center_crop.py` to verify the center-crop logic works correctly for all three cube dimension groups. This is a unit test of the `_center_crop` helper and does NOT require actual data files.

```python
"""Tests for center-crop spatial normalization."""
import numpy as np
import pytest

from solarflare_data.loader import _center_crop


TARGET_H, TARGET_W = 437, 877


class TestCenterCrop:
    """Verify center-crop produces correct dimensions and centering."""

    @pytest.mark.parametrize("orig_h,orig_w,label", [
        (440, 884, "Group A (cubes 0-3)"),
        (627, 877, "Group B (cubes 4-6)"),
        (520, 1044, "Group C (cubes 7-13)"),
    ])
    def test_output_shape(self, orig_h, orig_w, label):
        """Each cube group crops to exactly 437x877."""
        arr = np.random.rand(10, orig_h, orig_w).astype(np.float32)
        result = _center_crop(arr, TARGET_H, TARGET_W)
        assert result.shape == (10, TARGET_H, TARGET_W), f"Failed for {label}"

    def test_center_alignment(self):
        """Crop is centered: known pixel at center survives."""
        T, H, W = 5, 440, 884
        arr = np.zeros((T, H, W), dtype=np.float32)
        center_y, center_x = H // 2, W // 2
        arr[:, center_y, center_x] = 1.0
        result = _center_crop(arr, TARGET_H, TARGET_W)
        # After crop, the original center should still be in the result
        new_center_y = TARGET_H // 2
        new_center_x = TARGET_W // 2
        # The offset: y0 = (440-437)//2 = 1, x0 = (884-877)//2 = 3
        # So original center (220, 442) maps to (220-1, 442-3) = (219, 439)
        assert result[0, center_y - 1, center_x - 3] == 1.0

    def test_no_crop_needed(self):
        """If cube already matches crop_size, return unchanged."""
        arr = np.random.rand(10, TARGET_H, TARGET_W).astype(np.float32)
        result = _center_crop(arr, TARGET_H, TARGET_W)
        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_crop_too_large_raises(self):
        """Crop larger than input raises ValueError."""
        arr = np.random.rand(10, 100, 100).astype(np.float32)
        with pytest.raises(ValueError, match="exceeds"):
            _center_crop(arr, TARGET_H, TARGET_W)

    def test_temporal_dim_preserved(self):
        """Time dimension is not affected by crop."""
        for T in [1, 50, 200]:
            arr = np.random.rand(T, 627, 877).astype(np.float32)
            result = _center_crop(arr, TARGET_H, TARGET_W)
            assert result.shape[0] == T

    def test_data_integrity(self):
        """Cropped region contains the correct subset of original data."""
        arr = np.arange(5 * 440 * 884, dtype=np.float32).reshape(5, 440, 884)
        result = _center_crop(arr, TARGET_H, TARGET_W)
        y0 = (440 - TARGET_H) // 2  # 1
        x0 = (884 - TARGET_W) // 2  # 3
        expected = arr[:, y0:y0 + TARGET_H, x0:x0 + TARGET_W]
        np.testing.assert_array_equal(result, expected)
```

Run the new test file to verify all pass.
  </action>
  <verify>
    <automated>python3 -m pytest tests/test_center_crop.py -v --tb=short && python3 -m pytest tests/ -x --tb=short -q</automated>
  </verify>
  <done>
- All center-crop tests pass for all three cube dimension groups
- All 206+ tests pass (existing + new)
- Tests cover: output shape per group, center alignment, no-op when already correct size, error on too-small input, temporal dimension preservation, data integrity
  </done>
</task>

</tasks>

<verification>
1. `python3 -m pytest tests/ -x --tb=short -q` -- all tests pass (existing + new center-crop tests)
2. `grep -n "crop_size" config.yaml` -- shows `crop_size: [437, 877]`
3. `grep -n "target_size" config.yaml main.py solarflare_data/loader.py` -- returns NO matches (fully replaced)
4. `grep -n "F.interpolate\|bilinear" solarflare_data/loader.py` -- returns NO matches in the crop blocks (the model predictor.py still uses F.interpolate for its own upsampling -- that is correct and unrelated)
5. `grep -n "_center_crop" solarflare_data/loader.py` -- shows helper definition and two call sites
</verification>

<success_criteria>
- config.yaml uses `crop_size: [437, 877]` with no `target_size` key remaining
- Both load paths (raw and preprocessed) in loader.py use numpy center-crop slicing, not bilinear interpolation
- main.py reads `crop_size` and passes it through correctly
- All existing tests pass unchanged
- New center-crop tests verify correctness for all three cube dimension groups
- No torch import needed for cropping (pure numpy operation)
</success_criteria>

<output>
After completion, create `.planning/quick/3-fix-spatial-dimension-mismatch-by-center/3-SUMMARY.md`
</output>
