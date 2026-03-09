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
