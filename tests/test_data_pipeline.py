"""Tests for data pipeline: mmap loading, augmentation, normalization, splitting (TST-04)."""
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_npy_files(tmp_path):
    """Create 3 synthetic .npy files of shape (20, 32, 32) and return paths + assignments."""
    rng = np.random.RandomState(42)
    paths = []
    for i in range(3):
        data = rng.uniform(0, 10, size=(20, 32, 32)).astype(np.float32)
        p = tmp_path / f"cube_{i:04d}.npy"
        np.save(p, data)
        paths.append(str(p))

    # All 3 files assigned to train split
    file_assignments = {"train": [0, 1, 2], "val": [], "test": []}
    return paths, file_assignments, tmp_path


@pytest.fixture
def flare_npy_files(tmp_path):
    """Create files with controlled extreme values for flare detection testing.

    File 0: All values low (0-3), no flares at any threshold >= 5.0
    File 1: High values (>5.0) in output frames only (frames 4-5 for t_in=4, t_out=2)
    File 2: High values (>5.0) in input frames only (frames 0-3), NOT in output frames
    """
    paths = []

    # File 0: All low values, no flares
    data0 = np.full((20, 4, 4), 2.0, dtype=np.float32)
    p0 = tmp_path / "cube_0000.npy"
    np.save(p0, data0)
    paths.append(str(p0))

    # File 1: Extreme values in output frames (frames 4 and 5)
    data1 = np.full((20, 4, 4), 2.0, dtype=np.float32)
    data1[4, :, :] = 10.0  # Output frame for window_start=0 with t_in=4
    data1[5, :, :] = 10.0  # Output frame for window_start=0 with t_in=4
    p1 = tmp_path / "cube_0001.npy"
    np.save(p1, data1)
    paths.append(str(p1))

    # File 2: Extreme values in input frames only (frames 0-3)
    data2 = np.full((20, 4, 4), 2.0, dtype=np.float32)
    data2[0, :, :] = 10.0  # Input frame only
    data2[1, :, :] = 10.0  # Input frame only
    data2[2, :, :] = 10.0  # Input frame only
    data2[3, :, :] = 10.0  # Input frame only
    # Output frames (4, 5) stay at 2.0
    p2 = tmp_path / "cube_0002.npy"
    np.save(p2, data2)
    paths.append(str(p2))

    file_assignments = {"train": [0, 1, 2], "val": [], "test": []}
    return paths, file_assignments, tmp_path


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    """Verify SolarFluxDataset loads mmap files correctly."""

    def test_dataset_len_matches_index(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)
        assert len(ds) == len(index)

    def test_dataset_getitem_shape(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)

        X_in, Y_out, info = ds[0]
        assert X_in.shape == (1, 4, 32, 32), f"X_in shape: {X_in.shape}"
        assert Y_out.shape == (1, 2, 32, 32), f"Y_out shape: {Y_out.shape}"
        assert isinstance(info, tuple) and len(info) == 2

    def test_dataset_getitem_dual_channel(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(
            file_paths=paths, index=index, t_in=4, t_out=2,
            dual_channel=True, extreme_threshold=5.0,
        )

        X_in, Y_out, info = ds[0]
        assert X_in.shape == (2, 4, 32, 32), f"X_in dual shape: {X_in.shape}"
        assert Y_out.shape == (2, 2, 32, 32), f"Y_out dual shape: {Y_out.shape}"

    def test_dataset_values_are_finite(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)

        X_in, Y_out, _ = ds[0]
        assert torch.isfinite(X_in).all(), "X_in contains non-finite values"
        assert torch.isfinite(Y_out).all(), "Y_out contains non-finite values"

    def test_dataset_returns_none_on_bad_file(self, tmp_path):
        from solarflare_data.dataset import SolarFluxDataset

        # Create a corrupt file (write random bytes with .npy extension)
        bad_path = tmp_path / "corrupt.npy"
        bad_path.write_bytes(b"not a valid npy file at all" * 10)

        # Build a manual index referencing this corrupt file
        index = [(0, 0, 0)]
        ds = SolarFluxDataset(file_paths=[str(bad_path)], index=index, t_in=4, t_out=2)

        result = ds[0]
        assert result is None, "Expected None for corrupt file"


# ---------------------------------------------------------------------------
# build_index tests
# ---------------------------------------------------------------------------


class TestBuildIndex:
    """Verify build_index computes correct window counts."""

    def test_build_index_count_no_augmentation(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        # Each file: 20 timesteps, t_in=4, t_out=2 -> max_start = 20-4-2+1 = 15
        # 3 files * 15 = 45
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        assert len(index) == 45, f"Expected 45, got {len(index)}"

    def test_build_index_balanced_augmentation_multiplies(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        # balanced has 3 aug codes (NONE, HFLIP, VFLIP)
        # 3 files * 15 windows * 3 augs = 135
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="balanced", split="train")
        assert len(index) == 135, f"Expected 135, got {len(index)}"

    def test_build_index_val_split_no_augmentation(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, _, _ = sample_npy_files
        # Move all files to val split
        val_assignments = {"train": [], "val": [0, 1, 2], "test": []}
        index, flare_flags = build_index(paths, val_assignments, t_in=4, t_out=2, stride=1, augmentation="balanced", split="val")
        # Val never gets augmentation -> 3 * 15 = 45
        assert len(index) == 45, f"Expected 45, got {len(index)}"

    def test_build_index_stride(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        # stride=2: windows at 0,2,4,6,8,10,12,14 -> 8 per file
        # 3 * 8 = 24
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2, stride=2, augmentation="none", split="train")
        assert len(index) == 24, f"Expected 24, got {len(index)}"


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------


class TestAugmentation:
    """Verify augmentation transforms produce correct results."""

    def test_augmentation_hflip(self):
        from solarflare_data.dataset import SolarFluxDataset, AUG_HFLIP

        data = np.arange(32).reshape(2, 4, 4).astype(np.float32)
        result = SolarFluxDataset._apply_augmentation(data, AUG_HFLIP)

        # Horizontal flip reverses axis=2 (W dimension)
        expected = np.flip(data, axis=2).copy()
        np.testing.assert_array_equal(result, expected)

    def test_augmentation_none_is_copy(self):
        from solarflare_data.dataset import SolarFluxDataset, AUG_NONE

        data = np.arange(32).reshape(2, 4, 4).astype(np.float32)
        result = SolarFluxDataset._apply_augmentation(data, AUG_NONE)

        np.testing.assert_array_equal(result, data)
        assert result is not data, "AUG_NONE should return a copy, not the same object"


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestNormalization:
    """Verify on-the-fly normalization produces expected results."""

    def test_normalization_asinh_on_the_fly(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")

        # Dataset without normalization
        ds_raw = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)
        X_raw, _, _ = ds_raw[0]

        # Dataset with asinh normalization
        ds_norm = SolarFluxDataset(
            file_paths=paths, index=index, t_in=4, t_out=2,
            norm_params={"asinh_softening": 0.005, "scale": 3.0},
            norm_method="asinh",
        )
        X_norm, _, _ = ds_norm[0]

        # Normalized values should differ from raw
        assert not torch.allclose(X_raw, X_norm), "Normalization had no effect"
        assert torch.isfinite(X_norm).all(), "Asinh normalization produced non-finite values"

    def test_normalization_linear_on_the_fly(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")

        # Data is uniform [0, 10], center=5.0, scale=2.5
        # Expected: (x - 5) / 2.5 -> values in [-2, 2], mean near 0
        ds_norm = SolarFluxDataset(
            file_paths=paths, index=index, t_in=4, t_out=2,
            norm_params={"center": 5.0, "scale": 2.5},
            norm_method="linear",
        )
        X_norm, _, _ = ds_norm[0]

        mean_val = X_norm.mean().item()
        assert abs(mean_val) < 1.0, f"Expected mean near 0, got {mean_val}"
        assert torch.isfinite(X_norm).all(), "Linear normalization produced non-finite values"


# ---------------------------------------------------------------------------
# Flare detection tests (TRAIN-04)
# ---------------------------------------------------------------------------


class TestFlareDetection:
    """Verify build_index returns flare_flags alongside index."""

    def test_build_index_returns_tuple(self, sample_npy_files):
        """build_index always returns (index, flare_flags) tuple."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        result = build_index(paths, assignments, t_in=4, t_out=2, stride=1,
                             augmentation="none", split="train")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-element tuple, got {len(result)}"

    def test_flare_flags_length_matches_index(self, sample_npy_files):
        """len(flare_flags) == len(index) always."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train")
        assert len(flare_flags) == len(index), (
            f"flare_flags length {len(flare_flags)} != index length {len(index)}"
        )

    def test_flare_flags_length_with_augmentation(self, sample_npy_files):
        """flare_flags length matches index with balanced augmentation."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="balanced",
                                         split="train")
        assert len(flare_flags) == len(index), (
            f"flare_flags length {len(flare_flags)} != index length {len(index)}"
        )

    def test_no_threshold_all_flags_false(self, sample_npy_files):
        """Without extreme_threshold (None), all flare_flags are False."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=None)
        assert not any(flare_flags), "All flags should be False when threshold is None"

    def test_detects_output_extremes(self, flare_npy_files):
        """Sequence with output pixel > threshold -> flare_flags[i] == True."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = flare_npy_files
        # File 1 has extreme values (10.0) in output frames at window_start=0
        # threshold=5.0 should detect them
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=5.0)
        # Find entries from file 1 (file_idx=1) at window_start=0
        flare_detected = False
        for i, (file_idx, ws, aug) in enumerate(index):
            if file_idx == 1 and ws == 0:
                assert flare_flags[i], (
                    f"Expected flare flag True for file 1, window_start=0, got False"
                )
                flare_detected = True
        assert flare_detected, "Did not find expected file 1 window_start=0 entry"

    def test_does_not_detect_input_only_extremes(self, flare_npy_files):
        """Sequence with input extremes but no output extremes -> False."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = flare_npy_files
        # File 2 has extremes only in input frames (0-3), output frames (4-5) are 2.0
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=5.0)
        # For file 2, window_start=0: input = frames 0-3 (extreme), output = frames 4-5 (low)
        for i, (file_idx, ws, aug) in enumerate(index):
            if file_idx == 2 and ws == 0:
                assert not flare_flags[i], (
                    f"Expected flare flag False for file 2 (input-only extremes), got True"
                )

    def test_no_flares_in_low_value_file(self, flare_npy_files):
        """File with all low values -> no flare flags set."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = flare_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=5.0)
        # File 0 has all values = 2.0, none above threshold 5.0
        for i, (file_idx, ws, aug) in enumerate(index):
            if file_idx == 0:
                assert not flare_flags[i], (
                    f"Expected no flare flag for file 0 (all low), got True at ws={ws}"
                )

    def test_augmented_copies_share_flare_flag(self, flare_npy_files):
        """All aug variants of same window share same flare flag."""
        from solarflare_data.dataset import build_index

        paths, assignments, _ = flare_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="balanced",
                                         split="train",
                                         extreme_threshold=5.0)
        # For file 1, window_start=0: all 3 aug variants should have same flag
        flags_for_window = []
        for i, (file_idx, ws, aug) in enumerate(index):
            if file_idx == 1 and ws == 0:
                flags_for_window.append(flare_flags[i])
        assert len(flags_for_window) == 3, f"Expected 3 aug variants, got {len(flags_for_window)}"
        assert all(f == flags_for_window[0] for f in flags_for_window), (
            f"Aug variants have different flare flags: {flags_for_window}"
        )

    def test_val_split_still_computes_flare_flags(self, flare_npy_files):
        """Val/test splits still get flare_flags computed (for logging)."""
        from solarflare_data.dataset import build_index

        paths, _, _ = flare_npy_files
        val_assignments = {"train": [], "val": [0, 1, 2], "test": []}
        index, flare_flags = build_index(paths, val_assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="balanced",
                                         split="val",
                                         extreme_threshold=5.0)
        assert len(flare_flags) == len(index)
        # File 1 should still have flare flags even in val split
        has_flare = any(
            flare_flags[i] for i, (file_idx, ws, aug) in enumerate(index)
            if file_idx == 1 and ws == 0
        )
        assert has_flare, "Flare flags should still be computed for val split"


# ---------------------------------------------------------------------------
# Weighted sampler tests (TRAIN-04)
# ---------------------------------------------------------------------------


class TestWeightedSampler:
    """Verify create_dataloaders integrates WeightedRandomSampler."""

    def test_sampler_replaces_shuffle_when_enabled(self, flare_npy_files):
        """When flare_flags and weight > 1.0, sampler replaces shuffle."""
        from torch.utils.data import WeightedRandomSampler
        from solarflare_data.dataset import SolarFluxDataset, build_index
        from solarflare_data.loader import create_dataloaders

        paths, assignments, _ = flare_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=5.0)
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)

        # Empty val/test datasets
        empty_idx, _ = build_index(paths, {"train": [], "val": [], "test": []},
                                   t_in=4, t_out=2, split="val")
        ds_empty = SolarFluxDataset(file_paths=paths, index=empty_idx, t_in=4, t_out=2)

        train_loader, val_loader, test_loader = create_dataloaders(
            ds, ds_empty, ds_empty,
            batch_size=1,
            train_flare_flags=flare_flags,
            flare_oversample_weight=3.0,
        )
        # Train loader should use sampler, not shuffle
        assert train_loader.sampler is not None
        assert isinstance(train_loader.sampler, WeightedRandomSampler)

    def test_no_sampler_when_weight_lte_one(self, flare_npy_files):
        """When flare_oversample_weight <= 1.0, no sampler (shuffle=True)."""
        from torch.utils.data import WeightedRandomSampler
        from solarflare_data.dataset import SolarFluxDataset, build_index
        from solarflare_data.loader import create_dataloaders

        paths, assignments, _ = flare_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=5.0)
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)

        empty_idx, _ = build_index(paths, {"train": [], "val": [], "test": []},
                                   t_in=4, t_out=2, split="val")
        ds_empty = SolarFluxDataset(file_paths=paths, index=empty_idx, t_in=4, t_out=2)

        train_loader, _, _ = create_dataloaders(
            ds, ds_empty, ds_empty,
            batch_size=1,
            train_flare_flags=flare_flags,
            flare_oversample_weight=1.0,
        )
        # Should NOT use WeightedRandomSampler
        assert not isinstance(train_loader.sampler, WeightedRandomSampler)

    def test_no_sampler_when_flags_none(self, flare_npy_files):
        """When train_flare_flags is None, no sampler (existing behavior)."""
        from torch.utils.data import WeightedRandomSampler
        from solarflare_data.dataset import SolarFluxDataset, build_index
        from solarflare_data.loader import create_dataloaders

        paths, assignments, _ = flare_npy_files
        index, _ = build_index(paths, assignments, t_in=4, t_out=2,
                                stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)

        empty_idx, _ = build_index(paths, {"train": [], "val": [], "test": []},
                                   t_in=4, t_out=2, split="val")
        ds_empty = SolarFluxDataset(file_paths=paths, index=empty_idx, t_in=4, t_out=2)

        train_loader, _, _ = create_dataloaders(
            ds, ds_empty, ds_empty, batch_size=1,
        )
        assert not isinstance(train_loader.sampler, WeightedRandomSampler)

    def test_sampler_weights_correct(self, flare_npy_files):
        """Sampler weights: flare_oversample_weight for flare, 1.0 for non-flare."""
        from solarflare_data.dataset import build_index
        from solarflare_data.loader import create_dataloaders, _build_sampler_weights

        paths, assignments, _ = flare_npy_files
        index, flare_flags = build_index(paths, assignments, t_in=4, t_out=2,
                                         stride=1, augmentation="none",
                                         split="train",
                                         extreme_threshold=5.0)
        weight = 3.0
        weights = _build_sampler_weights(flare_flags, weight)
        for i, flag in enumerate(flare_flags):
            expected = weight if flag else 1.0
            assert weights[i] == expected, (
                f"Weight at {i}: expected {expected}, got {weights[i]}"
            )

    def test_val_test_loaders_unchanged(self, flare_npy_files):
        """Val/test loaders remain shuffle=False, no sampler."""
        from torch.utils.data import WeightedRandomSampler
        from solarflare_data.dataset import SolarFluxDataset, build_index
        from solarflare_data.loader import create_dataloaders

        paths, _, _ = flare_npy_files
        all_val = {"train": [0], "val": [1, 2], "test": []}
        train_idx, train_flags = build_index(paths, all_val, t_in=4, t_out=2,
                                             stride=1, augmentation="none",
                                             split="train",
                                             extreme_threshold=5.0)
        val_idx, _ = build_index(paths, all_val, t_in=4, t_out=2,
                                 stride=1, augmentation="none",
                                 split="val",
                                 extreme_threshold=5.0)
        test_idx, _ = build_index(paths, {"train": [], "val": [], "test": []},
                                  t_in=4, t_out=2, split="test")

        ds_train = SolarFluxDataset(file_paths=paths, index=train_idx, t_in=4, t_out=2)
        ds_val = SolarFluxDataset(file_paths=paths, index=val_idx, t_in=4, t_out=2)
        ds_test = SolarFluxDataset(file_paths=paths, index=test_idx, t_in=4, t_out=2)

        _, val_loader, test_loader = create_dataloaders(
            ds_train, ds_val, ds_test,
            batch_size=1,
            train_flare_flags=train_flags,
            flare_oversample_weight=3.0,
        )
        assert not isinstance(val_loader.sampler, WeightedRandomSampler)
        assert not isinstance(test_loader.sampler, WeightedRandomSampler)
