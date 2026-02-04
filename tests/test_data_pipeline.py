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


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    """Verify SolarFluxDataset loads mmap files correctly."""

    def test_dataset_len_matches_index(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)
        assert len(ds) == len(index)

    def test_dataset_getitem_shape(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        ds = SolarFluxDataset(file_paths=paths, index=index, t_in=4, t_out=2)

        X_in, Y_out, info = ds[0]
        assert X_in.shape == (1, 4, 32, 32), f"X_in shape: {X_in.shape}"
        assert Y_out.shape == (1, 2, 32, 32), f"Y_out shape: {Y_out.shape}"
        assert isinstance(info, tuple) and len(info) == 2

    def test_dataset_getitem_dual_channel(self, sample_npy_files):
        from solarflare_data.dataset import SolarFluxDataset, build_index

        paths, assignments, _ = sample_npy_files
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
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
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
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
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")
        assert len(index) == 45, f"Expected 45, got {len(index)}"

    def test_build_index_balanced_augmentation_multiplies(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        # balanced has 3 aug codes (NONE, HFLIP, VFLIP)
        # 3 files * 15 windows * 3 augs = 135
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="balanced", split="train")
        assert len(index) == 135, f"Expected 135, got {len(index)}"

    def test_build_index_val_split_no_augmentation(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, _, _ = sample_npy_files
        # Move all files to val split
        val_assignments = {"train": [], "val": [0, 1, 2], "test": []}
        index = build_index(paths, val_assignments, t_in=4, t_out=2, stride=1, augmentation="balanced", split="val")
        # Val never gets augmentation -> 3 * 15 = 45
        assert len(index) == 45, f"Expected 45, got {len(index)}"

    def test_build_index_stride(self, sample_npy_files):
        from solarflare_data.dataset import build_index

        paths, assignments, _ = sample_npy_files
        # stride=2: windows at 0,2,4,6,8,10,12,14 -> 8 per file
        # 3 * 8 = 24
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=2, augmentation="none", split="train")
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
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")

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
        index = build_index(paths, assignments, t_in=4, t_out=2, stride=1, augmentation="none", split="train")

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
