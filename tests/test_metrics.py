"""Unit tests for evaluation metric functions in utils/metrics.py."""
import pytest
import torch
import math

from utils.metrics import (
    compute_csi,
    compute_hss,
    accumulate_contingency,
    compute_persistence_prediction,
    compute_persistence_skill,
    compute_ssim_per_timestep,
    compute_peak_flux_error,
    compute_temporal_variation_ratio,
    compute_rmse_per_timestep,
    compute_correlation_per_timestep,
)


# ---------------------------------------------------------------------------
# CSI (Critical Success Index) tests
# ---------------------------------------------------------------------------
class TestComputeCSI:
    def test_normal_case(self):
        """CSI = TP / (TP + FP + FN) = 10/15 ~ 0.6667."""
        result = compute_csi(10, 2, 3)
        assert abs(result - 10.0 / 15.0) < 1e-6

    def test_no_events(self):
        """When TP=FP=FN=0, CSI should be 0.0 (guard zero denominator)."""
        assert compute_csi(0, 0, 0) == 0.0

    def test_perfect(self):
        """Perfect prediction: TP=100, FP=0, FN=0 -> CSI=1.0."""
        assert compute_csi(100, 0, 0) == 1.0


# ---------------------------------------------------------------------------
# HSS (Heidke Skill Score) tests
# ---------------------------------------------------------------------------
class TestComputeHSS:
    def test_normal_case(self):
        """HSS formula: 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))."""
        tp, fp, fn, tn = 10, 2, 3, 85
        expected_num = 2 * (tp * tn - fp * fn)
        expected_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        expected = expected_num / expected_den
        result = compute_hss(tp, fp, fn, tn)
        assert abs(result - expected) < 1e-4

    def test_no_events(self):
        """When no events predicted or observed, HSS=0.0."""
        assert compute_hss(0, 0, 0, 100) == 0.0

    def test_perfect(self):
        """Perfect predictions: HSS=1.0."""
        assert compute_hss(50, 0, 0, 50) == 1.0


# ---------------------------------------------------------------------------
# Contingency Table tests
# ---------------------------------------------------------------------------
class TestAccumulateContingency:
    def test_known_tensors(self):
        """Verify TP/FP/FN/TN with known values at a given threshold."""
        # Shape: (B=1, C=1, T=2, H=2, W=2)
        # Threshold = 0.5
        # pred:  t=0 all 1.0 (above threshold), t=1 all 0.0 (below threshold)
        pred = torch.zeros(1, 1, 2, 2, 2)
        pred[:, :, 0, :, :] = 1.0  # t=0: all above
        pred[:, :, 1, :, :] = 0.0  # t=1: all below

        # target: t=0 mixed (2 above, 2 below), t=1 all 1.0 (above)
        target = torch.zeros(1, 1, 2, 2, 2)
        target[0, 0, 0, 0, 0] = 1.0  # above
        target[0, 0, 0, 0, 1] = 1.0  # above
        target[0, 0, 0, 1, 0] = 0.0  # below
        target[0, 0, 0, 1, 1] = 0.0  # below
        target[:, :, 1, :, :] = 1.0  # all above

        result = accumulate_contingency(pred, target, threshold=0.5)
        assert len(result) == 2

        # t=0: pred all above, target 2 above 2 below
        # TP=2, FP=2, FN=0, TN=0
        tp0, fp0, fn0, tn0 = result[0]
        assert tp0 == 2
        assert fp0 == 2
        assert fn0 == 0
        assert tn0 == 0

        # t=1: pred all below, target all above
        # TP=0, FP=0, FN=4, TN=0
        tp1, fp1, fn1, tn1 = result[1]
        assert tp1 == 0
        assert fp1 == 0
        assert fn1 == 4
        assert tn1 == 0

    def test_returns_list_of_tuples(self):
        """Result should be a list of (tp, fp, fn, tn) tuples with length T."""
        pred = torch.randn(2, 1, 3, 4, 4)
        target = torch.randn(2, 1, 3, 4, 4)
        result = accumulate_contingency(pred, target, threshold=0.5)
        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert len(item) == 4
            # All values should be non-negative integers
            for val in item:
                assert isinstance(val, int)
                assert val >= 0


# ---------------------------------------------------------------------------
# Persistence Baseline tests
# ---------------------------------------------------------------------------
class TestComputePersistencePrediction:
    def test_output_shape(self):
        """Output shape should match (B, output_channels, T_out, H, W)."""
        # X_in: (B=2, C=2, T_in=10, H=8, W=8)
        X_in = torch.randn(2, 2, 10, 8, 8)
        result = compute_persistence_prediction(X_in, output_channels=1, T_out=4)
        assert result.shape == (2, 1, 4, 8, 8)

    def test_repeats_last_frame(self):
        """All output timesteps should equal the last input frame."""
        X_in = torch.randn(2, 2, 10, 8, 8)
        result = compute_persistence_prediction(X_in, output_channels=1, T_out=4)
        last_frame = X_in[:, :1, -1:, :, :]  # (B, 1, 1, H, W)
        for t in range(4):
            torch.testing.assert_close(result[:, :, t:t+1, :, :], last_frame)


class TestComputePersistenceSkill:
    def test_positive_skill(self):
        """Model better than persistence -> positive skill."""
        result = compute_persistence_skill(model_mae=0.5, persistence_mae=1.0)
        assert abs(result - 50.0) < 1e-6

    def test_zero_persistence_mae(self):
        """When persistence MAE is near zero, return 0.0 gracefully."""
        result = compute_persistence_skill(model_mae=0.1, persistence_mae=0.0)
        assert result == 0.0

    def test_model_worse_than_persistence(self):
        """Model worse than persistence -> negative skill."""
        result = compute_persistence_skill(model_mae=2.0, persistence_mae=1.0)
        assert result < 0.0


# ---------------------------------------------------------------------------
# Standalone SSIM per-timestep tests
# ---------------------------------------------------------------------------
class TestComputeSSIMPerTimestep:
    def test_identical_tensors(self):
        """Identical pred/target should produce SSIM close to 1.0."""
        tensor = torch.randn(2, 1, 3, 32, 32)
        result = compute_ssim_per_timestep(tensor, tensor)
        assert len(result) == 3
        for val in result:
            assert val > 0.95  # Should be very close to 1.0

    def test_different_tensors(self):
        """Different tensors should produce lower SSIM."""
        pred = torch.randn(2, 1, 3, 32, 32)
        target = torch.randn(2, 1, 3, 32, 32)
        result = compute_ssim_per_timestep(pred, target)
        assert len(result) == 3
        for val in result:
            assert val < 0.95  # Should be lower than identical

    def test_returns_list_of_floats(self):
        """Should return a list of T float values."""
        pred = torch.randn(2, 1, 4, 32, 32)
        target = torch.randn(2, 1, 4, 32, 32)
        result = compute_ssim_per_timestep(pred, target)
        assert isinstance(result, list)
        assert len(result) == 4
        for val in result:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Peak Flux Error tests
# ---------------------------------------------------------------------------
class TestComputePeakFluxError:
    def test_identical_tensors(self):
        """Identical tensors should have 0.0 peak flux error."""
        tensor = torch.randn(2, 1, 3, 8, 8)
        result = compute_peak_flux_error(tensor, tensor)
        assert len(result) == 3
        for val in result:
            assert abs(val) < 1e-6

    def test_known_difference(self):
        """Known max difference should be correctly computed."""
        pred = torch.zeros(1, 1, 2, 4, 4)
        target = torch.zeros(1, 1, 2, 4, 4)
        # Set known maximums
        pred[0, 0, 0, 0, 0] = 5.0
        target[0, 0, 0, 0, 0] = 3.0
        pred[0, 0, 1, 0, 0] = 2.0
        target[0, 0, 1, 0, 0] = 2.0
        result = compute_peak_flux_error(pred, target)
        assert len(result) == 2
        assert abs(result[0] - 2.0) < 1e-6  # |5-3|=2
        assert abs(result[1] - 0.0) < 1e-6  # |2-2|=0

    def test_returns_list_of_floats(self):
        """Should return a list of T float values."""
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.randn(2, 1, 4, 8, 8)
        result = compute_peak_flux_error(pred, target)
        assert isinstance(result, list)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Temporal Variation Ratio tests
# ---------------------------------------------------------------------------
class TestComputeTemporalVariationRatio:
    def test_identical_varying_tensors(self):
        """Identical pred/target with variation should return ~1.0."""
        tensor = torch.randn(2, 1, 4, 8, 8)
        result = compute_temporal_variation_ratio(tensor, tensor)
        assert abs(result - 1.0) < 1e-5

    def test_static_prediction(self):
        """Static prediction (no variation) should return ~0.0."""
        pred = torch.ones(2, 1, 4, 8, 8) * 0.5  # constant
        target = torch.randn(2, 1, 4, 8, 8)  # varying
        result = compute_temporal_variation_ratio(pred, target)
        assert result < 0.01  # near zero

    def test_single_frame(self):
        """Single-frame input (T=1) should return 1.0."""
        pred = torch.randn(2, 1, 1, 8, 8)
        target = torch.randn(2, 1, 1, 8, 8)
        result = compute_temporal_variation_ratio(pred, target)
        assert result == 1.0

    def test_zero_target_variation(self):
        """When target has no variation, return 0.0."""
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.ones(2, 1, 4, 8, 8) * 0.5  # constant
        result = compute_temporal_variation_ratio(pred, target)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Per-Timestep RMSE tests
# ---------------------------------------------------------------------------
class TestComputeRMSEPerTimestep:
    def test_identical_tensors(self):
        """Identical tensors should produce RMSE of 0.0."""
        tensor = torch.randn(2, 1, 3, 8, 8)
        result = compute_rmse_per_timestep(tensor, tensor)
        assert len(result) == 3
        for val in result:
            assert abs(val) < 1e-6

    def test_known_values(self):
        """Check RMSE with known values."""
        pred = torch.zeros(1, 1, 1, 2, 2)
        target = torch.ones(1, 1, 1, 2, 2)
        result = compute_rmse_per_timestep(pred, target)
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-6  # sqrt(mean((0-1)^2)) = 1.0

    def test_returns_list_of_floats(self):
        """Should return a list of T float values."""
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.randn(2, 1, 4, 8, 8)
        result = compute_rmse_per_timestep(pred, target)
        assert isinstance(result, list)
        assert len(result) == 4
        for val in result:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Per-Timestep Correlation tests
# ---------------------------------------------------------------------------
class TestComputeCorrelationPerTimestep:
    def test_identical_tensors(self):
        """Identical tensors should produce correlation of 1.0."""
        tensor = torch.randn(2, 1, 3, 8, 8)
        result = compute_correlation_per_timestep(tensor, tensor)
        assert len(result) == 3
        for val in result:
            assert abs(val - 1.0) < 1e-5

    def test_zero_variance(self):
        """Zero-variance input should return 0.0."""
        pred = torch.ones(2, 1, 2, 8, 8) * 3.0
        target = torch.randn(2, 1, 2, 8, 8)
        result = compute_correlation_per_timestep(pred, target)
        assert len(result) == 2
        for val in result:
            assert val == 0.0

    def test_returns_list_of_floats(self):
        """Should return a list of T float values."""
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.randn(2, 1, 4, 8, 8)
        result = compute_correlation_per_timestep(pred, target)
        assert isinstance(result, list)
        assert len(result) == 4
        for val in result:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Integration tests: validate() returns structured dict
# ---------------------------------------------------------------------------
class TestValidateReturnsDict:
    """Integration tests verifying validate() returns a dict with all metric keys."""

    @pytest.fixture
    def tiny_model_and_loader(self):
        """Create a tiny model and synthetic dataloader for integration testing."""
        from models import SolarFluxPredictor
        from torch.utils.data import DataLoader, TensorDataset

        model = SolarFluxPredictor(
            input_channels=1,
            output_channels=1,
            t_out=2,
            channels=[4, 8, 16],
            kernel_size=3,
            use_checkpointing=False,
            dropout_rate=0.0,
        )
        device = torch.device("cpu")
        model = model.to(device)

        # Create synthetic data: B=2, C=1, T_in=4, H=32, W=32
        X = torch.randn(2, 1, 4, 32, 32)
        # Y: B=2, C=1, T_out=2, H=32, W=32
        Y = torch.randn(2, 1, 2, 32, 32)
        # Metadata placeholder (dataset_id, start_idx)
        meta1 = torch.tensor([0, 0])
        meta2 = torch.tensor([0, 1])

        dataset = TensorDataset(X, Y, torch.stack([meta1, meta2]))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        return model, loader, device

    def test_validate_returns_dict_with_all_keys(self, tiny_model_and_loader):
        """validate() must return a dict with all expected metric keys."""
        from training.trainer import validate

        model, loader, device = tiny_model_and_loader

        result = validate(
            model, loader, device,
            use_amp=False,
            show_progress=False,
            output_channels=1,
            extreme_threshold=0.277,
            ssim_data_range=2.0,
        )

        assert isinstance(result, dict), f"validate() should return dict, got {type(result)}"

        expected_keys = [
            'val_loss',
            'val_mae_per_timestep',
            'val_rmse_per_timestep',
            'val_correlation_per_timestep',
            'val_csi',
            'val_csi_per_timestep',
            'val_hss',
            'val_hss_per_timestep',
            'val_ssim',
            'val_ssim_per_timestep',
            'persistence_mae_per_timestep',
            'persistence_skill_per_timestep',
            'persistence_csi',
            'persistence_hss',
            'peak_flux_error_per_timestep',
            'temporal_variation_ratio',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_validate_dict_value_types(self, tiny_model_and_loader):
        """Validate that dict values have correct types (float vs list)."""
        from training.trainer import validate

        model, loader, device = tiny_model_and_loader

        result = validate(
            model, loader, device,
            use_amp=False,
            show_progress=False,
            output_channels=1,
            extreme_threshold=0.277,
            ssim_data_range=2.0,
        )

        # Scalar metrics should be float
        for key in ['val_loss', 'val_csi', 'val_hss', 'val_ssim',
                     'persistence_csi', 'persistence_hss', 'temporal_variation_ratio']:
            assert isinstance(result[key], float), f"{key} should be float, got {type(result[key])}"

        # Per-timestep metrics should be lists
        for key in ['val_mae_per_timestep', 'val_rmse_per_timestep',
                     'val_correlation_per_timestep', 'val_csi_per_timestep',
                     'val_hss_per_timestep', 'val_ssim_per_timestep',
                     'persistence_mae_per_timestep', 'persistence_skill_per_timestep',
                     'peak_flux_error_per_timestep']:
            assert isinstance(result[key], list), f"{key} should be list, got {type(result[key])}"
            assert len(result[key]) == 2, f"{key} should have 2 elements (T_out=2)"


class TestHistoryNewKeys:
    """Verify the history dict structure includes new metric keys."""

    def test_history_has_new_keys(self):
        """The history dict initialized in train_model() should have new metric keys."""
        # We can test this by checking the expected keys exist in the template
        expected_new_keys = [
            'val_csi', 'val_hss', 'val_ssim', 'val_ssim_per_timestep',
            'persistence_skill_per_timestep', 'persistence_csi', 'persistence_hss',
            'peak_flux_error_per_timestep', 'temporal_variation_ratio',
            'val_rmse_per_timestep', 'val_correlation_per_timestep',
            'val_csi_per_timestep', 'val_hss_per_timestep',
            'persistence_mae_per_timestep',
        ]
        # Import and check the code has the new keys
        import inspect
        from training.trainer import train_model
        source = inspect.getsource(train_model)
        for key in expected_new_keys:
            assert f"'{key}'" in source, f"History key '{key}' not found in train_model source"
