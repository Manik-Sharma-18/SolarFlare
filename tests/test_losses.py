"""Unit tests for loss functions in training/losses.py (TST-02).

Tests cover:
- SSIM: identical inputs, different inputs, symmetry, NaN resilience
- MS-SSIM: identical inputs, finite output
- Gaussian kernel: shape, sum-to-one, caching
- WeightedMAE: zero error, extreme weighting
- CompositeLoss: scalar output, finite, components dict, 5D temporal input
- get_loss_function factory: all types + unknown raises ValueError
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.losses import (
    ssim,
    ms_ssim,
    gaussian_kernel,
    _KERNEL_CACHE,
    WeightedMAELoss,
    CompositeLoss,
    get_loss_function,
)


@pytest.fixture(autouse=True)
def clear_kernel_cache():
    """Clear the kernel cache before and after each test."""
    _KERNEL_CACHE.clear()
    yield
    _KERNEL_CACHE.clear()


# ---------------------------------------------------------------------------
# SSIM tests
# ---------------------------------------------------------------------------

class TestSSIM:
    def test_ssim_identical_inputs(self):
        """ssim(x, x) should return ~1.0 for identical inputs."""
        x = torch.rand(1, 1, 32, 32)
        val = ssim(x, x)
        assert abs(val.item() - 1.0) < 0.01, f"Expected ~1.0, got {val.item()}"

    def test_ssim_different_inputs(self):
        """ssim(x, y) for random x, y should be in [0, 1) and < 1.0."""
        torch.manual_seed(0)
        x = torch.rand(1, 1, 32, 32)
        y = torch.rand(1, 1, 32, 32)
        val = ssim(x, y)
        assert 0.0 <= val.item() <= 1.0, f"SSIM out of range: {val.item()}"
        assert val.item() < 1.0, "SSIM of different inputs should be < 1.0"

    def test_ssim_symmetry(self):
        """ssim(x, y) == ssim(y, x) within tolerance."""
        torch.manual_seed(1)
        x = torch.rand(1, 1, 32, 32)
        y = torch.rand(1, 1, 32, 32)
        val_xy = ssim(x, y)
        val_yx = ssim(y, x)
        assert torch.allclose(val_xy, val_yx, atol=1e-6), (
            f"SSIM not symmetric: {val_xy.item()} vs {val_yx.item()}"
        )

    def test_ssim_nan_input_no_crash(self):
        """ssim with NaN input should not raise an exception."""
        x = torch.rand(1, 1, 32, 32)
        x_nan = x.clone()
        x_nan[0, 0, 5, 5] = float("nan")
        y = torch.rand(1, 1, 32, 32)
        # Should not raise -- result may be NaN, which is acceptable
        result = ssim(x_nan, y)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# MS-SSIM tests
# ---------------------------------------------------------------------------

class TestMSSSIM:
    def test_ms_ssim_identical_inputs(self):
        """ms_ssim(x, x) should return ~1.0."""
        x = torch.rand(1, 1, 64, 64)
        val = ms_ssim(x, x)
        assert abs(val.item() - 1.0) < 0.05, f"Expected ~1.0, got {val.item()}"

    def test_ms_ssim_returns_finite(self):
        """ms_ssim(x, y) should return a finite value."""
        torch.manual_seed(2)
        x = torch.rand(1, 1, 64, 64)
        y = torch.rand(1, 1, 64, 64)
        val = ms_ssim(x, y)
        assert torch.isfinite(val), f"MS-SSIM not finite: {val.item()}"


# ---------------------------------------------------------------------------
# Gaussian kernel tests
# ---------------------------------------------------------------------------

class TestGaussianKernel:
    def test_gaussian_kernel_shape(self):
        """gaussian_kernel should return (size, size) tensor."""
        kernel = gaussian_kernel(11, 1.5, torch.device("cpu"))
        assert kernel.shape == (11, 11)

    def test_gaussian_kernel_sums_to_one(self):
        """Kernel values should sum to ~1.0."""
        kernel = gaussian_kernel(11, 1.5, torch.device("cpu"))
        assert abs(kernel.sum().item() - 1.0) < 0.01, (
            f"Kernel sum: {kernel.sum().item()}"
        )

    def test_gaussian_kernel_caching(self):
        """Same args should return the same cached object."""
        k1 = gaussian_kernel(11, 1.5, torch.device("cpu"))
        k2 = gaussian_kernel(11, 1.5, torch.device("cpu"))
        assert k1 is k2, "Kernel caching failed -- different objects returned"


# ---------------------------------------------------------------------------
# WeightedMAE tests
# ---------------------------------------------------------------------------

class TestWeightedMAE:
    def test_weighted_mae_zero_error(self):
        """pred == target should produce loss ~0."""
        loss_fn = WeightedMAELoss(base_weight=1.0, extreme_weight=2.0)
        x = torch.rand(1, 1, 32, 32)
        val = loss_fn(x, x)
        assert val.item() < 1e-6, f"Expected ~0, got {val.item()}"

    def test_weighted_mae_extreme_higher_weight(self):
        """Extreme target values should produce higher loss than uniform L1."""
        loss_fn = WeightedMAELoss(base_weight=1.0, extreme_weight=2.0)
        torch.manual_seed(3)

        # Target with mix of normal (0.1) and extreme (10.0) values
        target = torch.full((1, 1, 32, 32), 0.1)
        target[:, :, :16, :] = 10.0  # extreme region

        # Pred with uniform offset
        pred = target + 0.5

        weighted_loss = loss_fn(pred, target)
        plain_l1 = F.l1_loss(pred, target)

        assert weighted_loss.item() > plain_l1.item(), (
            f"Weighted loss ({weighted_loss.item()}) should exceed "
            f"plain L1 ({plain_l1.item()})"
        )


# ---------------------------------------------------------------------------
# CompositeLoss tests
# ---------------------------------------------------------------------------

class TestCompositeLoss:
    def test_composite_loss_returns_scalar(self):
        """CompositeLoss output should be a 0-dim tensor."""
        loss_fn = CompositeLoss(use_ms_ssim=False)
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        val = loss_fn(pred, target)
        assert val.dim() == 0, f"Expected scalar, got dim={val.dim()}"

    def test_composite_loss_returns_finite(self):
        """CompositeLoss output should be finite."""
        loss_fn = CompositeLoss(use_ms_ssim=False)
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        val = loss_fn(pred, target)
        assert torch.isfinite(val), f"CompositeLoss not finite: {val.item()}"

    def test_composite_loss_components(self):
        """return_components=True should return dict with expected keys."""
        loss_fn = CompositeLoss(use_ms_ssim=False)
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        result = loss_fn(pred, target, return_components=True)

        assert isinstance(result, dict)
        expected_keys = {"total", "l1", "ssim", "ssim_val", "extreme"}
        assert set(result.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )
        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"

    def test_composite_loss_5d_input(self):
        """5D tensors (B, C, T, H, W) should be handled via temporal flattening."""
        loss_fn = CompositeLoss(use_ms_ssim=False)
        pred = torch.rand(1, 1, 2, 32, 32)
        target = torch.rand(1, 1, 2, 32, 32)
        val = loss_fn(pred, target)
        assert torch.isfinite(val), f"5D composite loss not finite: {val.item()}"
        assert val.dim() == 0


# ---------------------------------------------------------------------------
# get_loss_function factory tests
# ---------------------------------------------------------------------------

class TestGetLossFunction:
    def test_get_loss_function_l1(self):
        """type='l1' should return nn.L1Loss."""
        fn = get_loss_function({"type": "l1"})
        assert isinstance(fn, nn.L1Loss)

    def test_get_loss_function_composite(self):
        """type='composite' should return CompositeLoss."""
        fn = get_loss_function({"type": "composite"})
        assert isinstance(fn, CompositeLoss)

    def test_get_loss_function_weighted(self):
        """type='weighted' should return WeightedMAELoss."""
        fn = get_loss_function({"type": "weighted"})
        assert isinstance(fn, WeightedMAELoss)

    def test_get_loss_function_unknown_raises(self):
        """Unknown loss type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function({"type": "mse"})
