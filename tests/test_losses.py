"""Unit tests for loss functions in training/losses.py (TST-02).

Tests cover:
- SSIM: identical inputs, different inputs, symmetry, NaN resilience
- MS-SSIM: identical inputs, finite output
- Gaussian kernel: shape, sum-to-one, caching
- WeightedMAE: zero error, extreme weighting, absolute threshold, boundary
- AsymmetricExtremeLoss: asymmetric underestimation, symmetric below, zero error
- Temporal diff loss: nonzero for different dynamics, zero for identical, 5D handling
- Temporal variation penalty: negative value, capped at target variation, T=1
- Temporal weighting: later timesteps weighted more, broadcast shape
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
    AsymmetricExtremeLoss,
    compute_temporal_diff_loss,
    compute_temporal_var_penalty,
    apply_temporal_weights,
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
        loss_fn = WeightedMAELoss(base_weight=1.0, extreme_weight=3.0, threshold=0.3456)
        x = torch.rand(1, 1, 32, 32)
        val = loss_fn(x, x)
        assert val.item() < 1e-6, f"Expected ~0, got {val.item()}"

    def test_weighted_mae_extreme_higher_weight(self):
        """Extreme target values (above threshold) should produce higher loss than uniform L1."""
        loss_fn = WeightedMAELoss(base_weight=1.0, extreme_weight=3.0, threshold=0.3456)

        # Target with mix of normal (0.1, below threshold) and extreme (0.5, above threshold)
        target = torch.full((1, 1, 32, 32), 0.1)
        target[:, :, :16, :] = 0.5  # above threshold=0.3456

        # Pred with uniform offset
        pred = target + 0.05

        weighted_loss = loss_fn(pred, target)
        plain_l1 = F.l1_loss(pred, target)

        assert weighted_loss.item() > plain_l1.item(), (
            f"Weighted loss ({weighted_loss.item()}) should exceed "
            f"plain L1 ({plain_l1.item()})"
        )

    def test_weighted_mae_absolute_threshold(self):
        """Same target value at different batch max values should produce identical loss.

        Binary weighting with absolute threshold means the loss should NOT depend
        on the maximum target value in the batch (unlike the old relative approach).
        """
        loss_fn = WeightedMAELoss(base_weight=1.0, extreme_weight=3.0, threshold=0.3456)

        # Batch 1: target has values 0.5 (above threshold), max is 0.5
        target1 = torch.full((1, 1, 8, 8), 0.5)
        pred1 = torch.full((1, 1, 8, 8), 0.4)

        # Batch 2: same target at 0.5, but add a larger value to change max
        target2 = torch.full((1, 1, 8, 8), 0.5)
        target2[0, 0, 0, 0] = 2.0  # different batch max
        pred2 = torch.full((1, 1, 8, 8), 0.4)
        pred2[0, 0, 0, 0] = 1.9  # same error at extreme pixel

        loss1 = loss_fn(pred1, target1)
        loss2 = loss_fn(pred2, target2)

        # With absolute threshold, both should use extreme_weight for all pixels
        # (all target values > 0.3456), so losses should be identical
        assert torch.allclose(loss1, loss2, atol=1e-5), (
            f"Absolute threshold failed: loss1={loss1.item():.6f} vs loss2={loss2.item():.6f} "
            f"should be identical for same error magnitudes"
        )

    def test_weighted_mae_threshold_boundary(self):
        """Values at threshold boundary should get correct weights.

        Below threshold: weight = base_weight (1.0)
        Above threshold: weight = extreme_weight (3.0)
        """
        loss_fn = WeightedMAELoss(base_weight=1.0, extreme_weight=3.0, threshold=0.35)

        # All below threshold: weight should be base_weight=1.0
        target_below = torch.full((1, 1, 4, 4), 0.34)
        pred_below = torch.full((1, 1, 4, 4), 0.24)  # error = 0.1
        loss_below = loss_fn(pred_below, target_below)
        # Expected: base_weight * 0.1 = 1.0 * 0.1 = 0.1
        assert abs(loss_below.item() - 0.1) < 1e-5, (
            f"Below threshold: expected 0.1, got {loss_below.item()}"
        )

        # All above threshold: weight should be extreme_weight=3.0
        target_above = torch.full((1, 1, 4, 4), 0.36)
        pred_above = torch.full((1, 1, 4, 4), 0.26)  # error = 0.1
        loss_above = loss_fn(pred_above, target_above)
        # Expected: extreme_weight * 0.1 = 3.0 * 0.1 = 0.3
        assert abs(loss_above.item() - 0.3) < 1e-5, (
            f"Above threshold: expected 0.3, got {loss_above.item()}"
        )


# ---------------------------------------------------------------------------
# AsymmetricExtremeLoss tests
# ---------------------------------------------------------------------------

class TestAsymmetricExtremeLoss:
    def test_asymmetric_underestimation_above_threshold(self):
        """Above threshold, underestimation loss is alpha times overestimation loss.

        With alpha=2.0, underestimating an extreme region by 0.2 should produce
        2x the loss compared to overestimating by the same magnitude.
        """
        loss_fn = AsymmetricExtremeLoss(alpha=2.0, threshold=0.3456)

        # Target above threshold
        target = torch.full((1, 1, 4, 4), 0.5)

        # Underestimate by 0.2
        pred_under = torch.full((1, 1, 4, 4), 0.3)
        loss_under = loss_fn(pred_under, target)

        # Overestimate by 0.2
        pred_over = torch.full((1, 1, 4, 4), 0.7)
        loss_over = loss_fn(pred_over, target)

        # Underestimation loss should be alpha (2.0) times overestimation loss
        ratio = loss_under.item() / loss_over.item()
        assert abs(ratio - 2.0) < 0.01, (
            f"Expected ratio ~2.0, got {ratio:.4f} "
            f"(under={loss_under.item():.6f}, over={loss_over.item():.6f})"
        )

    def test_symmetric_below_threshold(self):
        """Below threshold, underestimation and overestimation produce equal loss."""
        loss_fn = AsymmetricExtremeLoss(alpha=2.0, threshold=0.3456)

        # Target below threshold
        target = torch.full((1, 1, 4, 4), 0.1)

        # Underestimate by 0.1
        pred_under = torch.full((1, 1, 4, 4), 0.0)
        loss_under = loss_fn(pred_under, target)

        # Overestimate by 0.1
        pred_over = torch.full((1, 1, 4, 4), 0.2)
        loss_over = loss_fn(pred_over, target)

        assert torch.allclose(loss_under, loss_over, atol=1e-6), (
            f"Below threshold should be symmetric: "
            f"under={loss_under.item():.6f} vs over={loss_over.item():.6f}"
        )

    def test_zero_error(self):
        """pred == target should produce loss ~0."""
        loss_fn = AsymmetricExtremeLoss(alpha=2.0, threshold=0.3456)
        target = torch.full((1, 1, 8, 8), 0.5)
        pred = target.clone()
        loss_val = loss_fn(pred, target)
        assert loss_val.item() < 1e-6, f"Expected ~0, got {loss_val.item()}"


# ---------------------------------------------------------------------------
# Temporal diff loss tests
# ---------------------------------------------------------------------------

class TestTemporalDiffLoss:
    def test_nonzero_for_different_dynamics(self):
        """When pred has different frame-to-frame changes than target, loss > 0.

        Pred changes linearly, target changes quadratically.
        """
        # Shape (B, C, T, H, W) = (1, 1, 4, 8, 8)
        pred = torch.zeros(1, 1, 4, 8, 8)
        target = torch.zeros(1, 1, 4, 8, 8)

        # Pred: linear changes (diffs are constant)
        for t in range(4):
            pred[:, :, t, :, :] = 0.1 * t

        # Target: quadratic changes (diffs increase)
        for t in range(4):
            target[:, :, t, :, :] = 0.1 * t * t

        loss_val = compute_temporal_diff_loss(pred, target)
        assert loss_val.item() > 0, (
            f"Different dynamics should produce loss > 0, got {loss_val.item()}"
        )

    def test_zero_for_identical_dynamics(self):
        """When pred = target + constant offset, frame-to-frame diffs match, loss ~0."""
        target = torch.zeros(1, 1, 4, 8, 8)
        for t in range(4):
            target[:, :, t, :, :] = 0.1 * t

        # Pred = target + constant offset (diffs are identical)
        pred = target + 0.5

        loss_val = compute_temporal_diff_loss(pred, target)
        assert loss_val.item() < 1e-6, (
            f"Identical dynamics should produce loss ~0, got {loss_val.item()}"
        )

    def test_3d_input_handles(self):
        """5D tensor (B,C,T,H,W) should work. Also test T=2 (minimum for 1 diff)."""
        # T=2: minimum valid temporal dimension for 1 diff
        pred = torch.rand(2, 1, 2, 8, 8)
        target = torch.rand(2, 1, 2, 8, 8)
        loss_val = compute_temporal_diff_loss(pred, target)
        assert torch.isfinite(loss_val), f"T=2 should work, got {loss_val.item()}"
        assert loss_val.dim() == 0, "Should return scalar"

        # T=4: standard case
        pred4 = torch.rand(1, 1, 4, 8, 8)
        target4 = torch.rand(1, 1, 4, 8, 8)
        loss_val4 = compute_temporal_diff_loss(pred4, target4)
        assert torch.isfinite(loss_val4), f"T=4 should work, got {loss_val4.item()}"


# ---------------------------------------------------------------------------
# Temporal variation penalty tests
# ---------------------------------------------------------------------------

class TestTemporalVarPenalty:
    def test_negative_value(self):
        """Result should be negative (rewards variation)."""
        # Both pred and target have frame variation
        pred = torch.zeros(1, 1, 4, 8, 8)
        target = torch.zeros(1, 1, 4, 8, 8)
        for t in range(4):
            pred[:, :, t, :, :] = 0.1 * t
            target[:, :, t, :, :] = 0.15 * t

        penalty = compute_temporal_var_penalty(pred, target, lambda_val=0.1)
        assert penalty.item() < 0, (
            f"Penalty should be negative, got {penalty.item()}"
        )

    def test_capped_at_target_variation(self):
        """When pred variation exceeds target, penalty is capped at target level.

        penalty = -lambda * min(pred_var, target_var)
        If pred has 10x more variation, penalty magnitude should equal
        lambda * target_var, NOT lambda * pred_var.
        """
        target = torch.zeros(1, 1, 4, 8, 8)
        for t in range(4):
            target[:, :, t, :, :] = 0.1 * t  # small variation

        # Pred has 10x more variation
        pred = torch.zeros(1, 1, 4, 8, 8)
        for t in range(4):
            pred[:, :, t, :, :] = 1.0 * t  # large variation

        lambda_val = 0.1
        penalty = compute_temporal_var_penalty(pred, target, lambda_val=lambda_val)

        # target_var = mean(|diffs|) = 0.1 (constant diffs of 0.1)
        # Expected penalty = -0.1 * 0.1 = -0.01 (capped at target, not pred)
        target_diffs = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        expected_target_var = target_diffs.abs().mean().item()
        expected_penalty = -lambda_val * expected_target_var

        assert abs(penalty.item() - expected_penalty) < 1e-5, (
            f"Penalty should be capped at target variation: "
            f"expected {expected_penalty:.6f}, got {penalty.item():.6f}"
        )

    def test_zero_for_single_frame(self):
        """T=1 input (no diffs possible) should return 0."""
        pred = torch.rand(1, 1, 1, 8, 8)
        target = torch.rand(1, 1, 1, 8, 8)
        penalty = compute_temporal_var_penalty(pred, target, lambda_val=0.1)
        assert penalty.item() == 0.0, (
            f"T=1 should return 0, got {penalty.item()}"
        )


# ---------------------------------------------------------------------------
# Temporal weighting tests
# ---------------------------------------------------------------------------

class TestTemporalWeighting:
    def test_later_timesteps_weighted_more(self):
        """With weights [1.0, 1.5, 2.0, 2.5], error at later timesteps contributes more.

        If loss tensor is 1.0 everywhere, weighted mean should be
        (1.0+1.5+2.0+2.5)/4 = 1.75, not 1.0 (unweighted).
        """
        loss_tensor = torch.ones(1, 1, 4, 8, 8)
        weights = [1.0, 1.5, 2.0, 2.5]
        weighted_loss = apply_temporal_weights(loss_tensor, weights)

        expected = (1.0 + 1.5 + 2.0 + 2.5) / 4.0  # = 1.75
        assert abs(weighted_loss.item() - expected) < 1e-5, (
            f"Expected weighted mean ~{expected}, got {weighted_loss.item()}"
        )

    def test_broadcast_shape(self):
        """Function should accept (B,C,T,H,W) and return a scalar."""
        loss_tensor = torch.rand(2, 3, 4, 16, 16)
        weights = [1.0, 1.5, 2.0, 2.5]
        result = apply_temporal_weights(loss_tensor, weights)
        assert result.dim() == 0, f"Expected scalar, got dim={result.dim()}"
        assert torch.isfinite(result), f"Result not finite: {result.item()}"


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
        """return_components=True should return dict with all 8 keys."""
        loss_fn = CompositeLoss(use_ms_ssim=False)
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        result = loss_fn(pred, target, return_components=True)

        assert isinstance(result, dict)
        expected_keys = {
            "total", "l1", "ssim", "ssim_val", "extreme",
            "temporal_diff", "temporal_var", "asymmetric",
        }
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}. "
            f"Missing: {expected_keys - set(result.keys())}"
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

    def test_composite_loss_5d_temporal_terms(self):
        """5D input produces non-zero temporal_diff and temporal_var components."""
        loss_fn = CompositeLoss(use_ms_ssim=False)

        # Create pred and target with DIFFERENT frame-to-frame dynamics
        pred = torch.zeros(1, 1, 4, 32, 32)
        target = torch.zeros(1, 1, 4, 32, 32)

        # Pred: linear changes (constant diffs)
        for t in range(4):
            pred[:, :, t, :, :] = 0.1 * t

        # Target: quadratic changes (increasing diffs)
        for t in range(4):
            target[:, :, t, :, :] = 0.1 * t * t

        result = loss_fn(pred, target, return_components=True)

        # temporal_diff should be > 0 (different dynamics)
        assert result["temporal_diff"].item() > 0, (
            f"temporal_diff should be > 0 for different dynamics, "
            f"got {result['temporal_diff'].item()}"
        )

        # temporal_var should be < 0 (negative penalty)
        assert result["temporal_var"].item() < 0, (
            f"temporal_var should be < 0 (negative penalty), "
            f"got {result['temporal_var'].item()}"
        )

    def test_composite_loss_4d_backward_compat(self):
        """4D input still works, temporal terms are 0.0."""
        loss_fn = CompositeLoss(use_ms_ssim=False)
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        result = loss_fn(pred, target, return_components=True)

        assert result["temporal_diff"].item() == 0.0, (
            f"4D temporal_diff should be 0.0, got {result['temporal_diff'].item()}"
        )
        assert result["temporal_var"].item() == 0.0, (
            f"4D temporal_var should be 0.0, got {result['temporal_var'].item()}"
        )

    def test_composite_loss_temporal_weights_applied(self):
        """With per-timestep weights, later timesteps affect loss more than earlier.

        Two CompositeLoss instances: one with uniform weights [1,1,1,1],
        one with [1,1.5,2,2.5]. Feed same 5D input where error concentrates
        at last timestep. Second instance should have higher total loss.
        """
        # Create input where error is concentrated at the last timestep
        pred = torch.zeros(1, 1, 4, 32, 32)
        target = torch.zeros(1, 1, 4, 32, 32)

        # Only the last timestep has significant error
        target[:, :, 3, :, :] = 0.5  # large target at last timestep
        # pred stays zero everywhere -> error concentrated at t=3

        loss_uniform = CompositeLoss(
            use_ms_ssim=False,
            temporal_weights=[1.0, 1.0, 1.0, 1.0],
        )
        loss_ramped = CompositeLoss(
            use_ms_ssim=False,
            temporal_weights=[1.0, 1.5, 2.0, 2.5],
        )

        val_uniform = loss_uniform(pred, target)
        val_ramped = loss_ramped(pred, target)

        assert val_ramped.item() > val_uniform.item(), (
            f"Ramped weights should produce higher loss when error is at last timestep: "
            f"ramped={val_ramped.item():.6f} vs uniform={val_uniform.item():.6f}"
        )


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

    def test_get_loss_function_composite_new_params(self):
        """Factory reads temporal_diff_weight, temporal_var_lambda, asymmetric params from config."""
        config = {
            "type": "composite",
            "l1_weight": 1.0,
            "ssim_weight": 0.3,
            "extreme_weight": 3.0,
            "use_ms_ssim": False,
            "temporal_diff_weight": 1.0,
            "temporal_var_lambda": 0.1,
            "asymmetric_weight": 0.5,
            "asymmetric_alpha": 2.0,
            "extreme_threshold": 0.3456,
            "temporal_weights": [1.0, 1.5, 2.0, 2.5],
        }
        fn = get_loss_function(config)
        assert isinstance(fn, CompositeLoss)

        # Verify the new parameters were set correctly
        assert fn.temporal_diff_weight == 1.0
        assert fn.temporal_var_lambda == 0.1
        assert fn.asymmetric_weight == 0.5
        assert hasattr(fn, "asymmetric_extreme")
        assert fn.temporal_weights == [1.0, 1.5, 2.0, 2.5]

        # Verify it works end-to-end with 5D input
        pred = torch.rand(1, 1, 4, 32, 32)
        target = torch.rand(1, 1, 4, 32, 32)
        result = fn(pred, target, return_components=True)
        assert "temporal_diff" in result
        assert "asymmetric" in result

    def test_get_loss_function_unknown_raises(self):
        """Unknown loss type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function({"type": "mse"})
