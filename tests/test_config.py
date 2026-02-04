"""Tests for utils/config_validator.py -- config validation and error accumulation."""
import copy

import pytest

from utils.config_validator import validate_config, ConfigValidationError


# ------------------------------------------------------------------ #
# Valid config
# ------------------------------------------------------------------ #


def test_valid_config_passes(base_config):
    """A valid base_config passes validate_config() without raising."""
    validate_config(base_config)  # Should not raise


# ------------------------------------------------------------------ #
# Top-level field validation
# ------------------------------------------------------------------ #


def test_missing_device_raises(base_config):
    """Missing 'device' key raises ConfigValidationError."""
    del base_config["device"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("'device' is required" in e for e in exc_info.value.errors)


def test_invalid_device_raises(base_config):
    """device='tpu' raises ConfigValidationError."""
    base_config["device"] = "tpu"
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("device" in e for e in exc_info.value.errors)


def test_missing_seed_raises(base_config):
    """Missing 'seed' raises ConfigValidationError."""
    del base_config["seed"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("'seed' is required" in e for e in exc_info.value.errors)


def test_negative_seed_raises(base_config):
    """seed=-1 raises ConfigValidationError."""
    base_config["seed"] = -1
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("seed" in e and ">= 0" in e for e in exc_info.value.errors)


# ------------------------------------------------------------------ #
# Data section validation
# ------------------------------------------------------------------ #


def test_missing_data_section_raises(base_config):
    """Missing 'data' section raises ConfigValidationError."""
    del base_config["data"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("'data' section is required" in e for e in exc_info.value.errors)


def test_missing_t_in_raises(base_config):
    """Missing data.t_in raises ConfigValidationError."""
    del base_config["data"]["t_in"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("data.t_in" in e for e in exc_info.value.errors)


def test_invalid_split_ratios_sum(base_config):
    """split_ratios that don't sum to ~1.0 raise ConfigValidationError."""
    base_config["data"]["split_ratios"] = [0.5, 0.1, 0.1]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("split_ratios" in e and "sum" in e for e in exc_info.value.errors)


def test_invalid_augmentation_mode(base_config):
    """augmentation='random' raises ConfigValidationError."""
    base_config["data"]["augmentation"] = "random"
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("augmentation" in e for e in exc_info.value.errors)


# ------------------------------------------------------------------ #
# Cross-field validation
# ------------------------------------------------------------------ #


def test_dual_channel_input_channels_mismatch(base_config):
    """dual_channel=True with input_channels=1 raises ConfigValidationError."""
    base_config["data"]["dual_channel"] = True
    base_config["model"]["input_channels"] = 1
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("dual_channel" in e and "input_channels" in e for e in exc_info.value.errors)


def test_amp_on_cpu_raises(base_config):
    """device='cpu' with use_amp=True raises ConfigValidationError."""
    base_config["device"] = "cpu"
    base_config["training"]["use_amp"] = True
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("AMP" in e and "cpu" in e for e in exc_info.value.errors)


# ------------------------------------------------------------------ #
# Model section validation
# ------------------------------------------------------------------ #


def test_missing_model_channels_raises(base_config):
    """Missing model.channels raises ConfigValidationError."""
    del base_config["model"]["channels"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("model.channels" in e for e in exc_info.value.errors)


def test_even_kernel_size_raises(base_config):
    """kernel_size=4 (even) raises ConfigValidationError."""
    base_config["model"]["kernel_size"] = 4
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("kernel_size" in e and "odd" in e for e in exc_info.value.errors)


# ------------------------------------------------------------------ #
# Error accumulation
# ------------------------------------------------------------------ #


def test_multiple_errors_accumulated(base_config):
    """Introducing 3+ errors at once results in len(e.errors) >= 3."""
    del base_config["device"]
    del base_config["seed"]
    del base_config["data"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert len(exc_info.value.errors) >= 3


# ------------------------------------------------------------------ #
# Backward compatibility
# ------------------------------------------------------------------ #


def test_backward_compat_augment_bool(base_config):
    """data.augment=True converts to data.augmentation='balanced'."""
    base_config["data"]["augment"] = True
    del base_config["data"]["augmentation"]
    # Should not raise
    validate_config(base_config)
    assert base_config["data"]["augmentation"] == "balanced"


# ------------------------------------------------------------------ #
# Phase 5 config fields (SSIM tiling, uncertainty)
# ------------------------------------------------------------------ #


def test_ssim_tiling_threshold_too_small(base_config):
    """ssim_tiling_threshold=16 raises error about >= 32."""
    base_config["loss"]["ssim_tiling_threshold"] = 16
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("ssim_tiling_threshold" in e and "32" in e for e in exc_info.value.errors)


def test_uncertainty_n_samples_too_low(base_config):
    """uncertainty.n_samples=1 raises error about >= 2."""
    base_config["uncertainty"] = {"n_samples": 1}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(base_config)
    assert any("n_samples" in e and ">= 2" in e for e in exc_info.value.errors)
