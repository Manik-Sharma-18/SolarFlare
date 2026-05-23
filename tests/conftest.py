"""Shared fixtures and pytest configuration for SolarFlare tests."""
import pytest


def pytest_configure(config):
    """Register custom markers for device-specific tests."""
    config.addinivalue_line("markers", "mps: requires MPS device")
    config.addinivalue_line("markers", "cuda: requires CUDA device")


def pytest_collection_modifyitems(config, items):
    """Auto-skip device-specific tests when hardware is unavailable."""
    import torch

    skip_mps = None
    skip_cuda = None

    for item in items:
        if "mps" in item.keywords:
            if skip_mps is None:
                if not torch.backends.mps.is_available():
                    skip_mps = pytest.mark.skip(reason="MPS not available")
                else:
                    skip_mps = False
            if skip_mps:
                item.add_marker(skip_mps)

        if "cuda" in item.keywords:
            if skip_cuda is None:
                if not torch.cuda.is_available():
                    skip_cuda = pytest.mark.skip(reason="CUDA not available")
                else:
                    skip_cuda = False
            if skip_cuda:
                item.add_marker(skip_cuda)


@pytest.fixture
def base_config():
    """Return a minimal valid config dict that passes validate_config()."""
    return {
        "device": "cpu",
        "seed": 42,
        "data": {
            "data_dir": "./data",
            "t_in": 4,
            "t_out": 2,
            "split_ratios": [0.7, 0.2, 0.1],
            "augmentation": "none",
            "stride": 1,
            "num_workers": 0,
        },
        "model": {
            "input_channels": 1,
            "channels": [8, 16, 32],
            "kernel_size": 3,
        },
        "training": {
            "batch_size": 1,
            "epochs": 5,
            "patience": 3,
            "lr": 0.001,
            "grad_clip": 1.0,
            "use_amp": False,
        },
        "loss": {
            "type": "l1",
        },
        "normalization": {
            "method": "asinh",
        },
    }


@pytest.fixture(scope="session")
def device():
    """Return the universal CPU test device."""
    import torch
    return torch.device("cpu")


@pytest.fixture
def tiny_model_config():
    """Return minimal model params for creating a SolarFluxPredictor."""
    return {
        "input_channels": 1,
        "output_channels": 1,
        "t_out": 2,
        "channels": [4, 8, 16],
        "kernel_size": 3,
        "use_checkpointing": False,
        "dropout_rate": 0.0,
    }


@pytest.fixture
def sa_model_config():
    """Return model params with all v3.0 ARCH features enabled (tiny channels for speed)."""
    return {
        "input_channels": 1,
        "output_channels": 1,
        "t_out": 2,
        "channels": [4, 8, 16],
        "kernel_size": 3,
        "use_checkpointing": False,
        "dropout_rate": 0.15,
        "use_sa_convlstm": True,
        "temporal_attention": True,
        "attention_gate": True,
        "delta_scale_init": 100.0,
    }
