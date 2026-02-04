"""Tests for utils/device.py -- device resolution, AMP context, DummyGradScaler."""
import pytest
import torch

from utils.device import (
    resolve_device,
    get_amp_context,
    get_grad_scaler,
    clear_device_cache,
    _DummyGradScaler,
)


# ------------------------------------------------------------------ #
# resolve_device tests
# ------------------------------------------------------------------ #


def test_resolve_device_cpu():
    """resolve_device('cpu') returns torch.device('cpu')."""
    dev = resolve_device("cpu")
    assert dev == torch.device("cpu")


def test_resolve_device_auto_returns_valid_device():
    """resolve_device('auto') returns a device with type in (cuda, mps, cpu)."""
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")


def test_resolve_device_invalid_raises_valueerror():
    """resolve_device('tpu') raises ValueError with 'Unknown device'."""
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_resolve_device_unavailable_cuda_raises_runtime():
    """When CUDA unavailable, resolve_device('cuda') raises RuntimeError."""
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS is available")
def test_resolve_device_unavailable_mps_raises_runtime():
    """When MPS unavailable, resolve_device('mps') raises RuntimeError."""
    with pytest.raises(RuntimeError, match="MPS is not available"):
        resolve_device("mps")


@pytest.mark.mps
def test_resolve_device_mps():
    """resolve_device('mps') returns torch.device('mps') -- MPS hardware only."""
    dev = resolve_device("mps")
    assert dev == torch.device("mps")


# ------------------------------------------------------------------ #
# get_amp_context tests
# ------------------------------------------------------------------ #


def test_get_amp_context_disabled():
    """get_amp_context(False, device) returns a nullcontext instance."""
    from contextlib import nullcontext

    ctx = get_amp_context(False, torch.device("cpu"))
    assert isinstance(ctx, nullcontext)


def test_get_amp_context_enabled_cpu():
    """get_amp_context(True, cpu) returns a context manager."""
    ctx = get_amp_context(True, torch.device("cpu"))
    # Verify it works as a context manager
    with ctx:
        pass


# ------------------------------------------------------------------ #
# get_grad_scaler tests
# ------------------------------------------------------------------ #


def test_get_grad_scaler_cpu():
    """get_grad_scaler(False, cpu) returns _DummyGradScaler."""
    scaler = get_grad_scaler(False, torch.device("cpu"))
    assert isinstance(scaler, _DummyGradScaler)


def test_get_grad_scaler_amp_cpu():
    """get_grad_scaler(True, cpu) returns _DummyGradScaler (not real scaler)."""
    scaler = get_grad_scaler(True, torch.device("cpu"))
    assert isinstance(scaler, _DummyGradScaler)


# ------------------------------------------------------------------ #
# _DummyGradScaler tests
# ------------------------------------------------------------------ #


def test_dummy_grad_scaler_scale_passthrough():
    """scaler.scale(loss) returns loss unchanged."""
    scaler = _DummyGradScaler()
    loss = torch.tensor(3.14, requires_grad=True)
    assert scaler.scale(loss) is loss


def test_dummy_grad_scaler_step_calls_optimizer():
    """scaler.step(optimizer) updates parameters."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Capture params before
    params_before = [p.clone().detach() for p in model.parameters()]

    # Forward + backward
    x = torch.randn(1, 2)
    loss = model(x).sum()
    loss.backward()

    scaler = _DummyGradScaler()
    scaler.step(optimizer)

    # Params should have changed
    for before, after in zip(params_before, model.parameters()):
        assert not torch.equal(before, after.detach()), "Parameters should change after step"


def test_dummy_grad_scaler_update_noop():
    """scaler.update() does not raise."""
    scaler = _DummyGradScaler()
    scaler.update()  # Should not raise


def test_dummy_grad_scaler_skips_nan_grads():
    """When gradients contain NaN, scaler.step() does NOT call optimizer.step()."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Set up NaN gradients
    for p in model.parameters():
        p.grad = torch.full_like(p, float("nan"))

    params_before = [p.clone().detach() for p in model.parameters()]

    scaler = _DummyGradScaler()
    scaler.step(optimizer)

    # Params should NOT have changed
    for before, after in zip(params_before, model.parameters()):
        assert torch.equal(before, after.detach()), "Parameters should not change with NaN grads"


# ------------------------------------------------------------------ #
# clear_device_cache tests
# ------------------------------------------------------------------ #


def test_clear_device_cache_cpu_noop():
    """clear_device_cache(cpu) does not raise."""
    clear_device_cache(torch.device("cpu"))
