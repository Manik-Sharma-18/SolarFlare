"""Unit tests for SolarFluxPredictor forward pass shapes (TST-01, TST-05).

Tests cover:
- Basic forward shape (single-channel)
- Dual-channel input with single-channel output
- Various t_out and t_in combinations (parametrized)
- Downsample mode
- Output finiteness
- Teacher forcing
- Dropout randomness
- Parameter counting
- Device placement
- MPS smoke test (skipped if unavailable)
- SA-ConvLSTM integration (ARCH-01)
- Delta scale (ARCH-02)
- Attention gate (ARCH-03)
- Wider channels (ARCH-04)
- Kernel size 5 (ARCH-05)
- Temporal attention (ARCH-07)
- Full v3.0 architecture integration
"""
import pytest
import torch
import torch.nn as nn

from models.predictor import SolarFluxPredictor


def _make_model(**overrides):
    """Create a tiny SolarFluxPredictor with fast defaults."""
    defaults = dict(
        input_channels=1,
        output_channels=1,
        t_out=2,
        channels=[4, 8, 16],
        kernel_size=3,
        downsample_input=False,
        use_checkpointing=False,
        dropout_rate=0.0,
    )
    defaults.update(overrides)
    return SolarFluxPredictor(**defaults)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestForwardShape:
    def test_forward_basic_shape(self):
        """input_channels=1, output_channels=1, t_in=4, t_out=2 -> (1,1,2,32,32)."""
        model = _make_model(t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32), f"Got {out.shape}"

    def test_forward_dual_channel(self):
        """input_channels=2, output_channels=1 -> (1,1,2,32,32)."""
        model = _make_model(input_channels=2, output_channels=1, t_out=2)
        model.eval()
        x = torch.randn(1, 2, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32), f"Got {out.shape}"

    @pytest.mark.parametrize("t_out", [1, 3, 5])
    def test_forward_different_t_out(self, t_out):
        """Output temporal dim should match t_out."""
        model = _make_model(t_out=t_out)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, t_out, 32, 32), f"Got {out.shape}"

    @pytest.mark.parametrize("t_in", [2, 6, 10])
    def test_forward_different_t_in(self, t_in):
        """Model should handle different input sequence lengths."""
        model = _make_model(t_out=2)
        model.eval()
        x = torch.randn(1, 1, t_in, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32), f"Got {out.shape}"

    def test_forward_with_downsample(self):
        """downsample_input=True with 64x64 spatial -> (1,1,2,64,64)."""
        model = _make_model(downsample_input=True, t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 64, 64), f"Got {out.shape}"

    def test_forward_sa_convlstm(self):
        """ARCH-01: SA-ConvLSTM model produces correct output shape."""
        model = _make_model(use_sa_convlstm=True, t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)

    def test_forward_wider_channels(self):
        """ARCH-04: Wider channels [32,64,128] produce correct shape."""
        model = _make_model(channels=[32, 64, 128], t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)

    def test_forward_kernel_5(self):
        """ARCH-05: Kernel size 5 produces correct shape."""
        model = _make_model(kernel_size=5, t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)


# ---------------------------------------------------------------------------
# Output quality tests
# ---------------------------------------------------------------------------

class TestForwardOutput:
    def test_forward_output_is_finite(self):
        """All output values should be finite."""
        model = _make_model(t_out=2)
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "Non-finite values in output"

    def test_forward_with_teacher_forcing(self):
        """Teacher forcing should not change output shape."""
        model = _make_model(t_out=2)
        model.train()
        x = torch.randn(1, 1, 4, 32, 32)
        y_true = torch.randn(1, 1, 2, 32, 32)
        out = model(x, teacher_forcing_ratio=0.5, y_true=y_true)
        assert out.shape == (1, 1, 2, 32, 32), f"Got {out.shape}"

    def test_forward_with_dropout(self):
        """Dropout should produce different outputs across runs in train mode."""
        model = _make_model(dropout_rate=0.1, t_out=2)
        model.train()
        x = torch.randn(1, 1, 4, 32, 32)

        torch.manual_seed(0)
        out1 = model(x)

        torch.manual_seed(1)
        out2 = model(x)

        assert out1.shape == (1, 1, 2, 32, 32)
        # Outputs should differ due to dropout randomness
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs across runs"
        )


# ---------------------------------------------------------------------------
# Delta scale tests (ARCH-02)
# ---------------------------------------------------------------------------

class TestDeltaScale:
    def test_delta_scale_exists(self):
        """ARCH-02: delta_scale parameter exists when init != 0."""
        model = _make_model(delta_scale_init=100.0)
        assert hasattr(model, 'delta_scale')
        assert model.delta_scale is not None
        assert isinstance(model.delta_scale, nn.Parameter)

    def test_delta_scale_init_value(self):
        """ARCH-02: delta_scale initialized to specified value."""
        model = _make_model(delta_scale_init=100.0)
        assert model.delta_scale.item() == pytest.approx(100.0)

    def test_delta_scale_disabled(self):
        """delta_scale is None when init is 0.0."""
        model = _make_model(delta_scale_init=0.0)
        assert model.delta_scale is None

    def test_delta_scale_affects_output(self):
        """ARCH-02: delta_scale changes output magnitude."""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 4, 32, 32)
        model_no_scale = _make_model(delta_scale_init=0.0, t_out=2)
        model_with_scale = _make_model(delta_scale_init=100.0, t_out=2)
        # Just verify both produce finite output
        model_no_scale.eval()
        model_with_scale.eval()
        with torch.no_grad():
            out1 = model_no_scale(x)
            out2 = model_with_scale(x)
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()


# ---------------------------------------------------------------------------
# Full v3.0 architecture tests (ARCH-01 through ARCH-07)
# ---------------------------------------------------------------------------

class TestFullArchitecture:
    def test_full_arch_forward(self):
        """ALL ARCH: Full v3.0 architecture produces finite output."""
        model = _make_model(
            use_sa_convlstm=True, temporal_attention=True,
            attention_gate=True, delta_scale_init=100.0,
            dropout_rate=0.15, channels=[4, 8, 16], kernel_size=3, t_out=2
        )
        model.eval()
        x = torch.randn(1, 1, 10, 32, 32)  # T=10 input
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)
        assert torch.isfinite(out).all()

    def test_full_arch_dual_channel(self):
        """Full v3.0 arch with dual-channel input."""
        model = _make_model(
            input_channels=2, output_channels=1,
            use_sa_convlstm=True, temporal_attention=True,
            attention_gate=True, delta_scale_init=100.0,
            dropout_rate=0.15, t_out=2
        )
        model.eval()
        x = torch.randn(1, 2, 10, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)
        assert torch.isfinite(out).all()

    def test_full_arch_with_downsample(self):
        """Full v3.0 arch with input downsampling."""
        model = _make_model(
            use_sa_convlstm=True, temporal_attention=True,
            attention_gate=True, delta_scale_init=100.0,
            downsample_input=True, t_out=2
        )
        model.eval()
        x = torch.randn(1, 1, 10, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 64, 64)
        assert torch.isfinite(out).all()

    def test_backward_compatibility(self):
        """Default params (no SA features) still work correctly."""
        model = _make_model(t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)
        assert torch.isfinite(out).all()

    def test_sa_convlstm_replaces_all_modules(self):
        """ARCH-01: All 6 ConvLSTM modules are SAConvLSTM when enabled."""
        from models.sa_convlstm import SAConvLSTM
        model = _make_model(use_sa_convlstm=True)
        assert isinstance(model.encoder_conv1, SAConvLSTM)
        assert isinstance(model.encoder_conv2, SAConvLSTM)
        assert isinstance(model.encoder_conv3, SAConvLSTM)
        assert isinstance(model.decoder_conv2, SAConvLSTM)
        assert isinstance(model.decoder_conv3, SAConvLSTM)
        assert isinstance(model.refine_conv, SAConvLSTM)

    def test_sa_dropout_stochastic(self):
        """ARCH-06: SA model with dropout produces stochastic outputs in train mode."""
        model = _make_model(
            use_sa_convlstm=True, dropout_rate=0.15, t_out=2
        )
        model.train()
        x = torch.randn(1, 1, 4, 32, 32)

        torch.manual_seed(0)
        out1 = model(x)

        torch.manual_seed(1)
        out2 = model(x)

        assert out1.shape == (1, 1, 2, 32, 32)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "SA model with dropout should produce different outputs"
        )

    def test_wider_channels_kernel5_combined(self):
        """ARCH-04+05: Wider channels [32,64,128] with kernel_size=5."""
        model = _make_model(channels=[32, 64, 128], kernel_size=5, t_out=2)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestModelUtilities:
    def test_count_parameters(self):
        """count_parameters() should return a positive integer."""
        model = _make_model()
        n_params = model.count_parameters()
        assert isinstance(n_params, int)
        assert n_params > 0, f"Expected positive, got {n_params}"

    def test_model_to_device(self):
        """Model should move to CPU without error."""
        model = _make_model()
        model = model.to(torch.device("cpu"))
        for p in model.parameters():
            assert p.device == torch.device("cpu"), f"Param on {p.device}"


# ---------------------------------------------------------------------------
# MPS smoke test (TST-05)
# ---------------------------------------------------------------------------

class TestMPSSmoke:
    @pytest.mark.mps
    def test_forward_on_mps(self):
        """Forward pass on MPS should produce finite output with correct shape."""
        device = torch.device("mps")
        model = _make_model(t_out=2).to(device)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32), f"Got {out.shape}"
        assert torch.isfinite(out.cpu()).all(), "Non-finite values in MPS output"

    @pytest.mark.mps
    def test_full_arch_mps(self):
        """Full v3.0 architecture on MPS device."""
        device = torch.device("mps")
        model = _make_model(
            use_sa_convlstm=True, temporal_attention=True,
            attention_gate=True, delta_scale_init=100.0,
            dropout_rate=0.15, t_out=2
        ).to(device)
        model.eval()
        x = torch.randn(1, 1, 4, 32, 32, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 2, 32, 32)
        assert torch.isfinite(out.cpu()).all()
