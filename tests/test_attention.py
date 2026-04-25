"""Unit tests for attention modules: TemporalAttention and AttentionGate."""
import pytest
import torch
import inspect


# Test dimensions (small for speed)
BATCH = 1
CHANNELS = 16
ENCODER_CHANNELS = 8
DECODER_CHANNELS = 16
H, W = 8, 8


@pytest.fixture
def temporal_attention():
    """Create a TemporalAttention module."""
    from models.attention import TemporalAttention
    return TemporalAttention(channels=CHANNELS, t_max=20)


@pytest.fixture
def attention_gate():
    """Create an AttentionGate module."""
    from models.attention import AttentionGate
    return AttentionGate(
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
    )


class TestTemporalAttention:
    """Tests for the TemporalAttention module."""

    def test_output_shapes(self, temporal_attention):
        """TemporalAttention returns (context, attn_weights) with correct shapes."""
        decoder_state = torch.randn(BATCH, CHANNELS, H, W)
        T = 5
        encoder_states = [torch.randn(BATCH, CHANNELS, H, W) for _ in range(T)]
        context, attn_weights = temporal_attention(decoder_state, encoder_states)
        assert context.shape == (BATCH, CHANNELS, H, W)
        assert attn_weights.shape == (BATCH, T)

    def test_attn_weights_sum_to_one(self, temporal_attention):
        """Attention weights sum to 1.0 along T dimension."""
        decoder_state = torch.randn(BATCH, CHANNELS, H, W)
        T = 5
        encoder_states = [torch.randn(BATCH, CHANNELS, H, W) for _ in range(T)]
        _, attn_weights = temporal_attention(decoder_state, encoder_states)
        weight_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_context_finite(self, temporal_attention):
        """Context output is finite and correct shape."""
        decoder_state = torch.randn(BATCH, CHANNELS, H, W)
        T = 3
        encoder_states = [torch.randn(BATCH, CHANNELS, H, W) for _ in range(T)]
        context, _ = temporal_attention(decoder_state, encoder_states)
        assert torch.isfinite(context).all()

    @pytest.mark.parametrize("T", [1, 5, 10])
    def test_variable_sequence_lengths(self, temporal_attention, T):
        """TemporalAttention works with T=1, T=5, T=10 encoder states."""
        decoder_state = torch.randn(BATCH, CHANNELS, H, W)
        encoder_states = [torch.randn(BATCH, CHANNELS, H, W) for _ in range(T)]
        context, attn_weights = temporal_attention(decoder_state, encoder_states)
        assert context.shape == (BATCH, CHANNELS, H, W)
        assert attn_weights.shape == (BATCH, T)

    def test_uses_manual_attention(self, temporal_attention):
        """TemporalAttention uses torch.bmm, NOT F.scaled_dot_product_attention."""
        source = inspect.getsource(temporal_attention.__class__)
        assert "torch.bmm" in source or "bmm(" in source
        assert "scaled_dot_product_attention" not in source

    def test_pos_embed_is_parameter_and_zeros(self):
        """pos_embed should be a learnable parameter initialized to zeros."""
        from models.attention import TemporalAttention
        attn = TemporalAttention(channels=CHANNELS, t_max=10)
        assert hasattr(attn, 'pos_embed')
        # Should be a parameter
        param_names = {n for n, _ in attn.named_parameters()}
        assert 'pos_embed' in param_names, "pos_embed should be a learnable parameter"
        # Initialized to zeros
        assert torch.allclose(attn.pos_embed, torch.zeros_like(attn.pos_embed)), (
            "pos_embed should be initialized to zeros"
        )
        assert attn.pos_embed.shape == (10, CHANNELS)



class TestAttentionGate:
    """Tests for the AttentionGate module."""

    def test_output_shape(self, attention_gate):
        """AttentionGate returns gated_x with encoder_channels shape."""
        g = torch.randn(BATCH, DECODER_CHANNELS, H, W)
        x = torch.randn(BATCH, ENCODER_CHANNELS, H, W)
        gated_x = attention_gate(g, x)
        assert gated_x.shape == (BATCH, ENCODER_CHANNELS, H, W)

    def test_output_range(self, attention_gate):
        """AttentionGate output in valid range (0 to max of input x)."""
        g = torch.randn(BATCH, DECODER_CHANNELS, H, W)
        x = torch.abs(torch.randn(BATCH, ENCODER_CHANNELS, H, W))  # positive input
        gated_x = attention_gate(g, x)
        # Since gate is sigmoid * x, output should be between 0 and x
        assert (gated_x >= 0).all()
        assert (gated_x <= x + 1e-6).all()

    def test_zero_decoder_near_uniform_gate(self, attention_gate):
        """AttentionGate with all-zero decoder input produces near-uniform gating."""
        g = torch.zeros(BATCH, DECODER_CHANNELS, H, W)
        x = torch.ones(BATCH, ENCODER_CHANNELS, H, W)
        gated_x = attention_gate(g, x)
        # With g=0, gate ≈ sigmoid(W_x(x) bias terms) -- should be near 0.5 * x
        # The gate should be roughly uniform (not collapsed to 0 or 1)
        gate_values = gated_x / (x + 1e-8)
        assert gate_values.mean() > 0.1, "Gate is collapsed to near-zero"
        assert gate_values.mean() < 0.9, "Gate is collapsed to near-one"

    def test_gate_not_collapsed(self, attention_gate):
        """AttentionGate is not all 0 or all 1 for non-trivial inputs."""
        g = torch.randn(BATCH, DECODER_CHANNELS, H, W)
        x = torch.ones(BATCH, ENCODER_CHANNELS, H, W)
        gated_x = attention_gate(g, x)
        # With x=ones, gated_x = gate values directly
        assert not torch.allclose(gated_x, torch.zeros_like(gated_x), atol=1e-3), (
            "Gate collapsed to all zeros"
        )
        assert not torch.allclose(gated_x, torch.ones_like(gated_x), atol=1e-3), (
            "Gate collapsed to all ones"
        )

    def test_no_batchnorm(self, attention_gate):
        """AttentionGate does not contain BatchNorm (batch_size=1 instability)."""
        for name, module in attention_gate.named_modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)), (
                f"Found BatchNorm at {name} -- not allowed per CONTEXT.md"
            )
