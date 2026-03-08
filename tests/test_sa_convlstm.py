"""Unit tests for SA-ConvLSTM module: SelfAttentionMemory, SAConvLSTMCell, SAConvLSTM."""
import pytest
import torch
import inspect


# Test dimensions (small for speed)
BATCH = 1
HIDDEN_DIM = 16
ATTN_DIM = 8
INPUT_DIM = 4
KERNEL_SIZE = 3
H, W = 8, 8
T = 3


@pytest.fixture
def sam():
    """Create a SelfAttentionMemory module."""
    from models.sa_convlstm import SelfAttentionMemory
    return SelfAttentionMemory(hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM)


@pytest.fixture
def sa_cell():
    """Create an SAConvLSTMCell."""
    from models.sa_convlstm import SAConvLSTMCell
    return SAConvLSTMCell(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE)


@pytest.fixture
def sa_convlstm():
    """Create an SAConvLSTM wrapper."""
    from models.sa_convlstm import SAConvLSTM
    return SAConvLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE)


class TestSelfAttentionMemory:
    """Tests for the SelfAttentionMemory module."""

    def test_output_shapes(self, sam):
        """SelfAttentionMemory returns (h_out, m_new) with correct shapes."""
        h = torch.randn(BATCH, HIDDEN_DIM, H, W)
        m_prev = torch.randn(BATCH, HIDDEN_DIM, H, W)
        h_out, m_new = sam(h, m_prev)
        assert h_out.shape == (BATCH, HIDDEN_DIM, H, W)
        assert m_new.shape == (BATCH, HIDDEN_DIM, H, W)

    def test_outputs_finite(self, sam):
        """SelfAttentionMemory outputs are all finite (no NaN/inf)."""
        h = torch.randn(BATCH, HIDDEN_DIM, H, W)
        m_prev = torch.randn(BATCH, HIDDEN_DIM, H, W)
        h_out, m_new = sam(h, m_prev)
        assert torch.isfinite(h_out).all()
        assert torch.isfinite(m_new).all()

    def test_uses_manual_attention(self, sam):
        """SelfAttentionMemory uses torch.bmm, NOT F.scaled_dot_product_attention."""
        source = inspect.getsource(sam.__class__)
        assert "torch.bmm" in source or "bmm(" in source
        assert "scaled_dot_product_attention" not in source

    def test_residual_connection(self, sam):
        """h_out differs from h_input due to residual addition but preserves shape."""
        h = torch.randn(BATCH, HIDDEN_DIM, H, W)
        m_prev = torch.randn(BATCH, HIDDEN_DIM, H, W)
        h_out, _ = sam(h, m_prev)
        # h_out should be h + some_projection, so it differs from h
        assert h_out.shape == h.shape
        assert not torch.allclose(h_out, h, atol=1e-6)

    def test_parameter_count(self, sam):
        """SAM parameter count ~3.5 * C^2 (within 20%)."""
        total_params = sum(p.numel() for p in sam.parameters())
        expected = 3.5 * HIDDEN_DIM ** 2  # 3.5 * 256 = 896
        # Allow 20% tolerance plus bias terms
        assert total_params == pytest.approx(expected, rel=0.2), (
            f"Expected ~{expected} params, got {total_params}"
        )


class TestSAConvLSTMCell:
    """Tests for the SAConvLSTMCell."""

    def test_output_3tuple(self, sa_cell):
        """SAConvLSTMCell returns (h, c, m) 3-tuple."""
        x = torch.randn(BATCH, INPUT_DIM, H, W)
        h = torch.zeros(BATCH, HIDDEN_DIM, H, W)
        c = torch.zeros(BATCH, HIDDEN_DIM, H, W)
        m = torch.zeros(BATCH, HIDDEN_DIM, H, W)
        result = sa_cell(x, h, c, m)
        assert len(result) == 3

    def test_output_shapes(self, sa_cell):
        """SAConvLSTMCell output shapes match hidden_dim."""
        x = torch.randn(BATCH, INPUT_DIM, H, W)
        h = torch.zeros(BATCH, HIDDEN_DIM, H, W)
        c = torch.zeros(BATCH, HIDDEN_DIM, H, W)
        m = torch.zeros(BATCH, HIDDEN_DIM, H, W)
        h_out, c_out, m_out = sa_cell(x, h, c, m)
        assert h_out.shape == (BATCH, HIDDEN_DIM, H, W)
        assert c_out.shape == (BATCH, HIDDEN_DIM, H, W)
        assert m_out.shape == (BATCH, HIDDEN_DIM, H, W)

    def test_composition_pattern(self, sa_cell):
        """SAConvLSTMCell wraps ConvLSTMCell via composition."""
        from models.convlstm import ConvLSTMCell
        assert hasattr(sa_cell, "convlstm_cell")
        assert isinstance(sa_cell.convlstm_cell, ConvLSTMCell)


class TestSAConvLSTM:
    """Tests for the SAConvLSTM wrapper."""

    def test_output_shapes(self, sa_convlstm):
        """SAConvLSTM returns outputs=(B,hidden_dim,T,H,W) and hidden_state as list of 3-tuples."""
        x = torch.randn(BATCH, INPUT_DIM, T, H, W)
        outputs, hidden_state = sa_convlstm(x)
        assert outputs.shape == (BATCH, HIDDEN_DIM, T, H, W)
        assert isinstance(hidden_state, list)
        assert len(hidden_state) == 1  # num_layers=1
        assert len(hidden_state[0]) == 3  # (h, c, m)

    def test_hidden_state_shapes(self, sa_convlstm):
        """Hidden state 3-tuples have correct shapes."""
        x = torch.randn(BATCH, INPUT_DIM, T, H, W)
        _, hidden_state = sa_convlstm(x)
        h, c, m = hidden_state[0]
        assert h.shape == (BATCH, HIDDEN_DIM, H, W)
        assert c.shape == (BATCH, HIDDEN_DIM, H, W)
        assert m.shape == (BATCH, HIDDEN_DIM, H, W)

    def test_init_hidden_3tuples(self, sa_convlstm):
        """_init_hidden returns list of 3-tuples (h, c, m) all zeros."""
        hidden = sa_convlstm._init_hidden(BATCH, H, W, torch.device("cpu"))
        assert len(hidden) == 1
        assert len(hidden[0]) == 3
        h, c, m = hidden[0]
        assert torch.all(h == 0)
        assert torch.all(c == 0)
        assert torch.all(m == 0)

    def test_none_hidden_init(self, sa_convlstm):
        """SAConvLSTM with hidden_state=None produces finite output."""
        x = torch.randn(BATCH, INPUT_DIM, T, H, W)
        outputs, _ = sa_convlstm(x, hidden_state=None)
        assert torch.isfinite(outputs).all()

    def test_continue_from_hidden(self, sa_convlstm):
        """SAConvLSTM can continue from provided hidden_state (3-tuple format)."""
        x = torch.randn(BATCH, INPUT_DIM, T, H, W)
        _, hidden_state = sa_convlstm(x)
        # Run second sequence with previous hidden state
        x2 = torch.randn(BATCH, INPUT_DIM, T, H, W)
        outputs2, hidden_state2 = sa_convlstm(x2, hidden_state=hidden_state)
        assert outputs2.shape == (BATCH, HIDDEN_DIM, T, H, W)
        assert torch.isfinite(outputs2).all()
        assert len(hidden_state2[0]) == 3
