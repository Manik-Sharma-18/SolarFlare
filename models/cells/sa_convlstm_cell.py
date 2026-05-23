"""SA-ConvLSTM cell — ConvLSTM cell + Self-Attention Memory."""
import torch
import torch.nn as nn
from typing import Tuple

from .convlstm_cell import ConvLSTMCell
from .sam import SelfAttentionMemory


class SAConvLSTMCell(nn.Module):
    """Composition of ConvLSTMCell + SelfAttentionMemory.

    Returns ``(h, c, m)`` 3-tuple. Callers must thread ``m`` through time.

    Args:
        input_dim: Input channels.
        hidden_dim: Hidden state channels.
        kernel_size: Conv kernel size.
        attn_dim: SAM attention dim. Defaults to ``hidden_dim // 2``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        attn_dim: int = None,
    ):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim // 2
        self.convlstm_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.sam = SelfAttentionMemory(hidden_dim, attn_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, c = self.convlstm_cell(x, h_prev, c_prev)
        h_out, m_new = self.sam(h, m_prev)
        return h_out, c, m_new
