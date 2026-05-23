"""ConvLSTM cell — single-timestep recurrent unit with spatial conv gates."""
import torch
import torch.nn as nn
from typing import Tuple


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell.

    Replaces LSTM's matrix multiplications with 2D convolutions to preserve
    spatial structure. All four gates (i, f, g, o) are produced by a single
    fused conv over ``cat(x, h_prev)``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        self._init_forget_bias()

    def _init_forget_bias(self):
        """Set forget-gate bias to 1.0 for better gradient flow."""
        with torch.no_grad():
            if self.conv.bias is not None:
                self.conv.bias[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One ConvLSTM step.

        Args:
            x: ``(B, input_dim, H, W)`` input frame.
            h_prev, c_prev: previous hidden / cell state, ``(B, hidden_dim, H, W)``.

        Returns:
            ``(h_next, c_next)``.
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
