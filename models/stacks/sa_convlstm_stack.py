"""Multi-layer SA-ConvLSTM stack."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..cells import SAConvLSTMCell
from .base import rnn_sequence_loop, zero_states_3tuple


def _step(
    cell: SAConvLSTMCell,
    x_t: torch.Tensor,
    state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
):
    h_prev, c_prev, m_prev = state
    h_next, c_next, m_next = cell(x_t, h_prev, c_prev, m_prev)
    return h_next, (h_next, c_next, m_next)


class SAConvLSTM(nn.Module):
    """Stack of :class:`SAConvLSTMCell`.

    Same API as :class:`ConvLSTM` but per-layer state is the ``(h, c, m)``
    3-tuple used by Self-Attention Memory.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        attn_dim: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for layer_idx in range(num_layers):
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(
                SAConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, attn_dim)
            )
        self.cell_list = nn.ModuleList(cells)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        B, _, _, H, W = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(B, H, W, x.device)
        return rnn_sequence_loop(self.cell_list, x, hidden_state, _step)

    def _init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return zero_states_3tuple(
            self.num_layers, batch_size, self.hidden_dim, height, width, device
        )
