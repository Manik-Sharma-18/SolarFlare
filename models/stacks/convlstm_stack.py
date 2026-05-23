"""Multi-layer ConvLSTM stack."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..cells import ConvLSTMCell
from .base import rnn_sequence_loop, zero_states_2tuple


def _step(cell: ConvLSTMCell, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
    h_prev, c_prev = state
    h_next, c_next = cell(x_t, h_prev, c_prev)
    return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    """Stack of :class:`ConvLSTMCell` over time.

    Layer 0 takes ``input_dim``; later layers take ``hidden_dim`` (uniform).
    No dtype argument — always float; callers must cast for mixed precision.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for layer_idx in range(num_layers):
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        self.cell_list = nn.ModuleList(cells)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Process a sequence.

        Args:
            x: ``(B, C, T, H, W)``.
            hidden_state: initial states; auto-zero if ``None``.

        Returns:
            ``(outputs, hidden_state)`` where ``outputs`` is
            ``(B, hidden_dim, T, H, W)``.
        """
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
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return zero_states_2tuple(
            self.num_layers, batch_size, self.hidden_dim, height, width, device
        )
