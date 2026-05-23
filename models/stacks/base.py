"""Shared recurrent-stack scaffolding.

Both :class:`ConvLSTM` and :class:`SAConvLSTM` share the same outer
``for t in range(T): for layer in range(L): ...`` loop. This module factors
that out so each stack only supplies its per-cell step and per-layer state
initializer.
"""
from typing import Callable, List, Tuple

import torch


StepFn = Callable[[torch.nn.Module, torch.Tensor, Tuple], Tuple[torch.Tensor, Tuple]]


def rnn_sequence_loop(
    cells: torch.nn.ModuleList,
    x: torch.Tensor,
    hidden_state: List[Tuple],
    step_cell: StepFn,
) -> Tuple[torch.Tensor, List[Tuple]]:
    """Drive a multi-layer recurrent stack over a temporal sequence.

    Args:
        cells: per-layer cells.
        x: input sequence ``(B, C, T, H, W)``.
        hidden_state: per-layer state tuples (mutated in place — pass clones if
            you need the original preserved).
        step_cell: ``(cell, x_t, layer_state) → (h_next, new_layer_state)``.

    Returns:
        ``(outputs, hidden_state)`` where ``outputs`` is the per-timestep
        hidden state of the *last* layer, stacked as ``(B, hidden_dim, T, H, W)``.
    """
    T = x.size(2)
    outputs = []
    h_next = None
    for t in range(T):
        x_t = x[:, :, t]
        for layer_idx, cell in enumerate(cells):
            h_next, hidden_state[layer_idx] = step_cell(cell, x_t, hidden_state[layer_idx])
            x_t = h_next
        outputs.append(h_next)
    return torch.stack(outputs, dim=2), hidden_state


def zero_states_2tuple(
    num_layers: int,
    batch_size: int,
    hidden_dim: int,
    height: int,
    width: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Zero (h, c) for each layer."""
    return [
        (
            torch.zeros(batch_size, hidden_dim, height, width, device=device),
            torch.zeros(batch_size, hidden_dim, height, width, device=device),
        )
        for _ in range(num_layers)
    ]


def zero_states_3tuple(
    num_layers: int,
    batch_size: int,
    hidden_dim: int,
    height: int,
    width: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Zero (h, c, m) for each layer."""
    return [
        (
            torch.zeros(batch_size, hidden_dim, height, width, device=device),
            torch.zeros(batch_size, hidden_dim, height, width, device=device),
            torch.zeros(batch_size, hidden_dim, height, width, device=device),
        )
        for _ in range(num_layers)
    ]
