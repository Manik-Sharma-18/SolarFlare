"""Decoder state plumbing.

Both ConvLSTM and SA-ConvLSTM stacks have the same outer state shape
``list[layer_state]``; only the layer-state arity differs (``(h, c)`` vs
``(h, c, m)``). The helpers here clone encoder states into decoder states
so the AR loop never branches on ``use_sa_convlstm``.
"""
from typing import List, Tuple

import torch


LayerState2 = Tuple[torch.Tensor, torch.Tensor]
LayerState3 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def clone_layer_state(state: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Element-wise ``.clone()`` of a layer state tuple.

    The encoder returns its own state lists; the decoder mutates state in
    place during the AR loop, so we clone to avoid storage sharing.
    """
    return tuple(t.clone() for t in state)


def init_decoder_state_from_encoder(
    encoder_state: List[Tuple[torch.Tensor, ...]],
) -> List[Tuple[torch.Tensor, ...]]:
    """Clone an encoder state list into a single-layer decoder state list.

    Decoder ConvLSTMs are single-layer (``num_layers=1`` by construction in
    :class:`SolarFluxPredictor`), so we use the encoder's *final* layer state
    as the decoder seed.
    """
    return [clone_layer_state(encoder_state[-1])]
