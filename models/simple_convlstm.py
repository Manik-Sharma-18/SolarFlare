"""SimpleConvLSTM — minimal 2-layer ConvLSTM seq2seq forecaster.

Canonical nowcasting recipe (Shi et al. 2015 encoder-forecaster; Keras
next-frame example): a small stack of ConvLSTM layers at flat hidden
width, BatchNorm between layers, a 1x1 Conv2d head. No attention, no
residual delta-scale, no MC-dropout, no downsampling — the deliberately
simple baseline to weigh against the 6-layer ``SolarFluxPredictor``.

API-compatible with ``SolarFluxPredictor.forward(x, teacher_forcing_ratio,
y_true) -> (B, out_ch, T_out, H, W)`` so it drops into the existing
trainer unchanged. Decoding is pure autoregressive: ``teacher_forcing_ratio``
and ``y_true`` are accepted for signature compatibility but ignored.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .cells import ConvLSTMCell

State = List[Tuple[torch.Tensor, torch.Tensor]]


class SimpleConvLSTM(nn.Module):
    """Two-layer ConvLSTM encoder-forecaster at native resolution.

    Args:
        input_channels: Channels of input frames (1 = flux only).
        output_channels: Channels to predict (usually 1).
        t_out: Output horizon (frames to forecast).
        hidden_dim: Flat hidden width for every ConvLSTM layer.
        kernel_size: ConvLSTM conv kernel.
        num_layers: Stacked ConvLSTM layers in encoder and decoder.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        t_out: int = 4,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
    ):
        super().__init__()
        self.t_out = t_out
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_cells = self._make_stack(input_channels, hidden_dim, kernel_size, num_layers)
        self.decoder_cells = self._make_stack(output_channels, hidden_dim, kernel_size, num_layers)
        self.enc_norms = nn.ModuleList(nn.BatchNorm2d(hidden_dim) for _ in range(num_layers))
        self.dec_norms = nn.ModuleList(nn.BatchNorm2d(hidden_dim) for _ in range(num_layers))
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    @staticmethod
    def _make_stack(in_ch: int, hidden: int, k: int, n: int) -> nn.ModuleList:
        """Layer 0 consumes ``in_ch``; deeper layers consume ``hidden``."""
        return nn.ModuleList(
            ConvLSTMCell(in_ch if i == 0 else hidden, hidden, k) for i in range(n)
        )

    def _init_states(self, ref: torch.Tensor) -> State:
        """Zero (h, c) per layer, shaped from a reference frame ``(B, _, H, W)``."""
        B, _, H, W = ref.shape
        z = lambda: torch.zeros(B, self.hidden_dim, H, W, device=ref.device, dtype=ref.dtype)
        return [(z(), z()) for _ in range(self.num_layers)]

    def _step(
        self,
        cells: nn.ModuleList,
        norms: nn.ModuleList,
        x: torch.Tensor,
        states: State,
    ) -> Tuple[torch.Tensor, State]:
        """One timestep up the stack; returns top hidden + new states."""
        new_states: State = []
        inp = x
        for i, cell in enumerate(cells):
            h, c = cell(inp, states[i][0], states[i][1])
            h = norms[i](h)
            new_states.append((h, c))
            inp = h
        return inp, new_states

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode ``T_in`` frames, then autoregressively forecast ``t_out``.

        Args:
            x: ``(B, C, T_in, H, W)``.
            teacher_forcing_ratio, y_true: accepted, ignored (pure AR decode).

        Returns:
            ``(B, output_channels, t_out, H, W)``.
        """
        T_in = x.shape[2]
        states = self._init_states(x[:, :, 0])
        for t in range(T_in):
            _, states = self._step(self.encoder_cells, self.enc_norms, x[:, :, t], states)

        frame = x[:, : self.output_channels, -1]  # last flux frame
        preds = []
        for _ in range(self.t_out):
            h, states = self._step(self.decoder_cells, self.dec_norms, frame, states)
            frame = self.head(h)
            preds.append(frame)

        return torch.stack(preds, dim=2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
