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
        residual: bool = False,
        norm_type: str = "batch",
        enable_classifier_head: bool = False,
        kernel_dilations=None,
        recurrent_init: str = "default",
        depthwise_separable: bool = False,
        ss_per_step: bool = False,
    ):
        super().__init__()
        self.t_out = t_out
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.norm_type = norm_type
        self.enable_classifier_head = enable_classifier_head
        # Per-step scheduled sampling: ramp TF prob DOWN across the decode
        # rollout so later steps train autoregressively even early in training.
        # Mirrors deployment (each committed frame feeds the next) and curbs
        # long-horizon error compounding (S48 diverged; S47 fast-decay didn't).
        self.ss_per_step = ss_per_step

        dilations = list(kernel_dilations) if kernel_dilations is not None else [1] * num_layers
        assert len(dilations) == num_layers, (
            f"kernel_dilations length {len(dilations)} must == num_layers {num_layers}")
        self.encoder_cells = self._make_stack(input_channels, hidden_dim, kernel_size, dilations, recurrent_init, depthwise_separable)
        self.decoder_cells = self._make_stack(output_channels, hidden_dim, kernel_size, dilations, recurrent_init, depthwise_separable)
        self.enc_norms = nn.ModuleList(self._make_norm(hidden_dim) for _ in range(num_layers))
        self.dec_norms = nn.ModuleList(self._make_norm(hidden_dim) for _ in range(num_layers))
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)
        # Per-pixel binary "is this pixel extreme at t+k?" classifier (logits).
        # Placed on the last decoder activation `h`, NOT on the predicted frame,
        # so the regression and classification gradients don't fight on shared
        # output. Used by the dual_head loss; eval CSI reads its sigmoid directly.
        self.ext_head = (nn.Conv2d(hidden_dim, 1, kernel_size=1)
                         if enable_classifier_head else None)

    def _make_norm(self, hidden: int) -> nn.Module:
        """Norm layer. ``group`` (GroupNorm) has no running stats, so it is
        safe to reuse across the unrolled decode loop — ``batch`` (BatchNorm)
        updates running stats in-place and crashes the multi-step backward
        (autograd inplace-version error). ``batch`` kept as the default for
        checkpoint compatibility with the S0–S4 runs."""
        if self.norm_type == "group":
            groups = 8 if hidden % 8 == 0 else 1
            return nn.GroupNorm(groups, hidden)
        return nn.BatchNorm2d(hidden)

    @staticmethod
    def _make_stack(in_ch: int, hidden: int, k: int, dilations,
                    recurrent_init: str = "default",
                    depthwise_separable: bool = False) -> nn.ModuleList:
        """Layer 0 consumes ``in_ch``; deeper layers consume ``hidden``.
        Per-layer dilation from ``dilations``. ``recurrent_init`` applied to
        every cell's recurrent slice. ``depthwise_separable`` swaps the
        fused gate conv for a (depthwise + pointwise) factorization."""
        return nn.ModuleList(
            ConvLSTMCell(in_ch if i == 0 else hidden, hidden, k,
                         dilation=d, recurrent_init=recurrent_init,
                         depthwise_separable=depthwise_separable)
            for i, d in enumerate(dilations)
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
            teacher_forcing_ratio: per-step prob of feeding ``y_true`` during
                training (0.0 ⇒ pure autoregressive).
            y_true: ``(B, C, t_out, H, W)`` ground truth, used only for TF.

        Returns:
            ``(B, output_channels, t_out, H, W)``.
        """
        T_in = x.shape[2]
        states = self._init_states(x[:, :, 0])
        for t in range(T_in):
            _, states = self._step(self.encoder_cells, self.enc_norms, x[:, :, t], states)

        frame = x[:, : self.output_channels, -1]  # last flux frame
        preds = []
        ext_logits_list = [] if self.enable_classifier_head else None
        for t in range(self.t_out):
            h, states = self._step(self.decoder_cells, self.dec_norms, frame, states)
            # Residual: predict Δ added to the previous frame, so the model
            # falls back to persistence (Δ→0 = copy last frame) instead of
            # collapsing to ~0. Else predict the absolute frame from scratch.
            pred = frame + self.head(h) if self.residual else self.head(h)
            preds.append(pred)
            if self.enable_classifier_head:
                ext_logits_list.append(self.ext_head(h))
            # Teacher forcing: during training, feed the ground-truth frame
            # as the next step's input (and residual base) with probability
            # teacher_forcing_ratio — anchors the rollout to truth and curbs
            # autoregressive error compounding.
            # Per-step ramp: step 0 keeps the full ratio, the last step → 0,
            # so the tail of every rollout is autoregressive regardless of
            # epoch. Plain mode uses one ratio for all steps.
            step_tf = teacher_forcing_ratio
            if self.ss_per_step and self.t_out > 1:
                step_tf *= 1.0 - t / (self.t_out - 1)
            if (self.training and y_true is not None
                    and step_tf > 0.0
                    and torch.rand(()) < step_tf):
                frame = y_true[:, : self.output_channels, t]
            else:
                frame = pred

        pred_stack = torch.stack(preds, dim=2)
        if self.enable_classifier_head:
            # (B, 1, t_out, H, W) logits — shape matches pred for downstream
            # per-pixel BCEWithLogitsLoss against ``|target|>extreme_threshold``.
            return pred_stack, torch.stack(ext_logits_list, dim=2)
        return pred_stack

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
