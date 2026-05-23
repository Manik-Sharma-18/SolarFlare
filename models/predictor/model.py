"""SolarFluxPredictor — encoder-decoder ConvLSTM with autoregressive decode.

State_dict layout matches V3.0 production checkpoints
(``encoder_conv1.*``, ``downsample1.*``, ``decoder_conv2.*``, etc.) so
existing ``best_model.pt`` files load unchanged.

Forward dataflow::

    x → preprocess → encoder (3 ConvLSTM stacks) → encoder_h3_packed (opt)
                                                  ↘ skip h1
    for t in range(t_out):
      input_frame → decoder → (temporal_attn?) → upsample
                  → (attention_gate?) → refine → output_conv → delta_scale?
                  → flux + delta = next prediction
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..stacks import ConvLSTM, SAConvLSTM
from .encoder import encoder_forward
from .decoder import decoder_step
from .head import apply_head
from .state import init_decoder_state_from_encoder
from .teacher_forcing import advance_input_frame
from .wiring import attach_dropout, attach_optional_modules


class SolarFluxPredictor(nn.Module):
    """ConvLSTM-based autoregressive solar flux predictor.

    Predicts residuals (Δ) added to the last input flux frame for training
    stability. The legacy ``downsample_input=True`` branch was removed
    (never used in production after 2026-05-21); model now runs the
    encoder/decoder at native window resolution.

    Args:
        input_channels: 1 for flux-only, 2 for flux + extreme indicator.
        output_channels: Usually 1 (flux prediction).
        t_out: Output horizon.
        channels: ``[c1, c2, c3]`` — encoder channel progression.
        kernel_size: ConvLSTM kernel size.
        use_checkpointing: Gradient checkpointing on the encoder.
        dropout_rate: MC-Dropout rate; ``0.0`` ⇒ ``nn.Identity``.
        use_sa_convlstm: Replace ConvLSTM with SA-ConvLSTM (ARCH-01).
        temporal_attention: Cross-attend decoder to encoder states (ARCH-07).
        attention_gate: Attention U-Net gate on skip (ARCH-03).
        delta_scale_init: Learnable residual multiplier (ARCH-02);
            ``0.0`` ⇒ disabled.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        t_out: int = 3,
        channels: List[int] = [16, 32, 64],
        kernel_size: int = 3,
        use_checkpointing: bool = False,
        dropout_rate: float = 0.0,
        use_sa_convlstm: bool = False,
        temporal_attention: bool = False,
        attention_gate: bool = False,
        delta_scale_init: float = 0.0,
    ):
        super().__init__()
        self.t_out = t_out
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_checkpointing = use_checkpointing
        self.dropout_rate = dropout_rate
        self.use_sa_convlstm = use_sa_convlstm
        self.use_temporal_attention = temporal_attention
        self.use_attention_gate = attention_gate

        c1, c2, c3 = channels
        RNN = SAConvLSTM if use_sa_convlstm else ConvLSTM

        self.preprocess = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_conv1 = RNN(c1, c1, kernel_size)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.encoder_conv2 = RNN(c2, c2, kernel_size)
        self.encoder_conv3 = RNN(c2, c3, kernel_size)

        self.decoder_input_conv = nn.Conv2d(input_channels, c1, kernel_size=3, padding=1)
        self.decoder_proj = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.decoder_conv2 = RNN(c2, c2, kernel_size)
        self.decoder_conv3 = RNN(c2, c3, kernel_size)

        self.upsample = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.refine_conv = RNN(c2 + c1, c1, kernel_size)
        self.output_conv = nn.Conv2d(c1, output_channels, kernel_size=1)

        attach_dropout(self, dropout_rate)
        attach_optional_modules(
            self, c1, c2, c3, temporal_attention, attention_gate, delta_scale_init
        )

    def _preprocess_sequence(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)
        x_prep = self.preprocess(x_flat)
        return x_prep.view(B, -1, T, H, W)

    def _run_encoder(self, x_prep: torch.Tensor, T_in: int):
        if self.use_checkpointing and self.training:
            return checkpoint(encoder_forward, self, x_prep, T_in, use_reentrant=False)
        return encoder_forward(self, x_prep, T_in)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive forward.

        Args:
            x: ``(B, C, T_in, H, W)``.
            teacher_forcing_ratio: Probability of feeding ground truth.
            y_true: ``(B, C, T_out, H, W)`` for teacher forcing.

        Returns:
            Predictions ``(B, output_channels, T_out, H, W)``.
        """
        B, C, T_in, H_orig, W_orig = x.size()

        x_prep = self._preprocess_sequence(x)
        h1_skip, h2_states, h3_states, encoder_h3_packed = self._run_encoder(x_prep, T_in)

        encoder_h3_states = (
            [encoder_h3_packed[:, t] for t in range(encoder_h3_packed.shape[1])]
            if encoder_h3_packed is not None
            else None
        )

        decoder_state2 = init_decoder_state_from_encoder(h2_states)
        decoder_state3 = init_decoder_state_from_encoder(h3_states)
        refine_state = None

        input_frame = x[:, :, -1]
        predictions = []
        for t in range(self.t_out):
            refined, decoder_state2, decoder_state3, refine_state = decoder_step(
                self, input_frame, decoder_state2, decoder_state3, refine_state,
                h1_skip, encoder_h3_states,
            )
            pred_flux = apply_head(self, refined, input_frame, (H_orig, W_orig))
            predictions.append(pred_flux)

            input_frame = advance_input_frame(
                input_frame, pred_flux, y_true, t, teacher_forcing_ratio,
                self.output_channels, C,
            )

        return torch.stack(predictions, dim=2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
