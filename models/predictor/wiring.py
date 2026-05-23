"""Module-construction helpers for :class:`SolarFluxPredictor`.

Pulled into a separate file purely to keep ``model.py`` under the
200-line cap. Each helper attaches modules onto the model instance using
the flat attribute names expected by V3.0 checkpoints.
"""
import torch
import torch.nn as nn

from ..attention import TemporalAttention, AttentionGate


def attach_dropout(model: nn.Module, dropout_rate: float) -> None:
    """Three MC-Dropout slots; ``nn.Identity`` when disabled (no overhead).

    ``nn.Dropout`` (not ``Dropout2d``) — 5D inputs trigger a deprecation
    warning on the 2D variant.
    """
    if dropout_rate > 0.0:
        model.dropout_enc1 = nn.Dropout(dropout_rate)
        model.dropout_enc2 = nn.Dropout(dropout_rate)
        model.dropout_dec = nn.Dropout(dropout_rate)
    else:
        model.dropout_enc1 = nn.Identity()
        model.dropout_enc2 = nn.Identity()
        model.dropout_dec = nn.Identity()


def attach_optional_modules(
    model: nn.Module,
    c1: int,
    c2: int,
    c3: int,
    temporal_attention: bool,
    attention_gate: bool,
    delta_scale_init: float,
) -> None:
    """Attach the v3.0 architecture flags (each is no-op when disabled)."""
    if temporal_attention:
        model.temporal_attn = TemporalAttention(channels=c3, t_max=20)
    if attention_gate:
        model.attn_gate = AttentionGate(encoder_channels=c1, decoder_channels=c2)

    if delta_scale_init != 0.0:
        model.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init)))
    else:
        model.delta_scale = None
