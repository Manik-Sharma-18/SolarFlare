"""Model architectures for Solar Flare Prediction.

See submodules: cells/, stacks/, attention/, predictor/, uncertainty/.
"""
from .cells import ConvLSTMCell, SelfAttentionMemory, SAConvLSTMCell
from .stacks import ConvLSTM, SAConvLSTM
from .attention import TemporalAttention, AttentionGate
from .predictor import SolarFluxPredictor
from .simple_convlstm import SimpleConvLSTM
from .factory import build_forecaster
from .uncertainty import (
    predict_with_uncertainty,
    predict_with_confidence_intervals,
    uncertainty_weighted_loss,
)

__all__ = [
    "ConvLSTMCell",
    "ConvLSTM",
    "SelfAttentionMemory",
    "SAConvLSTMCell",
    "SAConvLSTM",
    "TemporalAttention",
    "AttentionGate",
    "SolarFluxPredictor",
    "SimpleConvLSTM",
    "build_forecaster",
    "predict_with_uncertainty",
    "predict_with_confidence_intervals",
    "uncertainty_weighted_loss",
]
