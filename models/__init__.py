"""Model architectures for Solar Flare Prediction."""
from .convlstm import ConvLSTMCell, ConvLSTM
from .sa_convlstm import SelfAttentionMemory, SAConvLSTMCell, SAConvLSTM
from .attention import TemporalAttention, AttentionGate
from .predictor import SolarFluxPredictor
from .uncertainty import (
    predict_with_uncertainty,
    predict_with_confidence_intervals,
    uncertainty_weighted_loss
)
