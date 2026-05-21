"""Wind-flux probe heads for V5 JEPA encoder feature evaluation.

Frozen-encoder readout that maps spatially-pooled per-frame features [B, D]
to scalar spatial-mean ⟨|wind|⟩ prediction.

Two heads:
- LinearProbe: pure linear readout — defensible "feature linearly readable" claim.
- MLPProbe:    2-layer MLP — nonlinear upper bound; head can do extra work.

Target convention (probe trainer + eval are responsible):
- Train target = z-scored log10(y+1); μ/σ computed on train cubes only.
- Eval inverts to original units before reporting R²/Pearson r.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Linear readout: [B, D] → [B]."""

    def __init__(self, dim: int = 192) -> None:
        super().__init__()
        self.dim = dim
        self.head = nn.Linear(dim, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[-1] != self.dim:
            raise ValueError(f"Expected [B, {self.dim}]; got {tuple(z.shape)}")
        return self.head(z).squeeze(-1)


class MLPProbe(nn.Module):
    """2-layer MLP readout: [B, D] → [B]."""

    def __init__(self, dim: int = 192, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[-1] != self.dim:
            raise ValueError(f"Expected [B, {self.dim}]; got {tuple(z.shape)}")
        return self.net(z).squeeze(-1)


def build_probe(kind: str, dim: int = 192, hidden: int = 256, dropout: float = 0.1) -> nn.Module:
    k = kind.lower()
    if k == "linear":
        return LinearProbe(dim=dim)
    if k == "mlp":
        return MLPProbe(dim=dim, hidden=hidden, dropout=dropout)
    raise ValueError(f"Unknown probe kind: {kind!r} (expected 'linear' or 'mlp')")
