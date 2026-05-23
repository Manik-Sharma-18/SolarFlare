"""Self-Attention Memory (SAM) — channel-attention memory for SA-ConvLSTM.

Implements the SAM mechanism from SA-ConvLSTM (Lin et al., AAAI 2020) but
uses channel attention instead of spatial attention. At latent resolution
~110x221, spatial attention costs O((HW)^2) ≈ 24K x 24K matrices. Channel
attention operates on C ∈ [32, 128] dims, so it is trivially cheap.

MPS-safe: hand-rolled bmm + softmax (no F.scaled_dot_product_attention).
"""
import torch
import torch.nn as nn
from typing import Tuple


class SelfAttentionMemory(nn.Module):
    """Channel-attention SAM module.

    Maintains a memory state M updated each timestep via a gated mix of
    self-attention on h and cross-attention against M.

    Args:
        hidden_dim: Channels in the hidden state.
        attn_dim: Attention projection dim. Defaults to ``hidden_dim // 2``.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = None):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim // 2
        self.attn_dim = attn_dim

        self.query_h = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.key_h = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.value_h = nn.Conv2d(hidden_dim, attn_dim, 1)

        self.key_m = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.value_m = nn.Conv2d(hidden_dim, attn_dim, 1)

        self.gate = nn.Conv2d(attn_dim * 2, attn_dim, 1)
        self.output_proj = nn.Conv2d(attn_dim, hidden_dim, 1)
        self.memory_proj = nn.Conv2d(attn_dim, hidden_dim, 1)

        self.scale = attn_dim ** -0.5

    def _channel_attend(
        self,
        q_pool: torch.Tensor,
        k_pool: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Channel attention: softmax(q ⊗ k · scale) @ flatten(v).

        Args:
            q_pool: ``(B, attn_dim)`` pooled query.
            k_pool: ``(B, attn_dim)`` pooled key.
            v: ``(B, attn_dim, H, W)`` value features.

        Returns:
            ``(B, attn_dim, H, W)`` attended values.
        """
        B, _, H, W = v.shape
        q_flat = q_pool.unsqueeze(2)   # (B, attn_dim, 1)
        k_flat = k_pool.unsqueeze(1)   # (B, 1, attn_dim)
        attn = torch.softmax(q_flat * k_flat * self.scale, dim=-1)  # (B, attn_dim, attn_dim)
        v_flat = v.view(B, self.attn_dim, H * W)
        return torch.bmm(attn, v_flat).view(B, self.attn_dim, H, W)

    def forward(
        self,
        h: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine hidden state via channel attention with memory.

        Args:
            h: Current hidden state ``(B, hidden_dim, H, W)``.
            m_prev: Previous memory ``(B, hidden_dim, H, W)``.

        Returns:
            ``(h_out, m_new)`` both ``(B, hidden_dim, H, W)``.
        """
        q_h = self.query_h(h)
        k_h = self.key_h(h)
        v_h = self.value_h(h)
        q_pool = q_h.mean(dim=(-2, -1))
        k_pool = k_h.mean(dim=(-2, -1))
        z_h = self._channel_attend(q_pool, k_pool, v_h)

        k_m = self.key_m(m_prev)
        v_m = self.value_m(m_prev)
        k_m_pool = k_m.mean(dim=(-2, -1))
        z_m = self._channel_attend(q_pool, k_m_pool, v_m)

        combined = torch.cat([z_h, z_m], dim=1)
        gate_val = torch.sigmoid(self.gate(combined))
        z_fused = gate_val * z_h + (1 - gate_val) * z_m

        m_new = self.memory_proj(z_fused)
        h_out = h + self.output_proj(z_fused)
        return h_out, m_new
