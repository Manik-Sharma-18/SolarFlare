"""SA-ConvLSTM: Self-Attention ConvLSTM with channel attention memory.

Implements the Self-Attention Memory (SAM) mechanism from SA-ConvLSTM
(Lin et al., AAAI 2020) using channel attention to avoid O(H*W * H*W)
spatial attention cost. Uses manual bmm+softmax for MPS compatibility.

Classes:
    SelfAttentionMemory: Channel-attention memory module.
    SAConvLSTMCell: ConvLSTM cell augmented with SAM (composition pattern).
    SAConvLSTM: Multi-step wrapper processing sequences through SA cells.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .convlstm import ConvLSTMCell


class SelfAttentionMemory(nn.Module):
    """Channel-attention Self-Attention Memory module.

    Uses channel attention (not spatial) to avoid O(H*W * H*W) cost.
    At latent resolution 110x221, spatial attention would produce 24K x 24K
    matrices. Channel attention operates on C dimensions (32-128), which
    is trivially cheap.

    The module maintains a memory state M that is updated at each timestep
    via gated combination of self-attention on h and cross-attention with M.

    Args:
        hidden_dim: Number of channels in the hidden state.
        attn_dim: Attention projection dimension. Defaults to hidden_dim // 2.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = None):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim // 2
        self.attn_dim = attn_dim

        # Q/K/V projections for hidden state self-attention (1x1 Conv2d)
        self.query_h = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.key_h = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.value_h = nn.Conv2d(hidden_dim, attn_dim, 1)

        # K/V projections for memory cross-attention
        self.key_m = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.value_m = nn.Conv2d(hidden_dim, attn_dim, 1)

        # Gated combination of z_h and z_m
        self.gate = nn.Conv2d(attn_dim * 2, attn_dim, 1)

        # Output projection back to hidden_dim
        self.output_proj = nn.Conv2d(attn_dim, hidden_dim, 1)

        # Memory projection back to hidden_dim
        self.memory_proj = nn.Conv2d(attn_dim, hidden_dim, 1)

        self.scale = attn_dim ** -0.5

    def forward(
        self, h: torch.Tensor, m_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine hidden state using channel attention with memory.

        Args:
            h: Current hidden state from ConvLSTM (B, hidden_dim, H, W).
            m_prev: Previous memory state (B, hidden_dim, H, W).

        Returns:
            h_out: Refined hidden state (B, hidden_dim, H, W).
            m_new: Updated memory state (B, hidden_dim, H, W).
        """
        B, _, H, W = h.shape

        # Project h for self-attention
        q_h = self.query_h(h)   # (B, attn_dim, H, W)
        k_h = self.key_h(h)     # (B, attn_dim, H, W)
        v_h = self.value_h(h)   # (B, attn_dim, H, W)

        # Global average pool for channel descriptors
        q_pool = q_h.mean(dim=(-2, -1))  # (B, attn_dim)
        k_pool = k_h.mean(dim=(-2, -1))  # (B, attn_dim)

        # Channel attention weights via outer product
        q_flat = q_pool.unsqueeze(2)   # (B, attn_dim, 1)
        k_flat = k_pool.unsqueeze(1)   # (B, 1, attn_dim)
        attn_weights_h = torch.softmax(
            q_flat * k_flat * self.scale, dim=-1
        )  # (B, attn_dim, attn_dim)

        # Apply attention to value features
        v_flat = v_h.view(B, self.attn_dim, H * W)  # (B, attn_dim, H*W)
        z_h = torch.bmm(attn_weights_h, v_flat).view(
            B, self.attn_dim, H, W
        )  # (B, attn_dim, H, W)

        # Cross-attention with memory
        k_m = self.key_m(m_prev)
        v_m = self.value_m(m_prev)
        k_m_pool = k_m.mean(dim=(-2, -1))
        k_m_flat = k_m_pool.unsqueeze(1)  # (B, 1, attn_dim)
        attn_weights_m = torch.softmax(
            q_flat * k_m_flat * self.scale, dim=-1
        )  # (B, attn_dim, attn_dim)
        v_m_flat = v_m.view(B, self.attn_dim, H * W)
        z_m = torch.bmm(attn_weights_m, v_m_flat).view(
            B, self.attn_dim, H, W
        )  # (B, attn_dim, H, W)

        # Gate-controlled combination
        combined = torch.cat([z_h, z_m], dim=1)  # (B, 2*attn_dim, H, W)
        gate_val = torch.sigmoid(self.gate(combined))  # (B, attn_dim, H, W)
        z_fused = gate_val * z_h + (1 - gate_val) * z_m

        # Memory update (project back to hidden_dim)
        m_new = self.memory_proj(z_fused)

        # Output: residual connection
        h_out = h + self.output_proj(z_fused)

        return h_out, m_new


class SAConvLSTMCell(nn.Module):
    """SA-ConvLSTM cell: ConvLSTM + Self-Attention Memory.

    Wraps existing ConvLSTMCell via composition (not inheritance).
    Returns (h, c, m) 3-tuple instead of (h, c) 2-tuple.

    Args:
        input_dim: Number of input channels.
        hidden_dim: Number of hidden state channels.
        kernel_size: Size of convolutional kernel.
        attn_dim: SAM attention dimension. Defaults to hidden_dim // 2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        attn_dim: int = None,
    ):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim // 2
        self.convlstm_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.sam = SelfAttentionMemory(hidden_dim, attn_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one timestep through ConvLSTM then SAM.

        Args:
            x: Input tensor (B, input_dim, H, W).
            h_prev: Previous hidden state (B, hidden_dim, H, W).
            c_prev: Previous cell state (B, hidden_dim, H, W).
            m_prev: Previous memory state (B, hidden_dim, H, W).

        Returns:
            h_out: Refined hidden state (B, hidden_dim, H, W).
            c: Cell state from ConvLSTM (B, hidden_dim, H, W).
            m_new: Updated memory (B, hidden_dim, H, W).
        """
        h, c = self.convlstm_cell(x, h_prev, c_prev)
        h_out, m_new = self.sam(h, m_prev)
        return h_out, c, m_new


class SAConvLSTM(nn.Module):
    """Multi-step SA-ConvLSTM that processes a sequence of spatial inputs.

    Same API as ConvLSTM but hidden_state is List[Tuple[h, c, m]] (3-tuples)
    instead of List[Tuple[h, c]] (2-tuples).

    Args:
        input_dim: Number of input channels.
        hidden_dim: Number of hidden channels (same for all layers).
        kernel_size: Convolution kernel size.
        num_layers: Number of stacked SA-ConvLSTM layers.
        attn_dim: SAM attention dimension. Defaults to hidden_dim // 2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        attn_dim: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for layer_idx in range(num_layers):
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(
                SAConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, attn_dim)
            )
        self.cell_list = nn.ModuleList(cells)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """Process a sequence through all layers.

        Args:
            x: Input sequence (B, C, T, H, W).
            hidden_state: Optional initial states, list of (h, c, m)
                tuples per layer.

        Returns:
            outputs: Output sequence from final layer (B, hidden_dim, T, H, W).
            hidden_state: Final (h, c, m) states for each layer.
        """
        B, _, T, H, W = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(B, H, W, x.device)

        outputs = []
        for t in range(T):
            x_t = x[:, :, t]  # (B, C, H, W)

            for layer_idx, cell in enumerate(self.cell_list):
                h_prev, c_prev, m_prev = hidden_state[layer_idx]
                h_next, c_next, m_next = cell(x_t, h_prev, c_prev, m_prev)
                hidden_state[layer_idx] = (h_next, c_next, m_next)
                x_t = h_next

            outputs.append(h_next)

        outputs = torch.stack(outputs, dim=2)
        return outputs, hidden_state

    def _init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Initialize hidden states to zeros (h, c, m 3-tuples)."""
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            )
            for _ in range(self.num_layers)
        ]
