"""ConvLSTM cell — single-timestep recurrent unit with spatial conv gates."""
import torch
import torch.nn as nn
from typing import Tuple


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell.

    Replaces LSTM's matrix multiplications with 2D convolutions to preserve
    spatial structure. All four gates (i, f, g, o) are produced by a single
    fused conv over ``cat(x, h_prev)``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int,
                 dilation: int = 1, bias: bool = True,
                 recurrent_init: str = "default",
                 depthwise_separable: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size // 2)
        self.depthwise_separable = depthwise_separable
        ch = input_dim + hidden_dim

        if depthwise_separable:
            self.depthwise = nn.Conv2d(
                in_channels=ch, out_channels=ch,
                kernel_size=kernel_size, padding=self.padding,
                dilation=dilation, groups=ch, bias=False,
            )
            self.pointwise = nn.Conv2d(
                in_channels=ch, out_channels=4 * hidden_dim,
                kernel_size=1, bias=bias,
            )
            self.conv = None
        else:
            self.conv = nn.Conv2d(
                in_channels=ch, out_channels=4 * hidden_dim,
                kernel_size=kernel_size, padding=self.padding,
                dilation=dilation, bias=bias,
            )
            self.depthwise = self.pointwise = None

        self._init_forget_bias()
        if recurrent_init == "orthogonal":
            self._init_recurrent_orthogonal()

    def _init_forget_bias(self):
        """Set forget-gate bias to 1.0 for better gradient flow.
        Bias lives on the output-producing conv (pointwise for DW path,
        fused gate conv otherwise)."""
        out_conv = self.pointwise if self.depthwise_separable else self.conv
        with torch.no_grad():
            if out_conv.bias is not None:
                out_conv.bias[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def _init_recurrent_orthogonal(self):
        """Orthogonal init on the recurrent slice. For the fused path this
        is `conv.weight[:, input_dim:, :, :]`. For the depthwise path,
        recurrent channel mixing happens at the pointwise layer, so
        orthogonalize `pointwise.weight[:, input_dim:, :, :]`."""
        out_conv = self.pointwise if self.depthwise_separable else self.conv
        with torch.no_grad():
            W = out_conv.weight  # (4H, in+H, k, k) or (4H, in+H, 1, 1)
            recur = W[:, self.input_dim:, :, :].contiguous()
            shape = recur.shape
            flat = recur.view(shape[0], -1)
            nn.init.orthogonal_(flat)
            W[:, self.input_dim:, :, :] = flat.view(shape)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One ConvLSTM step.

        Args:
            x: ``(B, input_dim, H, W)`` input frame.
            h_prev, c_prev: previous hidden / cell state, ``(B, hidden_dim, H, W)``.

        Returns:
            ``(h_next, c_next)``.
        """
        combined = torch.cat([x, h_prev], dim=1)
        if self.depthwise_separable:
            gates = self.pointwise(self.depthwise(combined))
        else:
            gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
