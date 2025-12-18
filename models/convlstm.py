"""
ConvLSTM Cell and Layer implementations.

ConvLSTM extends LSTM to handle spatial data by replacing matrix multiplications
with convolutions. This preserves spatial structure while modeling temporal dynamics.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    
    Combines CNN (spatial feature extraction) with LSTM (temporal memory).
    Uses 2D convolutions instead of fully connected layers.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        """
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden state channels
            kernel_size: Size of convolutional kernel (use odd number for same padding)
            bias: Whether to include bias terms
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Single convolution computes all four gates at once
        # Input: concatenation of input and previous hidden state
        # Output: 4 * hidden_dim (for i, f, g, o gates)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        self._init_forget_bias()
    
    def _init_forget_bias(self):
        """Initialize forget gate bias to 1.0 for better gradient flow."""
        with torch.no_grad():
            if self.conv.bias is not None:
                # Gates order: input, forget, cell, output
                # Set forget gate bias to 1.0
                self.conv.bias[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: torch.Tensor, 
        c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one timestep.
        
        Args:
            x: Input tensor (B, input_dim, H, W)
            h_prev: Previous hidden state (B, hidden_dim, H, W)
            c_prev: Previous cell state (B, hidden_dim, H, W)
        
        Returns:
            h_next: New hidden state
            c_next: New cell state
        """
        # Concatenate input and previous hidden state along channel dimension
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute all gates in one convolution
        gates = self.conv(combined)
        
        # Split into individual gates
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate: what new info to store
        f = torch.sigmoid(f)  # Forget gate: what old info to discard
        g = torch.tanh(g)     # Cell gate: new candidate values
        o = torch.sigmoid(o)  # Output gate: what to output
        
        # Update cell state: forget old + add new
        c_next = f * c_prev + i * g
        
        # Compute output hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM that processes a sequence of spatial inputs.
    
    Stacks multiple ConvLSTMCells for deeper temporal modeling.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        kernel_size: int = 3, 
        num_layers: int = 1
    ):
        """
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels (same for all layers)
            kernel_size: Convolution kernel size
            num_layers: Number of stacked ConvLSTM layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create cell for each layer
        cells = []
        for layer_idx in range(num_layers):
            # First layer takes input_dim, rest take hidden_dim
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        
        self.cell_list = nn.ModuleList(cells)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process a sequence through all layers.
        
        Args:
            x: Input sequence (B, C, T, H, W)
            hidden_state: Optional initial states, list of (h, c) tuples per layer
        
        Returns:
            outputs: Output sequence from final layer (B, hidden_dim, T, H, W)
            last_state: Final (h, c) states for each layer
        """
        B, _, T, H, W = x.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(B, H, W, x.device)
        
        outputs = []
        
        # Process each timestep
        for t in range(T):
            x_t = x[:, :, t]  # (B, C, H, W)
            
            # Pass through all layers
            for layer_idx, cell in enumerate(self.cell_list):
                h_prev, c_prev = hidden_state[layer_idx]
                h_next, c_next = cell(x_t, h_prev, c_prev)
                hidden_state[layer_idx] = (h_next, c_next)
                x_t = h_next  # Output of this layer is input to next
            
            outputs.append(h_next)
        
        # Stack outputs: (B, hidden_dim, T, H, W)
        outputs = torch.stack(outputs, dim=2)
        
        return outputs, hidden_state
    
    def _init_hidden(
        self, 
        batch_size: int, 
        height: int, 
        width: int, 
        device: torch.device
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states to zeros."""
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            )
            for _ in range(self.num_layers)
        ]

