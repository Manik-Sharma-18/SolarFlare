"""
Solar Flux Predictor Model.

Encoder-decoder architecture with ConvLSTM for spatiotemporal prediction.
Uses autoregressive decoding with optional teacher forcing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

from .convlstm import ConvLSTM


class SolarFluxPredictor(nn.Module):
    """
    ConvLSTM-based autoregressive solar flux predictor.
    
    Architecture:
    1. Optional input downsampling (2x) to reduce memory
    2. Encoder: processes input sequence to extract temporal features
    3. Decoder: autoregressively generates future frames
    4. Skip connection from encoder to decoder for fine details
    
    The model predicts residuals (changes) rather than absolute values,
    which helps with training stability.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        t_out: int = 3,
        channels: List[int] = [16, 32, 64],
        kernel_size: int = 3,
        downsample_input: bool = True
    ):
        """
        Args:
            input_channels: Number of input channels (1 for flux, 2 for flux + extreme)
            output_channels: Number of output channels (typically 1 for flux prediction)
            t_out: Number of output frames to predict
            channels: Channel progression [enc1, enc2, latent]
            kernel_size: Kernel size for ConvLSTM cells
            downsample_input: Apply 2x spatial downsampling at input
        """
        super().__init__()
        self.t_out = t_out
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsample_input = downsample_input
        
        c1, c2, c3 = channels  # e.g., [16, 32, 64]
        
        # Optional input downsampling to reduce memory usage
        if downsample_input:
            self.input_down = nn.Sequential(
                nn.Conv2d(input_channels, c1, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
            self.input_up = nn.ConvTranspose2d(input_channels, input_channels, 
                                                kernel_size=4, stride=2, padding=1)
            enc_input_dim = c1
        else:
            self.input_down = None
            enc_input_dim = input_channels
        
        # Preprocessing conv (at reduced resolution if downsampling)
        self.preprocess = nn.Sequential(
            nn.Conv2d(enc_input_dim, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ENCODER
        # ConvLSTM1 at full (or reduced) resolution
        self.encoder_conv1 = ConvLSTM(c1, c1, kernel_size)
        
        # Spatial downsampling
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        
        # ConvLSTM2 and ConvLSTM3 at reduced resolution
        self.encoder_conv2 = ConvLSTM(c2, c2, kernel_size)
        self.encoder_conv3 = ConvLSTM(c2, c3, kernel_size)
        
        # DECODER
        # Map single frame to decoder input (uses all input channels)
        self.decoder_input_conv = nn.Conv2d(input_channels, c1, kernel_size=3, padding=1)
        
        # Decoder path (mirrors encoder)
        self.decoder_proj = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.decoder_conv2 = ConvLSTM(c2, c2, kernel_size)
        self.decoder_conv3 = ConvLSTM(c2, c3, kernel_size)
        
        # Upsample from latent space
        self.upsample = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        
        # Refinement with skip connection
        self.refine_conv = ConvLSTM(c2 + c1, c1, kernel_size)  # +c1 for skip
        
        # Final output head (predicts residual for output_channels only)
        if downsample_input:
            self.output_conv = nn.Sequential(
                nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, output_channels, kernel_size=1)
            )
        else:
            self.output_conv = nn.Conv2d(c1, output_channels, kernel_size=1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        teacher_forcing_ratio: float = 0.0, 
        y_true: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with autoregressive decoding.
        
        Args:
            x: Input sequence (B, C, T_in, H, W)
            teacher_forcing_ratio: Probability of using ground truth during training
            y_true: Ground truth future frames (B, C, T_out, H, W) for teacher forcing
        
        Returns:
            predictions: Predicted frames (B, C, T_out, H, W)
        """
        B, C, T_in, H_orig, W_orig = x.size()
        
        # Optional input downsampling
        if self.downsample_input:
            # Reshape for batch processing
            x_flat = x.view(B * T_in, C, H_orig, W_orig)
            x_down = self.input_down(x_flat)
            _, c1, H_down, W_down = x_down.shape
            x_prep = x_down.view(B, c1, T_in, H_down, W_down)
        else:
            x_flat = x.view(B * T_in, C, H_orig, W_orig)
            x_prep = self.preprocess(x_flat)
            _, c1, H_down, W_down = x_prep.shape
            x_prep = x_prep.view(B, c1, T_in, H_down, W_down)
        
        # Preprocess
        x_prep_flat = x_prep.view(B * T_in, -1, H_down, W_down)
        x_prep_out = self.preprocess(x_prep_flat)
        x_prep = x_prep_out.view(B, -1, T_in, H_down, W_down)
        
        # ENCODER
        # ConvLSTM1 at current resolution
        h1_seq, h1_states = self.encoder_conv1(x_prep)
        h1_skip = h1_states[0][0]  # Save for skip connection (B, c1, H_down, W_down)
        
        # Downsample spatially
        h1_down = self.downsample1(h1_seq[:, :, -1])
        H_latent, W_latent = h1_down.shape[-2:]
        h1_down = h1_down.unsqueeze(2).expand(-1, -1, T_in, -1, -1)
        
        # ConvLSTM2 and ConvLSTM3
        h2_seq, h2_states = self.encoder_conv2(h1_down)
        h3_seq, h3_states = self.encoder_conv3(h2_seq)
        
        # DECODER (Autoregressive)
        predictions = []
        input_frame = x[:, :, -1]  # Last input frame (B, C, H, W) - all channels
        
        # For residual prediction, we only use the flux channel (first channel)
        # This allows multi-channel input but single-channel output
        flux_channel_idx = 0
        
        # Initialize decoder states from encoder
        decoder_state2 = [(h2_states[0][0].clone(), h2_states[0][1].clone())]
        decoder_state3 = [(h3_states[0][0].clone(), h3_states[0][1].clone())]
        refine_state = None
        
        for t in range(self.t_out):
            # Map input frame to decoder representation (uses all channels)
            if self.downsample_input:
                dec_input = self.input_down(input_frame)
            else:
                dec_input = self.decoder_input_conv(input_frame)
            
            # Downsample to latent resolution
            dec_down = self.decoder_proj(dec_input)
            dec_down = dec_down.unsqueeze(2)  # (B, c2, 1, H_latent, W_latent)
            
            # Decoder ConvLSTMs
            dec_h2, decoder_state2 = self.decoder_conv2(dec_down, decoder_state2)
            dec_h3, decoder_state3 = self.decoder_conv3(dec_h2, decoder_state3)
            
            # Upsample
            dec_up = self.upsample(dec_h3[:, :, 0])  # (B, c2, H_down, W_down)
            
            # Ensure dimensions match (handles odd spatial sizes from stride-2 ops)
            if dec_up.shape[2:] != h1_skip.shape[2:]:
                dec_up = F.interpolate(dec_up, size=h1_skip.shape[2:], mode='nearest')
            
            # Skip connection and refinement
            dec_concat = torch.cat([dec_up, h1_skip], dim=1)
            dec_concat = dec_concat.unsqueeze(2)
            
            refined, refine_state = self.refine_conv(dec_concat, refine_state)
            
            # Output residual prediction (output_channels, typically 1)
            delta = self.output_conv(refined[:, :, 0])
            
            # Ensure delta matches original input dimensions
            if delta.shape[2:] != (H_orig, W_orig):
                delta = F.interpolate(delta, size=(H_orig, W_orig), mode='nearest')
            
            # Predict frame as input flux + residual
            # Only use flux channel for residual computation
            input_flux = input_frame[:, flux_channel_idx:flux_channel_idx+self.output_channels]
            pred_flux = input_flux + delta
            predictions.append(pred_flux)
            
            # Teacher forcing: use ground truth with some probability
            use_teacher = (
                teacher_forcing_ratio > 0 and 
                y_true is not None and 
                np.random.rand() < teacher_forcing_ratio
            )
            
            if use_teacher:
                # Use ground truth flux as next input
                # For multi-channel: concatenate predicted flux with other channels from input
                if C > self.output_channels:
                    # Keep extreme indicator channel from input, update flux
                    next_flux = y_true[:, :self.output_channels, t]
                    other_channels = input_frame[:, self.output_channels:]
                    input_frame = torch.cat([next_flux, other_channels], dim=1)
                else:
                    input_frame = y_true[:, :, t]
            else:
                # Use prediction as next input
                if C > self.output_channels:
                    # Recompute extreme indicator for predicted flux
                    other_channels = input_frame[:, self.output_channels:]
                    input_frame = torch.cat([pred_flux, other_channels], dim=1)
                else:
                    input_frame = pred_flux
        
        # Stack predictions: (B, C, T_out, H, W)
        predictions = torch.stack(predictions, dim=2)
        
        return predictions
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

