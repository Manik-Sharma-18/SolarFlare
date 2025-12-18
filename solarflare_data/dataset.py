"""
Dataset class for solar flux prediction.

Handles sliding window sampling and data augmentation.
Supports dual-channel mode for background + extreme event detection.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class SolarFluxDataset(Dataset):
    """
    PyTorch Dataset for solar flux prediction with sliding windows.
    
    Takes pre-normalized flux cubes and creates (input, output) pairs
    using sliding windows. Supports basic augmentation (flips).
    
    Dual-channel mode:
        - Channel 1: Asinh-normalized flux values
        - Channel 2: Extreme event indicator (|flux| > threshold)
    """
    
    def __init__(
        self,
        samples: List[Tuple[int, int]],
        datasets: List[np.ndarray],
        t_in: int = 8,
        t_out: int = 3,
        augment: bool = True,
        dual_channel: bool = False,
        extreme_threshold: Optional[float] = None
    ):
        """
        Args:
            samples: List of (dataset_id, start_idx) tuples identifying each sample
            datasets: List of normalized flux cubes, each (T, H, W)
            t_in: Number of input timesteps
            t_out: Number of output timesteps
            augment: Whether to apply random augmentations
            dual_channel: If True, output 2 channels (flux + extreme indicator)
            extreme_threshold: Threshold for extreme events (in normalized space)
        """
        self.samples = samples
        self.datasets = datasets
        self.t_in = t_in
        self.t_out = t_out
        self.augment = augment
        self.dual_channel = dual_channel
        self.extreme_threshold = extreme_threshold
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Get a single sample.
        
        Returns:
            X_in: Input tensor (C, T_in, H, W) where C=1 or 2
            Y_out: Output tensor (C, T_out, H, W) where C=1 or 2
            info: Tuple of (dataset_id, start_idx) for reference
        """
        dataset_id, start_idx = self.samples[idx]
        data = self.datasets[dataset_id]
        
        # Extract input and output sequences
        X_in = data[start_idx:start_idx + self.t_in]  # (T_in, H, W)
        Y_out = data[start_idx + self.t_in:start_idx + self.t_in + self.t_out]  # (T_out, H, W)
        
        # Apply augmentations (consistent across input/output)
        if self.augment:
            if np.random.rand() > 0.5:
                # Horizontal flip
                X_in = np.flip(X_in, axis=2).copy()
                Y_out = np.flip(Y_out, axis=2).copy()
            if np.random.rand() > 0.5:
                # Vertical flip
                X_in = np.flip(X_in, axis=1).copy()
                Y_out = np.flip(Y_out, axis=1).copy()
        
        if self.dual_channel and self.extreme_threshold is not None:
            # Create dual-channel output:
            # Channel 1: Flux values (already normalized)
            # Channel 2: Extreme event indicator (soft threshold)
            X_extreme = self._compute_extreme_channel(X_in)
            Y_extreme = self._compute_extreme_channel(Y_out)
            
            # Stack channels: (2, T, H, W)
            X_in = np.stack([X_in, X_extreme], axis=0)
            Y_out = np.stack([Y_out, Y_extreme], axis=0)
            
            X_in = torch.from_numpy(X_in.copy()).float()
            Y_out = torch.from_numpy(Y_out.copy()).float()
        else:
            # Single channel: (1, T, H, W)
            X_in = torch.from_numpy(X_in.copy()).float().unsqueeze(0)
            Y_out = torch.from_numpy(Y_out.copy()).float().unsqueeze(0)
        
        return X_in, Y_out, (dataset_id, start_idx)
    
    def _compute_extreme_channel(self, flux: np.ndarray) -> np.ndarray:
        """
        Compute extreme event indicator channel.
        
        Uses a soft sigmoid-like activation to highlight regions with
        extreme flux values, providing gradient information for training.
        
        Args:
            flux: Normalized flux values (T, H, W)
        
        Returns:
            Extreme indicator (T, H, W) in range [0, 1]
        """
        # Soft threshold: sigmoid around extreme_threshold
        # This provides smooth gradients instead of hard binary mask
        abs_flux = np.abs(flux)
        # Scale so values at threshold map to ~0.5
        scaled = (abs_flux - self.extreme_threshold) / (self.extreme_threshold * 0.5 + 1e-6)
        # Sigmoid activation
        extreme = 1.0 / (1.0 + np.exp(-scaled * 2))
        return extreme.astype(np.float32)

