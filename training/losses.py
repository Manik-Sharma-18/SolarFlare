"""
Composite loss functions for solar flare prediction.

Combines multiple loss components:
- L1 (MAE): Basic reconstruction loss
- MS-SSIM: Multi-scale structural similarity for sharper predictions
- Weighted MAE: Higher weight for extreme flux values (solar flares)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,  # For normalized data in [-1, 1]
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute SSIM between prediction and target.
    
    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian
        data_range: Dynamic range of data
        size_average: If True, return mean SSIM
    
    Returns:
        SSIM value(s)
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    window = gaussian_kernel(window_size, sigma, pred.device)
    window = window.expand(pred.size(1), 1, window_size, window_size)
    
    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1)) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    return ssim_map


def ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Multi-scale SSIM.
    
    Computes SSIM at multiple scales by progressively downsampling.
    
    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian
        data_range: Dynamic range of data
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    
    Returns:
        MS-SSIM value
    """
    if weights is None:
        # Default weights from original MS-SSIM paper
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=pred.device)
    
    levels = len(weights)
    ms_ssim_val = torch.ones(1, device=pred.device)
    
    for i in range(levels):
        ssim_val = ssim(pred, target, window_size, sigma, data_range, size_average=True)
        
        if i < levels - 1:
            # Downsample for next scale
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)
            
            # Weight only contrast/structure for intermediate scales
            ms_ssim_val = ms_ssim_val * (ssim_val ** weights[i])
        else:
            # Final scale includes luminance
            ms_ssim_val = ms_ssim_val * (ssim_val ** weights[i])
    
    return ms_ssim_val


class WeightedMAELoss(nn.Module):
    """
    MAE loss with higher weight for extreme values.
    
    Helps the model focus on predicting high-magnitude solar flare regions.
    """
    
    def __init__(self, base_weight: float = 1.0, extreme_weight: float = 2.0):
        """
        Args:
            base_weight: Weight for normal regions
            extreme_weight: Additional weight multiplier for extreme values
        """
        super().__init__()
        self.base_weight = base_weight
        self.extreme_weight = extreme_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MAE.
        
        Weight increases with absolute target value, emphasizing extreme regions.
        """
        abs_error = torch.abs(pred - target)
        
        # Weight based on target magnitude: higher flux = higher weight
        # Use smooth weighting based on absolute target value
        abs_target = torch.abs(target)
        max_target = abs_target.max() + 1e-6
        normalized_magnitude = abs_target / max_target
        
        # Weight: base_weight + extreme_weight * normalized_magnitude
        weights = self.base_weight + self.extreme_weight * normalized_magnitude
        
        weighted_error = weights * abs_error
        return weighted_error.mean()


class CompositeLoss(nn.Module):
    """
    Composite loss combining L1, MS-SSIM, and weighted extreme loss.
    
    Total loss = l1_weight * L1 + ssim_weight * (1 - MS_SSIM) + extreme_weight * WeightedMAE
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        extreme_weight: float = 1.0,
        use_ms_ssim: bool = True,
        ssim_data_range: float = 2.0
    ):
        """
        Args:
            l1_weight: Weight for L1 (MAE) loss
            ssim_weight: Weight for SSIM loss (as 1 - SSIM)
            extreme_weight: Weight for extreme value loss
            use_ms_ssim: Use multi-scale SSIM (True) or single-scale (False)
            ssim_data_range: Data range for SSIM computation
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.extreme_weight = extreme_weight
        self.use_ms_ssim = use_ms_ssim
        self.ssim_data_range = ssim_data_range
        
        self.weighted_mae = WeightedMAELoss(base_weight=1.0, extreme_weight=2.0)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss.
        
        Args:
            pred: Predicted tensor (B, C, T, H, W) or (B, C, H, W)
            target: Target tensor (same shape as pred)
            return_components: If True, return dict with individual loss components
        
        Returns:
            Total loss value (or dict if return_components=True)
        """
        # Flatten temporal dimension if present
        if pred.dim() == 5:
            B, C, T, H, W = pred.shape
            pred = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM loss (1 - SSIM, so lower is better)
        if self.use_ms_ssim and min(pred.shape[-2:]) >= 32:
            ssim_val = ms_ssim(pred, target, data_range=self.ssim_data_range)
        else:
            ssim_val = ssim(pred, target, data_range=self.ssim_data_range)
        ssim_loss = 1.0 - ssim_val
        
        # Weighted extreme loss
        extreme_loss = self.weighted_mae(pred, target)
        
        # Combine
        total_loss = (
            self.l1_weight * l1_loss + 
            self.ssim_weight * ssim_loss + 
            self.extreme_weight * extreme_loss
        )
        
        if return_components:
            return {
                'total': total_loss,
                'l1': l1_loss,
                'ssim': ssim_loss,
                'ssim_val': ssim_val,
                'extreme': extreme_loss
            }
        
        return total_loss


def get_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Loss configuration dict with keys:
            - type: 'l1', 'composite', or 'weighted'
            - l1_weight, ssim_weight, extreme_weight (for composite)
    
    Returns:
        Loss function module
    """
    loss_type = config.get('type', 'l1')
    
    if loss_type == 'l1':
        return nn.L1Loss()
    
    elif loss_type == 'composite':
        return CompositeLoss(
            l1_weight=config.get('l1_weight', 1.0),
            ssim_weight=config.get('ssim_weight', 0.5),
            extreme_weight=config.get('extreme_weight', 1.0),
            use_ms_ssim=config.get('use_ms_ssim', True),
            ssim_data_range=config.get('ssim_data_range', 2.0)
        )
    
    elif loss_type == 'weighted':
        return WeightedMAELoss(
            base_weight=config.get('base_weight', 1.0),
            extreme_weight=config.get('extreme_weight', 2.0)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

