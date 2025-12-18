"""Device management with CUDA toggle and AMP handling."""
import torch
from contextlib import nullcontext


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device based on config and availability.
    
    Returns CPU if use_cuda is False or CUDA is unavailable.
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        if use_cuda and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
        else:
            print("Using CPU (CUDA disabled in config)")
    
    return device


def get_amp_context(use_amp: bool, device: torch.device):
    """
    Get the appropriate autocast context for mixed precision.
    
    Returns nullcontext if AMP is disabled or on CPU.
    """
    if use_amp and device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda')
    elif use_amp and device.type == 'cpu':
        # CPU AMP is supported in newer PyTorch but less beneficial
        return torch.amp.autocast(device_type='cpu')
    else:
        return nullcontext()


def get_grad_scaler(use_amp: bool, device: torch.device):
    """
    Get gradient scaler for mixed precision training.
    
    Returns a no-op scaler if AMP is disabled or on CPU.
    """
    if use_amp and device.type == 'cuda':
        return torch.amp.GradScaler()
    else:
        # Return a dummy scaler that does nothing
        return _DummyGradScaler()


class _DummyGradScaler:
    """A no-op gradient scaler for CPU or non-AMP training."""
    
    def scale(self, loss):
        return loss
    
    def step(self, optimizer):
        optimizer.step()
    
    def update(self):
        pass
    
    def unscale_(self, optimizer):
        pass

