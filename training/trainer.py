"""
Training loop and validation for the solar flux predictor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from utils.device import get_amp_context, get_grad_scaler
from utils.metrics import compute_metrics
from .losses import get_loss_function, CompositeLoss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    teacher_forcing_ratio: float,
    epoch: int,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    show_progress: bool = True,
    output_channels: int = 1
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for AMP
        device: Device to train on
        teacher_forcing_ratio: TF ratio for this epoch
        epoch: Current epoch number
        loss_fn: Loss function (default: L1)
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping norm
        show_progress: Show progress bar
        output_channels: Number of output channels (for dual-channel: only compare first N)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if show_progress else dataloader
    
    for X_in, Y_out, _ in iterator:
        X_in = X_in.to(device)
        Y_out = Y_out.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with get_amp_context(use_amp, device):
            predictions = model(X_in, teacher_forcing_ratio, Y_out)
            
            # For dual-channel: compare only flux channel (first output_channels)
            # Model outputs: (B, output_channels, T, H, W)
            # Y_out may be: (B, 1 or 2, T, H, W)
            Y_target = Y_out[:, :output_channels]
            
            loss = loss_fn(predictions, Y_target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if show_progress:
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = True,
    show_progress: bool = True,
    output_channels: int = 1
) -> tuple:
    """
    Validate model on a dataset.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to use
        loss_fn: Loss function (default: L1)
        use_amp: Use automatic mixed precision
        show_progress: Show progress bar
        output_channels: Number of output channels
    
    Returns:
        avg_loss: Average validation loss
        avg_mae_per_timestep: MAE for each output timestep
    """
    model.eval()
    total_loss = 0.0
    all_mae_per_timestep = []
    
    # Default to L1 loss if none provided
    if loss_fn is None:
        loss_fn = nn.L1Loss()
    
    iterator = tqdm(dataloader, desc="Validating") if show_progress else dataloader
    
    with torch.no_grad():
        for X_in, Y_out, _ in iterator:
            X_in = X_in.to(device)
            Y_out = Y_out.to(device)
            
            with get_amp_context(use_amp, device):
                predictions = model(X_in, teacher_forcing_ratio=0.0)
                
                # For dual-channel: compare only flux channel
                Y_target = Y_out[:, :output_channels]
                loss = loss_fn(predictions, Y_target)
            
            total_loss += loss.item()
            metrics = compute_metrics(predictions, Y_target)
            all_mae_per_timestep.append(metrics['mae_per_timestep'])
    
    avg_loss = total_loss / len(dataloader)
    avg_mae_per_timestep = np.mean(all_mae_per_timestep, axis=0)
    
    return avg_loss, avg_mae_per_timestep


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, List]:
    """
    Main training loop with early stopping and checkpointing.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Device to train on
    
    Returns:
        Training history dict with losses and metrics
    """
    # Extract config
    epochs = config.get('epochs', 25)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    tf_start = config.get('tf_start', 0.5)
    patience = config.get('patience', 8)
    use_amp = config.get('use_amp', True)
    grad_clip = config.get('grad_clip', 1.0)
    save_dir = Path(config.get('save_dir', './outputs'))
    checkpoint_name = config.get('checkpoint_name', 'best_model.pt')
    show_progress = config.get('show_progress', True)
    output_channels = config.get('output_channels', 1)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create loss function from config
    loss_config = config.get('loss', {'type': 'l1'})
    loss_fn = get_loss_function(loss_config)
    loss_fn = loss_fn.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Gradient scaler for mixed precision
    scaler = get_grad_scaler(use_amp, device)
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_per_timestep': [],
        'learning_rate': []
    }
    
    print(f"\nStarting training for {epochs} epochs")
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    print(f"  Loss: {loss_config.get('type', 'l1')}")
    print(f"  Teacher forcing: {tf_start} → 0.0")
    print()
    
    for epoch in range(1, epochs + 1):
        # Teacher forcing schedule: linear decay
        tf_ratio = max(0.0, tf_start * (1 - epoch / epochs))
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.2e} | TF: {tf_ratio:.3f}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            tf_ratio, epoch, loss_fn, use_amp, grad_clip, show_progress, output_channels
        )
        
        # Validate
        val_loss, val_mae_per_timestep = validate(
            model, val_loader, device, loss_fn, use_amp, show_progress, output_channels
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val MAE:    {val_mae_per_timestep}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_per_timestep'].append(val_mae_per_timestep.tolist())
        history['learning_rate'].append(current_lr)
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / checkpoint_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        print()
    
    # Save training history
    if config.get('save_history', True):
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {history_path}")
    
    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """
    Load model from checkpoint.
    
    Returns:
        Checkpoint dict with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    
    return checkpoint

