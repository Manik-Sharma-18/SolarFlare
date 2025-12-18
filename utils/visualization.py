"""Visualization utilities for predictions."""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def visualize_predictions(
    model,
    dataset,
    device: torch.device,
    n_samples: int = 3,
    save_path: str = 'predictions.png',
    use_amp: bool = False
):
    """
    Visualize model predictions vs ground truth.
    
    Creates a grid showing input frame, predicted frames, and ground truth.
    """
    model.eval()
    
    t_out = dataset.t_out
    n_cols = 2 + t_out  # input + predictions + final ground truth
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Determine value range for consistent coloring
    vmin, vmax = -1, 1
    
    with torch.no_grad():
        for i in range(n_samples):
            # Sample evenly from dataset
            idx = i * len(dataset) // n_samples
            X_in, Y_out, (ds_id, start_idx) = dataset[idx]
            X_in = X_in.unsqueeze(0).to(device)
            Y_out = Y_out.unsqueeze(0).to(device)
            
            # Get predictions
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    predictions = model(X_in, teacher_forcing_ratio=0.0)
            else:
                predictions = model(X_in, teacher_forcing_ratio=0.0)
            
            # Plot last input frame
            ax = axes[i, 0]
            im = ax.imshow(X_in[0, 0, -1].cpu().numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'Input (t={dataset.t_in})')
            ax.axis('off')
            
            # Plot predictions
            for t in range(t_out):
                ax = axes[i, 1 + t]
                ax.imshow(predictions[0, 0, t].cpu().numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'Pred t+{t+1}')
                ax.axis('off')
            
            # Plot final ground truth
            ax = axes[i, -1]
            ax.imshow(Y_out[0, 0, -1].cpu().numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'GT t+{t_out}')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MAE)')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Per-timestep MAE
    if 'val_mae_per_timestep' in history and len(history['val_mae_per_timestep']) > 0:
        mae_array = np.array(history['val_mae_per_timestep'])
        for t in range(mae_array.shape[1]):
            axes[1].plot(mae_array[:, t], label=f't+{t+1}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation MAE per Timestep')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def visualize_with_uncertainty(
    mean_pred: torch.Tensor,
    uncertainty: torch.Tensor,
    ground_truth: torch.Tensor,
    save_path: str = 'uncertainty.png',
    timestep: int = -1
):
    """
    Visualize predictions with uncertainty maps.
    
    Creates a 3-row visualization:
    - Row 1: Mean prediction for each timestep
    - Row 2: Uncertainty (std) for each timestep  
    - Row 3: Ground truth for each timestep
    
    Args:
        mean_pred: Mean prediction (B, C, T, H, W) or (C, T, H, W)
        uncertainty: Uncertainty/std (B, C, T, H, W) or (C, T, H, W)
        ground_truth: Ground truth (B, C, T, H, W) or (C, T, H, W)
        save_path: Where to save the visualization
        timestep: Specific timestep to show (-1 for all)
    """
    # Handle batch dimension
    if mean_pred.dim() == 5:
        mean_pred = mean_pred[0]  # Take first sample
        uncertainty = uncertainty[0]
        ground_truth = ground_truth[0]
    
    # Convert to numpy
    mean_np = mean_pred[0].cpu().numpy()  # (T, H, W)
    unc_np = uncertainty[0].cpu().numpy()
    gt_np = ground_truth[0].cpu().numpy()
    
    if timestep >= 0:
        # Show single timestep
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(mean_np[timestep], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title(f'Prediction t+{timestep+1}')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        im1 = axes[1].imshow(unc_np[timestep], cmap='hot', vmin=0)
        axes[1].set_title(f'Uncertainty t+{timestep+1}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        im2 = axes[2].imshow(gt_np[timestep], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title(f'Ground Truth t+{timestep+1}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        # Show all timesteps
        T = mean_np.shape[0]
        fig, axes = plt.subplots(3, T, figsize=(4 * T, 12))
        
        if T == 1:
            axes = axes.reshape(3, 1)
        
        for t in range(T):
            # Row 1: Predictions
            im = axes[0, t].imshow(mean_np[t], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, t].set_title(f'Pred t+{t+1}')
            axes[0, t].axis('off')
            
            # Row 2: Uncertainty
            im = axes[1, t].imshow(unc_np[t], cmap='hot', vmin=0)
            axes[1, t].set_title(f'Uncertainty t+{t+1}')
            axes[1, t].axis('off')
            
            # Row 3: Ground truth
            im = axes[2, t].imshow(gt_np[t], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[2, t].set_title(f'GT t+{t+1}')
            axes[2, t].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Prediction', fontsize=12)
        axes[1, 0].set_ylabel('Uncertainty', fontsize=12)
        axes[2, 0].set_ylabel('Ground Truth', fontsize=12)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty visualization to {save_path}")


def visualize_uncertainty_statistics(
    uncertainty: torch.Tensor,
    save_path: str = 'uncertainty_stats.png'
):
    """
    Visualize uncertainty distribution and statistics.
    
    Args:
        uncertainty: Uncertainty tensor (B, C, T, H, W)
        save_path: Where to save
    """
    unc_np = uncertainty.cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(unc_np, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[0].axvline(unc_np.mean(), color='red', linestyle='--', label=f'Mean: {unc_np.mean():.4f}')
    axes[0].axvline(np.median(unc_np), color='orange', linestyle='--', label=f'Median: {np.median(unc_np):.4f}')
    axes[0].set_xlabel('Uncertainty (std)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Uncertainty Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = f"""
    Uncertainty Statistics
    ----------------------
    Mean:   {unc_np.mean():.6f}
    Std:    {unc_np.std():.6f}
    Min:    {unc_np.min():.6f}
    Max:    {unc_np.max():.6f}
    Median: {np.median(unc_np):.6f}
    
    Low uncertainty (<mean):  {(unc_np < unc_np.mean()).sum() / len(unc_np) * 100:.1f}%
    High uncertainty (>mean): {(unc_np >= unc_np.mean()).sum() / len(unc_np) * 100:.1f}%
    """
    
    axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes, 
                fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1].axis('off')
    axes[1].set_title('Statistics Summary')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty statistics to {save_path}")
