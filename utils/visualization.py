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

