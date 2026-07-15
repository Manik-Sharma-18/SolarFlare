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
            if isinstance(predictions, tuple):
                # Dual-head model returns (pred, ext_logits) — plot pred only.
                predictions = predictions[0]

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
    """Plot training and validation loss curves with comprehensive metric subplots.

    Layout: 3 rows x 3 cols = 9 subplots.
    Row 0: Loss curves, Per-timestep MAE, CSI & HSS over epochs.
    Row 1: SSIM over epochs, Persistence skill per timestep, Temporal variation ratio.
    Row 2: All loss components (log), Temporal terms, Extreme terms.

    Backward-compatible: new subplots are only populated when the corresponding
    metric keys exist in history (old history files still work).
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # --- (0,0) Loss curves ---
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MAE)')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- (0,1) Per-timestep MAE ---
    if 'val_mae_per_timestep' in history and len(history['val_mae_per_timestep']) > 0:
        mae_array = np.array(history['val_mae_per_timestep'])
        if mae_array.ndim == 2 and mae_array.shape[1] > 0:
            for t in range(mae_array.shape[1]):
                axes[0, 1].plot(mae_array[:, t], label=f't+{t+1}')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_title('Validation MAE per Timestep')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

    # --- (0,2) CSI and HSS over epochs ---
    has_csi = 'val_csi' in history and len(history['val_csi']) > 0
    has_hss = 'val_hss' in history and len(history['val_hss']) > 0
    if has_csi or has_hss:
        if has_csi:
            axes[0, 2].plot(history['val_csi'], label='CSI', color='tab:blue')
        if has_hss:
            axes[0, 2].plot(history['val_hss'], label='HSS', color='tab:orange')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('CSI & HSS over Epochs')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(bottom=0)

    # --- (1,0) SSIM over epochs ---
    if 'val_ssim' in history and len(history['val_ssim']) > 0:
        axes[1, 0].plot(history['val_ssim'], label='SSIM', color='tab:green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Validation SSIM over Epochs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # --- (1,1) Persistence skill per timestep over epochs ---
    if 'persistence_skill_per_timestep' in history and len(history['persistence_skill_per_timestep']) > 0:
        skill_array = np.array(history['persistence_skill_per_timestep'])
        if skill_array.ndim == 2 and skill_array.shape[1] > 0:
            for t in range(skill_array.shape[1]):
                axes[1, 1].plot(skill_array[:, t], label=f't+{t+1}')
            axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Skill')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Skill (%)')
            axes[1, 1].set_title('Persistence Skill per Timestep')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

    # --- (1,2) Temporal variation ratio over epochs ---
    if 'temporal_variation_ratio' in history and len(history['temporal_variation_ratio']) > 0:
        axes[1, 2].plot(history['temporal_variation_ratio'], label='Temporal Var Ratio', color='tab:purple')
        axes[1, 2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (1.0)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Ratio')
        axes[1, 2].set_title('Temporal Variation Ratio over Epochs')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    # --- Row 2: Loss component breakdown ---
    # Component color scheme (tab10)
    comp_colors = {
        'train_l1': 'tab:blue',
        'train_ssim': 'tab:orange',
        'train_extreme': 'tab:green',
        'train_temporal_diff': 'tab:red',
        'train_temporal_var': 'tab:purple',
        'train_asymmetric': 'tab:brown',
    }
    comp_labels = {
        'train_l1': 'L1',
        'train_ssim': 'SSIM',
        'train_extreme': 'Extreme',
        'train_temporal_diff': 'Temporal Diff',
        'train_temporal_var': 'Temporal Var (|v|)',
        'train_asymmetric': 'Asymmetric',
    }

    # --- (2,0) All loss components overlaid (log scale) ---
    has_components = 'train_l1' in history and len(history.get('train_l1', [])) > 0
    if has_components:
        for key, color in comp_colors.items():
            if key in history and len(history[key]) > 0:
                values = history[key]
                # Plot absolute values (temporal_var is negative)
                plot_vals = [abs(v) for v in values]
                axes[2, 0].plot(plot_vals, label=comp_labels[key], color=color)
        axes[2, 0].set_yscale('log')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Loss (log scale)')
        axes[2, 0].set_title('Loss Components')
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)

    # --- (2,1) Temporal terms ---
    has_tdiff = 'train_temporal_diff' in history and len(history.get('train_temporal_diff', [])) > 0
    has_tvar = 'train_temporal_var' in history and len(history.get('train_temporal_var', [])) > 0
    if has_tdiff or has_tvar:
        if has_tdiff:
            axes[2, 1].plot(history['train_temporal_diff'],
                            label='Temporal Diff', color='tab:red')
        if has_tvar:
            tvar_abs = [abs(v) for v in history['train_temporal_var']]
            axes[2, 1].plot(tvar_abs,
                            label='Temporal Var (|v|, negative=reward)', color='tab:purple')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].set_title('Temporal Terms')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)

    # --- (2,2) Extreme terms ---
    has_extreme = 'train_extreme' in history and len(history.get('train_extreme', [])) > 0
    has_asymmetric = 'train_asymmetric' in history and len(history.get('train_asymmetric', [])) > 0
    if has_extreme or has_asymmetric:
        if has_extreme:
            axes[2, 2].plot(history['train_extreme'],
                            label='Extreme (WeightedMAE)', color='tab:green')
        if has_asymmetric:
            axes[2, 2].plot(history['train_asymmetric'],
                            label='Asymmetric', color='tab:brown')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Loss')
        axes[2, 2].set_title('Extreme Terms')
        axes[2, 2].legend(fontsize=8)
        axes[2, 2].grid(True, alpha=0.3)

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
