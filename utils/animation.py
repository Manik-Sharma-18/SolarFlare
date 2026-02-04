"""
Animation utilities for visualizing solar flare evolution.

Provides tools to create:
- MP4 videos of flux evolution over time
- Interactive HTML viewers with time slider
- Side-by-side prediction vs ground truth comparisons
- Support for both normalized and raw unnormalized data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Optional, Union, List, Tuple
import warnings


def load_flux_data(
    file_path: Union[str, Path],
    use_raw: bool = False,
    percentile_clip: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Load flux data from either preprocessed NPZ or raw .npy files.
    
    Args:
        file_path: Path to .npz (preprocessed) or .npy (raw) file
        use_raw: If True, loads unnormalized data (from flux_raw or reconstructs from .npy)
        percentile_clip: Optional percentile for color clipping (e.g., 99 for 1-99th)
    
    Returns:
        flux_cube: (T, H, W) numpy array
        metadata: Dict with data range, shape, etc.
    
    Example:
        # Load normalized data
        flux, meta = load_flux_data('data_processed/cube_005.npz')
        
        # Load raw unnormalized data
        flux, meta = load_flux_data('data/windTotal.npy', use_raw=True, percentile_clip=99)
    """
    file_path = Path(file_path)
    metadata = {}
    
    if file_path.suffix == '.npy' or use_raw and file_path.suffix == '.npy':
        # Load raw structured array
        print(f"Loading raw .npy: {file_path}")
        data = np.load(file_path)
        
        if data.dtype.names and 'windTotal' in data.dtype.names:
            x_coords = np.unique(data['X'])
            y_coords = np.unique(data['Y'])
            times = np.unique(data['time'])
            
            H, W, T = len(y_coords), len(x_coords), len(times)
            
            x_to_idx = {x: i for i, x in enumerate(x_coords)}
            y_to_idx = {y: i for i, y in enumerate(y_coords)}
            time_to_idx = {t: i for i, t in enumerate(times)}
            
            flux_cube = np.zeros((T, H, W), dtype=np.float32)
            for i in range(len(data)):
                t_idx = time_to_idx[data['time'][i]]
                h_idx = y_to_idx[data['Y'][i]]
                w_idx = x_to_idx[data['X'][i]]
                flux_cube[t_idx, h_idx, w_idx] = data['windTotal'][i]
            
            metadata = {
                'source': 'raw_npy',
                'shape': (T, H, W),
                'x_range': (float(x_coords.min()), float(x_coords.max())),
                'y_range': (float(y_coords.min()), float(y_coords.max())),
                'time_range': (str(times[0]), str(times[-1]))
            }
        else:
            raise ValueError("Not a valid structured array with windTotal field")
    
    else:
        # Load preprocessed NPZ
        print(f"Loading NPZ: {file_path}")
        data = np.load(file_path)
        
        if use_raw and 'flux_raw' in data:
            flux_cube = data['flux_raw']
            metadata['source'] = 'npz_raw'
        elif 'flux' in data:
            flux_cube = data['flux']
            metadata['source'] = 'npz_normalized'
            if use_raw:
                print("Warning: No flux_raw in NPZ, using normalized flux")
        elif 'data' in data:
            flux_cube = data['data']
            metadata['source'] = 'npz_data'
        else:
            keys = list(data.keys())
            if keys:
                flux_cube = data[keys[0]]
                metadata['source'] = f'npz_{keys[0]}'
            else:
                raise ValueError("Could not find flux data in NPZ")
        
        metadata['shape'] = flux_cube.shape
    
    # Add data statistics
    metadata['min'] = float(flux_cube.min())
    metadata['max'] = float(flux_cube.max())
    metadata['mean'] = float(flux_cube.mean())
    metadata['std'] = float(flux_cube.std())
    
    # Apply percentile clipping if requested
    if percentile_clip is not None:
        low_percentile = 100 - percentile_clip
        vmin_clip = np.percentile(flux_cube, low_percentile)
        vmax_clip = np.percentile(flux_cube, percentile_clip)
        metadata['percentile_clip'] = percentile_clip
        metadata['vmin_suggested'] = float(vmin_clip)
        metadata['vmax_suggested'] = float(vmax_clip)
        print(f"Suggested color limits ({percentile_clip}th percentile): [{vmin_clip:.2e}, {vmax_clip:.2e}]")
    
    print(f"Loaded cube: {flux_cube.shape} | Range: [{metadata['min']:.2e}, {metadata['max']:.2e}]")
    
    return flux_cube, metadata


def animate_flare_sequence(
    data_cube: np.ndarray,
    output_path: str = 'flare_evolution.mp4',
    fps: int = 10,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title_prefix: str = 'Solar Flux',
    figsize: tuple = (10, 8),
    dpi: int = 100,
    colorbar: bool = True
):
    """
    Create MP4 animation of solar flux evolution.
    
    Args:
        data_cube: (T, H, W) numpy array of flux values
        output_path: Where to save the MP4 file
        fps: Frames per second
        cmap: Matplotlib colormap
        vmin, vmax: Color scale limits (None = auto from data min/max)
        title_prefix: Prefix for frame titles
        figsize: Figure size in inches
        dpi: Output resolution
        colorbar: Show colorbar
    
    Example:
        import numpy as np
        
        # Load preprocessed cube
        data = np.load('data_processed/cube_005.npz')
        flux = data['flux']  # (T, H, W)
        
        # Create animation with consistent colors
        animate_flare_sequence(flux, 'flare_video.mp4', fps=10)
    """
    # Calculate consistent color limits from ALL frames
    if vmin is None:
        vmin = float(data_cube.min())
        print(f"Auto vmin: {vmin:.4f}")
    if vmax is None:
        vmax = float(data_cube.max())
        print(f"Auto vmax: {vmax:.4f}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial frame with FIXED color limits
    im = ax.imshow(data_cube[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True, interpolation='nearest')
    ax.set_title(f'{title_prefix} - Frame 1/{len(data_cube)}')
    ax.axis('off')
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Flux')
    
    def update(frame):
        im.set_array(data_cube[frame])
        ax.set_title(f'{title_prefix} - Frame {frame+1}/{len(data_cube)}')
        return [im]
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(data_cube), 
        interval=1000/fps, blit=True
    )
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save animation
    try:
        ani.save(str(output_path), writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Saved animation to {output_path}")
    except Exception as e:
        # Fallback to pillow if ffmpeg not available
        try:
            gif_path = output_path.with_suffix('.gif')
            ani.save(str(gif_path), writer='pillow', fps=fps)
            print(f"ffmpeg not available, saved GIF to {gif_path}")
        except Exception as e2:
            print(f"Could not save animation: {e2}")
            print("Install ffmpeg: sudo apt install ffmpeg")
    
    plt.close()


def interactive_flare_viewer(
    data_cube: np.ndarray,
    timestamps: Optional[List[str]] = None,
    output_path: str = 'flare_viewer.html',
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = 'Interactive Solar Flux Evolution'
):
    """
    Create interactive HTML viewer with time slider (requires plotly).
    
    Args:
        data_cube: (T, H, W) numpy array
        timestamps: Optional list of timestamp strings for each frame
        output_path: Where to save HTML file
        cmap: Colorscale name (plotly compatible)
        vmin, vmax: Color scale limits (None = auto from data)
        title: Plot title
    
    Example:
        interactive_flare_viewer(flux_cube, output_path='viewer.html')
        # Open viewer.html in browser for interactive exploration
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return
    
    T = len(data_cube)
    
    # Calculate consistent color limits
    if vmin is None:
        vmin = float(data_cube.min())
    if vmax is None:
        vmax = float(data_cube.max())
    
    # Map matplotlib colormap to plotly (RdBu_r -> RdBu reversed)
    plotly_cmap = 'RdBu_r' if cmap == 'RdBu_r' else cmap
    
    # Create frames
    frames = []
    for i in range(T):
        label = timestamps[i] if timestamps else f'Frame {i+1}'
        frames.append(
            go.Frame(
                data=[go.Heatmap(
                    z=data_cube[i],
                    colorscale=plotly_cmap,
                    zmin=vmin,
                    zmax=vmax,
                    showscale=True
                )],
                name=str(i),
                layout=go.Layout(title=f'{title} - {label}')
            )
        )
    
    # Initial figure
    fig = go.Figure(
        data=[go.Heatmap(
            z=data_cube[0],
            colorscale=plotly_cmap,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(title='Flux')
        )],
        frames=frames
    )
    
    # Add play/pause buttons and slider
    fig.update_layout(
        title=title,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.1,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate'
                        )]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    prefix='Frame: ',
                    visible=True,
                    xanchor='center'
                ),
                steps=[
                    dict(
                        args=[[str(i)], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate'
                        )],
                        label=timestamps[i] if timestamps else str(i+1),
                        method='animate'
                    )
                    for i in range(T)
                ]
            )
        ]
    )
    
    # Invert y-axis for image-like display
    fig.update_yaxes(autorange='reversed')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_path))
    print(f"Saved interactive viewer to {output_path}")


def animate_prediction_vs_truth(
    model,
    dataset,
    device,
    sample_idx: int = 0,
    output_path: str = 'prediction_comparison.mp4',
    fps: int = 5,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    include_input: bool = True
):
    """
    Side-by-side animation comparing model prediction to ground truth.
    
    Args:
        model: Trained SolarFluxPredictor model
        dataset: Dataset containing samples
        device: torch device
        sample_idx: Which sample to visualize
        output_path: Where to save MP4
        fps: Frames per second
        cmap: Colormap
        vmin, vmax: Color limits (None = auto from data)
        include_input: Include input sequence before predictions
    
    Example:
        animate_prediction_vs_truth(model, test_dataset, device, 
                                   sample_idx=0, output_path='comparison.mp4')
    """
    import torch
    
    # Get sample
    X_in, Y_out, _ = dataset[sample_idx]
    X_in_batch = X_in.unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_in_batch, teacher_forcing_ratio=0.0)
    
    # Convert to numpy: (T, H, W)
    pred_np = predictions[0, 0].cpu().numpy()
    truth_np = Y_out[0].cpu().numpy() if Y_out.dim() == 4 else Y_out.cpu().numpy()
    input_np = X_in[0].cpu().numpy() if X_in.dim() == 4 else X_in.cpu().numpy()
    
    # Calculate consistent color limits from ALL data
    if vmin is None or vmax is None:
        all_data = np.concatenate([pred_np.flatten(), truth_np.flatten(), input_np.flatten()])
        if vmin is None:
            vmin = float(all_data.min())
        if vmax is None:
            vmax = float(all_data.max())
        print(f"Auto color limits: [{vmin:.4f}, {vmax:.4f}]")
    
    T_out = pred_np.shape[0]
    T_in = input_np.shape[0]
    
    # Determine total frames
    if include_input:
        total_frames = T_in + T_out
    else:
        total_frames = T_out
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def get_frame_data(frame_idx):
        """Get data for a given frame index."""
        if include_input:
            if frame_idx < T_in:
                # Input phase
                return input_np[frame_idx], input_np[frame_idx], f'Input t={frame_idx+1}', 'Input (same)'
            else:
                # Prediction phase
                pred_idx = frame_idx - T_in
                return pred_np[pred_idx], truth_np[pred_idx], f'Pred t+{pred_idx+1}', f'GT t+{pred_idx+1}'
        else:
            return pred_np[frame_idx], truth_np[frame_idx], f'Pred t+{frame_idx+1}', f'GT t+{frame_idx+1}'
    
    # Initialize with FIXED color limits
    pred_data, truth_data, pred_title, truth_title = get_frame_data(0)
    
    im1 = axes[0].imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].set_title(pred_title)
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Normalized Flux')
    
    im2 = axes[1].imshow(truth_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title(truth_title)
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Flux')
    
    def update(frame):
        pred_data, truth_data, pred_title, truth_title = get_frame_data(frame)
        
        im1.set_array(pred_data)
        im2.set_array(truth_data)
        axes[0].set_title(pred_title)
        axes[1].set_title(truth_title)
        
        return [im1, im2]
    
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000/fps, blit=True
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ani.save(str(output_path), writer='ffmpeg', fps=fps, dpi=100)
        print(f"Saved comparison animation to {output_path}")
    except Exception as e:
        gif_path = output_path.with_suffix('.gif')
        try:
            ani.save(str(gif_path), writer='pillow', fps=fps)
            print(f"Saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"Could not save animation: {e2}")
    
    plt.close()


def animate_with_uncertainty(
    mean_pred: np.ndarray,
    uncertainty: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str = 'uncertainty_animation.mp4',
    fps: int = 2,
    cmap_pred: str = 'RdBu_r',
    cmap_unc: str = 'hot'
):
    """
    Animate predictions with uncertainty maps.
    
    Creates a 3-panel animation:
    - Left: Mean prediction
    - Center: Uncertainty map
    - Right: Ground truth
    
    Args:
        mean_pred: (T, H, W) mean predictions
        uncertainty: (T, H, W) uncertainty values
        ground_truth: (T, H, W) ground truth
        output_path: Where to save
        fps: Frames per second
        cmap_pred: Colormap for predictions
        cmap_unc: Colormap for uncertainty
    """
    T = mean_pred.shape[0]
    
    # Calculate consistent color limits
    pred_vmin = min(mean_pred.min(), ground_truth.min())
    pred_vmax = max(mean_pred.max(), ground_truth.max())
    unc_vmax = uncertainty.max()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initialize with FIXED color limits
    im1 = axes[0].imshow(mean_pred[0], cmap=cmap_pred, vmin=pred_vmin, vmax=pred_vmax, interpolation='nearest')
    axes[0].set_title('Mean Prediction t+1')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046)
    cbar1.set_label('Flux')
    
    im2 = axes[1].imshow(uncertainty[0], cmap=cmap_unc, vmin=0, vmax=unc_vmax, interpolation='nearest')
    axes[1].set_title('Uncertainty t+1')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046)
    cbar2.set_label('Std Dev')
    
    im3 = axes[2].imshow(ground_truth[0], cmap=cmap_pred, vmin=pred_vmin, vmax=pred_vmax, interpolation='nearest')
    axes[2].set_title('Ground Truth t+1')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046)
    cbar3.set_label('Flux')
    
    def update(frame):
        im1.set_array(mean_pred[frame])
        im2.set_array(uncertainty[frame])
        im3.set_array(ground_truth[frame])
        
        axes[0].set_title(f'Mean Prediction t+{frame+1}')
        axes[1].set_title(f'Uncertainty t+{frame+1}')
        axes[2].set_title(f'Ground Truth t+{frame+1}')
        
        return [im1, im2, im3]
    
    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ani.save(str(output_path), writer='ffmpeg', fps=fps, dpi=100)
        print(f"Saved uncertainty animation to {output_path}")
    except Exception:
        gif_path = output_path.with_suffix('.gif')
        ani.save(str(gif_path), writer='pillow', fps=fps)
        print(f"Saved as GIF to {gif_path}")
    
    plt.close()


def create_difference_animation(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str = 'difference.mp4',
    fps: int = 2
):
    """
    Animate the difference between predictions and ground truth.
    
    Useful for identifying systematic errors.
    
    Args:
        predictions: (T, H, W) predicted values
        ground_truth: (T, H, W) ground truth values
        output_path: Where to save
        fps: Frames per second
    """
    diff = predictions - ground_truth
    T = diff.shape[0]
    
    # Calculate consistent color limits from ALL frames
    pred_vmin = min(predictions.min(), ground_truth.min())
    pred_vmax = max(predictions.max(), ground_truth.max())
    
    # Symmetric color limits for difference
    max_abs_diff = float(np.abs(diff).max())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Initialize with FIXED color limits
    im1 = axes[0].imshow(predictions[0], cmap='RdBu_r', vmin=pred_vmin, vmax=pred_vmax, interpolation='nearest')
    axes[0].set_title('Prediction t+1')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046)
    cbar1.set_label('Flux')
    
    im2 = axes[1].imshow(ground_truth[0], cmap='RdBu_r', vmin=pred_vmin, vmax=pred_vmax, interpolation='nearest')
    axes[1].set_title('Ground Truth t+1')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046)
    cbar2.set_label('Flux')
    
    im3 = axes[2].imshow(diff[0], cmap='seismic', vmin=-max_abs_diff, vmax=max_abs_diff, interpolation='nearest')
    axes[2].set_title('Difference t+1')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046)
    cbar3.set_label('Error')
    
    def update(frame):
        im1.set_array(predictions[frame])
        im2.set_array(ground_truth[frame])
        im3.set_array(diff[frame])
        
        axes[0].set_title(f'Prediction t+{frame+1}')
        axes[1].set_title(f'Ground Truth t+{frame+1}')
        axes[2].set_title(f'Difference t+{frame+1} (MAE={np.abs(diff[frame]).mean():.4f})')
        
        return [im1, im2, im3]
    
    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ani.save(str(output_path), writer='ffmpeg', fps=fps, dpi=100)
        print(f"Saved difference animation to {output_path}")
    except Exception:
        gif_path = output_path.with_suffix('.gif')
        ani.save(str(gif_path), writer='pillow', fps=fps)
        print(f"Saved as GIF to {gif_path}")
    
    plt.close()

