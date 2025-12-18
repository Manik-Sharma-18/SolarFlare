#!/usr/bin/env python3
"""
Standalone script for creating solar flare animations.

Creates animated visualizations of solar flux evolution from preprocessed data cubes.

Usage:
    # Create MP4 video
    python visualize_flares.py --cube data_processed/cube_005.npz --output flare.mp4 --fps 10
    
    # Create interactive HTML viewer
    python visualize_flares.py --cube data_processed/cube_005.npz --format html --output viewer.html
    
    # Specify frame range
    python visualize_flares.py --cube data_processed/cube_005.npz --start 0 --end 50 --fps 5
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from utils.animation import animate_flare_sequence, interactive_flare_viewer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create animations of solar flare evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic video (normalized data)
    python visualize_flares.py --cube data_processed/cube_005.npz
    
    # Raw unnormalized data from .npy file
    python visualize_flares.py --cube data/windTotal_MM_2024-10-02_0000_2348.npy --raw --percentile-clip 99
    
    # Unnormalized from preprocessed cube (if available)
    python visualize_flares.py --cube data_processed/cube_005.npz --unnormalized
    
    # Interactive HTML viewer
    python visualize_flares.py --cube data_processed/cube_005.npz --format html
    
    # Custom output and FPS
    python visualize_flares.py --cube data_processed/cube_005.npz --output my_video.mp4 --fps 15
    
    # Animate specific frame range
    python visualize_flares.py --cube data_processed/cube_005.npz --start 10 --end 60
        """
    )
    
    parser.add_argument(
        '--cube', '-c',
        required=True,
        help='Path to preprocessed cube NPZ file or raw .npy file'
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Load raw unnormalized data from .npy structured array'
    )
    parser.add_argument(
        '--unnormalized',
        action='store_true',
        help='Use unnormalized flux values (raw magnitudes, not preprocessed)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output path (default: flare_animation.mp4 or flare_viewer.html)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['mp4', 'html', 'gif'],
        default='mp4',
        help='Output format (default: mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for video (default: 10)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Start frame index (default: 0)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End frame index (default: last frame)'
    )
    parser.add_argument(
        '--cmap',
        default='RdBu_r',
        help='Colormap (default: RdBu_r)'
    )
    parser.add_argument(
        '--vmin',
        type=float,
        default=None,
        help='Minimum value for colorscale (default: auto from data, or percentile for raw)'
    )
    parser.add_argument(
        '--vmax',
        type=float,
        default=None,
        help='Maximum value for colorscale (default: auto from data, or percentile for raw)'
    )
    parser.add_argument(
        '--percentile-clip',
        type=float,
        default=None,
        help='Clip color scale to percentiles (e.g., 99 for 1-99th percentile) for raw data'
    )
    parser.add_argument(
        '--title',
        default='Solar Flux Evolution',
        help='Title for the animation'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='Resolution for video output (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Load cube
    cube_path = Path(args.cube)
    if not cube_path.exists():
        print(f"Error: File not found: {cube_path}")
        sys.exit(1)
    
    print(f"Loading data from {cube_path}...")
    
    # Check if it's a raw .npy file or preprocessed NPZ
    if args.raw or cube_path.suffix == '.npy':
        print("Loading raw structured array...")
        data = np.load(cube_path)
        
        # Convert structured array to cube
        if data.dtype.names is not None and 'windTotal' in data.dtype.names:
            x_coords = np.unique(data['X'])
            y_coords = np.unique(data['Y'])
            times = np.unique(data['time'])
            
            H, W, T = len(y_coords), len(x_coords), len(times)
            print(f"Reconstructing cube: T={T}, H={H}, W={W}")
            
            x_to_idx = {x: i for i, x in enumerate(x_coords)}
            y_to_idx = {y: i for i, y in enumerate(y_coords)}
            time_to_idx = {t: i for i, t in enumerate(times)}
            
            flux_cube = np.zeros((T, H, W), dtype=np.float32)
            for i in range(len(data)):
                t_idx = time_to_idx[data['time'][i]]
                h_idx = y_to_idx[data['Y'][i]]
                w_idx = x_to_idx[data['X'][i]]
                flux_cube[t_idx, h_idx, w_idx] = data['windTotal'][i]
            
            print(f"Raw data range: [{flux_cube.min():.2e}, {flux_cube.max():.2e}]")
        else:
            print("Error: Not a valid structured array with windTotal field")
            sys.exit(1)
    else:
        # Load preprocessed NPZ
        data = np.load(cube_path)
        
        # Handle different NPZ structures
        if args.unnormalized and 'flux_raw' in data:
            flux_cube = data['flux_raw']
            print("Using unnormalized flux from preprocessed cube")
        elif 'flux' in data:
            flux_cube = data['flux']
            if args.unnormalized:
                print("Warning: No 'flux_raw' in cube, using normalized 'flux'")
        elif 'data' in data:
            flux_cube = data['data']
        else:
            # Try first array in file
            keys = list(data.keys())
            if keys:
                flux_cube = data[keys[0]]
                print(f"Using array '{keys[0]}' from NPZ file")
            else:
                print("Error: Could not find flux data in NPZ file")
                sys.exit(1)
        
        if not args.unnormalized:
            print(f"Normalized cube shape: {flux_cube.shape} (T, H, W)")
            print(f"Value range: [{flux_cube.min():.4f}, {flux_cube.max():.4f}]")
        else:
            print(f"Unnormalized cube shape: {flux_cube.shape} (T, H, W)")
            print(f"Value range: [{flux_cube.min():.2e}, {flux_cube.max():.2e}]")
    
    # Apply frame range
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else flux_cube.shape[0]
    
    flux_cube = flux_cube[start:end]
    print(f"Using frames {start} to {end} ({len(flux_cube)} frames)")
    
    # Handle color limits for raw data
    if (args.raw or args.unnormalized) and args.percentile_clip is not None:
        print(f"Applying {args.percentile_clip}th percentile clipping...")
        low_val = np.percentile(flux_cube, 100 - args.percentile_clip)
        high_val = np.percentile(flux_cube, args.percentile_clip)
        if args.vmin is None:
            args.vmin = low_val
            print(f"  vmin: {low_val:.2e}")
        if args.vmax is None:
            args.vmax = high_val
            print(f"  vmax: {high_val:.2e}")
    
    # Determine output path
    if args.output is None:
        if args.format == 'html':
            output_path = 'flare_viewer.html'
        elif args.format == 'gif':
            output_path = 'flare_animation.gif'
        else:
            output_path = 'flare_animation.mp4'
    else:
        output_path = args.output
    
    # Create animation
    if args.format == 'html':
        print("Creating interactive HTML viewer...")
        interactive_flare_viewer(
            flux_cube,
            output_path=output_path,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title
        )
    else:
        print(f"Creating {args.format.upper()} animation at {args.fps} FPS...")
        animate_flare_sequence(
            flux_cube,
            output_path=output_path,
            fps=args.fps,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            title_prefix=args.title,
            dpi=args.dpi
        )
    
    print("Done!")


if __name__ == '__main__':
    main()

