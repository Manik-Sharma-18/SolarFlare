import zarr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

zg = zarr.open('harp_11930.zarr', mode='r')

print("--- Analysis of wind data ---")
wind = zg['wind']
times = zg['Time']

print(f"Shape: {wind.shape} (Y={wind.shape[0]}, X={wind.shape[1]}, T={wind.shape[2]})")

# Calculate some basic stats over the whole dataset or a sample
t_idx = wind.shape[2] // 2
sample_wind = wind[..., t_idx]

print(f"Stats at T={t_idx}:")
print(f"  Min: {np.nanmin(sample_wind):.4f}")
print(f"  Max: {np.nanmax(sample_wind):.4f}")
print(f"  Mean: {np.nanmean(sample_wind):.4f}")
print(f"  Std: {np.nanstd(sample_wind):.4f}")
print(f"  Non-zero pixels: {np.count_nonzero(sample_wind)}")

# Plot a simple heatmap
plt.figure(figsize=(10, 6))
# Handle adaptive colormap scaling (98th percentile)
vmax = np.nanpercentile(np.abs(sample_wind), 98)
vmin = -vmax if vmax > 0 else -1
vmax = vmax if vmax > 0 else 1

im = plt.imshow(sample_wind, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
plt.colorbar(im, label="Wind (Units)")
plt.title(f"Wind Heatmap (HARP 11930) - T={t_idx}")
plt.savefig('wind_heatmap_T162.png', bbox_inches='tight', dpi=150)
print("Saved wind_heatmap_T162.png")

# Extract numpy arrays for wind, time, and axes
print("\nExtracting data into numpy arrays...")
wind_array = np.array(wind)
time_array = np.array(times)

# Generate axis data since the Zarr only provides raw indices
y_axis = np.arange(wind_array.shape[0])
x_axis = np.arange(wind_array.shape[1])

# Save the arrays to a compressed npz file for easy loading later
output_file = 'harp_11930_extracted.npz'
print(f"Saving extracted arrays to {output_file}...")
np.savez_compressed(output_file, wind=wind_array, time=time_array, y=y_axis, x=x_axis)
print("Done!")

# Generate a GIF map for all timesteps
print("\nGenerating GIF animation for all timesteps...")
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate global vmin/vmax from a subset to ensure consistent colormap across frames
subset = wind_array[:, :, ::10]
global_vmax = np.nanpercentile(np.abs(subset), 98)
global_vmin = -global_vmax if global_vmax > 0 else -1
global_vmax = global_vmax if global_vmax > 0 else 1

im = ax.imshow(wind_array[..., 0], cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax, origin='lower')
plt.colorbar(im, ax=ax, label="Wind (Units)")
title = ax.set_title("Wind Heatmap (HARP 11930) - T=0")

def update(frame):
    im.set_array(wind_array[..., frame])
    title.set_text(f"Wind Heatmap (HARP 11930) - T={frame}")
    return [im, title]

ani = animation.FuncAnimation(fig, update, frames=wind_array.shape[2], blit=True)
gif_output = 'wind_evolution.gif'
print(f"Saving {gif_output} (this may take a minute)...")
ani.save(gif_output, writer='pillow', fps=15)
print("GIF generation complete!")
