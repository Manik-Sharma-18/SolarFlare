# Solar Flare Prediction with ConvLSTM

A deep learning system for predicting solar wind flux using spatiotemporal data. This model uses **Convolutional LSTM (ConvLSTM)** networks to learn patterns in winding flux data cubes and predict future frames.

## Table of Contents

1. [Overview](#overview)
2. [What is a ConvLSTM?](#what-is-a-convlstm)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Data Pipeline](#data-pipeline)
5. [Project Structure](#project-structure)
6. [Configuration Guide](#configuration-guide)
7. [Usage](#usage)
8. [Memory & Performance](#memory--performance)
9. [Understanding the Output](#understanding-the-output)

---

## Overview

### The Problem

Solar flares and coronal mass ejections release enormous amounts of energy that can affect Earth's magnetic field, satellites, and power grids. Predicting these events requires understanding the complex dynamics of solar magnetic fields.

**Winding flux data** captures the magnetic field topology on the solar surface over time. By analyzing how these patterns evolve, we can potentially forecast future activity.

### The Solution

This model takes a sequence of past winding flux observations and predicts what the next few frames will look like. Think of it like predicting the next frames of a video.

```
INPUT: 8 frames of flux data    →    MODEL    →    OUTPUT: 3 predicted frames
       (what happened)                              (what will happen)
```

---

## What is a ConvLSTM?

To understand ConvLSTM, let's first understand its components:

### Regular Neural Networks

A basic neural network takes an input, processes it through layers, and produces an output. But it has no "memory" - each input is processed independently.

### LSTM (Long Short-Term Memory)

LSTMs are designed for **sequential data** (like time series). They have a "memory cell" that can:
- **Remember** important information from the past
- **Forget** irrelevant information
- **Update** the memory with new information
- **Output** relevant information

```
Time 1: Input → LSTM → Output (remembers something)
Time 2: Input → LSTM → Output (uses memory + new input)
Time 3: Input → LSTM → Output (continues building context)
```

### Convolutional Neural Networks (CNNs)

CNNs are designed for **spatial data** (like images). They use small filters that slide across the image to detect patterns like edges, textures, and shapes.

```
Image → [Conv Filter] → Feature Maps (detected patterns)
```

### ConvLSTM: The Best of Both Worlds

ConvLSTM combines LSTM's temporal memory with CNN's spatial processing:

- **Input**: A sequence of images (video frames)
- **Processing**: Each timestep uses convolutions instead of matrix multiplication
- **Output**: Maintains both spatial structure AND temporal context

```
Frame 1 → ConvLSTM → Spatial features + Memory
Frame 2 → ConvLSTM → Spatial features + Updated Memory
Frame 3 → ConvLSTM → Spatial features + Richer Memory
```

### The ConvLSTM Cell Equations

For those interested in the math, here's what happens at each timestep:

```
1. Input Gate (i):      What new information to store
   i = sigmoid(Conv([x, h_prev]))

2. Forget Gate (f):     What old information to discard
   f = sigmoid(Conv([x, h_prev]))

3. Cell Gate (g):       New candidate values
   g = tanh(Conv([x, h_prev]))

4. Output Gate (o):     What to output
   o = sigmoid(Conv([x, h_prev]))

5. Cell State Update:   c = f * c_prev + i * g

6. Hidden State:        h = o * tanh(c)
```

The `Conv([x, h_prev])` means we concatenate the current input with the previous hidden state and apply a convolution - this is what makes it "Convolutional" LSTM.

---

## Architecture Deep Dive

Our model uses an **Encoder-Decoder** architecture with **autoregressive prediction**.

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Input   │    │ConvLSTM │    │ Down-   │    │ConvLSTM │       │
│  │ Frames  │ →  │ Layer 1 │ →  │ sample  │ →  │Layer 2,3│       │
│  │(8 frames)    │(16 ch)  │    │ (2x)    │    │(32→64ch)│       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│                      ↓ skip                        ↓             │
│                  connection                   latent state       │
└─────────────────────────────────────────────────────────────────┘
                                                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Last    │    │ConvLSTM │    │ Up-     │    │ Output  │       │
│  │ Input   │ →  │ Layers  │ →  │ sample  │ →  │ + Skip  │ → Pred│
│  │ Frame   │    │(continues    │ (2x)    │    │         │       │
│  └─────────┘    │ from enc)    └─────────┘    └─────────┘       │
│       ↑              │                                           │
│       └──────────────┴── Autoregressive: pred becomes next input │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Input Downsampling (Optional)

**Purpose**: Reduce memory usage for large spatial dimensions.

```python
# If enabled, reduces H×W by half
input: (B, 1, 8, 440, 884) → (B, 16, 8, 220, 442)
```

This is crucial for fitting the model in GPU memory. Without it, a single batch might use 4+ GB of VRAM.

#### 2. Encoder

The encoder processes the input sequence to build a rich representation:

**ConvLSTM Layer 1** (at full resolution):
- Processes all 8 input frames
- Captures local spatial patterns and short-term temporal dynamics
- Outputs features AND maintains hidden state for decoder

**Spatial Downsampling**:
- Reduces resolution by 2x (stride=2 convolution)
- Allows deeper layers to have larger receptive field
- Reduces computation

**ConvLSTM Layers 2 & 3** (at reduced resolution):
- Build increasingly abstract representations
- Layer 3 output is the "latent state" - compressed representation of the input

#### 3. Skip Connection

**Purpose**: Preserve fine-grained details that might be lost in downsampling.

The hidden state from encoder layer 1 is concatenated with upsampled decoder output. This helps the model:
- Maintain sharp edges and small features
- Combine high-level semantics with low-level details

#### 4. Decoder (Autoregressive)

The decoder generates predictions one frame at a time:

```
Step 1: Use last input frame → Predict frame 1
Step 2: Use predicted frame 1 → Predict frame 2
Step 3: Use predicted frame 2 → Predict frame 3
```

**Why autoregressive?** Each prediction depends on the previous one, which:
- Maintains temporal consistency
- Allows the model to "build" on its predictions
- More realistic than predicting all frames at once

#### 5. Residual Prediction

Instead of predicting the full frame, the model predicts the **change** (residual):

```python
predicted_frame = input_frame + delta
```

**Benefits**:
- Easier to learn (smaller values)
- Stable training (predictions stay close to input)
- Works well when consecutive frames are similar

### Teacher Forcing

During training, we sometimes feed the ground truth instead of predictions:

```
Training with 50% teacher forcing:
  - 50% of the time: Use predicted frame as next input
  - 50% of the time: Use actual ground truth as next input
```

This helps training converge faster. The ratio decreases over epochs (curriculum learning).

---

## Data Pipeline

### Input Data Format

The model expects `.npy` files containing structured arrays with fields:

| Field | Type | Description |
|-------|------|-------------|
| `X` | float64 | X coordinate on solar surface |
| `Y` | float64 | Y coordinate on solar surface |
| `windTotal` | float64 | Winding flux value |
| `time` | datetime64 | Timestamp of observation |

### Data Processing Steps

```
1. LOAD: Read .npy structured array
   └─ Shape: (N_records,) with fields

2. RESHAPE: Convert to dense cube
   └─ Shape: (T, H, W) where T=timesteps, H=height, W=width

3. NORMALIZE: Scale values to [-1, 1] range
   └─ Robust method: (value - median) / (p99 - p1)
   └─ Handles outliers better than simple min-max

4. SPLIT: Divide temporally
   └─ Train: first 70% of timesteps
   └─ Val: next 15%
   └─ Test: final 15%

5. WINDOW: Create sliding windows
   └─ Each sample: 8 input frames + 3 target frames
   └─ Stride: 1 (overlapping windows)

6. AUGMENT: Random flips during training
   └─ Horizontal flip (50% chance)
   └─ Vertical flip (50% chance)
```

### Why This Normalization?

Solar flux data has extreme outliers (values up to 224 million!). Simple normalization like `x / max` would squash most values near zero.

**Robust normalization** uses percentiles:
- Center at median (not affected by outliers)
- Scale by the range between 1st and 99th percentile
- Most values end up in [-1, 1], outliers may exceed this

---

## Project Structure

```
SolarFlare/
├── config.yaml              # All hyperparameters and settings
├── main.py                  # Entry point - run this to train
├── README.md                # This documentation
│
├── models/                  # Neural network architectures
│   ├── __init__.py
│   ├── convlstm.py          # ConvLSTMCell and ConvLSTM classes
│   └── predictor.py         # SolarFluxPredictor (full model)
│
├── solarflare_data/         # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset class
│   └── loader.py            # Data loading and normalization
│
├── training/                # Training loop and validation
│   ├── __init__.py
│   └── trainer.py           # train_model, train_epoch, validate
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── device.py            # CUDA/CPU selection, AMP handling
│   ├── metrics.py           # Loss and evaluation metrics
│   └── visualization.py     # Plotting predictions and history
│
├── data/                    # Your .npy data files go here
│   └── windTotal_*.npy
│
└── outputs/                 # Generated during training
    ├── best_model.pt        # Model checkpoint
    ├── training_history.json
    ├── predictions.png
    └── metadata.json
```

### Module Details

#### `models/convlstm.py`

Contains the building blocks:

- **`ConvLSTMCell`**: Single cell that processes one timestep
  - Takes: current input + previous hidden state
  - Returns: new hidden state + new cell state

- **`ConvLSTM`**: Multi-layer wrapper
  - Stacks multiple cells
  - Processes entire sequence
  - Handles hidden state initialization

#### `models/predictor.py`

The complete model:

- **`SolarFluxPredictor`**: Encoder-decoder architecture
  - Configurable channels: `[16, 32, 64]`
  - Optional input downsampling
  - Autoregressive prediction with teacher forcing

#### `solarflare_data/loader.py`

Data preparation:

- **`load_and_prepare_data`**: Main function
  - Loads all .npy files
  - Applies normalization
  - Creates train/val/test splits
  - Returns PyTorch datasets

#### `training/trainer.py`

Training logic:

- **`train_model`**: Main loop with early stopping
- **`train_epoch`**: Single epoch with progress bar
- **`validate`**: Evaluation without gradients

#### `utils/device.py`

Hardware abstraction:

- **`get_device`**: Returns CUDA or CPU based on config
- **`get_amp_context`**: Mixed precision context manager
- **`get_grad_scaler`**: Gradient scaling for AMP

---

## Configuration Guide

All settings are in `config.yaml`. This section explains each parameter and how changing it affects the model.

---

### Device Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `use_cuda` | `true` | Use GPU for training | `false`: Forces CPU (10-50x slower but useful for debugging) |

---

### Data Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `data_dir` | `"./data"` | Path to raw .npy files | Change to point to your data location |
| `use_preprocessed` | `true` | Use preprocessed cubes | `true`: Fast loading (recommended). `false`: Convert on-the-fly (slow) |
| `preprocessed_dir` | `"./data_processed"` | Path to preprocessed cubes | Change after running `preprocess_data.py` with different output |
| `t_in` | `10` | Input sequence length | **Higher**: More temporal context, more memory, potentially better predictions. **Lower**: Faster, less context |
| `t_out` | `4` | Output prediction length | **Higher**: Predict further into future, harder task, more error accumulation. **Lower**: Easier, more accurate |
| `train_split` | `0.7` | Training data fraction | **Higher**: More training data, risk of less validation. **Lower**: More validation, less training |
| `val_split` | `0.15` | Validation data fraction | Remainder (1 - train - val) goes to test set |
| `augment` | `false` | Random flips during training | `true`: More data diversity, may help generalization. `false`: Faster training |
| `dual_channel` | `true` | Enable 2-channel input | `true`: Flux + extreme indicator (helps with flare prediction). `false`: Single-channel flux only |

---

### Normalization Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `method` | `"asinh"` | Normalization method | `"asinh"`: Preserves extreme values (recommended for flares). `"robust"`: Clips extremes. `"fixed"`: Simple division |
| `asinh_softening` | `1000.0` | Softening for asinh transform | **Higher (10000+)**: Less compression, more linear. **Lower (100-500)**: More compression, extremes more compressed |
| `extreme_threshold_percentile` | `99.5` | Percentile for extreme detection | **Higher (99.9)**: Only very extreme events flagged. **Lower (95)**: More events flagged as extreme |
| `percentile_low` | `1` | Lower percentile (robust method) | For `method: "robust"` only |
| `percentile_high` | `99` | Upper percentile (robust method) | For `method: "robust"` only |
| `fixed_factor` | `40000.0` | Division factor (fixed method) | For `method: "fixed"` only |

**How to choose `asinh_softening`:**
- Look at your data's typical (non-flare) magnitude
- Set softening ≈ typical magnitude
- Values below softening behave linearly, above behave logarithmically

---

### Model Architecture

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `input_channels` | `2` | Input channels | `1`: Single-channel (flux only). `2`: Dual-channel (flux + extreme indicator) |
| `output_channels` | `1` | Output channels | Typically `1` (predict flux). Must match what you want to predict |
| `channels` | `[16, 32, 64]` | Channel progression | **Higher [32,64,128]**: More capacity, more memory, potentially better. **Lower [8,16,32]**: Less memory, faster, may underfit |
| `kernel_size` | `3` | ConvLSTM kernel size | **Larger (5,7)**: Larger receptive field, more params. **Smaller (3)**: Standard, efficient |
| `downsample_input` | `true` | 2x spatial downsampling | `true`: Halves H,W, saves ~4x memory. `false`: Full resolution, more memory |
| `use_checkpointing` | `false` | Gradient checkpointing | `true`: Saves ~40% VRAM, ~20% slower training. Enable for larger models |
| `dropout_rate` | `0.0` | Dropout for uncertainty | `0.0`: No dropout. `0.1-0.2`: Enable MC Dropout for uncertainty estimation |

**Channel progression impact:**

| Config | Parameters | Memory | Use Case |
|--------|-----------|--------|----------|
| `[8, 16, 32]` | ~175K | ~1-2 GB | Limited GPU (4GB) |
| `[16, 32, 64]` | ~695K | ~3-4 GB | Standard GPU (8GB) |
| `[32, 64, 128]` | ~2.7M | ~8-12 GB | Large GPU (16GB+) |

---

### Training Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `batch_size` | `1` | Samples per batch | **Higher**: More stable gradients, more memory. Keep at 1 for large images |
| `num_workers` | `2` | DataLoader workers | `0`: Required for Google Colab. `2-4`: Faster on local machines |
| `epochs` | `25` | Maximum epochs | **Higher**: More training, risk of overfitting. **Lower**: Faster, may underfit |
| `lr` | `0.001` | Learning rate | **Higher (0.01)**: Faster learning, may overshoot. **Lower (0.0001)**: Slower, more stable |
| `weight_decay` | `0.00001` | L2 regularization | **Higher (0.001)**: More regularization, prevents overfitting. **Lower**: Less constraint on weights |
| `tf_start` | `0.5` | Initial teacher forcing | **Higher (0.8)**: More guidance early, may not learn error recovery. **Lower (0.2)**: Harder training, better autoregressive |
| `patience` | `8` | Early stopping patience | **Higher (15)**: More chances to improve. **Lower (5)**: Faster termination |
| `use_amp` | `true` | Mixed precision training | `true`: ~2x faster, less memory. `false`: Full precision, more stable |
| `grad_clip` | `1.0` | Gradient clipping norm | **Lower (0.5)**: More aggressive clipping, stabler. **Higher (5.0)**: Less clipping |

**Learning rate guidance:**

| Symptom | Action |
|---------|--------|
| Loss oscillating wildly | Decrease `lr` by 10x |
| Loss decreasing very slowly | Increase `lr` by 2-5x |
| Loss goes to NaN | Decrease `lr`, enable `grad_clip` |
| Validation loss increasing while train decreases | Overfitting: increase `weight_decay` |

---

### Loss Function Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `type` | `"composite"` | Loss function type | `"l1"`: Simple MAE. `"composite"`: L1 + SSIM + weighted. `"weighted"`: Weighted MAE only |
| `l1_weight` | `1.0` | Weight for L1 loss | **Higher**: Prioritize numerical accuracy. **Lower**: Let other losses dominate |
| `ssim_weight` | `0.5` | Weight for SSIM loss | **Higher (1.0+)**: Sharper predictions, may have artifacts. **Lower (0.1)**: Less structural focus |
| `extreme_weight` | `1.0` | Weight for extreme value loss | **Higher (2.0+)**: Focus on flare regions. **Lower (0.5)**: More balanced |
| `use_ms_ssim` | `true` | Multi-scale SSIM | `true`: Better for varying feature sizes. `false`: Single-scale, faster |
| `ssim_data_range` | `2.0` | Data range for SSIM | Should be 2.0 for data normalized to [-1, 1] |

**Loss weight tuning guide:**

| Problem | Solution |
|---------|----------|
| Predictions too blurry | Increase `ssim_weight` to 1.0-2.0 |
| Missing extreme events | Increase `extreme_weight` to 2.0-3.0 |
| Artifacts in predictions | Decrease `ssim_weight`, increase `l1_weight` |
| Background looks wrong | Decrease `extreme_weight` |

---

### Output Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `save_dir` | `"./outputs"` | Output directory | Change to organize different experiments |
| `checkpoint_name` | `"best_model.pt"` | Model checkpoint filename | Change to keep multiple checkpoints |
| `save_history` | `true` | Save training history JSON | `false`: Skip saving (not recommended) |
| `save_visualizations` | `true` | Save prediction plots | `false`: Skip plots (faster) |

---

### Uncertainty Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `enabled` | `false` | Enable uncertainty quantification | `true`: Run MC Dropout during evaluation. Requires `dropout_rate > 0` |
| `n_samples` | `20` | MC Dropout samples | **Higher (50+)**: Better estimates, slower. **Lower (10)**: Faster, noisier |
| `save_uncertainty_maps` | `true` | Save uncertainty visualizations | `false`: Skip uncertainty plots |

**Uncertainty interpretation:**

| Uncertainty Level | Meaning | Action |
|-------------------|---------|--------|
| Low (< mean) | Model is confident | Trust prediction |
| High (> mean) | Multiple outcomes possible | Verify with other data |
| Spatially varying | Edges/transitions uncertain | Focus on high-uncertainty regions |

---

### Logging Settings

| Parameter | Default | Description | Effect of Changing |
|-----------|---------|-------------|-------------------|
| `progress_bar` | `true` | Show tqdm progress bars | `false`: Cleaner logs for batch jobs |
| `log_interval` | `10` | Log every N batches | Only used if `progress_bar: false` |

---

### Quick Configuration Recipes

**For Solar Flare Detection (Recommended):**
```yaml
normalization:
  method: "asinh"
  asinh_softening: 1000.0
data:
  dual_channel: true
model:
  input_channels: 2
loss:
  type: "composite"
  extreme_weight: 2.0
```

**For Fast Prototyping:**
```yaml
model:
  channels: [8, 16, 32]
training:
  epochs: 10
  patience: 5
```

**For Maximum Accuracy (Large GPU):**
```yaml
model:
  channels: [32, 64, 128]
  downsample_input: false
training:
  epochs: 50
  patience: 15
loss:
  type: "composite"
  ssim_weight: 1.0
```

**For Google Colab:**
```yaml
training:
  num_workers: 0
  use_amp: true
```

---

## Usage

### Prerequisites

```bash
pip install torch numpy pyyaml matplotlib tqdm
```

### Training

1. Place your `.npy` files in the `data/` directory

2. Adjust `config.yaml` if needed (especially `data_dir`)

3. Run training:

```bash
python main.py
```

Or with a custom config:

```bash
python main.py --config my_config.yaml
```

### Inference

```python
from main import load_config, run_inference

config = load_config("config.yaml")
model = run_inference(config, "outputs/best_model.pt")

# model is now ready for predictions
# input shape: (batch, 1, t_in, H, W)
# output shape: (batch, 1, t_out, H, W)
```

### Example: Making Predictions

```python
import torch
from models import SolarFluxPredictor
from training.trainer import load_checkpoint
from utils import get_device

# Setup
config = load_config("config.yaml")
device = get_device(config['device']['use_cuda'])

# Create and load model
model = SolarFluxPredictor(
    t_out=3,
    channels=[16, 32, 64],
    downsample_input=True
).to(device)

load_checkpoint(model, "outputs/best_model.pt", device)
model.eval()

# Prepare input (8 frames of your data)
# Shape: (1, 1, 8, H, W) - batch, channel, time, height, width
input_sequence = torch.randn(1, 1, 8, 220, 442).to(device)

# Predict
with torch.no_grad():
    predictions = model(input_sequence)
    # predictions shape: (1, 1, 3, H, W)
```

---

## Memory & Performance

### GPU Memory Usage

With default settings (RTX 3050, 8GB VRAM):

| Component | Memory |
|-----------|--------|
| Model parameters | ~2 MB |
| Input batch | ~16 MB |
| Forward pass activations | ~1-2 GB |
| Backward pass gradients | ~1-2 GB |
| **Total** | **~3-4 GB** |

### Optimizations Enabled

1. **Input Downsampling**: Reduces spatial dims by 2x
2. **Reduced Channels**: [16, 32, 64] instead of [32, 64, 128]
3. **Mixed Precision (AMP)**: Uses float16 where safe
4. **Gradient Checkpointing**: Can be added if needed

### Tuning for Different GPUs

**4GB VRAM** (e.g., GTX 1650):
```yaml
model:
  channels: [8, 16, 32]
  downsample_input: true
training:
  use_amp: true
```

**8GB VRAM** (e.g., RTX 3050, 3060):
```yaml
model:
  channels: [16, 32, 64]
  downsample_input: true
training:
  use_amp: true
```

**16GB+ VRAM** (e.g., RTX 4080):
```yaml
model:
  channels: [32, 64, 128]
  downsample_input: false
training:
  batch_size: 2
```

### Training Time Estimates

On RTX 3050 with default settings:
- ~2 minutes per epoch
- ~25 epochs = ~50 minutes total
- Early stopping may finish sooner

---

## Understanding the Output

### Training Progress

```
Epoch 1/25 | LR: 1.00e-03 | TF: 0.48
Epoch 1: 100%|██████████| 150/150 [02:15<00:00]
  Train Loss: 0.152340
  Val Loss:   0.143210
  Val MAE:    [0.12, 0.14, 0.16]  ← MAE for each predicted timestep
  ✓ Saved best model
```

### Predictions Visualization

The `predictions.png` shows:
- **Column 1**: Last input frame (what the model saw)
- **Columns 2-4**: Predicted frames (t+1, t+2, t+3)
- **Column 5**: Ground truth (what actually happened)

### Metrics

- **MAE (Mean Absolute Error)**: Average pixel-wise difference
  - Lower is better
  - In normalized units (divide by scale to get original units)

- **Per-timestep MAE**: How error grows with prediction horizon
  - Expect increasing error for further predictions
  - e.g., [0.12, 0.14, 0.16] means t+3 is hardest to predict

---

## Troubleshooting

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

Solutions:
1. Enable `downsample_input: true`
2. Reduce channels: `[8, 16, 32]`
3. Enable AMP: `use_amp: true`
4. Use CPU: `use_cuda: false` (slow but works)

### Training Loss Not Decreasing

1. Check normalization - values should be roughly in [-1, 1]
2. Reduce learning rate: `lr: 0.0001`
3. Increase teacher forcing: `tf_start: 0.8`

### NaN Losses

1. Reduce learning rate
2. Enable gradient clipping: `grad_clip: 0.5`
3. Check for extreme values in your data

---

## Recent Improvements

### Asinh Normalization (Recommended)

Solar flux data has extreme dynamic range (values from -36M to +70M). The previous "robust" normalization clipped extreme values, losing critical solar flare signals.

**Asinh transformation** preserves extreme values:

```yaml
normalization:
  method: "asinh"
  asinh_softening: 1000.0  # Higher = less compression
```

Benefits:
- **No clipping**: Extreme values are compressed but preserved
- **Symmetric**: Handles negative values naturally
- **Standard in astrophysics**: Used for wide dynamic range data

### Dual-Channel Input Mode

Enable dual-channel mode to help the model focus on both background activity and extreme events:

```yaml
data:
  dual_channel: true

model:
  input_channels: 2  # Channel 1: flux, Channel 2: extreme indicator
  output_channels: 1  # Predict flux only
```

- **Channel 1**: Asinh-normalized flux values
- **Channel 2**: Soft extreme event indicator (sigmoid around threshold)

### Composite Loss Function

The new composite loss combines multiple objectives for better predictions:

```yaml
loss:
  type: "composite"
  l1_weight: 1.0      # Basic reconstruction
  ssim_weight: 0.5    # Structural similarity (sharper predictions)
  extreme_weight: 1.0 # Higher weight for extreme values
```

Components:
- **L1 (MAE)**: Basic reconstruction loss
- **MS-SSIM**: Multi-scale structural similarity for sharp edges
- **Weighted MAE**: Extra penalty for errors in high-flux regions

---

## Gradient Checkpointing

Save ~40% GPU memory by recomputing activations during backward pass:

```yaml
model:
  use_checkpointing: true  # Enable memory savings
```

**Trade-offs:**
- Saves ~40% VRAM
- ~20% slower training
- No change to inference

**When to use:**
- Training larger models on limited VRAM
- Increasing channel sizes: `[32, 64, 128]`

---

## Uncertainty Quantification (MC Dropout)

Get confidence estimates for predictions using Monte Carlo Dropout:

### Configuration

```yaml
model:
  dropout_rate: 0.1  # Enable dropout (required for UQ)

uncertainty:
  enabled: true       # Enable UQ during evaluation
  n_samples: 20       # More samples = better estimate, slower
  save_uncertainty_maps: true
```

### How It Works

1. During training, dropout regularizes the model
2. During inference, dropout is kept ON
3. Multiple forward passes produce a distribution
4. Standard deviation = model uncertainty

### Usage in Code

```python
from models import SolarFluxPredictor, predict_with_uncertainty

model = SolarFluxPredictor(dropout_rate=0.1, ...)
# ... train model ...

# Get predictions with uncertainty
mean_pred, uncertainty = predict_with_uncertainty(model, x, n_samples=20)

# High uncertainty = model is less confident
```

### Interpreting Results

- **Low uncertainty**: Model is confident in prediction
- **High uncertainty**: Multiple possible outcomes, proceed with caution
- **Spatially varying**: Shows where model is unsure (often at flare edges)

---

## Animation Tools

Create animated visualizations of solar flare evolution:

### Command Line

```bash
# Create MP4 video
python visualize_flares.py --cube data_processed/cube_005.npz --output flare.mp4 --fps 10

# Create interactive HTML viewer (requires plotly)
python visualize_flares.py --cube data_processed/cube_005.npz --format html

# Animate specific frames
python visualize_flares.py --cube data_processed/cube_005.npz --start 10 --end 60 --fps 5
```

### Python API

```python
from utils.animation import (
    animate_flare_sequence,
    interactive_flare_viewer,
    animate_prediction_vs_truth
)

# Load data
import numpy as np
data = np.load('data_processed/cube_005.npz')
flux = data['flux']

# Create MP4
animate_flare_sequence(flux, 'evolution.mp4', fps=10)

# Create interactive HTML
interactive_flare_viewer(flux, output_path='viewer.html')

# Compare prediction vs ground truth
animate_prediction_vs_truth(model, test_dataset, device, output_path='comparison.mp4')
```

### Visualization Options

| Function | Output | Use Case |
|----------|--------|----------|
| `animate_flare_sequence` | MP4/GIF | General evolution video |
| `interactive_flare_viewer` | HTML | Interactive exploration |
| `animate_prediction_vs_truth` | MP4 | Model evaluation |
| `animate_with_uncertainty` | MP4 | Uncertainty visualization |
| `create_difference_animation` | MP4 | Error analysis |

---

## Future Improvements (Planned)

The following improvements are planned for future releases:

### Training Optimizations

1. **Gradient Accumulation**
   - Simulate larger batch sizes (e.g., 16) with batch_size=1
   - More stable gradients, better convergence

2. **OneCycleLR Scheduler**
   - Replace CosineAnnealingLR for faster convergence
   - Super-convergence with cyclical learning rates

### Architecture Enhancements

3. **Attention Mechanism**
   - Spatial attention to focus on active regions
   - Better prediction of localized solar flares

### Advanced Loss Functions

4. **Gradient Difference Loss (GDL)**
   - Penalize blurry edges explicitly
   - Sharper spatial boundaries

### Advanced Architectures

5. **PredRNN-V2 Cell Replacement**
   - Memory decoupling for better long-term dynamics
   - Reverse Scheduled Sampling (RSS)

---

## Preprocessing with New Normalization

To use the new asinh normalization, re-preprocess your data:

```bash
python preprocess_data.py --input ./data --output ./data_processed --method asinh --softening 1000.0
```

This will:
1. Apply asinh transformation to preserve extreme values
2. Compute extreme event threshold for dual-channel mode
3. Save normalized cubes and metadata

---

## License

MIT License - feel free to use and modify for your research.

## References

- Shi, X., et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
- Wang, Y., et al. (2022). "PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning"
- PyTorch Documentation: https://pytorch.org/docs/stable/

---

*Built for solar physics research. Contributions welcome!*

