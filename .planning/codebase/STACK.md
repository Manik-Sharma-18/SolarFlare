# Technology Stack

**Analysis Date:** 2026-02-02

## Languages

**Primary:**
- Python 3.9.6 - All application code, model training, inference, preprocessing

## Runtime

**Environment:**
- Python 3.9.6
- CUDA-enabled (optional, CPU fallback supported)

**Package Manager:**
- pip
- Lockfile: requirements.txt (present, minimal)

## Frameworks

**Core ML:**
- PyTorch >= 2.0.0 - Deep learning framework, ConvLSTM model, training/inference
- NumPy >= 1.21.0 - Numerical operations, array manipulation, normalization

**Data Processing:**
- PyTorch DataLoader - Batching, shuffling, multi-worker loading (in `solarflare_data/loader.py`)
- Custom Dataset classes - Sliding window sampling, augmentation (in `solarflare_data/dataset.py`)

**Visualization:**
- Matplotlib >= 3.5.0 - Training plots, prediction visualizations, animations
  - Used in: `utils/visualization.py`, `utils/animation.py`

**Configuration:**
- PyYAML >= 6.0 - YAML config parsing (config.yaml)

**Utilities:**
- tqdm >= 4.64.0 - Progress bars for training and data processing
- pathlib - Cross-platform file path handling
- json - Metadata and results serialization

## Key Dependencies

**Critical:**
- torch >= 2.0.0 - Entire model architecture, training loop, inference
  - ConvLSTM cells, encoder-decoder, attention mechanisms
  - Automatic mixed precision (AMP) support
  - Gradient checkpointing for memory efficiency
- numpy >= 1.21.0 - Data normalization (asinh, robust, fixed methods), statistics

**Infrastructure:**
- matplotlib >= 3.5.0 - Training history plots, heatmaps, uncertainty visualizations
- pyyaml >= 6.0 - Configuration management
- tqdm >= 4.64.0 - Training/inference progress indication

## Configuration

**Environment:**
- YAML-based configuration: `config.yaml`
- Config sections:
  - `device`: CUDA enable/disable flag
  - `data`: Directory paths, preprocessing params (t_in, t_out, splits, augmentation)
  - `normalization`: Method selection (asinh, robust, fixed) with parameters
  - `model`: Architecture params (input/output channels, layer sizes, kernel size)
  - `training`: Batch size, learning rate, epochs, scheduler type, dropout
  - `loss`: Loss function type (l1, composite, weighted) with weights
  - `output`: Checkpoint and visualization save paths
  - `logging`: Progress bar and logging settings
  - `uncertainty`: MC Dropout configuration for uncertainty quantification

**Key Configurations:**
- `config['device']['use_cuda']` - GPU support toggle
- `config['data']['use_preprocessed']` - Fast loading from cached .npz files
- `config['data']['dual_channel']` - Enable extreme event detection channel
- `config['model']['use_checkpointing']` - Memory-saving gradient checkpointing
- `config['model']['dropout_rate']` - MC Dropout for uncertainty

## Build/Dev

- No build system (pure Python, no compilation)
- Development: Direct Python execution
- Entry points:
  - `main.py` - Training pipeline (loads config, trains, evaluates, visualizes)
  - `inference.py` - Model loading and prediction on new data
  - `visualize_flares.py` - Interactive visualization and animation
  - `preprocess_data.py` - Data conversion from structured arrays to dense cubes

## Hardware Requirements

**Development/Training:**
- GPU strongly recommended (NVIDIA CUDA with torch support)
  - RTX 3050+ for 1-hour training runs (batch_size=1, 25 epochs)
  - CPU fallback supported but slow
- RAM: 8GB+ (large spatial dimensions, 256x256+ typical)
- Storage: Raw data (several GB), preprocessed cubes, model checkpoints

**Production/Inference:**
- GPU recommended for real-time predictions
- CPU inference possible with reduced throughput

## Platform Requirements

**Development:**
- macOS, Linux, or Windows with Python 3.9+
- CUDA Toolkit (for GPU support)
- cudnn (if using CUDA)

**Production:**
- Deployment target: Any machine with PyTorch and NumPy
- Model inference runs standalone with trained checkpoint

---

*Stack analysis: 2026-02-02*
