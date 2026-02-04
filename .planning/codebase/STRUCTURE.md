# Codebase Structure

**Analysis Date:** 2026-02-02

## Directory Layout

```
SolarFlare/
├── main.py                   # Primary entry point: training pipeline
├── inference.py              # Inference script: load model and predict
├── visualize_flares.py       # Visualization script: create animations
├── config.yaml               # Training configuration (model, data, loss, etc)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── improvements.md           # Development notes and improvements log
├── ConvLSTM.py              # Legacy ConvLSTM implementation (deprecated)
├── preprocess_data.py       # Data preprocessing utilities
│
├── models/                   # Model architectures and components
│   ├── __init__.py          # Exports: SolarFluxPredictor, ConvLSTM, uncertainty functions
│   ├── predictor.py         # SolarFluxPredictor: main encoder-decoder model
│   ├── convlstm.py          # ConvLSTMCell and ConvLSTM layer implementations
│   └── uncertainty.py        # MC Dropout uncertainty quantification
│
├── solarflare_data/         # Data loading and preprocessing
│   ├── __init__.py          # Exports: SolarFluxDataset, load_and_prepare_data, load_preprocessed_data
│   ├── dataset.py           # SolarFluxDataset: PyTorch Dataset with sliding windows
│   └── loader.py            # Data loading, normalization (asinh/robust/fixed), train/val/test split
│
├── training/                # Training loop and loss functions
│   ├── __init__.py          # Exports: train_model, train_epoch, validate, loss functions
│   ├── trainer.py           # train_epoch(), validate(), train_model(), load_checkpoint()
│   └── losses.py            # L1, MS-SSIM, weighted MAE, composite loss functions
│
├── utils/                   # Utility functions and helpers
│   ├── __init__.py          # Exports: device, metrics, visualization, animation utilities
│   ├── device.py            # get_device(), get_amp_context(), get_grad_scaler()
│   ├── metrics.py           # compute_metrics(): MAE per timestep, MSE, etc
│   ├── visualization.py     # visualize_predictions(), plot_training_history(), uncertainty visualization
│   └── animation.py         # animate_flare_sequence(), animate_prediction_vs_truth()
│
├── data/                    # Raw input data directory
│   └── *.npy               # Solar flux time series files (loaded by loader.py)
│
├── data_processed/         # Preprocessed cube cache for faster loading
│   └── *.npz               # Preprocessed flux cubes with metadata
│
├── outputs/                # Training outputs (generated)
│   ├── best_model.pt       # Best model checkpoint (model + optimizer state)
│   ├── metadata.json       # Normalization parameters and data statistics
│   ├── training_history.json  # Loss/MAE per epoch for analysis
│   ├── test_results.json   # Test loss and MAE per timestep
│   ├── training_history.png    # Loss curves visualization
│   ├── predictions.png     # Sample predictions vs ground truth
│   └── uncertainty_sample_*.png # MC Dropout uncertainty maps (if enabled)
│
└── .planning/              # GSD planning documents
    └── codebase/           # Codebase analysis documents
        ├── ARCHITECTURE.md # Architecture patterns and data flow
        └── STRUCTURE.md    # This file
```

## Directory Purposes

**models/**
- Purpose: Model architecture definitions for solar flux prediction
- Contains: Encoder-decoder with ConvLSTM, uncertainty quantification
- Key files: `models/predictor.py` (main model), `models/convlstm.py` (core cell)

**solarflare_data/**
- Purpose: Data pipeline from raw files to training-ready datasets
- Contains: Data loaders, normalization (asinh/robust/fixed), dataset classes
- Key files: `solarflare_data/loader.py` (loading & preprocessing), `solarflare_data/dataset.py` (PyTorch Dataset)

**training/**
- Purpose: Training loop, validation, and loss functions
- Contains: Epoch training, validation, early stopping, checkpointing
- Key files: `training/trainer.py` (training loop), `training/losses.py` (loss functions)

**utils/**
- Purpose: Shared utilities for device management, metrics, and visualization
- Contains: GPU/CPU device setup, AMP context managers, metric computation, visualization helpers
- Key files: `utils/device.py` (device/AMP), `utils/visualization.py` (plotting)

**data/**
- Purpose: Raw solar flux data storage
- Contains: `.npy` files with time series data (shape: T × H × W)
- Key files: `windTotal*.npy` pattern preferred

**data_processed/**
- Purpose: Cache of preprocessed data cubes for faster training startup
- Contains: `.npz` files with normalized flux cubes and metadata
- Key files: Generated during first run if `use_preprocessed: true` in config

**outputs/**
- Purpose: Store training artifacts and results
- Contains: Model checkpoints, training history, test metrics, visualizations
- Key files: `best_model.pt` (model weights), `metadata.json` (normalization params)

## Key File Locations

**Entry Points:**
- `main.py`: Primary training entry point (data loading → model creation → training → testing → visualization)
- `inference.py`: Standalone inference script for making predictions on new data
- `visualize_flares.py`: Post-training animation generation

**Configuration:**
- `config.yaml`: All hyperparameters, paths, and training options (YAML format)

**Core Logic:**
- `models/predictor.py`: `SolarFluxPredictor` class - encoder-decoder architecture with autoregressive decoding
- `models/convlstm.py`: `ConvLSTMCell` - spatiotemporal feature extraction via convolution + LSTM gates
- `training/trainer.py`: `train_model()` - main training loop with early stopping and checkpointing
- `solarflare_data/loader.py`: `load_and_prepare_data()` - loads .npy files, normalizes, creates datasets

**Testing:**
- No dedicated test directory; validation occurs during training via `training/trainer.py:validate()`
- Test set created by `load_and_prepare_data()` (last `1 - train_split - val_split` fraction)

## Naming Conventions

**Files:**
- `*.py`: Python source modules
- `config.yaml`: YAML configuration
- `*.npy`: NumPy binary arrays (raw data)
- `*.npz`: NumPy compressed archives (preprocessed data)
- `*.pt`: PyTorch checkpoint files
- `*.json`: Metadata and logs
- `*.png`: Visualization images

**Directories:**
- Lowercase snake_case: `solarflare_data`, `data_processed`, `training`
- Common names: `models`, `utils`, `data`, `outputs`

**Python Modules:**
- Classes: PascalCase (e.g., `SolarFluxPredictor`, `ConvLSTMCell`, `SolarFluxDataset`)
- Functions: snake_case (e.g., `load_and_prepare_data`, `train_epoch`, `visualize_predictions`)
- Constants: UPPER_CASE (e.g., in config: `train_split`, used directly as-is in code)

**Variables:**
- Model state: `model.state_dict()`, `model.eval()`, `model.train()`
- Tensors: Shape notation in comments (e.g., `(B, C, T, H, W)` for batch, channels, time, height, width)
- Loss/metrics: `train_loss`, `val_loss`, `mae_per_timestep`
- Data: `X_in` (input), `Y_out` (output), `Y_target` (target for loss)

## Where to Add New Code

**New Feature (e.g., skip connections, attention mechanism):**
- Primary code: Modify `models/predictor.py` (update `SolarFluxPredictor.__init__()` and `forward()`)
- Tests: Add validation code in `main.py:run_training()` (e.g., test with/without feature on small subset)
- Configuration: Add config options to `config.yaml`

**New Loss Function:**
- Implementation: Add function to `training/losses.py`
- Export: Update `training/__init__.py` to export function
- Configuration: Register in `training/losses.py:get_loss_function()` dispatcher
- Usage: Reference in `config.yaml` `loss.type` and `main.py:train_config`

**New Normalization Method:**
- Implementation: Add to `solarflare_data/loader.py` (extend `_normalize()` and `_unnormalize()` functions)
- Registration: Update dispatcher in `load_and_prepare_data()` to handle new method name
- Configuration: Add option to `config.yaml` `normalization.method`

**New Visualization:**
- Implementation: Add function to `utils/visualization.py`
- Export: Update `utils/__init__.py`
- Usage: Call from `main.py:run_training()` after training complete

**New Utility Function:**
- For device/AMP: Add to `utils/device.py`
- For metrics: Add to `utils/metrics.py`
- For animation: Add to `utils/animation.py`
- Export: Update `utils/__init__.py`

**New Data Augmentation:**
- Implementation: Add to `solarflare_data/dataset.py:__getitem__()` method (apply consistent to X_in and Y_out)
- Configuration: Add toggle to `config.yaml` `data.augment`

## Special Directories

**outputs/:**
- Purpose: Training artifacts and results
- Generated: Yes (created by `main.py` and `training/trainer.py`)
- Committed: No (in `.gitignore`)
- Clean up: `rm -rf outputs/*` to restart training fresh

**data_processed/:**
- Purpose: Cache of preprocessed numpy cubes for faster reloading
- Generated: Yes (created on first run if `use_preprocessed: true`)
- Committed: No (in `.gitignore`)
- Refresh: Delete to force reprocessing from raw data files

**data/**
- Purpose: Raw solar flux data files
- Generated: No (user-provided)
- Committed: No (in `.gitignore` - files too large)
- Location: Referenced in `config.yaml` `data.data_dir`

**__pycache__/**
- Purpose: Python bytecode cache
- Generated: Yes (automatic)
- Committed: No (in `.gitignore`)
- Clean up: Python clears automatically; manually delete if needed

**.git/**
- Purpose: Git version control
- Generated: N/A
- Committed: N/A

**Notebooks (*.ipynb):**
- Purpose: Development and exploration (SolarFlare_Colab.ipynb, SolarFlare_Colab_v2.ipynb)
- Generated: No (checked in)
- Committed: Yes
- Role: Reference implementations and interactive development; not part of production pipeline
