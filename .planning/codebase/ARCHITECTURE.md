# Architecture

**Analysis Date:** 2026-02-02

## Pattern Overview

**Overall:** Encoder-Decoder with Spatiotemporal Convolutional LSTM

**Key Characteristics:**
- Autoregressive sequence-to-sequence prediction for spatiotemporal data (solar flux maps)
- Encoder processes input sequences; decoder generates future frames autoregressively
- ConvLSTM cells at multiple resolution levels for hierarchical feature learning
- Teacher forcing during training with scheduled decay to zero-forcing at test time
- Residual predictions to improve training stability

## Layers

**Data Loading Layer:**
- Purpose: Load, normalize, and prepare solar flux data for training
- Location: `solarflare_data/loader.py`, `solarflare_data/dataset.py`
- Contains: Data loading from `.npy` files, normalization (asinh/robust/fixed), train/val/test splitting, data augmentation
- Depends on: NumPy, PyTorch Dataset utilities
- Used by: Training pipeline via `main.py`

**Model Architecture Layer:**
- Purpose: Define and instantiate the core prediction model (ConvLSTM-based encoder-decoder)
- Location: `models/predictor.py`, `models/convlstm.py`
- Contains: `SolarFluxPredictor` (main encoder-decoder), `ConvLSTM` cells with multi-scale processing
- Depends on: PyTorch nn modules, gradient checkpointing utilities
- Used by: Training and inference pipelines

**Training Layer:**
- Purpose: Implement training loops, validation, loss computation, and checkpointing
- Location: `training/trainer.py`, `training/losses.py`
- Contains: `train_epoch()`, `validate()`, `train_model()` functions; composite loss functions (L1, MS-SSIM, weighted MAE)
- Depends on: Model, data loaders, optimizers, device management utilities
- Used by: `main.py` entry point

**Inference Layer:**
- Purpose: Load trained models and generate predictions on new data
- Location: `inference.py`, `models/uncertainty.py`
- Contains: Model loading, normalization/denormalization, prediction functions, MC Dropout uncertainty estimation
- Depends on: Trained checkpoints, normalization metadata
- Used by: Standalone inference scripts and `main.py` inference mode

**Visualization & Utilities Layer:**
- Purpose: Provide supporting tools for device management, metrics, visualization, and animations
- Location: `utils/device.py`, `utils/metrics.py`, `utils/visualization.py`, `utils/animation.py`
- Contains: Device/AMP setup, MAE/MSE metrics, prediction visualization, GIF animations
- Depends on: PyTorch, NumPy, Matplotlib, PIL
- Used by: Training pipeline and post-training analysis

## Data Flow

**Training Pipeline:**

1. **Configuration Loading** (`main.py:load_config`)
   - Read `config.yaml` with model, training, data, and normalization parameters

2. **Data Preparation** (`main.py:run_training`)
   - Call `load_and_prepare_data()` or `load_preprocessed_data()` from `solarflare_data/loader.py`
   - Load raw `.npy` files containing solar flux time series (shape: T × H × W)
   - Normalize using configured method (asinh, robust, or fixed scaling)
   - Create `SolarFluxDataset` instances with sliding windows (t_in → t_out)
   - Optional: Enable dual-channel mode (flux + extreme indicator)
   - Optional: Apply data augmentation (random flips)
   - Split into train/val/test loaders

3. **Model Creation** (`main.py:36-107`)
   - Instantiate `SolarFluxPredictor` with architecture from config
   - Optional: Enable gradient checkpointing for memory efficiency
   - Optional: Enable dropout for MC Dropout uncertainty

4. **Training Loop** (`training/trainer.py:train_model`)
   - For each epoch:
     - Decay teacher forcing ratio linearly from `tf_start` to 0
     - Update learning rate via scheduler (cosine/step/constant)
     - **Train phase**: Forward pass with teacher forcing, compute loss, backward pass with gradient clipping
     - **Validation phase**: No teacher forcing, compute metrics
     - **Checkpointing**: Save best model based on validation loss
     - **Early stopping**: Stop if no improvement for `patience` epochs

5. **Testing & Visualization** (`main.py:142-184`)
   - Load best checkpoint
   - Evaluate on test set
   - Generate prediction visualizations (3 test samples)
   - Optionally: Run MC Dropout for uncertainty quantification

**Inference Pipeline:**

1. Load trained model checkpoint with configuration
2. Load new input data (normalized to same statistics as training)
3. Forward pass with `teacher_forcing_ratio=0.0` (pure autoregressive)
4. Denormalize predictions to original flux values
5. Optional: Run MC Dropout multiple times for uncertainty bounds

**State Management:**

- **Model State**: Saved in PyTorch checkpoint (model weights, optimizer state, epoch, validation loss)
- **Data State**: Normalization parameters stored in `metadata.json` for consistent preprocessing
- **Training State**: Loss history and metrics logged in `training_history.json`
- **Device State**: Managed via `get_device()` and `get_amp_context()` utilities

## Key Abstractions

**SolarFluxPredictor:**
- Purpose: Main spatiotemporal prediction model combining encoder-decoder with ConvLSTM
- Examples: `models/predictor.py:21-260`
- Pattern: Encoder processes all input timesteps to extract features → Decoder autoregressively generates output timesteps using previous predictions (or teacher forcing)

**ConvLSTM Cell:**
- Purpose: Spatiotemporal sequence modeling using convolutions instead of fully connected layers
- Examples: `models/convlstm.py:12-100`
- Pattern: Combines spatial convolution with LSTM gating mechanism. Single Conv2d computes all four gates (input, forget, cell, output) concatenating input and hidden state

**SolarFluxDataset:**
- Purpose: PyTorch Dataset providing sliding window samples with optional augmentation
- Examples: `solarflare_data/dataset.py:13-102`
- Pattern: Stores list of (dataset_id, start_idx) samples; `__getitem__` extracts window, applies augmentations, converts to tensors

**Loss Functions:**
- Purpose: Multiple loss components for robust spatiotemporal prediction
- Examples: `training/losses.py:24-150`
- Pattern: L1 for reconstruction, MS-SSIM for structural similarity, weighted MAE emphasizing extreme values

**Normalization Strategies:**
- Purpose: Handle solar flux distribution with wide dynamic range and extreme values
- Examples: `solarflare_data/loader.py:150-220`
- Pattern: Asinh (recommended) uses `sinh^-1(x/softening)` to compress extreme values; robust uses percentile clipping; fixed uses manual scaling factor

## Entry Points

**Training Entry Point:**
- Location: `main.py:run_training(config)`
- Triggers: `python main.py` or `python main.py --config custom.yaml`
- Responsibilities: Load config, prepare data, create model, run training loop, test, visualize, optionally estimate uncertainty

**Inference Entry Point:**
- Location: `inference.py:load_model(), predict()`
- Triggers: Direct function calls or `python inference.py` as standalone script
- Responsibilities: Load checkpoint, initialize model, run forward passes with new data, return predictions in original scale

**Visualization Entry Point:**
- Location: `visualize_flares.py`
- Triggers: `python visualize_flares.py` after training
- Responsibilities: Create animations of predictions vs ground truth

## Error Handling

**Strategy:** Exceptions propagate with informative messages

**Patterns:**
- Data loading: FileNotFoundError if `.npy` files missing; ValueError if no datasets loaded successfully
- Normalization: Check for valid normalization method; use defaults if config missing
- Model: Warn if dropout_rate=0 when uncertainty requested; validate checkpoint compatibility
- Training: Early stopping on patience exhaustion; gradient clipping prevents explosion
- Device: Graceful fallback to CPU if CUDA unavailable despite config request

## Cross-Cutting Concerns

**Logging:**
- Standard print statements with section dividers (═══)
- Progress bars via tqdm for training/validation iterations
- Loss/metric values logged after each epoch
- Saved training history in JSON for post-analysis

**Validation:**
- Input shapes validated implicitly by PyTorch (shape mismatches raise runtime errors)
- Normalization parameters saved and validated at inference time
- Metadata JSON contains value statistics for data quality checks

**Authentication:**
- Not applicable (local training only)

**Device Management:**
- `get_device()` detects CUDA availability and prints GPU info
- `get_amp_context()` returns appropriate autocast context (cuda/cpu/none)
- `get_grad_scaler()` provides gradient scaling for mixed precision training
