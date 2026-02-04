# Coding Conventions

**Analysis Date:** 2026-02-02

## Naming Patterns

**Files:**
- Module files: `lowercase_with_underscores.py`
  - Examples: `trainer.py`, `convlstm.py`, `visualization.py`, `metrics.py`
- Entry point files: `main.py`, `inference.py`, `preprocess_data.py`, `visualize_flares.py`
- Package files: `__init__.py` at package roots
- Configuration file: `config.yaml` (single configuration file for the project)

**Functions:**
- Use `snake_case` for function names
  - Examples: `load_config()`, `run_training()`, `compute_metrics()`, `train_epoch()`, `load_and_prepare_data()`
- Private functions prefixed with single underscore: `_init_hidden()`, `_structured_to_cube()`, `_compute_norm_params()`, `_DummyGradScaler`
- All functions include comprehensive docstrings

**Variables:**
- Use `snake_case` for variables and parameters
  - Examples: `teacher_forcing_ratio`, `grad_clip`, `best_val_loss`, `patience_counter`, `t_in`, `t_out`
- Loop variables: `i`, `f` for simple iteration; descriptive names for complex loops
- Configuration keys: `lowercase_with_underscores` (e.g., `use_cuda`, `batch_size`, `dropout_rate`)

**Classes:**
- Use `PascalCase` for class names
  - Examples: `SolarFluxPredictor`, `ConvLSTMCell`, `ConvLSTM`, `SolarFluxDataset`, `WeightedMAELoss`, `CompositeLoss`, `_DummyGradScaler`
- Private classes: `_DummyGradScaler`

**Constants:**
- Module-level configuration uses YAML (`config.yaml`)
- Hardcoded constants in code use UPPER_CASE
  - Example: `C1 = (0.01 * data_range) ** 2` in `losses.py`

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc`, `pyproject.toml`, or `.style.yapf`)
- Appears to follow PEP 8 style conventions
- Line length: ~90 characters (inferred from code patterns)
- Indentation: 4 spaces
- Two blank lines between module-level functions and classes
- One blank line between class methods

**Linting:**
- No linting config file detected (no `.pylintrc`, `pyproject.toml`, or `setup.cfg`)
- No active linting tool configured in requirements
- Code follows implicit PEP 8 standards

**Import Organization:**

Order observed across all modules:
1. Standard library imports: `import sys`, `import json`, `from pathlib import Path`
2. Third-party imports: `import torch`, `import numpy as np`, `import yaml`, `from tqdm import tqdm`
3. Local package imports: `from models import SolarFluxPredictor`, `from .dataset import SolarFluxDataset`

Example from `main.py`:
```python
import sys
import json
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SolarFluxPredictor
from solarflare_data import load_and_prepare_data, load_preprocessed_data
from training import train_model, validate
```

**Path Aliases:**
- No formal alias configuration; uses `sys.path.insert(0, str(Path(__file__).parent))` in `main.py` to enable root-relative imports
- Relative imports used within packages: `from .dataset import SolarFluxDataset`, `from .losses import get_loss_function`

## Error Handling

**Strategy:** Explicit exception raising with descriptive messages. No broad exception suppression.

**Patterns:**

1. **File/Directory validation:**
   - Always check existence before loading
   - Raise `FileNotFoundError` with full path message
   - Example from `loader.py`:
   ```python
   if not data_path.exists():
       raise FileNotFoundError(f"Data directory not found: {data_path}")
   ```

2. **Configuration validation:**
   - Use `ValueError` for invalid config values
   - Example from `trainer.py`:
   ```python
   if scheduler_type == 'cosine':
       # ...
   elif scheduler_type == 'step':
       # ...
   else:
       raise ValueError(f"Unknown scheduler type: {scheduler_type}")
   ```

3. **Data loading with graceful degradation:**
   - Catch exceptions during file loading, log, and continue
   - Example from `loader.py`:
   ```python
   for file_path in npy_files:
       try:
           data = np.load(file_path)
           # ...
       except Exception as e:
           print(f"  Error: {e}")
           continue
   ```

4. **Post-operation validation:**
   - Validate results after bulk operations
   - Example from `loader.py`:
   ```python
   if len(datasets) == 0:
       raise ValueError("No datasets loaded successfully")
   ```

5. **Optional parameter with defaults:**
   - Use `None` as sentinel, check, then provide default
   - Example from `trainer.py`:
   ```python
   if loss_fn is None:
       loss_fn = nn.L1Loss()
   ```

## Logging

**Framework:** `print()` statements and optional progress bars via `tqdm`

**Patterns:**

1. **Section headers:**
   ```python
   print("\n" + "=" * 60)
   print("SECTION NAME")
   print("=" * 60)
   ```

2. **Informational messages:**
   ```python
   print(f"Found {len(npy_files)} data files:")
   for f in npy_files:
       print(f"  {f.name}")
   ```

3. **Progress tracking with tqdm:**
   ```python
   iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if show_progress else dataloader
   for X_in, Y_out, _ in iterator:
       # ...
       if show_progress:
           iterator.set_postfix({'loss': f'{loss.item():.6f}'})
   ```

4. **Device and config info:**
   ```python
   print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   print(f"Total trainable parameters: {total_params:,}")
   ```

5. **Numeric formatting:**
   - Loss values: `.6f` (6 decimal places)
   - Learning rate: `.2e` (scientific notation)
   - Teacher forcing: `.3f` (3 decimal places)

## Comments

**When to Comment:**
- Document module purpose at the top with docstring
- Do not repeat code in comments; only explain *why* when non-obvious
- Explain complex algorithms: "Teacher forcing schedule: linear decay" with formula

**JSDoc/TSDoc:**
- Comprehensive docstrings for all public functions and classes
- Format: Google-style docstrings
- Always include `Args:`, `Returns:`, and descriptions

**Example from `trainer.py`:**
```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    # ... more params ...
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        # ... more args ...

    Returns:
        Average loss for the epoch
    """
```

## Function Design

**Size:**
- Keep functions focused on single task
- Typical range: 20-80 lines
- Longer functions (trainer.py): `train_model()` ~160 lines, justified by complex logic with multiple stages

**Parameters:**
- Use type hints on all parameters and returns
- Maximum 8-10 parameters; complex state passed via config dicts
- Example from `train_epoch()`:
  ```python
  def train_epoch(
      model: nn.Module,
      dataloader: DataLoader,
      optimizer: torch.optim.Optimizer,
      scaler,
      device: torch.device,
      teacher_forcing_ratio: float,
      epoch: int,
      loss_fn: Optional[nn.Module] = None,
      use_amp: bool = True,
      grad_clip: float = 1.0,
      show_progress: bool = True,
      output_channels: int = 1
  ) -> float:
  ```

**Return Values:**
- Single return value preferred (either scalar or tuple)
- Return tuples for related values: `return avg_loss, avg_mae_per_timestep`
- Dict returns for multiple components: `return {'total': total_loss, 'l1': l1_loss, ...}`
- Return configs/metadata as dicts: `return checkpoint` with nested keys

## Module Design

**Exports:**
- Public functions and classes at module level
- Use `from .module import Class` to expose in `__init__.py`
- Example from `training/__init__.py`:
  ```python
  from .trainer import train_model, validate
  from .losses import get_loss_function
  ```

**Barrel Files:**
- `__init__.py` files in each package export public API
- Pattern in `models/__init__.py`, `training/__init__.py`, `solarflare_data/__init__.py`
- Enables clean imports: `from models import SolarFluxPredictor`

**Organization:**
- Separate concerns: `trainer.py` handles training loop, `losses.py` handles loss functions
- Utilities in `utils/`: device management, metrics, visualization
- Data handling in `solarflare_data/`: dataset, loading, preprocessing
- Models in `models/`: network architectures and related utilities

---

*Convention analysis: 2026-02-02*
