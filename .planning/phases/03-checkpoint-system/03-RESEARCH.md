# Phase 3: Checkpoint System - Research

**Researched:** 2026-02-04
**Domain:** PyTorch checkpoint save/load, atomic file I/O, cross-device tensor remapping
**Confidence:** HIGH

## Summary

This phase replaces the current minimal checkpointing (single best-model save, no resume support) with a production-grade checkpoint system: atomic writes, full training state resume, cross-device portability, and normalization params embedded in the checkpoint file. The emergency checkpoint infrastructure from Phase 2 already saves model/optimizer/scheduler state -- this phase standardizes the format, adds resume logic, and hardens writes against corruption.

The standard approach uses `torch.save()` / `torch.load()` with `map_location` for cross-device portability, `tempfile.NamedTemporaryFile` + `os.replace()` for atomic writes, and a version-stamped checkpoint dict for forward compatibility. No external libraries are needed -- everything is built into Python stdlib and PyTorch.

**Primary recommendation:** Centralize all checkpoint I/O into a single module (`utils/checkpoint.py`) that owns the dict schema, atomic write, versioned load, and device remapping. The trainer and inference code call this module rather than using `torch.save`/`torch.load` directly.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` (torch.save/torch.load) | 2.x | Serialize/deserialize state dicts | Built-in, handles all tensor types, GPU/CPU/MPS |
| `tempfile` (stdlib) | N/A | Create temp files for atomic writes | Built-in, cross-platform |
| `os` (os.replace, os.fsync) | N/A | Atomic rename, flush to disk | POSIX-atomic rename guarantee |
| `pathlib` | N/A | Path manipulation | Already used throughout codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `shutil` (stdlib) | N/A | File copy if needed | Only if backup copies are ever added |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.save` pickle | `safetensors` (HuggingFace) | Safer deserialization but adds dependency; overkill for self-generated checkpoints |
| Manual atomic write | `python-atomicwrites` | Library is deprecated; `os.replace` is sufficient |

**Installation:** No new dependencies required.

## Architecture Patterns

### Recommended Module Structure
```
utils/
├── checkpoint.py    # NEW: all checkpoint I/O (save, load, validate, atomic write)
├── device.py        # Existing: resolve_device, get_grad_scaler, etc.
└── config_validator.py  # Existing: validate_config
```

### Pattern 1: Centralized Checkpoint Module
**What:** A single `utils/checkpoint.py` module that defines the checkpoint dict schema, handles atomic saving, versioned loading, device remapping, and integrity validation. All other code (trainer, inference, emergency saves) calls this module.
**When to use:** Always -- avoids scattered `torch.save`/`torch.load` calls with inconsistent schemas.

### Pattern 2: Atomic Write via Temp File + os.replace
**What:** Write checkpoint to a `NamedTemporaryFile` in the same directory, `os.fsync()` to flush, then `os.replace()` to atomically swap.
**When to use:** Every checkpoint write (best model, latest, emergency).
**Example:**
```python
import os
import tempfile
import torch

def _atomic_save(state_dict: dict, filepath: Path) -> None:
    """Save checkpoint atomically: write to temp file, fsync, rename."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Temp file MUST be in same directory for atomic os.replace
    fd = tempfile.NamedTemporaryFile(
        dir=filepath.parent,
        prefix='.tmp_checkpoint_',
        suffix='.pt',
        delete=False
    )
    try:
        torch.save(state_dict, fd.name)
        # Flush to disk before rename
        with open(fd.name, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(fd.name, str(filepath))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(fd.name)
        except OSError:
            pass
        raise
    finally:
        fd.close()
```

### Pattern 3: CPU-Normalized Tensors for Portability
**What:** Before saving, move all tensors to CPU so the checkpoint file is device-neutral. On load, use `map_location='cpu'` and then move to target device.
**When to use:** Always -- ensures checkpoint works on any device.
**Example:**
```python
def _to_cpu(state_dict):
    """Recursively move all tensors in a state dict to CPU."""
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()}

# Save: model is on GPU, but checkpoint is CPU
checkpoint = {
    'model_state_dict': _to_cpu(model.state_dict()),
    'optimizer_state_dict': _to_cpu_nested(optimizer.state_dict()),
    ...
}
```

### Pattern 4: Optimizer State Device Remapping
**What:** `torch.load(map_location=device)` remaps all tensors. But if you later do `model.to(device)` after loading, the optimizer state tensors stay on the old device. Must manually remap optimizer state.
**When to use:** When loading a checkpoint to resume training (optimizer needed on target device).
**Example:**
```python
def _optimizer_to_device(optimizer, device):
    """Move optimizer state tensors to the target device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
```

**Critical note:** Since the decision is to always save tensors on CPU, `torch.load(map_location='cpu')` will load everything to CPU. After `model.to(device)` and `optimizer.load_state_dict(...)`, the optimizer state tensors will be on CPU. The manual `_optimizer_to_device` call is then needed to move them to the target device.

### Pattern 5: Version-Stamped Checkpoint Dict
**What:** Include a `checkpoint_version` integer in every checkpoint. On load, compare against current expected version and fail with a clear message if mismatched.
**When to use:** Always -- protects against silently loading incompatible checkpoints.

### Anti-Patterns to Avoid
- **Saving full model (not state_dict):** Pickles the class definition, breaks when code changes.
- **torch.save without atomic write:** Crash mid-write corrupts the checkpoint file irreversibly.
- **Not specifying map_location:** Defaults to original device, fails if device unavailable. Also triggers FutureWarning in recent PyTorch.
- **Not specifying weights_only:** PyTorch 2.x emits FutureWarning when `weights_only` is not set. Should use `weights_only=False` explicitly since checkpoint contains non-tensor data (config dict, version int, etc.). Using `weights_only=True` would fail because the checkpoint contains Python objects.
- **Saving tensors on GPU:** Checkpoint becomes device-specific, fails on different hardware.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Atomic file writes | Custom lock/copy mechanism | `tempfile.NamedTemporaryFile` + `os.replace` | `os.replace` is guaranteed atomic on POSIX; temp file in same dir guarantees same filesystem |
| Tensor serialization | Custom binary format | `torch.save` / `torch.load` | Handles all tensor dtypes, sparse, quantized, nested structures |
| Device remapping | Custom tensor-walking code | `torch.load(map_location=...)` for initial load | Built-in, handles recursive dicts/lists of tensors |

**Key insight:** The only place where manual tensor device work is needed is the optimizer state dict after loading, because `optimizer.load_state_dict()` does not honor the device of the model parameters -- it keeps tensors wherever they are in the loaded dict.

## Common Pitfalls

### Pitfall 1: Temp File on Different Filesystem
**What goes wrong:** `os.replace()` is only atomic when source and destination are on the same filesystem. If `tempfile.gettempdir()` returns `/tmp` but checkpoints are on `/data`, the rename becomes a copy+delete (not atomic).
**Why it happens:** Using `tempfile.NamedTemporaryFile()` without specifying `dir=`.
**How to avoid:** Always pass `dir=filepath.parent` to `NamedTemporaryFile`.
**Warning signs:** Checkpoint corruption after crashes; checkpoint file sometimes zero-length.

### Pitfall 2: Optimizer State Stuck on Wrong Device
**What goes wrong:** After `torch.load(map_location='cpu')` and `optimizer.load_state_dict(...)`, the optimizer's momentum/variance buffers are on CPU even after `model.to('cuda')`. Training runs but is extremely slow (constant CPU<->GPU transfers) or crashes.
**Why it happens:** PyTorch's optimizer doesn't have a `.to()` method. `load_state_dict` copies tensors as-is.
**How to avoid:** Explicitly iterate optimizer state and call `.to(device)` on each tensor after loading and moving model.
**Warning signs:** Training resumes but runs 10x slower than expected; CUDA out-of-memory from duplicate tensors.

### Pitfall 3: torch.load weights_only FutureWarning
**What goes wrong:** Recent PyTorch 2.x emits `FutureWarning: You are using torch.load with weights_only=False`. Will default to `True` in future, breaking checkpoints with non-tensor data.
**Why it happens:** Security hardening -- pickle-based loading can execute arbitrary code.
**How to avoid:** Explicitly pass `weights_only=False` when loading full training checkpoints (which contain config dicts, version ints, etc.). This is correct and intentional for self-generated checkpoints.
**Warning signs:** Warning spam in logs; future PyTorch upgrade breaks loading.

### Pitfall 4: Config Changed but Epoch Counter Resets
**What goes wrong:** Resume starts training from epoch 1 instead of continuing from the checkpoint's epoch. User gets 25 more epochs instead of the remaining 10.
**Why it happens:** Using `range(1, epochs + 1)` without adjusting for the checkpoint's starting epoch.
**How to avoid:** On resume, set `start_epoch = checkpoint['epoch'] + 1` and loop from `start_epoch` to `epochs + 1`. The `epochs` value comes from the **current** config (user may have changed it).
**Warning signs:** Training history shows duplicate epoch numbers; model trains far longer than expected.

### Pitfall 5: Scheduler State Not Restored
**What goes wrong:** Learning rate restarts from initial value instead of where it left off. For cosine annealing, this means the LR curve is wrong for the remaining epochs.
**Why it happens:** Forgetting to call `scheduler.load_state_dict()` or not saving scheduler state.
**How to avoid:** Always save and restore `scheduler.state_dict()`. The current emergency checkpoint code already does this conditionally.
**Warning signs:** LR jumps back to initial value on resume; unexpected training dynamics.

### Pitfall 6: Emergency Checkpoint Format Mismatch
**What goes wrong:** Emergency checkpoints from Phase 2 use a different dict schema than the new standard format. Resume fails when loading an emergency checkpoint.
**Why it happens:** Emergency save code was written before the checkpoint system was standardized.
**How to avoid:** The decision is to use the same full format for emergency checkpoints. Update the emergency save code to use the centralized `save_checkpoint()` function.
**Warning signs:** KeyError when loading an emergency checkpoint for resume.

### Pitfall 7: Best Model Deleted Before New One Written
**What goes wrong:** Old best model is deleted, then crash occurs before new best model is fully written. Both checkpoints are lost.
**Why it happens:** Deleting old file before atomic write of new file completes.
**How to avoid:** Write new best model atomically first, THEN delete old one. With atomic writes, the new file either fully exists or doesn't -- so the old file should only be deleted after `os.replace` succeeds.
**Warning signs:** No best_model.pt after a crash; training must restart from scratch.

## Code Examples

### Checkpoint Dict Schema (Recommended)
```python
CHECKPOINT_VERSION = 1

checkpoint = {
    # Versioning
    'checkpoint_version': CHECKPOINT_VERSION,

    # Training state (everything needed to resume)
    'epoch': epoch,
    'model_state_dict': model.state_dict(),   # moved to CPU before save
    'optimizer_state_dict': optimizer.state_dict(),  # moved to CPU before save
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'best_val_loss': best_val_loss,
    'patience_counter': patience_counter,

    # Normalization (self-contained, no metadata.json dependency)
    'normalization_params': normalization_params,

    # Config snapshot (for diffing on resume)
    'config': config,

    # Training history (losses, LRs per epoch)
    'history': history,

    # Metadata
    'emergency': False,  # True for emergency checkpoints
}
```

### Full Save Function
```python
def save_checkpoint(
    filepath: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
    patience_counter: int,
    normalization_params: dict,
    config: dict,
    history: dict,
    emergency: bool = False,
    emergency_reason: str = None,
) -> None:
    """Save a full training checkpoint atomically."""
    checkpoint = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'epoch': epoch,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'normalization_params': normalization_params,
        'config': config,
        'history': history,
        'emergency': emergency,
        'emergency_reason': emergency_reason,
    }
    # Move optimizer tensors to CPU
    _move_optimizer_state_to_cpu(checkpoint['optimizer_state_dict'])
    _atomic_save(checkpoint, filepath)
```

### Full Load Function
```python
def load_checkpoint(filepath: Path, device: torch.device) -> dict:
    """Load and validate a checkpoint, remapping to the target device."""
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    # Version check
    version = checkpoint.get('checkpoint_version')
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Checkpoint version mismatch: file has v{version}, "
            f"expected v{CHECKPOINT_VERSION}. Cannot resume."
        )

    return checkpoint
```

### Resume Logic in Trainer
```python
def resume_training(checkpoint, model, optimizer, scheduler, device):
    """Restore all training state from a checkpoint."""
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Optimizer state tensors are on CPU from checkpoint -- move to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint['patience_counter']
    history = checkpoint['history']

    return start_epoch, best_val_loss, patience_counter, history
```

### Config Diff on Resume
```python
def _diff_configs(saved_config: dict, current_config: dict, path: str = "") -> list:
    """Find differences between saved and current config."""
    diffs = []
    all_keys = set(list(saved_config.keys()) + list(current_config.keys()))
    for key in sorted(all_keys):
        full_key = f"{path}.{key}" if path else key
        old = saved_config.get(key)
        new = current_config.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            diffs.extend(_diff_configs(old, new, full_key))
        elif old != new:
            diffs.append(f"  {full_key}: {old!r} -> {new!r}")
    return diffs
```

### Cross-Device Load with Logging
```python
def load_model_for_inference(filepath: Path, device: torch.device = None):
    """Load model for inference with cross-device support."""
    if device is None:
        device = resolve_device('auto')

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    # Log device remapping
    saved_device = checkpoint.get('config', {}).get('device', 'unknown')
    if str(device) != saved_device and saved_device != 'unknown':
        logger.info("Remapping checkpoint from %s to %s", saved_device, device)

    # Build model from saved config and load weights
    model = _build_model_from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint.get('normalization_params')
```

### Architecture Mismatch Detection
```python
def _check_state_dict_compatibility(model, state_dict):
    """Check for missing/unexpected keys and fail with a helpful diff."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing or unexpected:
        lines = ["Architecture mismatch between model and checkpoint:"]
        if missing:
            lines.append(f"\n  Missing keys ({len(missing)}):")
            for k in sorted(missing):
                lines.append(f"    - {k}")
        if unexpected:
            lines.append(f"\n  Unexpected keys ({len(unexpected)}):")
            for k in sorted(unexpected):
                lines.append(f"    + {k}")
        raise RuntimeError("\n".join(lines))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.save(model, path)` (pickle whole model) | `torch.save(model.state_dict(), path)` | PyTorch 1.x+ | Avoids brittle class pickling |
| `torch.load(path)` (no map_location) | `torch.load(path, map_location='cpu')` | Always recommended, now warns | Cross-device portability |
| No `weights_only` param | `weights_only=False` explicit | PyTorch 2.0+ | Suppresses FutureWarning, intentional choice for full checkpoints |
| `os.rename` for atomic writes | `os.replace` | Python 3.3+ | `os.replace` works cross-platform, handles existing files |

**Deprecated/outdated:**
- `python-atomicwrites` library: Deprecated by maintainer. `os.replace` in stdlib is sufficient.
- `torch.save` with `_use_new_zipfile_serialization`: Default since PyTorch 1.6, no longer needs to be specified.

## Open Questions

1. **Optimizer state CPU conversion depth**
   - What we know: `optimizer.state_dict()` returns nested dicts with tensors at various depths. The `state` sub-dict contains per-parameter state (momentum, variance for Adam).
   - What's unclear: Whether a simple one-level iteration is sufficient or if deeper nesting exists in some optimizer types.
   - Recommendation: Walk the state dict recursively, converting any tensor encountered. AdamW (used in this project) has a flat per-parameter structure, so one level is sufficient, but a recursive approach is safer.

2. **GradScaler state in checkpoint**
   - What we know: The current code uses `get_grad_scaler` which returns either a real `GradScaler` (CUDA) or `DummyGradScaler` (MPS/CPU). `GradScaler` has a `state_dict()`.
   - What's unclear: Whether scaler state should be saved/restored for full reproducibility.
   - Recommendation: Save scaler state if using real GradScaler (CUDA AMP). The `DummyGradScaler` has no state. Add `scaler_state_dict` to checkpoint dict if applicable. This is a minor addition that improves CUDA resume fidelity.

3. **Normalization params availability during training**
   - What we know: Normalization params are computed during data loading (`load_and_prepare_data` returns `metadata` with `normalization` key) and saved to `metadata.json`. The trainer currently doesn't receive normalization params directly.
   - What's unclear: How to pass normalization params to the checkpoint save function.
   - Recommendation: Pass normalization params through the training config or as a separate argument to `train_model()`. The `metadata` dict is available in `main.py` -- either thread the normalization subset into `train_config` or add a `normalization_params` parameter.

## Codebase-Specific Findings

### Current Checkpoint State (What Exists)

**trainer.py `train_model()`:**
- Saves best model: `torch.save(checkpoint_dict, checkpoint_path)` -- NOT atomic
- Dict keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_loss`, `config`, optionally `scheduler_state_dict`
- Missing from current dict: `best_val_loss` as explicit field (only in `val_loss`), `patience_counter`, `normalization_params`, `checkpoint_version`, `history`
- Best model path: `save_dir / checkpoint_name` (e.g., `outputs/best_model.pt`)
- No "latest checkpoint" saved -- only best model

**trainer.py `_save_emergency_checkpoint()`:**
- Saves to `save_dir / f"EMERGENCY_{checkpoint_name}"`
- Dict keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_loss`, `config`, `emergency`, `reason`, optionally `scheduler_state_dict`
- NOT atomic (uses direct `torch.save`)
- Missing: `normalization_params`, `checkpoint_version`, `patience_counter`, `history`

**trainer.py `load_checkpoint()`:**
- Simple: `torch.load(path, map_location=device)`, loads model, optionally optimizer
- Does NOT check version or validate keys
- Does NOT handle architecture mismatches

**inference.py `load_model()`:**
- `torch.load(path, map_location=device)` -- no `weights_only` specified
- Reconstructs model from checkpoint's `config` dict
- Normalization params loaded separately from `metadata.json` (not self-contained)

### What Must Change

1. **trainer.py**: Replace inline `torch.save` calls (best model + emergency) with centralized `save_checkpoint()` from new `utils/checkpoint.py`
2. **trainer.py**: Add resume logic to `train_model()` -- accept `resume_from` path, load checkpoint, restore all state, continue from saved epoch
3. **trainer.py**: Save two checkpoints per the decision: `best_model.pt` and `checkpoint_epoch_NNN_valloss_X.XXXX.pt` in a `checkpoints/` subfolder
4. **trainer.py**: Delete old latest checkpoint when new one is saved; delete old best when new best found
5. **inference.py `load_model()`**: Use centralized `load_checkpoint()`, get normalization params from checkpoint itself (self-contained)
6. **main.py**: Pass normalization params into training config; add `resume_from` config field handling
7. **config.yaml**: Add `resume_from` field (null/absent by default)
8. **config_validator.py**: Validate `resume_from` if present (file exists, readable)

### File Management Details

Per the decisions:
- Two files max: `best_model.pt` and `checkpoint_epoch_015_valloss_0.0234.pt`
- Location: `{save_dir}/checkpoints/` subfolder
- Old best deleted immediately when new best is found
- Old latest deleted when new latest is saved (each epoch)
- Emergency checkpoints use same format, same directory, `EMERGENCY_` prefix

## Sources

### Primary (HIGH confidence)
- **Codebase analysis**: Direct reading of `trainer.py`, `main.py`, `inference.py`, `utils/device.py`, `config.yaml`, `config_validator.py`, `solarflare_data/loader.py`
- **PyTorch documentation**: `torch.save`, `torch.load`, `map_location`, `state_dict()` patterns are stable API since PyTorch 1.x, unchanged in 2.x
- **Python stdlib**: `os.replace` atomic guarantee (POSIX), `tempfile.NamedTemporaryFile` -- well-documented stdlib behavior

### Secondary (MEDIUM confidence)
- **PyTorch `weights_only` parameter**: Added in PyTorch 2.0, FutureWarning behavior confirmed via web search and training data
- **Optimizer state device remapping**: Well-known pattern, documented in PyTorch tutorials and community best practices
- **`os.replace` vs `os.rename`**: `os.replace` is preferred (handles existing destination on Windows), confirmed via Python docs and web search

### Tertiary (LOW confidence)
- None -- all findings verified with primary or secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- uses only PyTorch built-ins and Python stdlib, no external deps
- Architecture: HIGH -- patterns are well-established PyTorch checkpointing conventions, verified against current codebase
- Pitfalls: HIGH -- common PyTorch checkpointing issues well-documented in community

**Research date:** 2026-02-04
**Valid until:** 2026-06-04 (stable domain, PyTorch checkpoint API rarely changes)
