# Phase 4: Data Pipeline - Research

**Researched:** 2026-02-04
**Domain:** PyTorch DataLoader with numpy mmap, multi-worker safety, augmentation
**Confidence:** HIGH

## Summary

This phase replaces the current in-memory data loading pipeline with a memory-mapped approach suitable for 10-50GB datasets. The existing code in `solarflare_data/dataset.py` and `solarflare_data/loader.py` loads all data into RAM as numpy arrays, splits within files by time index, and uses random per-sample augmentation. The new pipeline must use `np.load(mmap_mode='r')` for lazy loading, whole-file train/test/val splitting, deterministic augmentation modes (balanced/aggressive via index multiplication), and safe multi-worker DataLoader configuration.

The critical technical challenge is making numpy mmap work safely with PyTorch DataLoader workers. On macOS (the development platform, Python 3.12), the default multiprocessing start method is already `spawn`, meaning worker processes do NOT inherit file descriptors from the parent. The dataset must use a lazy-open pattern: store only file paths in `__init__`, open mmap handles on first access in `__getitem__`. This ensures each worker gets its own independent file descriptor. On Linux, the CONTEXT.md decision says to use `fork`, which inherits mmap handles but risks copy-on-read memory duplication due to CPython reference counting -- the lazy-open pattern works safely for both start methods.

**Primary recommendation:** Use lazy-open mmap pattern (open per-worker on first `__getitem__` call), index-multiplication for deterministic augmentation, and the official PyTorch `seed_worker` pattern for reproducible numpy random state across workers.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | DataLoader, Dataset base class | Already in project |
| NumPy | 2.4.2 | `np.load(mmap_mode='r')` for memory-mapped arrays | Already in project |
| Python | 3.12.12 | `multiprocessing` module (spawn default on macOS) | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.utils.data.DataLoader` | (PyTorch 2.10) | Batching, shuffling, multi-worker loading | Always -- wraps Dataset |
| `torch.utils.data.Dataset` | (PyTorch 2.10) | Base class for map-style dataset | Always -- subclass for custom dataset |
| `platform` | stdlib | Detect OS for start method selection | At DataLoader creation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `np.load(mmap_mode='r')` | `numpy.memmap` directly | `np.load` with mmap_mode is simpler for .npy files, handles headers automatically |
| Lazy open in `__getitem__` | Open mmap in `worker_init_fn` | Both work; lazy open is simpler, no separate function needed |
| Index multiplication for augmentation | On-the-fly random augmentation | Index multiplication is deterministic, reproducible, and compatible with `len()` |

**Installation:**
No new dependencies needed. All libraries are already present.

## Architecture Patterns

### Recommended Project Structure
```
solarflare_data/
    __init__.py          # Public API exports
    dataset.py           # SolarFluxDataset (mmap-backed, index-multiplied augmentation)
    loader.py            # load_and_prepare_data(), create_dataloaders(), split logic
```

### Pattern 1: Lazy-Open Mmap Per Worker
**What:** Store file paths (not mmap handles) in Dataset.__init__. Open mmap lazily on first __getitem__ call. Each worker process gets its own independent file descriptor.
**When to use:** Always, for all multi-worker DataLoader configurations.
**Why:** With `spawn` (macOS default), workers start fresh -- no inherited state. With `fork` (Linux), inherited mmap handles cause copy-on-read memory bloat from CPython reference counting. Lazy open avoids both problems.
**Example:**
```python
# Source: PyTorch Forums + multiple verified community patterns
class SolarFluxDataset(Dataset):
    def __init__(self, file_paths, index, t_in, t_out, ...):
        self.file_paths = file_paths       # List[str] -- serializable
        self.index = index                 # List[(file_idx, window_start, aug_type)]
        self.t_in = t_in
        self.t_out = t_out
        self._mmap_cache = {}              # Will be populated per-worker lazily

    def _get_mmap(self, file_idx):
        if file_idx not in self._mmap_cache:
            self._mmap_cache[file_idx] = np.load(
                self.file_paths[file_idx], mmap_mode='r'
            )
        return self._mmap_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, window_start, aug_type = self.index[idx]
        data = self._get_mmap(file_idx)
        # Extract window and apply deterministic augmentation
        chunk = data[window_start:window_start + self.t_in + self.t_out]
        # .copy() converts mmap view to regular array, then to tensor
        chunk = chunk.copy()
        chunk = self._apply_augmentation(chunk, aug_type)
        ...
```

### Pattern 2: Index Multiplication for Deterministic Augmentation
**What:** At dataset init, expand the sample index to include augmentation variants. For "balanced" mode: each (file_idx, window_start) gets 3 entries: (file_idx, window_start, NONE), (file_idx, window_start, HFLIP), (file_idx, window_start, VFLIP). For "aggressive": 8 entries (identity + 3 flips + 4 rotations, or the specific set from CONTEXT: balanced + 90-degree rotations).
**When to use:** When augmentation mode is "balanced" or "aggressive" (training split only).
**Why:** Deterministic, reproducible, compatible with `len()`, no random state issues across workers. Augmentation type is encoded in the index, not sampled at runtime.
**Example:**
```python
def _build_index(file_assignments, t_in, t_out, stride, augmentation):
    """Build flat index of all (file_idx, window_start, aug_type) tuples."""
    AUG_NONE = 0
    AUG_HFLIP = 1
    AUG_VFLIP = 2
    AUG_ROT90 = 3
    AUG_ROT180 = 4
    AUG_ROT270 = 5
    AUG_HFLIP_ROT90 = 6   # Additional combinations for aggressive
    AUG_VFLIP_ROT90 = 7

    if augmentation == "none":
        aug_variants = [AUG_NONE]
    elif augmentation == "balanced":
        aug_variants = [AUG_NONE, AUG_HFLIP, AUG_VFLIP]
    elif augmentation == "aggressive":
        aug_variants = [AUG_NONE, AUG_HFLIP, AUG_VFLIP,
                        AUG_ROT90, AUG_ROT180, AUG_ROT270]
    else:
        aug_variants = [AUG_NONE]

    index = []
    for file_idx, file_path in file_assignments:
        shape = _get_file_shape(file_path)  # Read shape without loading
        T = shape[0]
        n_windows = (T - t_in - t_out) // stride + 1
        for w in range(0, n_windows * stride, stride):
            for aug in aug_variants:
                index.append((file_idx, w, aug))
    return index
```

### Pattern 3: Whole-File Split Assignment
**What:** Shuffle file list by seed, assign entire files to train/test/val splits by ratio.
**When to use:** Always -- locked decision from CONTEXT.md.
**Example:**
```python
import random

def assign_files_to_splits(file_paths, split_ratios, seed):
    """Assign entire files to splits. Extra files go to training."""
    rng = random.Random(seed)
    shuffled = list(file_paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * split_ratios[0])
    n_test = round(n * split_ratios[1])
    n_val = n - n_train - n_test  # Remainder

    # Ensure at least 1 file per split if possible
    # Extra from rounding goes to train
    if n_val < 0:
        n_val = 0
        n_test = n - n_train

    return {
        'train': shuffled[:n_train],
        'test': shuffled[n_train:n_train + n_test],
        'val': shuffled[n_train + n_test:],
    }
```

### Pattern 4: Platform-Aware Multiprocessing Context
**What:** Set `multiprocessing_context` on DataLoader based on OS. macOS uses `spawn` (already the default on Python 3.8+ macOS). Linux uses `fork` (user's decision).
**When to use:** When `num_workers > 0`.
**Example:**
```python
import platform
import sys

def get_multiprocessing_context():
    """Return multiprocessing context string based on platform."""
    if platform.system() == "Darwin":
        return "spawn"
    else:
        return "fork"  # Linux decision from CONTEXT.md
```

### Anti-Patterns to Avoid
- **Opening mmap in `__init__`:** Causes shared file descriptor issues with fork, and fails to serialize with spawn (mmap objects are not picklable).
- **Storing numpy arrays in Dataset fields:** With spawn, the entire array gets pickled and sent to each worker, multiplying memory. With fork, reference counting causes copy-on-read.
- **Using `np.random` without worker_init_fn:** All workers get identical numpy random state, producing identical augmentations. (Not relevant for index-multiplication augmentation, but important if any numpy randomness is used elsewhere.)
- **Using `pin_memory=True` on MPS/CPU:** Wastes memory on MPS (unified memory, no benefit) and has no effect on CPU. Only enable for CUDA.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Worker random seeding | Custom seed logic | PyTorch's `seed_worker` pattern from official docs | Handles torch + numpy + random correctly |
| Memory-mapped file access | Custom mmap management | `np.load(path, mmap_mode='r')` | Handles .npy headers, dtype, shape automatically |
| Multiprocessing start method | Manual `mp.set_start_method()` | `DataLoader(multiprocessing_context=...)` | Per-DataLoader, no global side effects |
| Batch collation | Custom batch assembly | PyTorch default collate | Handles nested tuples, tensors correctly |
| Shuffling with reproducibility | Manual index permutation | DataLoader shuffle=True with seeded Generator | Handles epoch-level re-shuffling correctly |

**Key insight:** PyTorch's DataLoader already handles most complexity (shuffling, batching, worker management). The Dataset's job is just: given an index, return one sample as tensors. Keep the Dataset simple.

## Common Pitfalls

### Pitfall 1: Mmap Handles Shared Across Fork
**What goes wrong:** Parent opens mmap in `__init__`, fork duplicates the file descriptor. Workers contend on the same fd, causing slowdowns (4-6x slower than num_workers=0). On macOS with spawn, the mmap object fails to pickle entirely.
**Why it happens:** `np.load(mmap_mode='r')` returns a `numpy.memmap` object which holds an open file descriptor. Fork inherits it; spawn can't serialize it.
**How to avoid:** Lazy-open pattern. Store only `str` paths in `__init__`. Open mmap on first `__getitem__` call per worker.
**Warning signs:** `num_workers > 0` is dramatically slower than `num_workers=0`, or workers crash with pickle errors.

### Pitfall 2: Forgetting `.copy()` on Mmap Slices
**What goes wrong:** Slicing a mmap returns a view, not a copy. Converting this view to a torch tensor may hold the entire mmap page in memory, or cause issues when the mmap is garbage-collected.
**Why it happens:** numpy mmap slices are lazy views by default.
**How to avoid:** Always call `.copy()` on the extracted window before converting to tensor: `chunk = data[start:end].copy()`
**Warning signs:** Unexpectedly high memory usage, or stale/corrupted data in tensors after mmap is closed.

### Pitfall 3: Identical Numpy Random State Across Workers
**What goes wrong:** All workers produce identical augmentations because they inherit the same numpy random seed.
**Why it happens:** Fork copies the parent's numpy random state. Spawn starts fresh but with default seed.
**How to avoid:** Use the official PyTorch `seed_worker` function as `worker_init_fn`. Or better: use index-multiplication augmentation (our chosen approach) which has no random state at all.
**Warning signs:** Augmented batches look suspiciously similar. Validation with different workers produces identical batch statistics.

### Pitfall 4: pin_memory on Non-CUDA Devices
**What goes wrong:** `pin_memory=True` on MPS does nothing useful (unified memory) and wastes RAM allocation overhead. On CPU, it's a pure waste.
**Why it happens:** PyTorch's default pin_memory behavior is CUDA-centric. If CUDA is unavailable, pin_memory silently becomes a no-op, but if explicitly set, it still allocates pinned pages.
**How to avoid:** `pin_memory = (device.type == "cuda")` -- conditional based on resolved device.
**Warning signs:** Higher-than-expected RAM usage on MPS/CPU training.

### Pitfall 5: Not Handling Corrupt/Unreadable Files in Workers
**What goes wrong:** A single bad .npy file crashes a worker, which may kill the entire DataLoader.
**Why it happens:** mmap open failure or corrupt data in one file propagates as an unhandled exception in the worker subprocess.
**How to avoid:** Wrap `__getitem__` file access in try/except. On error, log warning and return a sentinel or skip to next sample. Pre-flight validation at startup catches most issues early.
**Warning signs:** "DataLoader worker exited unexpectedly" errors during training.

### Pitfall 6: Wrong Augmentation Interpretation
**What goes wrong:** CONTEXT.md says balanced mode "triples training data (original + h-flip + v-flip)". This means 3 deterministic copies, NOT random per-sample flips with 50% probability (which is what the current code does).
**Why it happens:** Confusing deterministic augmentation modes with stochastic augmentation.
**How to avoid:** Index multiplication: each sample gets exactly 3 (or more) entries in the index, each with a fixed augmentation type.
**Warning signs:** `len(dataset)` doesn't change when augmentation is enabled (it should triple for balanced).

## Code Examples

### Complete Worker Init + DataLoader Setup
```python
# Source: PyTorch 2.10 docs (https://docs.pytorch.org/docs/stable/notes/randomness.html)
import torch
import numpy
import random as python_random
import platform

def seed_worker(worker_id):
    """Official PyTorch pattern for reproducible worker seeding."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    python_random.seed(worker_seed)

def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size, num_workers, device, seed):
    """Create DataLoaders with proper multi-worker and device configuration."""
    pin = (device.type == "cuda")
    mp_context = "spawn" if platform.system() == "Darwin" else "fork"

    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
        multiprocessing_context=mp_context if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return train_loader, val_loader, test_loader
```

### Applying Deterministic Augmentation in __getitem__
```python
def _apply_augmentation(self, chunk, aug_type):
    """Apply deterministic spatial augmentation to a (T, H, W) array.

    chunk is already a .copy() from mmap, safe to modify.
    """
    if aug_type == AUG_NONE:
        return chunk
    elif aug_type == AUG_HFLIP:
        return np.flip(chunk, axis=2).copy()  # flip W dimension
    elif aug_type == AUG_VFLIP:
        return np.flip(chunk, axis=1).copy()  # flip H dimension
    elif aug_type == AUG_ROT90:
        return np.rot90(chunk, k=1, axes=(1, 2)).copy()
    elif aug_type == AUG_ROT180:
        return np.rot90(chunk, k=2, axes=(1, 2)).copy()
    elif aug_type == AUG_ROT270:
        return np.rot90(chunk, k=3, axes=(1, 2)).copy()
    else:
        return chunk
```

### Reading .npy File Shape Without Loading
```python
def _get_npy_shape(file_path):
    """Read shape from .npy header without loading data. O(1) memory."""
    with open(file_path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
    return shape
```

### Error-Tolerant __getitem__
```python
def __getitem__(self, idx):
    try:
        file_idx, window_start, aug_type = self.index[idx]
        data = self._get_mmap(file_idx)
        chunk = data[window_start:window_start + self.t_in + self.t_out].copy()
        chunk = self._apply_augmentation(chunk, aug_type)
        X_in = torch.from_numpy(chunk[:self.t_in]).float().unsqueeze(0)
        Y_out = torch.from_numpy(chunk[self.t_in:]).float().unsqueeze(0)
        return X_in, Y_out, (file_idx, window_start)
    except Exception as e:
        logger.warning("Error loading sample %d: %s. Returning random neighbor.", idx, e)
        # Return a different valid sample
        fallback_idx = (idx + 1) % len(self)
        return self.__getitem__(fallback_idx)
```

## State of the Art

| Old Approach (Current Code) | New Approach (This Phase) | Impact |
|---|---|---|
| Load all data into RAM as numpy arrays | `np.load(mmap_mode='r')` lazy access | 10-50GB datasets fit in <4GB RAM |
| Split within files by time index | Whole-file assignment to splits | No data leakage between train/test |
| `pin_memory=True` always | `pin_memory = (device.type == "cuda")` | No wasted memory on MPS/CPU |
| Random per-sample augmentation (50% flip) | Deterministic index-multiplied augmentation | Reproducible, triples/multiplies dataset size |
| No multiprocessing context control | Platform-aware spawn/fork selection | Safe multi-worker on macOS and Linux |
| No worker seeding | `seed_worker` + Generator pattern | Reproducible across workers and epochs |

**Deprecated/outdated in current code:**
- `create_dataloaders()` in loader.py: hardcodes `pin_memory=True`, no multiprocessing_context, no worker_init_fn
- `SolarFluxDataset.__init__` takes `datasets: List[np.ndarray]` (in-memory arrays) -- must change to file paths
- Time-based splitting within files (lines 269-279 in loader.py) -- replaced by whole-file assignment
- Random augmentation in `__getitem__` (lines 73-81 in dataset.py) -- replaced by index-multiplication

## Open Questions

1. **Augmentation variant count for "aggressive" mode**
   - What we know: CONTEXT says "balanced plus 90-degree rotations -- further multiplies training data"
   - What's unclear: Exactly which combinations? Original + hflip + vflip + rot90 + rot180 + rot270 = 6x? Or include combined flip+rotation = 8x (all symmetries of a square)?
   - Recommendation: Use 6 variants (identity, hflip, vflip, rot90, rot180, rot270). The 8 dihedral symmetries of a square are: identity, 3 rotations, 2 flips, 2 flip+rotations. But CONTEXT specifically says "balanced plus 90-degree rotations" which reads as adding rotation variants on top of the balanced set. 6x is the safe interpretation.

2. **Structured array (.npy) vs preprocessed cube (.npz) format**
   - What we know: Current code supports both raw structured arrays and preprocessed .npz cubes. The new mmap pipeline works with dense .npy cubes (T, H, W).
   - What's unclear: Should the new pipeline support raw structured arrays directly, or only preprocessed dense cubes?
   - Recommendation: Require preprocessed dense .npy cubes for the mmap pipeline. Raw structured arrays need the slow `_structured_to_cube` conversion which defeats the purpose of mmap. The preprocessing step can remain separate.

3. **persistent_workers interaction with mmap**
   - What we know: `persistent_workers=True` keeps worker processes alive between epochs, avoiding re-spawn overhead.
   - What's unclear: With persistent workers and lazy-open mmap, the mmap handles stay open for the entire training run. For many large files this could hit OS file descriptor limits.
   - Recommendation: Use `persistent_workers=True` when `num_workers > 0`. The number of open fds equals `num_workers * num_files` which for typical use (4 workers, 10-50 files) is well under OS limits (default 256 on macOS, 1024 on Linux).

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10 official docs: [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) -- seed_worker pattern, Generator usage
- PyTorch 2.10 official docs: [Multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) -- spawn/fork for CUDA
- PyTorch 2.10 official docs: [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html) -- DataLoader API
- PyTorch 2.10 official tutorial: [pin_memory guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) -- CUDA-only recommendation
- NumPy 2.4 docs: [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html) -- mmap_mode parameter
- Python 3.14 docs: [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) -- start method defaults
- Local verification: Python 3.12 on macOS defaults to `spawn` (confirmed via `multiprocessing.get_start_method()`)
- Local verification: PyTorch 2.10.0, NumPy 2.4.2 installed

### Secondary (MEDIUM confidence)
- [Tanel Parnamaa: NumPy bug in DataLoader workers](https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/) -- identical augmentation bug
- [PyTorch Forums: DataLoader and memmaps](https://discuss.pytorch.org/t/dataloader-and-memmaps/180614) -- lazy open pattern
- [PyTorch Forums: numpy memmap throttles](https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274) -- performance with mmap
- [Yuxin Wu: Demystify RAM Usage in Multi-Process Data Loaders](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/) -- copy-on-read analysis
- [Python 3.14 fork() change](https://github.com/pytorch/pytorch/issues/169252) -- PyTorch issue tracking default change

### Tertiary (LOW confidence)
- None -- all findings verified with primary or secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, versions verified locally
- Architecture (mmap + lazy open): HIGH -- multiple official sources + community consensus
- Architecture (index multiplication augmentation): HIGH -- straightforward implementation, well-understood pattern
- Pitfalls (worker safety): HIGH -- extensively documented in PyTorch docs and community
- Pitfalls (pin_memory): HIGH -- official PyTorch tutorial explicitly covers this
- Augmentation variant count: MEDIUM -- interpretation of CONTEXT.md wording

**Research date:** 2026-02-04
**Valid until:** 2026-03-06 (30 days -- stable domain, no fast-moving changes expected)
