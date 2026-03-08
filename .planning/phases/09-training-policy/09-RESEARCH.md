# Phase 9: Training Policy - Research

**Researched:** 2026-03-08
**Domain:** PyTorch training configuration -- LR scheduling, data augmentation, teacher forcing, class-imbalanced sampling
**Confidence:** HIGH

## Summary

Phase 9 is primarily a configuration-activation phase. Four of the five requirements (TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-05) are satisfied by changing config.yaml values to activate already-implemented infrastructure: CosineAnnealingLR scheduler (trainer.py:489-494), balanced augmentation (dataset.py:268-269), teacher forcing elimination (tf_start=0.0), and epoch count increase. The only significant new code is TRAIN-04: WeightedRandomSampler integration, which requires (a) scanning output frames during `build_index()` to flag flare-containing sequences, (b) computing per-sample weights, and (c) modifying `create_dataloaders()` to accept and use a sampler instead of `shuffle=True`.

All components use PyTorch built-in classes (CosineAnnealingLR, WeightedRandomSampler) with no new dependencies. The verified PyTorch version is 2.10.0. Training data has 568 base windows (7 training files), expanding to 1,704 with balanced augmentation (3x). At batch_size=1, each epoch is 1,704 steps. With WeightedRandomSampler replacing shuffle, the sampler controls iteration order and the number of samples drawn per epoch.

**Primary recommendation:** Implement in three waves: (1) config updates for scheduler/augmentation/TF/epochs/patience, (2) WeightedRandomSampler with flare detection in build_index, (3) config validation cross-check warning. All changes are additive; no existing behavior is removed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Any pixel above 0.3456 threshold in any OUTPUT frame makes a sequence "flare-containing"
- Only output (target) frames are scanned -- we want to oversample WHERE flares happen, not where they existed in inputs
- Flare flags computed at `build_index()` time (dataset build), fitting existing index-multiplication pattern
- Configurable oversampling weight: `flare_oversample_weight: 3.0` in config `data:` section
- WeightedRandomSampler replaces `shuffle=True` in training DataLoader when oversampling is enabled
- Patience increased to 15-20 (from 8) to let cosine LR complete most of its 50-epoch cycle
- No warmup -- CosineAnnealingLR starts at configured LR (1e-4) and decays smoothly
- Val_loss remains the early stopping metric (composite loss already includes temporal terms from Phase 8)
- Single best model checkpoint saved by val_loss (no separate CSI checkpoint)
- 50 epochs with cosine schedule (T_max=50)
- Batch size stays at 1 (preserves memory headroom for Phase 10 architecture scaling)
- Learning rate stays at 1e-4 with AdamW
- With balanced augmentation (3x dataset), each epoch is ~1,704 steps at batch_size=1
- Update config.yaml in-place with v3.0 defaults (cosine scheduler, balanced augmentation, tf_start=0.0, 50 epochs)
- New keys go under `data:` section: `flare_oversample_weight: 3.0`
- Patience update goes in `training:` section
- Config validation cross-check: warn if `flare_oversample_weight > 1.0` but `augmentation: "none"` (suboptimal combo)

### Claude's Discretion
- Exact patience value within 15-20 range
- Internal implementation of WeightedRandomSampler integration (how weights array is built and passed)
- Whether to log flare sequence statistics at training start (count of flare vs non-flare sequences)
- Config validation warning message wording

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Cosine LR scheduler enabled (cosine annealing with eta_min=1e-6) | CosineAnnealingLR already implemented at trainer.py:489-494; activate via config `scheduler.type: "cosine"`. T_max set to epochs (50). eta_min already defaults to 1e-6. |
| TRAIN-02 | Balanced augmentation enabled (horizontal + vertical flips, 3x effective dataset) | Balanced augmentation already implemented in dataset.py:268-269; activate via config `augmentation: "balanced"`. Verified: 568 base windows become 1,704. |
| TRAIN-03 | Teacher forcing eliminated (tf_start=0.0) | TF decay formula at trainer.py:619 `tf_ratio = max(0.0, tf_start * (1 - epoch/epochs))`. Setting tf_start=0.0 makes tf_ratio=0.0 for all epochs. |
| TRAIN-04 | Class-imbalanced sampling via WeightedRandomSampler (flare-containing sequences oversampled 3x) | New code required: extend build_index() to flag flare sequences, compute weights array, create WeightedRandomSampler, pass to DataLoader (replacing shuffle=True). |
| TRAIN-05 | Training epochs increased to 50+ with cosine schedule | Config change: `training.epochs: 50`. CosineAnnealingLR T_max auto-set to epochs value. Patience increased to 15-20. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.optim.lr_scheduler.CosineAnnealingLR | PyTorch 2.10 | LR scheduling | Already implemented in trainer.py; PyTorch built-in, no dependencies |
| torch.utils.data.WeightedRandomSampler | PyTorch 2.10 | Class-imbalanced sampling | PyTorch standard approach for oversampling minority classes |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | existing | Mmap scanning for flare detection | Scanning output frames during build_index for threshold check |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| WeightedRandomSampler | Manual epoch-level oversampling via index duplication | Sampler is cleaner; index duplication would bloat memory and complicate augmentation interaction |
| CosineAnnealingLR | CosineAnnealingWarmRestarts | User decided no warmup/restarts; simple cosine decay is sufficient for 50 epochs |

**Installation:**
No new packages needed. All components are PyTorch built-ins.

## Architecture Patterns

### Recommended Change Map
```
config.yaml                    # Config updates (TRAIN-01,02,03,05)
solarflare_data/dataset.py     # build_index() flare scanning (TRAIN-04)
solarflare_data/loader.py      # create_dataloaders() sampler integration (TRAIN-04)
utils/config_validator.py      # Cross-check warning (TRAIN-04)
tests/test_data_pipeline.py    # Flare flag + sampler tests
tests/test_config.py           # Cross-check warning test
```

### Pattern 1: WeightedRandomSampler Integration

**What:** Replace `shuffle=True` with a WeightedRandomSampler that oversamples flare-containing sequences.
**When to use:** When TRAIN-04 flare_oversample_weight > 1.0 in config.
**Key constraint:** `sampler` and `shuffle` are mutually exclusive in PyTorch DataLoader. When a sampler is provided, `shuffle` must be False (or omitted).

**Implementation approach:**
```python
# In build_index(), scan output frames for flare content
# Return both index and flare_flags list
def build_index(..., extreme_threshold=None):
    index = []
    flare_flags = []
    for file_idx in file_assignments.get(split, []):
        mmap = np.load(file_paths[file_idx], mmap_mode="r")
        T = mmap.shape[0]
        max_start = T - t_in - t_out + 1
        for window_start in range(0, max_start, stride):
            # Check output frames for extreme values
            has_flare = False
            if extreme_threshold is not None:
                out_start = window_start + t_in
                out_end = out_start + t_out
                output_frames = mmap[out_start:out_end]
                has_flare = bool(np.any(output_frames > extreme_threshold))
            for aug in aug_codes:
                index.append((file_idx, window_start, aug))
                flare_flags.append(has_flare)
    return index, flare_flags
```

```python
# In create_dataloaders(), build sampler from flare_flags
from torch.utils.data import WeightedRandomSampler

def create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=1, num_workers=0, device=None, seed=42,
    flare_flags=None, flare_oversample_weight=1.0,
):
    # Build sampler if oversampling enabled
    sampler = None
    shuffle_train = True
    if flare_flags is not None and flare_oversample_weight > 1.0:
        weights = [flare_oversample_weight if f else 1.0 for f in flare_flags]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=g,
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        sampler=sampler,
        **common_kwargs
    )
```

**Verified API (PyTorch 2.10):**
- `WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)`
- `weights`: Sequence[float] -- one weight per sample
- `num_samples`: int -- how many samples to draw per epoch
- `replacement`: bool -- True allows same sample multiple times (required for oversampling)

### Pattern 2: Config-Driven Activation (Existing Pattern)

**What:** Most TRAIN requirements are activated by config changes alone.
**When to use:** TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-05.

The existing codebase already follows a pattern where config values control behavior:
- `scheduler.type: "cosine"` activates CosineAnnealingLR (trainer.py:489)
- `augmentation: "balanced"` activates H+V flip augmentation (dataset.py:268)
- `tf_start: 0.0` eliminates teacher forcing (trainer.py:619)
- `epochs: 50` sets training duration
- `patience: 18` sets early stopping patience

### Pattern 3: Config Cross-Check Warning (Phase 8 Pattern)

**What:** Warn when `flare_oversample_weight > 1.0` but `augmentation: "none"` -- oversampling without augmentation means the model sees identical flare sequences repeatedly, which is suboptimal.
**When to use:** In config_validator.py, following the same pattern as the Phase 8 loss/eval threshold mismatch warning (config_validator.py:336-344).

```python
# In validate_config(), after existing cross-checks
flare_weight = _get_nested(config, "data.flare_oversample_weight")
augmentation = _get_nested(config, "data.augmentation")
if (flare_weight is not None and isinstance(flare_weight, (int, float))
        and flare_weight > 1.0
        and isinstance(augmentation, str) and augmentation == "none"):
    warnings.append(
        f"data.flare_oversample_weight is {flare_weight} but "
        f"data.augmentation is 'none'; oversampling without augmentation "
        f"means repeated identical sequences -- consider enabling augmentation"
    )
```

### Anti-Patterns to Avoid
- **Setting both sampler and shuffle=True:** PyTorch raises `ValueError: sampler option is mutually exclusive with shuffle`. Always set `shuffle=False` when using a sampler.
- **Computing flare flags in __getitem__:** This would be called every iteration. Compute once at build_index time using mmap, consistent with existing patterns.
- **Using replacement=False with WeightedRandomSampler for oversampling:** Without replacement, you cannot oversample minority classes beyond their actual count. Use `replacement=True`.
- **Changing build_index return type without backward compatibility:** build_index currently returns `List[Tuple[int, int, int]]`. Adding flare_flags as a second return value changes the API. Must update all callers (loader.py uses build_index in two places).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LR scheduling | Custom decay function | CosineAnnealingLR | Already implemented; handles edge cases, integrates with optimizer |
| Class-imbalanced sampling | Manual index duplication or custom batch logic | WeightedRandomSampler | PyTorch native; handles sampling probability, replacement, generator seeding |
| Data augmentation | Runtime random flips | Existing index-multiplication augmentation | Deterministic, reproducible, no random state issues between workers |

**Key insight:** Phase 9 is fundamentally a configuration change, not a coding challenge. The only new logic is the flare detection scan in build_index and the plumbing to pass weights to the DataLoader.

## Common Pitfalls

### Pitfall 1: Sampler/Shuffle Mutual Exclusivity
**What goes wrong:** Passing both `sampler=sampler` and `shuffle=True` to DataLoader raises ValueError.
**Why it happens:** DataLoader enforces that iteration order is controlled by exactly one mechanism.
**How to avoid:** When sampler is provided, explicitly set `shuffle=False` (or omit it, since False is default). The current code at loader.py:762 hardcodes `shuffle=True` for train_loader -- this must be conditional.
**Warning signs:** ValueError at DataLoader construction time.

### Pitfall 2: build_index Return Type Change Breaking Callers
**What goes wrong:** build_index currently returns `List[Tuple[int, int, int]]`. If changed to return `(index, flare_flags)`, all existing callers break.
**Why it happens:** loader.py:493 and loader.py:634 both call `build_index(...)` and assign to `index` directly.
**How to avoid:** Two approaches: (a) Add `extreme_threshold` as optional param, return tuple only when provided, or (b) always return tuple and update all callers. Option (b) is cleaner -- update both load_and_prepare_data() and load_preprocessed_data() in loader.py, plus all tests.
**Warning signs:** TypeError on tuple unpacking.

### Pitfall 3: Flare Flags Must Match Augmented Index Length
**What goes wrong:** If flare flags are computed per base window but the index includes augmented copies, the weights array length won't match the dataset length.
**Why it happens:** Each base window generates 3 entries (balanced aug) -- flare flags must be replicated for each augmentation variant.
**How to avoid:** Append the flare flag inside the inner augmentation loop (for each `aug` in `aug_codes`), so flare_flags has exactly `len(index)` entries.
**Warning signs:** WeightedRandomSampler weights length != dataset length.

### Pitfall 4: Mmap Scan Performance for Flare Detection
**What goes wrong:** Scanning entire output frames at build time could be slow if done carelessly.
**Why it happens:** Each window requires reading t_out frames from the mmap to check for values above threshold.
**How to avoid:** Since build_index already opens mmap for shape checking (dataset.py:279), the data is already memory-mapped. The threshold check (`np.any(output_frames > threshold)`) is a simple comparison -- fast because numpy short-circuits on the first True. For 568 windows across 7 files, this is negligible overhead.
**Warning signs:** If build_index takes noticeably longer (unlikely with 568 windows).

### Pitfall 5: WeightedRandomSampler num_samples vs Dataset Length
**What goes wrong:** If `num_samples` differs from `len(dataset)`, each epoch has a different number of steps than expected.
**Why it happens:** WeightedRandomSampler's `num_samples` controls how many samples are drawn per epoch. Setting it to `len(dataset)` maintains the same epoch length.
**How to avoid:** Set `num_samples=len(train_dataset)` (or `len(weights)`). This maintains ~1,704 steps per epoch with balanced augmentation.
**Warning signs:** Epoch step count changes unexpectedly in training logs.

### Pitfall 6: Early Stopping Patience vs Cosine Schedule
**What goes wrong:** With patience=8 (current default) and 50 epochs of cosine annealing, early stopping may trigger during the cosine's natural high-LR phase before the model converges.
**Why it happens:** Cosine annealing starts with the full LR and decays to eta_min. The initial epochs may have higher variance in val_loss.
**How to avoid:** User locked decision: increase patience to 15-20. At 18 epochs patience, early stopping won't fire until at least epoch 18, by which point cosine has decayed significantly.
**Warning signs:** Training stops before epoch 30.

### Pitfall 7: Flare Detection Threshold in Normalized vs Raw Space
**What goes wrong:** The 0.3456 threshold is in normalized (asinh) space, but build_index scans raw mmap data (preprocessed cubes are already normalized).
**Why it happens:** Preprocessed cubes are already normalized during preprocess_data.py. The threshold 0.3456 was computed as a percentile of normalized data.
**How to avoid:** Since `use_preprocessed: true` is the default and preprocessed cubes are already in normalized space, the 0.3456 threshold is correct for scanning. If raw data were used, the threshold would need inverse transformation. Document this assumption.
**Warning signs:** Zero or all sequences flagged as flare-containing.

## Code Examples

### Config.yaml Changes (TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05)

```yaml
# Changes from current config.yaml:
data:
  augmentation: "balanced"      # was: "none" -- TRAIN-02
  flare_oversample_weight: 3.0  # new key -- TRAIN-04

training:
  epochs: 50                    # was: 25 -- TRAIN-05
  tf_start: 0.0                 # was: 0.5 -- TRAIN-03
  patience: 18                  # was: 8 -- TRAIN-05 (supports cosine schedule)

  scheduler:
    type: "cosine"              # was: "none" -- TRAIN-01
    cosine_eta_min: 0.000001    # unchanged (already 1e-6) -- TRAIN-01
```

### Flare Detection in build_index (TRAIN-04)

```python
# Source: PyTorch WeightedRandomSampler docs + existing build_index pattern
def build_index(
    file_paths, file_assignments, t_in, t_out,
    stride=1, augmentation="none", split="train",
    extreme_threshold=None,  # NEW param for flare detection
):
    index = []
    flare_flags = []  # NEW: parallel list of bool

    for file_idx in file_assignments.get(split, []):
        mmap = np.load(file_paths[file_idx], mmap_mode="r")
        T = mmap.shape[0]
        max_start = T - t_in - t_out + 1
        if max_start <= 0:
            continue

        for window_start in range(0, max_start, stride):
            # Detect flare in output frames
            has_flare = False
            if extreme_threshold is not None:
                out_start = window_start + t_in
                out_end = out_start + t_out
                has_flare = bool(np.any(mmap[out_start:out_end] > extreme_threshold))

            for aug in aug_codes:
                index.append((file_idx, window_start, aug))
                flare_flags.append(has_flare)

    return index, flare_flags
```

### WeightedRandomSampler in create_dataloaders (TRAIN-04)

```python
# Source: PyTorch 2.10 DataLoader + WeightedRandomSampler docs
from torch.utils.data import WeightedRandomSampler

def create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=1, num_workers=0, device=None, seed=42,
    train_flare_flags=None,        # NEW
    flare_oversample_weight=1.0,   # NEW
):
    g = torch.Generator()
    g.manual_seed(seed)

    sampler = None
    shuffle_train = True

    if (train_flare_flags is not None
            and flare_oversample_weight > 1.0
            and len(train_flare_flags) > 0):
        weights = [
            flare_oversample_weight if flag else 1.0
            for flag in train_flare_flags
        ]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=g,
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train if sampler is None else False,
        sampler=sampler,
        **common_kwargs,
    )
    # val/test loaders unchanged (shuffle=False, no sampler)
```

### Config Validation Cross-Check (TRAIN-04)

```python
# In validate_config(), following Phase 8's threshold cross-check pattern
data = config.get("data", {})
flare_weight = data.get("flare_oversample_weight")
augmentation_val = data.get("augmentation", "none")

if (flare_weight is not None
        and isinstance(flare_weight, (int, float))
        and flare_weight > 1.0
        and augmentation_val == "none"):
    warnings.append(
        f"data.flare_oversample_weight is {flare_weight} but "
        f"data.augmentation is 'none'; oversampling without augmentation "
        f"risks overfitting on repeated identical sequences"
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| scheduler: "none" | CosineAnnealingLR | Phase 9 | Smooth LR decay from 1e-4 to 1e-6 over 50 epochs |
| augmentation: "none" | augmentation: "balanced" | Phase 9 | 3x effective dataset (568 -> 1,704 samples) |
| tf_start: 0.5 | tf_start: 0.0 | Phase 9 | Model must predict autoregressively from step 1 |
| shuffle: True | WeightedRandomSampler | Phase 9 | Flare sequences seen 3x more often per epoch |
| patience: 8 | patience: 18 | Phase 9 | Allows cosine schedule to complete before early stop |

**Key interactions:**
- Balanced augmentation + WeightedRandomSampler: Augmentation creates 3 variants of each window (none, hflip, vflip). Flare flags are per-variant (all 3 share the same flag). The sampler oversamples all variants of flare sequences.
- tf_start=0.0 + cosine schedule: Without teacher forcing, the model generates outputs autoregressively. Combined with smooth LR decay, this allows gradual optimization without the crutch of ground-truth inputs during training.
- 50 epochs + patience=18: The cosine schedule needs most of 50 epochs to decay. Patience=18 ensures early stopping only fires if the model truly plateaus, not during natural LR fluctuations.

## Open Questions

1. **Exact flare sequence ratio in training data**
   - What we know: Threshold is 0.3456 in normalized space; preprocessed data is already normalized
   - What's unclear: How many of the 568 base training windows actually contain flare-level output values
   - Recommendation: Log flare/non-flare counts at training start (Claude's discretion per CONTEXT.md). If ratio is extreme (e.g., <5% flare), the 3x oversample may not be enough. This is informational -- no action needed now.

2. **WeightedRandomSampler + DataLoader generator interaction**
   - What we know: Both WeightedRandomSampler and DataLoader accept a `generator` parameter for reproducibility
   - What's unclear: Whether sharing the same Generator between sampler and DataLoader causes issues
   - Recommendation: Use the same Generator instance. PyTorch documentation shows this pattern. If reproducibility issues arise, create separate generators with related seeds.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | tests/conftest.py |
| Quick run command | `python -m pytest tests/test_data_pipeline.py tests/test_config.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | Cosine scheduler activated via config | unit | `python -m pytest tests/test_config.py -x -q` | Yes (extend) |
| TRAIN-02 | Balanced augmentation 3x sample count | unit | `python -m pytest tests/test_data_pipeline.py::TestBuildIndex -x -q` | Yes (existing passes) |
| TRAIN-03 | tf_start=0.0 produces tf_ratio=0.0 | unit | `python -m pytest tests/test_config.py -x -q` | Yes (config validation) |
| TRAIN-04 | Flare flags computed correctly; sampler weights correct; sampler replaces shuffle | unit | `python -m pytest tests/test_data_pipeline.py -x -q` | No -- Wave 0 |
| TRAIN-04 | Config cross-check warning for oversample+no-aug | unit | `python -m pytest tests/test_config.py -x -q` | No -- Wave 0 |
| TRAIN-05 | epochs=50, patience=18 in config | unit | `python -m pytest tests/test_config.py -x -q` | Yes (config validation) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_data_pipeline.py tests/test_config.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_data_pipeline.py::TestBuildIndex::test_build_index_returns_flare_flags` -- verifies flare_flags length matches index length
- [ ] `tests/test_data_pipeline.py::TestBuildIndex::test_flare_flags_detect_extreme_values` -- verifies sequences with values above threshold are flagged True
- [ ] `tests/test_data_pipeline.py::TestBuildIndex::test_flare_flags_only_check_output_frames` -- verifies input-only extremes don't trigger flare flag
- [ ] `tests/test_data_pipeline.py::TestWeightedSampler::test_sampler_replaces_shuffle` -- verifies DataLoader created with sampler, not shuffle=True
- [ ] `tests/test_data_pipeline.py::TestWeightedSampler::test_sampler_weights_match_flare_flags` -- verifies weight vector length and values
- [ ] `tests/test_config.py::test_flare_oversample_with_no_augmentation_warning` -- verifies cross-check warning fires
- [ ] `tests/test_config.py::test_flare_oversample_weight_valid` -- verifies valid config with flare_oversample_weight passes

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10.0 installed in project (verified via `python -c "import torch; print(torch.__version__)"`)
- `WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)` -- verified via inspect.signature
- `CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)` -- verified via inspect.signature
- Existing codebase: trainer.py (CosineAnnealingLR at L489-494), dataset.py (build_index), loader.py (create_dataloaders)
- Data verification: 568 training windows, 1,704 with balanced augmentation -- confirmed via direct data scan

### Secondary (MEDIUM confidence)
- [PyTorch DataLoader docs](https://docs.pytorch.org/docs/stable/data.html) -- sampler/shuffle mutual exclusivity confirmed
- [PyTorch Forums](https://discuss.pytorch.org/t/dataloader-shuffle-and-sampler/20279) -- community confirmation of sampler pattern

### Tertiary (LOW confidence)
None -- all findings verified against installed PyTorch and existing codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all PyTorch built-ins, verified API signatures against installed version
- Architecture: HIGH -- extending existing patterns (build_index, create_dataloaders, config_validator) with minimal new code
- Pitfalls: HIGH -- sampler/shuffle mutual exclusivity is well-documented; return type change is straightforward to manage

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable -- PyTorch APIs, project codebase patterns)
