# Phase 3: Checkpoint System - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Atomic checkpoint writes, full training resume from checkpoint, and cross-device checkpoint portability. Training crashes lose at most one epoch of progress. Checkpoint scheduling, distributed checkpointing, and cloud storage are out of scope.

</domain>

<decisions>
## Implementation Decisions

### Resume behavior
- User triggers resume via explicit `resume_from: path/to/checkpoint.pt` config field — no auto-detection
- If config has changed since checkpoint was saved, warn about differences but use current config values (user intended the change)
- Epoch counter continues from where it left off (resume at epoch 15, run until epoch 100)
- If `resume_from` points to missing or corrupted file, fail immediately at startup with clear message — never silently start from scratch

### Checkpoint content & format
- Single `.pt` file containing everything via `torch.save()` dict
- Full training state included: model weights, optimizer state, LR scheduler state, early stopping patience, best loss, normalization params, config snapshot, training metrics history
- Include `checkpoint_version` integer field — bump on format changes, fail with clear message on version mismatch
- Emergency checkpoints (from Phase 2 graceful shutdown) use the same full format — no special lighter format

### File management
- Keep two checkpoints: best model (lowest val loss) and most recent checkpoint — 2 files max
- Descriptive naming: `checkpoint_epoch_015_valloss_0.0234.pt` and `best_model.pt`
- Save inside `checkpoints/` subfolder within the experiment's output directory
- Old best checkpoint deleted immediately when new best is found (only one best_model file at any time)

### Cross-device loading
- Tensors always saved on CPU — checkpoint is device-neutral by default, loads anywhere without remapping
- Log an info message when loading on a different device than originally trained ("Remapping checkpoint from cuda to mps")
- Architecture mismatch (missing/unexpected state dict keys) fails at startup with a diff showing which keys are wrong
- `load_model()` for inference uses the same cross-device loading logic — portability works everywhere

### Claude's Discretion
- Atomic write implementation (temp file + rename pattern)
- Exact checkpoint dict key names
- How metrics history is structured in the checkpoint
- Validation of checkpoint integrity beyond version check

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-checkpoint-system*
*Context gathered: 2026-02-04*
