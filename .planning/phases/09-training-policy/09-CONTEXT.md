# Phase 9: Training Policy - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Training configuration maximizes the effectiveness of the new loss function through better learning rate scheduling, data exposure, and honest autoregressive predictions. Enables cosine LR, balanced augmentation, eliminates teacher forcing, and adds class-imbalanced sampling. Requirements: TRAIN-01 through TRAIN-05.

</domain>

<decisions>
## Implementation Decisions

### Flare sequence identification
- Any pixel above 0.3456 threshold in any OUTPUT frame makes a sequence "flare-containing"
- Only output (target) frames are scanned — we want to oversample WHERE flares happen, not where they existed in inputs
- Flare flags computed at `build_index()` time (dataset build), fitting existing index-multiplication pattern
- Configurable oversampling weight: `flare_oversample_weight: 3.0` in config `data:` section
- WeightedRandomSampler replaces `shuffle=True` in training DataLoader when oversampling is enabled

### Early stopping & scheduling
- Patience increased to 15-20 (from 8) to let cosine LR complete most of its 50-epoch cycle
- No warmup — CosineAnnealingLR starts at configured LR (1e-4) and decays smoothly
- Val_loss remains the early stopping metric (composite loss already includes temporal terms from Phase 8)
- Single best model checkpoint saved by val_loss (no separate CSI checkpoint)

### Training duration & hyperparameters
- 50 epochs with cosine schedule (T_max=50)
- Batch size stays at 1 (preserves memory headroom for Phase 10 architecture scaling)
- Learning rate stays at 1e-4 with AdamW
- With balanced augmentation (3x dataset), each epoch is ~1,704 steps at batch_size=1

### Config file strategy
- Update config.yaml in-place with v3.0 defaults (cosine scheduler, balanced augmentation, tf_start=0.0, 50 epochs)
- New keys go under `data:` section: `flare_oversample_weight: 3.0`
- Patience update goes in `training:` section
- Config validation cross-check: warn if `flare_oversample_weight > 1.0` but `augmentation: "none"` (suboptimal combo, consistent with Phase 8's threshold cross-check pattern)

### Claude's Discretion
- Exact patience value within 15-20 range
- Internal implementation of WeightedRandomSampler integration (how weights array is built and passed)
- Whether to log flare sequence statistics at training start (count of flare vs non-flare sequences)
- Config validation warning message wording

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `training/trainer.py:489-494`: CosineAnnealingLR already implemented — just activate via config `scheduler.type: "cosine"`
- `solarflare_data/dataset.py:268-269`: Balanced augmentation (H+V flip, 3x) already implemented — activate via config `augmentation: "balanced"`
- `solarflare_data/dataset.py:build_index()`: Index builder already mmaps files for shape — extend to scan for extreme values at build time
- `training/trainer.py:619`: Teacher forcing decay formula exists — set `tf_start: 0.0` eliminates TF entirely
- `solarflare_data/loader.py:create_dataloaders()`: Currently uses `shuffle=True` — replace with WeightedRandomSampler for training loader

### Established Patterns
- Index-multiplication augmentation: deterministic, no random state issues — flare flagging fits same pattern
- Config-driven construction via factory functions (get_loss_function, scheduler creation) — follow for sampler
- Config cross-check warnings: Phase 8 added loss/eval threshold mismatch warning — same pattern for oversampling+augmentation check
- `utils/config_validator.py` handles validation with accumulated errors

### Integration Points
- `build_index()` returns `List[Tuple[int, int, int]]` — may need to return or expose flare flags alongside
- `create_dataloaders()` needs `sampler` parameter instead of `shuffle=True` when oversampling enabled
- `config.yaml` `data:` section gets `flare_oversample_weight` key
- `config.yaml` `training:` section: `patience` updated, `tf_start` set to 0.0, `epochs` to 50
- `config.yaml` `training.scheduler.type` changed from "none" to "cosine"

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Most of Phase 9 is config-driven activation of existing infrastructure. The only significant new code is WeightedRandomSampler integration with flare detection at index build time.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-training-policy*
*Context gathered: 2026-03-08*
