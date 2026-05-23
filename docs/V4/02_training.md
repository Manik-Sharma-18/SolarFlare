# V4 Training Pipeline

Branch: `Version_4`. Single config-driven entrypoint, single trainer, two configs (pretrain magnetogram, fine-tune winding-flux).

## Entry point — `main.py`

CLI: `python3 main.py --config <path>`. Nothing else; every parameter is YAML.

Top-level config keys consumed: `device, seed, data, normalization, model, training, loss, evaluation, output, logging, error_handling, uncertainty, resume_from, transfer_learning`.

Flow: `resolve_device` → `set_seed` → build datasets via `solarflare_data.loader` → instantiate `SolarFluxPredictor` → `Trainer.fit()` → reload `best_model.pt` → `validate()` on test loader → emit `test_results.json` + visualizations.

## Trainer — `training/trainer.py`

- Optimizer: **AdamW**. Scheduler: `cosine` (default), `step`, or `constant`.
- Mixed precision: `get_amp_context` + `get_grad_scaler` (CUDA only — MPS/CPU get no-op scaler that skips step on NaN/Inf grad).
- No gradient accumulation. `batch_size` is whatever YAML says.
- NaN/Inf detection per step; abort if `max_consecutive_nan` exceeded.
- Teacher forcing decays linearly from `training.tf_start` to `0.0` across epochs.
- Graceful SIGINT / SIGTERM → writes `EMERGENCY_checkpoint_epoch_NNN.pt` then exits.
- Per-epoch: train loop, val loop (returns metrics dict from `utils/metrics.py`), checkpoint logic (best vs rolling), tensorboard if configured.

## Losses — `training/losses.py`

Factory: `get_loss_function(loss_type, **kwargs)`. Three types: `l1`, `ssim`, `composite`.

`CompositeLoss` — six weighted terms:

| Term | Default weight | Spec |
|---|---|---|
| Temporal-weighted L1 | `l1=1.0` | per-timestep weights `[1.0, 1.5, 2.0, 2.5]` for `t_out=4`; deeper future weighted heavier |
| `(1 − SSIM)` | `ssim=0.5` | MS-SSIM if `use_ms_ssim` + min spatial ≥ 32, else single-scale; tiled if H/W > `ssim_tiling_threshold`; MPS uses per-channel conv2d fallback (grouped conv bug) |
| `WeightedMAELoss` | `extreme=1.0` | binary mask `|target| > extreme_threshold` (default 0.277 normalised); flagged pixels get `extreme_pixel_weight=3.0` |
| L1 on frame-to-frame diffs | `temporal_diff=1.0` | matches predicted Δ-magnitude to GT Δ-magnitude |
| Variation reward (negative) | `temporal_var_lambda=0.1` | cap at target's variation; pushes predictions away from constant output (anti-persistence-trap) |
| `AsymmetricExtremeLoss` | `asymmetric=0.5` | `asymmetric_alpha=2.0`; penalty for under-predicting extremes harder than over-predicting |

Default config:
```
extreme_threshold=0.277  # normalized asinh space (corrected by quick-02; was 0.3456)
extreme_pixel_weight=3.0
temporal_weights=[1.0, 1.5, 2.0, 2.5]
```

## Optimizer + schedule

- `optimizer.name=adamw`, `lr=3e-4` (pretrain) / `1e-4` (fine-tune), `weight_decay=1e-5`.
- `delta_scale` parameter excluded from weight decay by name match.
- Scheduler: `cosine` with `eta_min=1e-6` (set in Phase 09-01).
- No warmup at V4 default.

## Checkpoints — `utils/checkpoint.py`

- `save_checkpoint(state, path)`: `torch.save → fsync → os.replace` (atomic).
- Always CPU-mapped on save; `map_location='cpu'` on load. Resume works cross-device.
- `CHECKPOINT_VERSION=1` enforced; mismatched versions raise on load.
- `best_model.pt` overwritten only when val loss improves.
- Rolling `checkpoint_epoch_NNN_valloss_XXXX.pt`; previous rolling file unlinked when replaced.
- `EMERGENCY_checkpoint_epoch_NNN.pt` on SIGINT/SIGTERM (best-effort, no fsync).

## Transfer learning — `utils/transfer.py`

Used by `finetune_winding_flux.yaml`. Two-phase flow:

1. **Load partial weights** from `transfer_learning.pretrained_checkpoint` (typically a magnetogram-pretrained `best_model.pt`):
   - State-dict key intersection; skip mismatched shapes.
   - Skipped 2D+ tensors → `kaiming_normal_` reinit; skipped 1D → `zeros_`.
   - Log per-key load/skip decision.
2. **Encoder freeze** (`freeze_encoder=True`): freezes `encoder_conv1.`, `encoder_conv2.`, `encoder_conv3.`, `downsample1.` parameters.
3. **Two LR groups** via `get_finetune_param_groups`:
   - Input-channel-dependent layers (`preprocess.`, `decoder_input_conv.`, `input_down.`, `input_up.`): `base_lr`.
   - Everything not in the frozen-encoder list: `base_lr * lr_scale_pretrained` (default 0.1).
4. **Unfreeze schedule**: after `unfreeze_after_epoch`, rebuild optimizer + cosine scheduler over remaining epochs.

## Config validation — `utils/config_validator.py`

Single-pass error accumulation; raises `ConfigValidationError` listing all problems at once.

Enforced:
- `dual_channel=True` ⇒ `model.input_channels ≥ 2`.
- `use_amp=True` incompatible with `device=cpu`.
- `resume_from` and `transfer_learning.pretrained_checkpoint` mutually exclusive.
- Deprecates `train_split` / `val_split` → replaced by `split_ratios`.
- Deprecates boolean `augment` → replaced by string `augment ∈ {none, balanced, aggressive}`.

## Two configs — diff table

| Field | `pretrain_magnetogram.yaml` | `finetune_winding_flux.yaml` |
|---|---|---|
| `data_dir` | magnetogram dataset | `data_processed/` (preprocessed cubes) |
| `data.stride` | 2 | 4 |
| `data.split_ratios` | 0.8 / 0.1 / 0.1 | 0.7 / 0.2 / 0.1 |
| `data.dual_channel` | false | true |
| `data.crop_size` | 256×256 | 437×877 |
| `data.num_workers` | 4 | 0 |
| `data.flare_oversample_weight` | 1.0 | 5.0 |
| `normalization.asinh_softening` | 500 | 1000 |
| `model.input_channels` | 1 | 2 |
| `model.use_checkpointing` | true | false |
| `training.batch_size` | 2 | 1 |
| `training.epochs` | 50 | 30 |
| `training.lr` | 3e-4 | 1e-4 |
| `training.tf_start` | 0.5 | 0.0 |
| `training.patience` | 15 | 18 |
| `loss.extreme_weight` | 1.0 | 3.0 |
| `loss.extreme_pixel_weight` | 10 | 25 |
| `transfer_learning` block | absent | present |

Architecture, optimizer (AdamW), scheduler (cosine), composite-loss weight skeleton — match between the two.

## Root `config.yaml`

**Not** the production entrypoint. Diagnostic L1+SSIM-only setup: extreme/temporal/asymmetric weights zeroed, constant LR, 5 epochs. Used for debugging individual loss terms.

## Inference — `inference.py`

Single-sequence CLI. Flow:
1. Load checkpoint + reconstruct `SolarFluxPredictor` from saved hyperparams.
2. Read raw `.npy` (asinh-normalised) or auto-detect pre-normalised `.npz` cube.
3. Center-crop to `data.crop_size`.
4. If `dual_channel`, build channel-2 = `sigmoid(extreme indicator)`.
5. `model(x, teacher_forcing_ratio=0.0)` under `torch.no_grad()`.
6. `unnormalize_asinh` to physical units.
7. Emit predictions + optional comparison plot.

## Files reference

| File:lines | Entity |
|---|---|
| `main.py` | CLI entry; orchestrates load → train → reload → test |
| `training/trainer.py` | `Trainer` class, `fit`, `train_epoch`, `validate` |
| `training/losses.py` | `CompositeLoss`, `WeightedMAELoss`, `AsymmetricExtremeLoss`, `get_loss_function` |
| `utils/checkpoint.py` | atomic `save_checkpoint`, `load_checkpoint`, version guard |
| `utils/transfer.py` | partial-load, `freeze_encoder`, `get_finetune_param_groups`, unfreeze |
| `utils/config_validator.py` | `ConfigValidationError`, single-pass validation |
| `configs/pretrain_magnetogram.yaml` | magnetogram pretraining config |
| `configs/finetune_winding_flux.yaml` | wind-flux fine-tune config (uses transfer block) |
| `config.yaml` | diagnostic config (L1+SSIM only, 5 epochs) |
| `inference.py` | single-sequence forward + denorm |
