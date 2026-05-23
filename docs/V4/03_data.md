# V4 — Data Pipeline

Branch: `Version_4`. This doc covers the ConvLSTM-era data path: raw structured-array `.npy` files to preprocessed `.npz` cubes to mmap-backed sliding-window `Dataset`.

**Scope note.** V4 does *not* consume the `*.zarr` HARP cubes. Zarr + `wind[H,W,T]` + `Time[T]` + `WIND_FLUX_CLIP` sentinel-style ingest is **V5 only** (`solarflare_data/zarr_loader.py`, archived). V4's raw layer is a structured numpy array — see below.

## Raw data — `windTotal_MM_*.npy` structured arrays

V4 ingests `.npy` files whose dtype carries four named fields:

| Field | Meaning |
|---|---|
| `X` | x-coordinate (pixel / grid index) |
| `Y` | y-coordinate |
| `time` | timestamp |
| `windTotal` | winding-flux value |

These are **sparse record lists**, not dense cubes. `loader.py::_structured_to_cube` and the identical helper in `preprocess_data.py` densify by taking `np.unique` on each of X/Y/time and scattering values into a `(T, H, W)` `float32` array (zero-initialised — missing records stay zero, no NaN convention).

Files on disk in `data/` matching the V4 pattern `windTotal_MM_<date>_<HHMM>_<HHMM>.npy`:

- **May 2024:** 2024-05-08, 09, 10 (full day 0000–2348), 2024-05-11 (0000–0400). 4 files.
- **Oct 2024:** 2024-10-01 (1800–2348), 10-02 (full), 10-03 (0000–1500). 3 files.
- **Nov 2025:** 2025-11-05 through 11-11 (seven full days). 7 files.

→ **14 raw windTotal files total** (matches the 14 `cube_*.npz` in `data_processed/`).

The 21 HARP-named `.zarr` directories under `data/` (harp_8, 17, 26, 43, 45, 49, 51, 54, 83, 86, 116, 156, 219, 221, 245, 274, 316, 318, 11930, may2024, nov2025) were added for V5 and are **not used by V4 code**. Listed here only as inventory.

The 12-min cadence, NaN convention, `Time==0` sentinel and per-pixel `WIND_FLUX_CLIP` described for V5 do **not** apply to V4 — V4 has no separate `Time` vector beyond the unique timestamps recovered from the structured array, and densification produces zeros (not NaNs) in untouched cells.

## Preprocess — `preprocess_data.py`

One-shot offline script: raw `.npy` to compressed `.npz` cubes in `data_processed/`.

Steps per file:

1. `structured_to_cube` — densify to `(T, H, W)` float32.
2. Stack all cubes once for global normalisation. `compute_normalization` supports three methods:
   - **`asinh`** (default): `arcsinh(x / softening) / scale`, then `clip(-1, 1)`. `softening = 1000.0`. `scale = arcsinh(p99.99(|x|) / softening)`. Also returns `extreme_threshold = p99.5(|x|)` and `extreme_threshold_normalized` for dual-channel mode.
   - **`robust`:** `(x - median) / ((p99 - p1) / 2)`.
   - **`fixed`:** `x / 40000.0`.
3. Apply normalisation per cube, save `cube_NNN.npz` with `data=` plus per-cube metadata. Write `metadata.json` containing `normalization`, `cubes[]`, `n_cubes`.

Output (current state of `data_processed/`): `cube_000.npz` through `cube_013.npz` plus `metadata.json` (14 cubes, in raw-file order before split shuffling).

**No spatial crop, no time-window slicing** in this script. Crops are applied later in the loader if `crop_size` is set (e.g. 437×877 center crop) — the script writes full-resolution normalised cubes.

## Dataset — `solarflare_data/dataset.py`

`SolarFluxDataset(Dataset)` — mmap-backed, lazy per-worker file handles, deterministic augmentation.

- **State:** only `file_paths`, precomputed `index = [(file_idx, window_start, aug_type), ...]`, `t_in`, `t_out`, `dual_channel`, `extreme_threshold`, `norm_params`, `norm_method`. No arrays or mmaps stored (spawn-safe).
- **`__getitem__`:** opens mmap on first touch per worker (`_get_mmap`), slices `X_in = mmap[s : s+t_in]`, `Y_out = mmap[s+t_in : s+t_in+t_out]`, **copies** before mutating (mmap is read-only). On-the-fly normalisation only when `norm_params` is provided (raw-loader path); preprocessed-loader path passes `norm_params=None` because cubes are already normalised on disk.
- **Augmentation codes** (integers in the index, no `__getitem__` RNG):
  `AUG_NONE=0, AUG_HFLIP=1, AUG_VFLIP=2, AUG_ROT90=3, AUG_ROT180=4, AUG_ROT270=5`.
  Sets: `balanced = {NONE, HFLIP, VFLIP}`, `aggressive = {NONE, HFLIP, VFLIP, ROT90, ROT180, ROT270}`. Train-split only; val/test always `AUG_NONE`.
- **Dual-channel mode:** ch1 = flux; ch2 = soft sigmoid extreme indicator from `_compute_extreme_channel` — `sigmoid(2 · (|x| − thr) / (0.5·thr))`. Output shape `(C, T, H, W)` with `C ∈ {1, 2}`.
- **Returns:** `(X_in, Y_out, (file_idx, window_start))` as float tensors, or `None` on read error.

`build_index` builds the sliding-window list per split. For `extreme_threshold is not None`, scans **output frames only** and flags `flare_density > 0.02` (default) — these flags drive the weighted train sampler downstream.

## Loader — `solarflare_data/loader.py`

Two public entry points; both share split logic and DataLoader construction.

- **`load_and_prepare_data(data_dir, ...)`:** raw structured-array path. Pre-flight scan rejects bad files (`failure_threshold=0.1`). If files are structured (have `dtype.names`), densifies via `_structured_to_cube` into a tempdir of `.npy` cubes; otherwise treats files as already-3D cubes. Optional `_center_crop` (e.g. `(437, 877)`) applied per cube. Computes `norm_params` from **training files only** (subsampled every 10th value via mmap). Builds three `SolarFluxDataset` instances with on-the-fly normalisation.
- **`load_preprocessed_data(preprocessed_dir, ...)`:** consumes `cube_*.npz` from `preprocess_data.py`. Re-extracts each cube's `data` to a tempdir `.npy` for mmap. Reads `metadata.json` to recover `extreme_threshold_normalized` for dual-channel mode. Datasets get `norm_params=None` (already normalised).
- **`load_harp_zarr_data(data_dir, cube_allowlist=None, window_size=None, window_stride=None, norm_method="zscore_per_cube", clip=1e8, ...)`** (`solarflare_data/harp_loader.py`): consumes `harp_*.zarr` cubes. Per cube, opens zarr `wind[H,W,T]` + `Time[T]`, drops frames with `Time<=0`, transposes to `(T,H,W)`, zeroes NaN and `|w|>clip` pixels (instrument artefacts), saves a densified `.npy` in a temp dir. Whole-file split is an **AR-identity** holdout (each zarr cube is one active region). HARP cubes have variable spatial extent (91×173 → 877×1061). Pass `window_size=128, window_stride=64` to enable the spatial sliding window: indices become 5-tuples `(file_idx, t_start, y_start, x_start, aug)`; cubes whose `min(H,W) < window_size` are skipped with a warning. Optional `cube_allowlist: list[str]` restricts to specific HARPs. The legacy `crop_size=(H,W)` parameter still exists for callers that want a one-shot center crop.

## Lessons from V5 (now applied to V4)

| Finding | V5 evidence | V4 action |
|---|---|---|
| Per-pixel physical max ~$10^7$; clip at $10^8$ (10× margin) | `archive/v5_jepa/docs/V5_JEPA/concepts/wind_flux_clipping.md`; F2 val loss 0.202 → 0.041 | `WIND_FLUX_CLIP = 1e8` in `harp_loader.py`; `bad = ~isfinite | |w|>clip → 0` |
| Quantity is a signed pseudoscalar — sign carries chirality | `archive/.../06_data.md` §11.1 | Normalisation preserves sign; `asinh` is odd → safe; `np.sign(x)·asinh(|x|/s)` for `signed_asinh` |
| Per-cube absolute-scale mismatch dominates novel-cube error | F9 — per-cube affine recovers novel-cube medAPE 22% → 9.9% | New `_compute_per_cube_norm` in `harp_loader.py`; `zscore_per_cube` is default `norm_method` |
| `Time == 0` is a sentinel for missing frames | `harp_loader._densify_harp_cube` | Drop those frames before windowing |
| NaN-safe stats everywhere | V5 used `nanmean`/`nanstd` throughout | `_compute_per_cube_norm` uses `np.nanmean / np.nanstd` |
| harp_8 has 14,220 pathological pixels up to 1.68e10 | F2 audit | Auto-zeroed by clip guard; reported in load output |
| D4 augmentations must pair spatial transforms with sign flips | `archive/.../06_data.md` §11.4 | **Resolved 2026-05-21**: `_apply_augmentation` now takes `is_pseudoscalar` and negates output for H-flip / V-flip / 90° / 270° (parity-odd); identity + 180° preserve sign. `harp_loader.load_harp_zarr_data` sets `is_pseudoscalar=True`. |
| Per-cube σ over zero-padded artefact pixels deflates scale | `_compute_per_cube_norm` ran on full population including the clip-zeroed pixels | **Resolved 2026-05-21**: stats over `sub != 0 & isfinite` only (fallback for cubes with < 100 non-zero samples). Per-cube norm dict now also carries `n_nz` and `n_sample`. |

## Spatial sliding window

When `window_size` is set, `solarflare_data/dataset.py::build_spatial_index` emits 5-tuples covering the **full** cube spatial extent — no data lost. Default `window_stride = window_size // 2` (50% overlap, MONAI-style). `_spatial_window_starts` appends a flush-right window when `(extent − win) % stride != 0` so the boundary is covered without zero-pad leakage. `SolarFluxDataset.__getitem__` recognises 5-tuple indices and slices `mmap[t_start:t_end, y_start:y_start+win, x_start:x_start+win]`.

Inference reassembly: `inference.py::infer_full_frame(model, input, window_size, stride)` tiles the full-resolution input, runs the model on each tile, and mean-aggregates overlapping predictions back into a `(T_out, H, W)` map.

Window must be square so `AUG_ROT90` / `AUG_ROT270` remain shape-compatible. `build_spatial_index` enforces this implicitly.
- **Splitting — `assign_files_to_splits`:** **whole-file** assignment via seeded shuffle. Default ratios `[0.7, 0.2, 0.1]` for train/test/val. No file is split across partitions, so within-file temporal leakage is impossible — but V4 predates the cube-level / AR-identity holdout discipline V5 enforces, so the same AR could in principle appear in both splits if it spans multiple raw files (not a concern here since all 14 raw files are distinct date windows).
- **DataLoader (`create_dataloaders`):**
  - `pin_memory = True` iff `device.type == "cuda"`.
  - `multiprocessing_context = "spawn"` on macOS (Darwin), default fork on Linux.
  - `worker_init_fn = _seed_worker` (seeds numpy + stdlib per worker from `torch.initial_seed()`).
  - `collate_fn = _skip_none_collate` drops `None` samples; returns `None` if entire batch failed.
  - `persistent_workers = True` when `num_workers > 0`.
  - Flare oversampling: if `train_flare_flags` given and `flare_oversample_weight > 1.0`, replaces `shuffle=True` with `WeightedRandomSampler` (flare windows get the higher weight, others 1.0).

## Magnetogram data

`data_magnetogram/` exists on disk but is **empty**. The V4 codebase has no magnetogram pretrain path — directory was created speculatively and never populated. No corresponding dataset class or channel-count contract in V4 code.

## Device handling — `utils/device.py` and `utils/mps_ops.py`

`resolve_device("auto"|"cuda"|"mps"|"cpu")` returns a `torch.device`. `auto` prefers CUDA → MPS → CPU. Forced devices raise `RuntimeError` if unavailable. `get_amp_context(use_amp, device)` returns `torch.amp.autocast(device_type=...)` or `nullcontext`. `get_grad_scaler` returns a real `torch.amp.GradScaler` only for CUDA + AMP; MPS and CPU always get `_DummyGradScaler` (no-op `scale` / `update`; `step` skips on NaN/Inf gradients). `clear_device_cache` does `gc.collect()` **before** `torch.{cuda,mps}.empty_cache()` to release dangling tensor refs — explicitly noted as critical for MPS, which leaks unreferenced tensors over long runs.

`mps_ops.py` provides correctness wrappers for two ops with known MPS bugs:

- **`safe_outer(a, b)`** — on MPS, replaces `torch.outer` with `a.unsqueeze(-1) * b.unsqueeze(0)` (Metal shader correctness bug). CUDA/CPU dispatch to `torch.outer`.
- **`safe_quantile(tensor, q, dim)`** — on MPS, sorts along `dim` then linearly interpolates between floor/ceil neighbours. CUDA/CPU dispatch to `torch.quantile`.

Both log a one-shot info line via `_log_mps_once()` when first triggered. `is_mps(...)` accepts tensors, devices, or string device specs.

## Visualisation and animation

`utils/visualization.py` — static matplotlib outputs:
- `visualize_predictions(model, dataset, device, n_samples, save_path)` — grid: last input frame + `t_out` predictions + final GT, per sample. Fixed `vmin/vmax = -1/1`. Source of the `comparison_*.png` / `predictions.png` files at repo root.
- `plot_training_history(history, save_path)` — 3×3 panel: loss curves, per-timestep MAE, CSI & HSS, SSIM, persistence skill per timestep, temporal variation ratio, all loss components (log), temporal terms, extreme terms. Each panel only drawn if the history dict carries its keys (backwards-compat).
- `visualize_with_uncertainty`, `visualize_uncertainty_statistics` — uncertainty-aware variants for ensemble / MC-dropout outputs.

`utils/animation.py` — temporal outputs:
- `load_flux_data(path, use_raw, percentile_clip)` — common loader for `.npy` raw or `.npz` preprocessed, returns `(cube, metadata)` with optional percentile-based colour-limit suggestion.
- `animate_flare_sequence(cube, output_path, fps, cmap='RdBu_r', vmin, vmax)` — single-panel MP4 of `(T, H, W)`. Writer = `ffmpeg` with `.gif` / pillow fallback. Source of `flare.gif` / `flare_evolution.mp4`.
- `animate_prediction_vs_truth(model, dataset, device, sample_idx, output_path, include_input=True)` — two-panel pred-vs-GT MP4 (input frames optionally prepended).
- `animate_with_uncertainty(mean, uncertainty, gt, ...)` — three-panel (mean | uncertainty | GT) MP4.
- `create_difference_animation(predictions, gt, ...)` — three-panel (pred | GT | diff) MP4 with symmetric `seismic` colormap on the difference.
- `interactive_flare_viewer(cube, timestamps, output_path)` — plotly HTML viewer with play/pause + slider (optional, requires plotly).

All MP4 calls fall back to pillow-GIF if `ffmpeg` is missing; output dirs are auto-created.

## Files reference

| Path | Role |
|---|---|
| `preprocess_data.py` | Offline densify + normalise → `data_processed/cube_*.npz` |
| `solarflare_data/__init__.py` | Re-export: `SolarFluxDataset`, `build_index`, `build_spatial_index`, `AUG_*`, loader fns, `load_harp_zarr_data`, `WIND_FLUX_CLIP` |
| `solarflare_data/dataset.py` | `SolarFluxDataset` (with `window_size`, `per_cube_norm`, `is_pseudoscalar`) + `build_index` + `build_spatial_index` |
| `solarflare_data/loader.py` | Legacy raw structured-`.npy` + preprocessed-`.npz` entry points, split, DataLoader factory |
| `solarflare_data/harp_loader.py` | HARP `*.zarr` entry point — densify, AR-identity split, per-cube norm (`_compute_per_cube_norm`), spatial sliding window |
| `main.py` | Branches on `data.loader: harp_zarr` to route through `load_harp_zarr_data`; legacy `use_preprocessed` and raw paths preserved |
| `utils/device.py` | Device resolution, AMP context, grad scaler, cache clear |
| `utils/mps_ops.py` | `safe_outer`, `safe_quantile`, `is_mps` |
| `utils/animation.py` | MP4 / GIF / HTML animations (`flare.gif`, prediction comparisons) |
| `utils/visualization.py` | Static PNGs (`comparison_*.png`, training history) |
| `data/windTotal_MM_*.npy` | 14 raw structured-array files (May 2024, Oct 2024, Nov 2025) |
| `data/*.zarr` | 21 HARP cubes — now consumed by V4 via `load_harp_zarr_data` |
| `data_processed/cube_*.npz` | 14 normalised dense cubes + `metadata.json` |
| `data_magnetogram/` | Present but empty — magnetogram pretrain not realised in V4 |
