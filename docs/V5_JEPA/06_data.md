# 06 — Data ingest + preprocessing (.zarr fp32)

## 11.1 Format change
- Old: `.npy` float64, structured array (X / Y / windTotal / time fields), 12-min cadence.
- New: `.zarr` fp32. Sample inspected: see §11.1a.
- Loader rewrite: `zarr.open` + lazy chunked reads. `xarray` wrapper not yet needed (no coord arrays in sample).

### 11.1a Sample inspection — `data/harp_11930.zarr`
Verified structure (HARP 11930, 2024-10-01 17:00 → 2024-10-04 09:48 UTC):
```
/
├── Time (325,) float64       — Unix epoch seconds, blosc-lz4
└── wind (627, 877, 325) f32  — [H, W, T] axes, blosc-zstd BITSHUFFLE
    chunks = [627, 877, 10]   — full-spatial × 10-frame time blocks
```
- **Axis convention:** `wind[H, W, T]` per locked spec (W=X, H=Y).
- **HARP ID** lives in directory name only — store explicitly during ingest.
- **No coordinate arrays** (X, Y absent). Pixel index → physical via constant 0.364 Mm/pixel.
- **No `.zattrs`** populated. Treat metadata as filename + locked priors.
- **Signed values.** Confirmed via `analyze_wind.py` (diverging RdBu_r colormap, symmetric `±vmax`). Range heavy-tailed; visualization uses `±p98(|x|)`.
- **NaNs present.** Script uses `np.nanmin / nanmax / nanmean / nanstd / nanpercentile` consistently — NaN is a real value class, not just zero.

### 11.1b Sparse zarr storage (per senior 2026-05-03)
- **Missing chunks ≠ data gap.** Default `fill_value=0`; chunks omitted when entire chunk is zero (or already-known-empty AR pixels). Reads fill silently.
- Implication: `wind[..., t]` always returns a tensor — including for absent chunks. NO retry / NO interpolation needed.
- Drop earlier "6-hr gap at chunk 10–12" interpretation — that was sparse-storage misread.
- **However:** zero/NaN reads are ambiguous between four classes: (a) sparse-chunk fill (zero), (b) off-AR padding within a present chunk (zero), (c) genuine near-zero winding flux (zero), (d) instrument-bad / off-disk pixel (NaN). Treat (a)–(c) as numerically identical at the model input; emit a `valid_mask` channel for (d). See §11.1d open questions.

### 11.1c Time + NaN handling
- **Time semantics:** 325 entries float64 epoch seconds. Sample contained 35 zero entries (sentinel).
- **Loader rule (time):** `valid_frame = Time > 0`. Filter index list before sampling windows; never sample a window straddling a `Time==0` slot.
- **Cadence verification:** median diff of valid Times ≈ 720 s (12 min). Loader asserts `|median(Δt) − 720| < 1 s`.
- **NaN handling:** `wind = np.where(np.isnan(wind), 0.0, wind)` after read. Emit companion `valid_pixel_mask = ~np.isnan(wind_orig)` as second channel into the encoder (or as attention mask). Loss reduces over `valid_pixel_mask` only.
- **NaN-safe stats:** all per-cube normalization uses `np.nanmean / nanstd / nanpercentile`, never plain `mean / std`.

### 11.1d Open questions remaining for senior (single follow-up)
1. **Time==0 sentinel semantics**: missing-frame, end-of-sequence pad, or instrument outage? Affects whether to interpolate or just skip.
2. **Zero-pixel ambiguity**: are off-AR pixels guaranteed exact-zero, or can flux genuinely be 0.0 inside the AR? If guaranteed, we get a free `valid_pixel_mask = wind != 0` channel (otherwise must rely on NaN-only mask).
3. **NaN policy**: NaN reserved for instrument-bad / off-disk only? Or also off-AR pad?
4. ~~**Sign convention**~~ **RESOLVED 2026-05-04**: chiral pseudoscalar — augmentation table in §11.4 enables sign-flipping transforms paired with explicit negation.
5. ~~**Future `.zattrs` / coord arrays**~~ **RESOLVED 2026-05-04**: NO — pixel-index forever. Only `wind` + `Time` arrays will be provided, ever. Locked.

## 11.2 Variable spatial dims
- Drop `crop_size: [437, 877]` from config.
- Each cube keeps native (H, W). Pad to nearest multiple of patch size (16) with zero + attention mask.
- Batching: per-cube random window OR same-shape grouping (bucketed sampler).

## 11.3 Preprocessing pipeline = under review

**Existing pipeline (asinh + percentile-based normalization):**
- Method: `asinh` with softening 1000.
- Scale percentile: 99.99 → scale=7.6346.
- Extreme threshold p99.5 normalized = 0.528.
- Center-crop to 437×877 (lost ~30% Nov 2025 area).

**Status:** flagged for removal / replacement. **Future research lever** — not committed.

**Sign-preserving requirement:** winding flux is signed (confirmed §11.1a). Any normalization that takes `|x|` or `log(x)` raw breaks the field. Candidate sign-preserving transforms:
- **Per-cube z-score** (default): `(x − μ) / σ` with `μ, σ` via `nanmean / nanstd`. Linear, sign-preserving, model-side LayerNorm complementary.
- **Signed asinh**: `sign(x) · asinh(|x|/s)` — equivalent to plain `asinh(x/s)` since asinh is odd. Compresses heavy tails, sign preserved by construction.
- **Symlog**: `sign(x) · log(1 + |x|/s)` — analogous, slightly stronger compression at large |x|.

**Open questions for preprocessing redesign:**
1. Per-cube z-score vs signed-asinh vs symlog vs unscaled fp32 (model-side LayerNorm only)?
2. Per-cube vs global normalization stats?
3. Two-channel split (sign, |x|) — adds capacity, doubles input but lets encoder choose; vs single signed channel?
4. How to handle heavy tails (±2e8 with median 0) without compressing flare-relevant outliers? Tail-clip at p99.9 after asinh?
5. Drop normalization entirely if encoder has RMSNorm / LayerNorm at input? Surya already LayerNorms internally.
6. Spatial padding strategy: zero-pad (current), edge-replicate, or reflect? Zero-pad collides with off-AR-zero ambiguity; reflect may inject phantom field structure.

**Recommendation:** start with **per-cube z-score + bf16 mixed precision** as simplest sign-preserving baseline. Treat signed-asinh and symlog as ablations against this floor. Asinh-magnitude (V4 method) DROPPED — broke sign.

## 11.4 Augmentation — chiral pseudoscalar (LOCKED 2026-05-04)

Winding flux is chiral pseudoscalar. Sign carries handedness. Augmentations that mirror or rotate-by-odd-90° MUST pair with explicit negation `x' = -T(x)` to preserve physics.

| Transform | Sign effect | Augmented sample |
|---|---|---|
| H-flip (mirror about vertical axis) | flips sign | `x' = -flip_H(x)` |
| V-flip (mirror about horizontal axis) | flips sign | `x' = -flip_V(x)` |
| 90° rotation | flips sign | `x' = -rot90(x)` |
| 180° rotation | preserves sign | `x' = rot180(x)` |
| 270° rotation | flips sign | `x' = -rot270(x)` |
| Identity | — | `x' = x` |
| Time reverse | physically wrong (forecast → hindcast) | OFF |
| Random crop | loses spatial extent | OFF (variable dims already provide diversity) |
| Additive Gaussian noise | scale-dependent | OFF until normalization strategy locked (§11.3) |

- **Full V5 augmentation set: 8× expansion** — 4 rotations × {flip, no-flip}, each with sign-corrected if needed. Symmetry group = D4 (dihedral, with chirality-aware action).
- Sanity check: applying augmentation should leave loss-on-augmented-pair invariant if encoder is equivariant. If not, add equivariance regularizer or restrict to identity + 180° rotation only.
- Mask augmentation = part of SSL pretext (§5 in `03_masks_and_pathak.md`), not data preprocessing.

## 11.5 Loader specification (deferred — placeholder)

Open infra/loader specs not yet committed. Each item resolved before first training run.

| # | Item | Notes |
|---|---|---|
| 1 | **Bucketed-shape sampler** | Group cubes by `(H, W)`. Within bucket, sample fixed-shape windows for batch coherence. Across buckets, gradient accumulation across heterogeneous batches. |
| 2 | **Window slicing rule** | Sample contiguous `[t, t + t_in + t_out)` slice; require ALL `Time > 0` in slice. Reject and resample otherwise. |
| 3 | **NaN cast policy** | Apply `nan_to_num(0.0)` at loader. Emit `valid_pixel_mask = ~isnan(orig)` as second loader output. Loss-side reduce over mask. |
| 4 | **bf16 cast boundary** | Loader returns fp32 (matches zarr). Model casts to bf16 at first compute. Norm stats computed in fp32. |
| 5 | **Encoder feature cache** | Frozen Surya encoder → cache embeddings to disk per cube. Variable shapes → cache per `(cube_id, t_start, t_in, H, W)` key, store as zarr or shelve. |
| 6 | **Padding-token RoPE coords** | Pad pixels still receive physical coords `(t·12, y·0.364, x·0.364)`. Attention mask zeros out their contribution; coords just for shape consistency. |
| 7 | **Cross-cube sampling balance** | Long cubes (1000+ frames) shouldn't dominate. Sample windows uniformly per-cube, not per-frame. |
| 8 | **SimVPv2 baseline target shape** | Conv stack needs fixed `(H, W)`. Resample all cubes to median `(H, W)` for baseline-only run. NOT used for V5 main path. |
| 9 | **Cube manifest** | JSON file listing `(harp_id, path, shape, valid_frame_count, time_span)`. Generated by ingestion-time scan. |
| 10 | **Test/val split** | Hold out cubes by HARP ID, not by time-window — prevents AR-identity leakage. |
