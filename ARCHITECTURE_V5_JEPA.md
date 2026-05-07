# V5 Architecture — JEPA + Frozen Encoder + Forecast Decoder

**Status:** Draft synthesis. Supersedes `ARCHITECTURE_V5.md` (Pathak Context Encoder draft).

**Date:** 2026-04-25 (revised after professor meeting same day; senior clarifications 2026-05-03/04).

> **2026-05-08 update:** Path A abandoned (Surya/HelioSpectFormer incompatible).
> Path B (JEPA-from-scratch) is now V5.0 primary and implemented. See
> [`docs/V5_JEPA/09_progress.md`](docs/V5_JEPA/09_progress.md) for active state.

This doc is split per IndraAstra 200-line cap. See `docs/V5_JEPA/`:

| File | Content |
|---|---|
| [00_overview.md](docs/V5_JEPA/00_overview.md) | Update log, locked assumptions (pixel scale 0.364 Mm/px, chiral pseudoscalar, no metadata), §0 TL;DR, §1 goal, §2 decision tree |
| [01_path_a.md](docs/V5_JEPA/01_path_a.md) | §3 Path A — LoRA on Surya (encoder, predictor, LoRA, loss, training recipe) |
| [02_path_b.md](docs/V5_JEPA/02_path_b.md) | §4 Path B — V-JEPA pretrain from scratch on SuryaBench |
| [03_masks_and_pathak.md](docs/V5_JEPA/03_masks_and_pathak.md) | §5 mask catalog, §6 why-not-Pathak |
| [04_multimodal_baselines.md](docs/V5_JEPA/04_multimodal_baselines.md) | §7 V5.2 multimodal extension, §8 diagnostic baselines |
| [05_open_questions.md](docs/V5_JEPA/05_open_questions.md) | §9 open questions, §10 next steps |
| [06_data.md](docs/V5_JEPA/06_data.md) | §11 data ingest + preprocessing (.zarr fp32, sparse storage, NaN, augmentation, loader spec) |
| [07_verify_summary.md](docs/V5_JEPA/07_verify_summary.md) | §12 verify dropping Pathak, §13 V4/V5.0/V5.1 summary table |
| [08_sources.md](docs/V5_JEPA/08_sources.md) | arXiv references |
| [09_progress.md](docs/V5_JEPA/09_progress.md) | **Implementation progress** — branch state, Path A→B pivot, files shipped, bugs hit, sanity results (MPS + CUDA) |

## Quick locks (the irreducible priors)
- **Pixel scale:** 0.364 Mm/pixel, constant for all data past/present/future.
- **Cadence:** 12 min.
- **Channels:** 1 (winding flux), signed fp32, NaN-aware.
- **Sign:** chiral pseudoscalar — augmentation must negate under H/V flip + 90°/270° rotation; 180° preserves.
- **Metadata:** only `wind` + `Time` arrays, ever.
- **Spatial dims:** variable per cube. No fixed (H, W).
- **Default path:** ~~Path A (LoRA on frozen Surya 366 M)~~ **Path B (JEPA-from-scratch, ViT-Small)** as of 2026-05-07. Path A unviable (Surya hard-locked to 4096/60min/13ch).
