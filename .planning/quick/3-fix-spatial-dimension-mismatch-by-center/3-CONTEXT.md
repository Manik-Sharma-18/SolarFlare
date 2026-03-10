# Quick Task 3: Fix spatial dimension mismatch by center-cropping all cubes to 437×877 - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Task Boundary

Fix spatial dimension mismatch across 14 data cubes. Three groups have different native dimensions (440×884, 627×877, ~520×1044) but share 0.36 Mm/pixel spacing. Current bilinear resize to 448×896 distorts physical scale differently per group, creating inconsistent Mm/pixel for ConvLSTM kernels and train/test domain gap. Fix by center-cropping all cubes to 437×877 (largest common dimensions), preserving physical scale perfectly.

</domain>

<decisions>
## Implementation Decisions

### Crop vs Resize Strategy
- Center-crop replaces bilinear resize entirely. No interpolation needed since all cubes share 0.36 Mm/pixel spacing.

### Pipeline Insertion Point
- Apply center-crop at load time in loader.py, replacing the existing F.interpolate call. Applied once per cube at load, before normalization.

### Config Handling
- Replace `target_size: [448, 896]` with `crop_size: [437, 877]` in config.yaml. Clean semantic change from resize to crop.

### Claude's Discretion
- None — all areas discussed.

</decisions>

<specifics>
## Specific Ideas

- Group A (cubes 0-3): 440×884 → crop to 437×877 (loses 3 rows height, 7 cols width)
- Group B (cubes 4-6): 627×877 → crop to 437×877 (loses 190 rows height, 0 cols width)
- Group C (cubes 7-13): ~520×1044 → crop to 437×877 (loses ~83 rows height, ~167 cols width)
- Both load paths (raw and preprocessed) in loader.py need updating
- The F.interpolate calls in load_and_prepare_data() and load_preprocessed_data() should be replaced with center-crop logic

</specifics>
