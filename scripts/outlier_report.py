"""Per-cube outlier audit for winding flux.

Physical reference: accepted peak winding flux ~1e13, very active regions ~1e14.
So real outliers are values >> 1e14, not 1e5.

Reports:
  - count of outlier pixels per cube
  - frame indices that contain them
  - timestamps (UTC) of those frames
  - peak |value| seen
  - fraction of total non-NaN pixels affected
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

BZ_CLIP = 1.0e7  # per-pixel winding flux physical max (per senior); integrated AR total ~1e13–1e14
DATA = Path("data")


def ts(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def audit_cube(zpath: Path) -> dict:
    z = zarr.open(str(zpath), "r")
    wind = z["wind"][:]              # (H, W, T)
    times = z["Time"][:]             # (T,)
    H, W, T = wind.shape

    finite = np.isfinite(wind)
    nan_count = int((~finite).sum())
    safe = wind[finite]

    extreme_mask = finite & (np.abs(wind) > BZ_CLIP)
    n_out = int(extreme_mask.sum())

    if n_out == 0:
        return {
            "cube": zpath.stem, "shape": (H, W, T), "T": T,
            "n_extreme": 0, "frac": 0.0, "peak": float(np.abs(safe).max()),
            "nan_pct": 100.0 * nan_count / wind.size,
            "frames": [], "first_ts": ts(times[0]), "last_ts": ts(times[-1]),
            "stats_before": (float(safe.mean()), float(safe.std()),
                             float(np.abs(safe).max())),
        }

    per_frame = extreme_mask.sum(axis=(0, 1))   # outliers in each frame
    bad_frames = np.where(per_frame > 0)[0]
    bad_ts = [(int(i), ts(times[i]), int(per_frame[i])) for i in bad_frames]

    peak_idx = np.unravel_index(np.argmax(np.abs(wind) * extreme_mask), wind.shape)
    peak_val = float(wind[peak_idx])
    peak_y, peak_x, peak_t = peak_idx

    masked = np.where(extreme_mask, 0.0, wind)
    masked_finite = np.isfinite(masked)
    safe_after = masked[masked_finite]

    return {
        "cube": zpath.stem, "shape": (H, W, T), "T": T,
        "n_extreme": n_out, "frac": 100.0 * n_out / wind.size,
        "peak": float(np.abs(wind[finite]).max()),
        "peak_val": peak_val, "peak_yx_t": (int(peak_y), int(peak_x), int(peak_t)),
        "peak_ts": ts(times[peak_t]),
        "nan_pct": 100.0 * nan_count / wind.size,
        "n_bad_frames": int(len(bad_frames)),
        "frames": bad_ts[:10],
        "first_ts": ts(times[0]), "last_ts": ts(times[-1]),
        "stats_before": (float(safe.mean()), float(safe.std()),
                         float(np.abs(safe).max())),
        "stats_after":  (float(safe_after.mean()), float(safe_after.std()),
                         float(np.abs(safe_after).max())),
    }


def main() -> None:
    cubes = sorted(DATA.glob("*.zarr"))
    print(f"OUTLIER_THRESHOLD = {BZ_CLIP:.0e}  (per-pixel winding flux; physical max per senior)\n")
    print(f"{'cube':<14} {'T':>4} {'n_extreme':>10} {'frac %':>8} "
          f"{'peak |G|':>12} {'σ before→after':>22}")
    print("-" * 80)
    rows = []
    for c in cubes:
        r = audit_cube(c)
        rows.append(r)
        if r["n_extreme"] == 0:
            print(f"{r['cube']:<14} {r['T']:>4} {0:>10} {0.0:>8.4f} "
                  f"{r['peak']:>12.2e}   (clean)")
        else:
            sb = r["stats_before"][1]; sa = r["stats_after"][1]
            print(f"{r['cube']:<14} {r['T']:>4} {r['n_extreme']:>10} "
                  f"{r['frac']:>8.4f} {r['peak']:>12.2e} "
                  f"{sb:>10.1f} → {sa:>9.1f}")

    print("\n=== Per-cube detail ===\n")
    for r in rows:
        if r["n_extreme"] == 0:
            continue
        print(f"[{r['cube']}]  shape={r['shape']}  cube span: {r['first_ts']} → {r['last_ts']}")
        print(f"  outlier pixels: {r['n_extreme']:,} ({r['frac']:.4f}% of cube)")
        print(f"  bad frames: {r['n_bad_frames']} of {r['T']}")
        print(f"  peak: {r['peak_val']:+.3e} G at (y={r['peak_yx_t'][0]}, "
              f"x={r['peak_yx_t'][1]}, t={r['peak_yx_t'][2]}) "
              f"timestamp={r['peak_ts']}")
        print(f"  σ before clip = {r['stats_before'][1]:.1f} G  →  "
              f"σ after = {r['stats_after'][1]:.1f} G")
        print(f"  first 10 bad frames (t_idx, UTC, n_pixels):")
        for t_i, t_s, n in r["frames"]:
            print(f"    t={t_i:>4}  {t_s}  n_pixels={n}")
        print()


if __name__ == "__main__":
    main()
