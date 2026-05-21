"""Batch-compute ⟨wind⟩(t) 1D targets for every cube in manifest.

Output: data/<harp_id>_wind_1d{_abs}{_clip1eN}.npy

Usage:
    python3 scripts/compute_wind_1d.py                       # signed + abs, clip 1e8
    python3 scripts/compute_wind_1d.py --recompute           # invalidate cache
    python3 scripts/compute_wind_1d.py --signed-only         # skip abs flavor
    python3 scripts/compute_wind_1d.py --clips 1e8 1e7 1e6   # sweep clips
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solarflare_data.wind_target import cache_path, load_or_compute_wind_1d  # noqa: E402
from solarflare_data.zarr_loader import WIND_FLUX_CLIP  # noqa: E402


def summarize(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"n": int(arr.size), "valid": 0, "min": None, "max": None, "mean": None}
    vals = arr[finite]
    return {
        "n": int(arr.size),
        "valid": int(finite.sum()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--signed-only", action="store_true")
    p.add_argument(
        "--clips",
        type=float,
        nargs="+",
        default=[WIND_FLUX_CLIP],
        help="Sequence of per-pixel clip thresholds.",
    )
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with manifest_path.open() as fh:
        manifest = json.load(fh)
    cubes = manifest.get("cubes", [])
    if not cubes:
        print(f"ERROR: empty cubes list in {manifest_path}", file=sys.stderr)
        return 1

    flavors = [("signed", False)]
    if not args.signed_only:
        flavors.append(("abs", True))

    n_jobs = len(cubes) * len(flavors) * len(args.clips)
    print(
        f"Computing wind_1d for {len(cubes)} cubes × {len(flavors)} flavors × {len(args.clips)} clips = {n_jobs} jobs."
    )
    print(f"  clips={[f'{c:.0e}' for c in args.clips]}  recompute={args.recompute}")
    print()

    total_t0 = time.time()
    rows = []
    for c in cubes:
        harp_id = c["harp_id"]
        cube_path = c["path"]
        if not Path(cube_path).exists():
            print(f"SKIP {harp_id}: {cube_path} missing")
            continue
        row = {"harp_id": harp_id}
        for flavor_name, abs_value in flavors:
            for clip in args.clips:
                cached = cache_path(cube_path, abs_value=abs_value, clip=clip).exists() and not args.recompute
                t0 = time.time()
                arr = load_or_compute_wind_1d(
                    cube_path, abs_value=abs_value, clip=clip, recompute=args.recompute
                )
                dt = time.time() - t0
                stats = summarize(arr)
                row.setdefault(flavor_name, {})[f"clip_{clip:.0e}"] = stats
                tag = "cache" if cached else "fresh"
                if stats["min"] is None:
                    print(f"  {harp_id:<14} {flavor_name:<6} clip={clip:.0e} EMPTY")
                else:
                    print(
                        f"  {harp_id:<14} {flavor_name:<6} clip={clip:.0e} {tag:<5} {dt:6.2f}s "
                        f"valid={stats['valid']:>4}/{stats['n']:<4} "
                        f"max={stats['max']:>10.3e} mean={stats['mean']:>10.3e}"
                    )
        rows.append(row)

    elapsed = time.time() - total_t0
    print()
    print(f"Done in {elapsed:.1f}s. Cubes processed: {len(rows)}/{len(cubes)}.")

    summary_path = manifest_path.parent / "wind_1d_summary.json"
    with summary_path.open("w") as fh:
        json.dump({"cubes": rows, "elapsed_s": elapsed}, fh, indent=2)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
