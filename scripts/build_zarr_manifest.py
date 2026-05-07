"""Scan data/*.zarr cubes; emit data/manifest.json.

Per cube: (harp_id, path, shape, valid_frame_count, time_span_s).

HARP ID = directory name (no metadata in zarr per locked spec).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr-dir", default="data/")
    p.add_argument("--out", default="data/manifest.json")
    p.add_argument("--cadence-s", type=float, default=720.0,
                   help="Expected median Δt between valid frames (12 min = 720 s).")
    p.add_argument("--cadence-tol", type=float, default=1.0)
    return p.parse_args()


def scan_cube(path: Path, cadence_s: float, cadence_tol: float) -> dict[str, Any]:
    z = zarr.open(str(path), mode="r")
    wind = z["wind"]            # [H, W, T]
    time = np.asarray(z["Time"][:], dtype=np.float64)

    h, w, t = wind.shape
    valid_mask = time > 0
    valid_count = int(valid_mask.sum())

    valid_time = time[valid_mask]
    if valid_count >= 2:
        dts = np.diff(valid_time)
        median_dt = float(np.median(dts))
        time_span = float(valid_time[-1] - valid_time[0])
    else:
        median_dt = -1.0
        time_span = 0.0

    cadence_ok = abs(median_dt - cadence_s) < cadence_tol if valid_count >= 2 else False

    return {
        "harp_id": path.stem,
        "path": str(path),
        "shape_hwt": [int(h), int(w), int(t)],
        "valid_frame_count": valid_count,
        "time_span_s": time_span,
        "median_dt_s": median_dt,
        "cadence_ok": bool(cadence_ok),
    }


def main() -> int:
    args = parse_args()
    zarr_dir = Path(args.zarr_dir)
    cubes = sorted(zarr_dir.glob("*.zarr"))
    if not cubes:
        print(f"[warn] No .zarr cubes under {zarr_dir.resolve()}")
        return 1

    entries: list[dict[str, Any]] = []
    for cube in cubes:
        try:
            entry = scan_cube(cube, args.cadence_s, args.cadence_tol)
            entries.append(entry)
            print(f"[ok] {entry['harp_id']:20s} shape={entry['shape_hwt']} "
                  f"valid={entry['valid_frame_count']} dt={entry['median_dt_s']:.1f}s "
                  f"{'' if entry['cadence_ok'] else '[CADENCE MISMATCH]'}")
        except Exception as exc:
            print(f"[err] {cube.name}: {exc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cubes": entries}, indent=2))
    print(f"[info] Wrote {out} ({len(entries)} cubes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
