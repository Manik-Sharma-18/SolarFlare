"""Convert legacy .npy winding-flux structured arrays to zarr cubes.

npy format: structured array [(X_Mm, Y_Mm, windTotal, time), ...]
zarr format: wind[H, W, T] fp32  chunks=(H, W, 10)  blosc-zstd-BITSHUFFLE
             Time[T] float64 epoch-seconds           blosc-lz4

Groups:
  may2024 → windTotal_MM_2024-05-*.npy  (440×884 px, ~342 frames)
  nov2025 → windTotal_MM_2025-11-*.npy  (437×1042 px, ~837 frames)
  oct2024 → harp_11930.zarr already superset (290 zarr ⊇ 191 npy). Skipped.

Usage:
  python scripts/convert_npy_to_zarr.py --group may2024
  python scripts/convert_npy_to_zarr.py --group nov2025 --name harp_nov2025
  python scripts/convert_npy_to_zarr.py --group oct2024   # prints superset check, exits
  python scripts/convert_npy_to_zarr.py --group may2024 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

GROUPS: dict[str, str] = {
    "may2024": "2024-05",
    "oct2024": "2024-10",
    "nov2025": "2025-11",
}
WIND_COMP = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
TIME_COMP = Blosc(cname="lz4",  clevel=5, shuffle=Blosc.SHUFFLE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=list(GROUPS), required=True)
    p.add_argument("--name", default=None,
                   help="Output zarr stem under --data-dir (default: harp_<group>)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without writing zarr")
    return p.parse_args()


def collect_all_timestamps(files: list[Path]) -> np.ndarray:
    """Union of all unique epoch-second timestamps across files, sorted."""
    ts: set[int] = set()
    for f in files:
        a = np.load(str(f), mmap_mode="r")
        ts.update(a["time"].astype("datetime64[s]").astype("int64").tolist())
    return np.array(sorted(ts), dtype=np.int64)


def build_grid(
    snap: np.ndarray, ux: np.ndarray, uy: np.ndarray
) -> np.ndarray:
    """Pivot flat (X, Y, windTotal) records → [H, W] float32.

    ux/uy: sorted unique coordinate arrays (Mm). searchsorted gives pixel index.
    Out-of-bounds records silently dropped (shouldn't happen in clean data).
    """
    H, W = len(uy), len(ux)
    row = np.searchsorted(uy, snap["Y"])
    col = np.searchsorted(ux, snap["X"])
    valid = (row < H) & (col < W)
    grid = np.zeros((H, W), dtype=np.float32)
    grid[row[valid], col[valid]] = snap["windTotal"][valid].astype(np.float32)
    return grid


def convert(files: list[Path], out: Path, dry_run: bool) -> None:
    print(f"[info] Scanning {len(files)} file(s) for coordinate grid + timestamps…")
    ref = np.load(str(files[0]), mmap_mode="r")
    ux = np.unique(ref["X"])   # sorted unique X (Mm)
    uy = np.unique(ref["Y"])   # sorted unique Y (Mm)
    H, W = len(uy), len(ux)
    times = collect_all_timestamps(files)
    T = len(times)
    t0 = np.datetime64(int(times[0]),  "s")
    t1 = np.datetime64(int(times[-1]), "s")
    print(f"[info] Grid {H}×{W} px | {T} frames | {t0} → {t1}")
    print(f"[info] Output: {out}")
    if dry_run:
        print("[dry-run] Skipping write.")
        return

    store = zarr.open(str(out), mode="w")
    z_wind = store.create_dataset(
        "wind", shape=(H, W, T), dtype="f4",
        chunks=(H, W, 10), compressor=WIND_COMP,
    )
    z_time = store.create_dataset(
        "Time", shape=(T,), dtype="f8",
        chunks=(T,), compressor=TIME_COMP,
    )
    z_time[:] = times.astype(np.float64)

    # Build epoch→time-index map for fast lookup
    t_idx = {int(t): i for i, t in enumerate(times)}

    for fi, fpath in enumerate(files):
        print(f"[info] Processing file {fi + 1}/{len(files)}: {fpath.name}", flush=True)
        a = np.load(str(fpath), mmap_mode="r")
        file_times = np.unique(a["time"].astype("datetime64[s]").astype("int64"))
        for t_epoch in file_times:
            ti = t_idx[int(t_epoch)]
            t_dt = np.datetime64(int(t_epoch), "s")
            snap = a[a["time"].astype("datetime64[s]") == t_dt]
            z_wind[:, :, ti] = build_grid(snap, ux, uy)
        print(f"       {len(file_times)} frames written.", flush=True)

    print(f"[ok] {out}  shape=({H}, {W}, {T})")


def check_oct2024_superset(data_dir: Path) -> None:
    """Oct 2024 npy is a strict subset of harp_11930.zarr — no action needed."""
    zpath = data_dir / "harp_11930.zarr"
    if not zpath.exists():
        print(f"[warn] {zpath} not found — cannot verify")
        return
    zz = zarr.open(str(zpath), "r")
    zarr_times = set(zz["Time"][:][zz["Time"][:] > 0].astype(int).tolist())
    npy_times: set[int] = set()
    for f in sorted(data_dir.glob("windTotal_MM_2024-10-*.npy")):
        a = np.load(str(f), mmap_mode="r")
        npy_times.update(a["time"].astype("datetime64[s]").astype("int64").tolist())
    npy_only = npy_times - zarr_times
    if npy_only:
        print(f"[WARN] oct2024: {len(npy_only)} npy timestamps NOT in zarr — needs merge!")
        for t in sorted(npy_only)[:5]:
            print(f"  {np.datetime64(t, 's')}")
    else:
        extra = len(zarr_times) - len(npy_times)
        print(
            f"[ok] oct2024: harp_11930.zarr is superset.\n"
            f"     zarr={len(zarr_times)} valid frames, npy={len(npy_times)} frames, "
            f"zarr has {extra} extra frames not in npy.\n"
            f"     No merge needed — harp_11930.zarr is already the canonical cube."
        )


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.group == "oct2024":
        check_oct2024_superset(data_dir)
        return 0

    prefix = GROUPS[args.group]
    files = sorted(data_dir.glob(f"windTotal_MM_{prefix}-*.npy"))
    if not files:
        print(f"[err] No files matching windTotal_MM_{prefix}-*.npy in {data_dir}")
        return 1

    name = args.name or f"harp_{args.group}"
    out = data_dir / f"{name}.zarr"
    if out.exists():
        print(f"[err] {out} already exists. Delete it first or choose --name.")
        return 1

    convert(files, out, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
