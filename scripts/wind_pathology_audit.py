"""Per-cube × clip pathology audit for ⟨|wind|⟩ probe target.

For each cube and clip ∈ {1e8, 1e7, 1e6}: valid frame count, abs mean, abs max,
percentile {p50, p90, p99}, and per-frame valid-pixel count via re-reading cube.

Emits:
    outputs_probe/wind_pathology.csv
    outputs_probe/wind_pathology.md   (markdown table for thesis)

Usage:
    python3 scripts/wind_pathology_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solarflare_data.wind_target import cache_path  # noqa: E402

CLIPS = [1e8, 1e7, 1e6]


def stats_for(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"valid": 0, "mean": np.nan, "max": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}
    v = arr[finite]
    return {
        "valid": int(finite.sum()),
        "mean": float(v.mean()),
        "max": float(v.max()),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
    }


def main() -> int:
    with Path("data/manifest.json").open() as fh:
        cubes = sorted(json.load(fh)["cubes"], key=lambda c: c["harp_id"])

    outdir = Path("outputs_probe")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "wind_pathology.csv"
    md_path = outdir / "wind_pathology.md"

    rows: list[dict] = []
    for c in cubes:
        harp_id = c["harp_id"]
        cube_path = c["path"]
        if not Path(cube_path).exists():
            continue
        row = {"harp_id": harp_id}
        for clip in CLIPS:
            cp = cache_path(cube_path, abs_value=True, clip=clip)
            if not cp.exists():
                continue
            arr = np.load(cp).astype(np.float32, copy=False)
            s = stats_for(arr)
            tag = f"{clip:.0e}"
            for k, v in s.items():
                row[f"{tag}_{k}"] = v
            row["n_frames"] = arr.size
        rows.append(row)

    fields = ["harp_id", "n_frames"]
    for clip in CLIPS:
        tag = f"{clip:.0e}"
        for k in ["valid", "mean", "max", "p50", "p90", "p99"]:
            fields.append(f"{tag}_{k}")

    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    lines = []
    lines.append("# Wind-target pathology audit — ⟨|wind|⟩ per-clip stats")
    lines.append("")
    lines.append("Per-cube valid-frame count, abs mean, abs max, p50/p90/p99 at three clip levels.")
    lines.append("Lower clip = tighter outlier suppression. Target column for probe: clip=1e6.")
    lines.append("")
    lines.append("| harp_id | T | clip=1e8 mean | max | p99 | clip=1e7 mean | max | p99 | clip=1e6 mean | max | p99 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cells = [r["harp_id"], str(r.get("n_frames", "—"))]
        for clip in CLIPS:
            tag = f"{clip:.0e}"
            for k in ["mean", "max", "p99"]:
                v = r.get(f"{tag}_{k}", np.nan)
                cells.append(f"{v:.2e}" if np.isfinite(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Pathology suppression deltas (clip 1e8 → 1e6)")
    lines.append("")
    lines.append("| harp_id | mean ratio 1e8→1e6 | max ratio 1e8→1e6 |")
    lines.append("|---|---|---|")
    for r in rows:
        m8 = r.get("1e+08_mean", np.nan)
        m6 = r.get("1e+06_mean", np.nan)
        x8 = r.get("1e+08_max", np.nan)
        x6 = r.get("1e+06_max", np.nan)
        mr = m6 / m8 if np.isfinite(m8) and m8 > 0 else np.nan
        xr = x6 / x8 if np.isfinite(x8) and x8 > 0 else np.nan
        mr_s = f"{mr:.3f}" if np.isfinite(mr) else "—"
        xr_s = f"{xr:.3f}" if np.isfinite(xr) else "—"
        lines.append(f"| {r['harp_id']} | {mr_s} | {xr_s} |")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    print()
    print(f"Cubes audited: {len(rows)}")

    pathological = [r for r in rows
                    if np.isfinite(r.get("1e+06_mean", np.nan))
                    and np.isfinite(r.get("1e+08_mean", np.nan))
                    and r["1e+08_mean"] > 0
                    and (r["1e+06_mean"] / r["1e+08_mean"]) < 0.5]
    print(f"Cubes with mean drop >2× from 1e8→1e6: {len(pathological)} → {[p['harp_id'] for p in pathological]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
