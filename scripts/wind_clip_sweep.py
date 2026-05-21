"""Fine-grained clip sweep for ⟨|wind|⟩ probe target.

10 log-spaced clips between 1e6 and 1e8 (descending). Single streaming pass
per cube: each frame read once, abs-mean computed at all 10 clip thresholds.

Outputs:
    outputs_probe/wind_clip_sweep_curve.png
    outputs_probe/wind_clip_sweep.md
    outputs_probe/wind_clip_sweep.npz  (per-cube mean/max curves)

Knee criterion: highest clip where harp_8 suppression < 0.5 of clip=1e8 mean
AND every healthy cube preserves > 0.85 of clip=1e8 mean.

Usage:
    python3 scripts/wind_clip_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solarflare_data.zarr_loader import open_cube  # noqa: E402

CLIPS = np.logspace(8, 6, 10)  # 1e8, ..., 1e6 descending
HEALTHY_RATIO_THRESH = 0.85
HARP8_RATIO_THRESH = 0.5


def per_cube_means(cube_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Single pass over cube. Returns (mean[clip,t], max[clip,t]) fp32."""
    cube = open_cube(cube_path)
    _, _, t = cube.shape
    means = np.full((len(CLIPS), t), np.nan, dtype=np.float32)
    maxes = np.full((len(CLIPS), t), np.nan, dtype=np.float32)
    for ti in range(t):
        if not cube.valid_frames[ti]:
            continue
        frame = np.asarray(cube.wind[:, :, ti], dtype=np.float32)
        absf = np.abs(frame)
        finite = np.isfinite(frame)
        for ci, c in enumerate(CLIPS):
            mask = finite & (absf <= c)
            if mask.any():
                vals = absf[mask]
                means[ci, ti] = float(vals.mean())
                maxes[ci, ti] = float(vals.max())
    return means, maxes


def aggregate(curve: np.ndarray) -> np.ndarray:
    """Time-average over valid frames, per clip."""
    out = np.full(curve.shape[0], np.nan, dtype=np.float64)
    for ci in range(curve.shape[0]):
        v = curve[ci]
        v = v[np.isfinite(v)]
        if v.size:
            out[ci] = float(v.mean())
    return out


def plot_sweep(summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    log_clips = np.log10(CLIPS)
    for harp_id, (m_curve, x_curve) in summary.items():
        m_ratio = m_curve / m_curve[0]
        x_ratio = x_curve / x_curve[0]
        is_harp8 = harp_id == "harp_8"
        kw = dict(color="C3" if is_harp8 else "C0",
                  lw=2.4 if is_harp8 else 0.8,
                  alpha=1.0 if is_harp8 else 0.45,
                  label=harp_id if is_harp8 else None)
        axes[0].plot(log_clips, m_ratio, **kw)
        axes[1].plot(log_clips, x_ratio, **kw)
    for ax, title in zip(axes, ["⟨|wind|⟩ time-mean ratio", "⟨|wind|⟩ time-max ratio"]):
        ax.set_xlabel("log10(clip)")
        ax.set_ylabel("ratio vs clip=1e8")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axhline(1.0, color="k", lw=0.3)
        ax.axhline(HEALTHY_RATIO_THRESH, color="g", lw=0.5, ls="--",
                   label=f"healthy floor {HEALTHY_RATIO_THRESH}")
        ax.axhline(HARP8_RATIO_THRESH, color="r", lw=0.5, ls="--",
                   label=f"harp_8 ceiling {HARP8_RATIO_THRESH}")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.suptitle("Clip sweep — red = harp_8, blue = healthy cubes (21 total)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_md(summary: dict, out_path: Path) -> tuple[float, list[str]]:
    lines = ["# Wind clip sweep — 10 log-spaced clips between 1e6 and 1e8", ""]
    lines.append(f"Clips (descending): {', '.join(f'{c:.2e}' for c in CLIPS)}")
    lines.append("")
    lines.append("## Mean ratio vs clip=1e8")
    lines.append("")
    header = "| harp_id | " + " | ".join(f"{np.log10(c):.2f}" for c in CLIPS) + " |"
    sep = "|---" * (len(CLIPS) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for harp_id, (m_curve, _) in summary.items():
        ratio = m_curve / m_curve[0]
        cells = " | ".join(f"{r:.3f}" if np.isfinite(r) else "—" for r in ratio)
        lines.append(f"| {harp_id} | {cells} |")
    lines.append("")
    lines.append("## Knee analysis")
    lines.append("")
    lines.append("| log10(clip) | clip | harp_8 ratio | min healthy ratio | passes knee? |")
    lines.append("|---|---|---|---|---|")
    healthy = [h for h in summary if h != "harp_8"]
    passing = []
    for ci, c in enumerate(CLIPS):
        h8 = summary["harp_8"][0][ci] / summary["harp_8"][0][0]
        hmin = min(summary[h][0][ci] / summary[h][0][0] for h in healthy)
        ok = (h8 < HARP8_RATIO_THRESH) and (hmin > HEALTHY_RATIO_THRESH)
        if ok:
            passing.append(c)
        lines.append(f"| {np.log10(c):.2f} | {c:.2e} | {h8:.3f} | {hmin:.3f} | {'✓' if ok else '✗'} |")
    lines.append("")
    if passing:
        rec = max(passing)
        lines.append(f"**Recommended clip:** {rec:.2e} (log10={np.log10(rec):.2f}) — "
                     f"largest clip satisfying harp_8<{HARP8_RATIO_THRESH} AND healthy>{HEALTHY_RATIO_THRESH}.")
    else:
        rec = np.nan
        lines.append("**No clip passes knee criterion.** Relax thresholds or accept tradeoff.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    return float(rec) if passing else np.nan, [f"{c:.2e}" for c in passing]


def main() -> int:
    with Path("data/manifest.json").open() as fh:
        cubes = sorted(json.load(fh)["cubes"], key=lambda c: c["harp_id"])

    outdir = Path("outputs_probe")
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print(f"Sweeping {len(CLIPS)} clips × {len(cubes)} cubes (streaming)…")
    for c in cubes:
        if not Path(c["path"]).exists():
            continue
        print(f"  {c['harp_id']:<14}", end="", flush=True)
        means, maxes = per_cube_means(c["path"])
        m_agg = aggregate(means)
        x_agg = aggregate(maxes)
        summary[c["harp_id"]] = (m_agg, x_agg)
        print(f"  mean@1e8={m_agg[0]:.2e}  mean@1e6={m_agg[-1]:.2e}")

    np.savez(outdir / "wind_clip_sweep.npz",
             clips=CLIPS,
             harp_ids=np.array(list(summary.keys())),
             means=np.stack([summary[h][0] for h in summary]),
             maxes=np.stack([summary[h][1] for h in summary]))
    print(f"wrote {outdir / 'wind_clip_sweep.npz'}")

    plot_sweep(summary, outdir / "wind_clip_sweep_curve.png")
    rec, passing = write_md(summary, outdir / "wind_clip_sweep.md")
    print()
    print(f"Knee-passing clips: {passing}")
    print(f"Recommended: {rec:.2e}" if np.isfinite(rec) else "Recommended: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
