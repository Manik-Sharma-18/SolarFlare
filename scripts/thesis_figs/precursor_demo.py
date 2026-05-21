"""Precursor demo for thesis Ch 1: harp_49 spatial-mean winding flux vs time
with GOES flare events overlaid and the 6-7 h precursor lead annotated.

Inputs:
  data/harp_49.zarr (wind, Time)
  data/hek_cache/11079_*.csv

Output:
  thesis/assets/figures/precursor_demo.pdf

Run: python3 scripts/thesis_figs/precursor_demo.py
"""
from __future__ import annotations
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ZARR = REPO / "data" / "harp_49.zarr"
HEK = REPO / "data" / "hek_cache" / "11079_20100608_20100613.csv"


def load_winding_mean(zpath: Path) -> tuple[np.ndarray, np.ndarray]:
    z = zarr.open(str(zpath), mode="r")
    # zarr layout: wind[H, W, T], Time[T]. Move T to axis 0.
    wind = np.transpose(np.asarray(z["wind"][:]), (2, 0, 1))
    time = np.asarray(z["Time"][:])
    keep = time > 1e8
    wind = wind[keep]
    time = time[keep]
    ts = np.array([datetime.utcfromtimestamp(float(t)) for t in time])
    mask_valid = np.isfinite(wind)
    num = np.where(mask_valid, wind, 0.0).reshape(len(time), -1).sum(axis=1)
    den = mask_valid.reshape(len(time), -1).sum(axis=1).astype(np.float64)
    w_mean = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return ts, w_mean


def load_flares(csvpath: Path) -> list[tuple[datetime, str]]:
    out = []
    with csvpath.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t = datetime.strptime(row["peak_time"], "%Y-%m-%d %H:%M:%S")
            out.append((t, row["class"]))
    return out


def main() -> None:
    ts, w = load_winding_mean(ZARR)
    flares = load_flares(HEK)

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(ts, w, color="#222222", linewidth=0.9, label=r"$\langle W \rangle(t)$")

    # event markers
    for t_f, cls in flares:
        color = "#d62728" if cls.startswith(("M", "X")) else "#ff7f0e"
        ax.axvline(t_f, color=color, linewidth=1.2, alpha=0.85)
        ax.text(t_f, ax.get_ylim()[1] if False else 0.98, cls,
                transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=8, color=color)

    # 6-7 h precursor window (Raphaldini 2022): shaded band before the M1.0
    m_event = next((t for t, c in flares if c.startswith("M")), None)
    if m_event is not None:
        t_lead_lo = m_event - timedelta(hours=7)
        t_lead_hi = m_event - timedelta(hours=6)
        ax.axvspan(t_lead_lo, t_lead_hi, color="#1f77b4", alpha=0.18,
                   label="Raphaldini 6--7 h precursor window")
        ax.annotate("", xy=(m_event, ax.get_ylim()[0]),
                    xytext=(t_lead_lo, ax.get_ylim()[0]),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                    linewidth=1.0))

    ax.set_xlabel("UTC")
    ax.set_ylabel(r"spatial-mean signed winding flux $\langle W \rangle$")
    ax.set_xlim(ts.min(), ts.max())
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path = OUT / "precursor_demo.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
