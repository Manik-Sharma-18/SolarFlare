"""Build per-timestep binary flare labels for each cube.

Pipeline:
  HARP -> NOAA AR (email_data/HARP_Merged_Statistics_Generated.xlsx)
  NOAA -> flare events (data/hek_cache/{NOAA}_*.csv or hek_{NOAA}_*.csv)
  cube timestamps (zarr Time array, epoch seconds)
  for each timestep t: label[t] = 1 if any flare class >= threshold in (t, t+window_hr]

Outputs (in data/):
  {harp_id}_labels_M_24h.npy   bool [T]   any M+ in next 24h
  {harp_id}_labels_X_24h.npy   bool [T]   any X+ in next 24h
  {harp_id}_labels_C_24h.npy   bool [T]   any C+ in next 24h (sanity check, dense)
  {harp_id}_labels_meta.json   dict       {noaa, hek_files, n_events_by_class, ...}
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl

from solarflare_data.zarr_loader import open_cube


def _load_harp2noaa(xlsx: Path) -> dict[int, int]:
    wb = openpyxl.load_workbook(str(xlsx), read_only=True, data_only=True)
    ws = wb["Sheet1"]
    out: dict[int, int] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        h, n, *_ = row
        if h is not None and n is not None:
            out[int(h)] = int(n)
    return out


def _hek_files_for(noaa: int, hek_dir: Path) -> list[Path]:
    """Return all csv files in hek_dir that begin with '{noaa}_' or 'hek_{noaa}_'."""
    pats = [f"{noaa}_", f"hek_{noaa}_"]
    return sorted([p for p in hek_dir.glob("*.csv") if any(p.name.startswith(x) for x in pats)])


def _load_events(files: list[Path]) -> list[tuple[float, str]]:
    """Return list of (peak_time_epoch_sec, class_letter)."""
    out: list[tuple[float, str]] = []
    seen: set[tuple[float, str]] = set()
    for f in files:
        with f.open() as fh:
            r = csv.DictReader(fh)
            for row in r:
                t = row.get("peak_time") or row.get("event_peaktime") or ""
                cls = (row.get("class") or row.get("fl_goescls") or "").strip()
                if not t or not cls:
                    continue
                try:
                    dt = datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        dt = datetime.strptime(t.strip(), "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        continue
                ep = dt.replace(tzinfo=timezone.utc).timestamp()
                k = (ep, cls)
                if k in seen:
                    continue
                seen.add(k)
                out.append(k)
    out.sort()
    return out


_CLASS_RANK = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


def _build_labels(times: np.ndarray, events: list[tuple[float, str]],
                  window_hr: float, min_class: str) -> np.ndarray:
    """Label = True if any event with class >= min_class falls in (t, t+window].
    times: epoch sec, may include zeros (sentinel) → those get False."""
    T = times.size
    out = np.zeros(T, dtype=bool)
    if not events:
        return out
    window_s = window_hr * 3600.0
    rank_thresh = _CLASS_RANK[min_class]
    ev_t = np.array([e[0] for e in events], dtype=np.float64)
    ev_r = np.array([_CLASS_RANK.get(e[1][0].upper(), -1) for e in events], dtype=np.int64)
    keep = ev_r >= rank_thresh
    ev_t = ev_t[keep]
    if ev_t.size == 0:
        return out
    for i in range(T):
        t0 = float(times[i])
        if t0 <= 0:
            continue
        t1 = t0 + window_s
        out[i] = bool(np.any((ev_t > t0) & (ev_t <= t1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--xlsx", default="email_data/HARP_Merged_Statistics_Generated.xlsx")
    ap.add_argument("--hek-dir", default="data/hek_cache")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--window-hr", type=float, default=24.0)
    args = ap.parse_args()

    harp2noaa = _load_harp2noaa(Path(args.xlsx))
    print(f"[info] HARP->NOAA map: {len(harp2noaa)} entries")
    manifest = json.loads(Path(args.manifest).read_text())
    cubes = manifest["cubes"]
    out_dir = Path(args.out_dir)
    hek_dir = Path(args.hek_dir)

    summary: list[dict] = []
    for entry in cubes:
        hid = entry["harp_id"]
        m = hid.replace("harp_", "")
        try:
            h_int = int(m)
        except ValueError:
            print(f"[skip] {hid} (date-named, no NOAA mapping)")
            summary.append({"harp_id": hid, "noaa": None, "skipped": "date-named"})
            continue
        noaa = harp2noaa.get(h_int)
        if noaa is None:
            print(f"[skip] {hid} (no NOAA in xlsx)")
            summary.append({"harp_id": hid, "noaa": None, "skipped": "no-noaa"})
            continue
        files = _hek_files_for(noaa, hek_dir)
        events = _load_events(files)
        n_by_cls = {c: sum(1 for _, k in events if k.startswith(c)) for c in "ABCMX"}

        cube = open_cube(Path(entry["path"]))
        times = cube.time
        T = times.size
        labels = {
            "C": _build_labels(times, events, args.window_hr, "C"),
            "M": _build_labels(times, events, args.window_hr, "M"),
            "X": _build_labels(times, events, args.window_hr, "X"),
        }

        for cls, arr in labels.items():
            np.save(out_dir / f"{hid}_labels_{cls}_{int(args.window_hr)}h.npy", arr)

        pos_M = int(labels["M"].sum())
        pos_X = int(labels["X"].sum())
        pos_C = int(labels["C"].sum())
        valid_T = int((times > 0).sum())
        meta = {
            "harp_id": hid, "noaa": noaa, "hek_files": [p.name for p in files],
            "n_events_by_class": n_by_cls, "T": T, "valid_T": valid_T,
            "window_hr": args.window_hr,
            "pos_rate": {"C": pos_C / max(valid_T, 1), "M": pos_M / max(valid_T, 1),
                          "X": pos_X / max(valid_T, 1)},
        }
        (out_dir / f"{hid}_labels_meta.json").write_text(json.dumps(meta, indent=2))
        summary.append(meta)
        print(f"  {hid:>15} NOAA={noaa} events C/M/X={n_by_cls['C']}/{n_by_cls['M']}/{n_by_cls['X']}"
              f"  pos[M]={pos_M}/{valid_T} ({100*pos_M/max(valid_T,1):.1f}%)"
              f"  pos[X]={pos_X}/{valid_T} ({100*pos_X/max(valid_T,1):.1f}%)")

    (out_dir / "_flare_labels_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary -> {out_dir / '_flare_labels_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
