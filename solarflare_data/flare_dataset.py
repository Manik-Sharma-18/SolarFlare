"""Flare-classifier dataset: cached encoder features [T,D] + binary flare labels [T].

For each (cube, frame_idx) returns (z [D], y_bin scalar, meta).
- Drops frames where feature row is NaN (Time==0 sentinel or all-padded pool).
- Drops frames where label is undefined (sentinel time → no forward window).
- Label files: `data/{harp}_labels_{cls}_{window}h.npy` produced by
  `scripts/build_flare_labels.py`. cls ∈ {'C','M','X'}; window ∈ {12,24,48}.

Cubes lacking a label file (e.g. date-named harps without NOAA mapping) are
silently dropped — caller is responsible for filtering the harp list upstream.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_feats(data_dir: Path, harp: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    feats = np.load(data_dir / f"{harp}_feat_{tag}.npy")
    valid = np.load(data_dir / f"{harp}_feat_{tag}_valid.npy").astype(bool)
    if feats.shape[0] != valid.shape[0]:
        raise ValueError(f"{harp}: feats T={feats.shape[0]} != valid T={valid.shape[0]}")
    return feats.astype(np.float32, copy=False), valid


def _load_labels(data_dir: Path, harp: str, cls: str, window_hr: int) -> np.ndarray | None:
    p = data_dir / f"{harp}_labels_{cls}_{window_hr}h.npy"
    if not p.exists():
        return None
    return np.load(p).astype(bool)


def compute_pos_weight(harps: list[str], data_dir: Path, cls: str, window_hr: int) -> float:
    """Class imbalance weight = N_neg / max(N_pos, 1). Computed on TRAIN cubes only."""
    n_pos, n_neg = 0, 0
    for h in harps:
        lab = _load_labels(data_dir, h, cls, window_hr)
        if lab is None:
            continue
        n_pos += int(lab.sum())
        n_neg += int(lab.size - lab.sum())
    return float(n_neg) / float(max(n_pos, 1))


def filter_labeled_harps(harps: list[str], data_dir: Path, cls: str, window_hr: int) -> list[str]:
    return [h for h in harps if _load_labels(data_dir, h, cls, window_hr) is not None]


class FlareFrameDataset(Dataset):
    """Per-frame (z [D], y_bin scalar, harp_id, t).

    Drops:
    - feature-NaN frames (feat_valid==False)
    - time-sentinel frames (zero Time → label undefined; encoded as valid==False already)
    """

    def __init__(self, harps: list[str], data_dir: str | Path, feat_tag: str,
                 cls: str, window_hr: int,
                 temporal_split: tuple[str, float] | None = None) -> None:
        """temporal_split: ('train', frac) keeps first frac of valid frames per cube;
        ('eval', frac) keeps trailing (1-frac). Cuts on per-cube valid-frame ordering,
        so train/eval are disjoint in time within each cube."""
        super().__init__()
        self.harps = list(harps)
        self.data_dir = Path(data_dir)
        self.feat_tag = feat_tag
        self.cls = cls
        self.window_hr = int(window_hr)

        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        rows: list[tuple[int, int]] = []  # (cube_local_idx, frame_t)
        for ci, h in enumerate(self.harps):
            f, v_feat = _load_feats(self.data_dir, h, feat_tag)
            lab = _load_labels(self.data_dir, h, cls, self.window_hr)
            if lab is None:
                raise FileNotFoundError(
                    f"Missing labels for {h}: data/{h}_labels_{cls}_{self.window_hr}h.npy")
            T = min(f.shape[0], lab.shape[0])
            f = f[:T]; lab = lab[:T]; v_feat = v_feat[:T]
            keep = v_feat & np.all(np.isfinite(f), axis=1)
            feats.append(f); labels.append(lab); valids.append(keep)
            valid_ts = np.where(keep)[0]
            if temporal_split is not None:
                mode, frac = temporal_split
                cut = int(round(frac * valid_ts.size))
                ts_sel = valid_ts[:cut] if mode == "train" else valid_ts[cut:]
            else:
                ts_sel = valid_ts
            for t in ts_sel:
                rows.append((ci, int(t)))

        self._feats = feats
        self._labels = labels
        self._valids = valids
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def dim(self) -> int:
        return int(self._feats[0].shape[1]) if self._feats else 0

    def pos_count(self) -> tuple[int, int]:
        n_pos = n = 0
        for ci, t in self._rows:
            n += 1
            n_pos += int(bool(self._labels[ci][t]))
        return n_pos, n

    def __getitem__(self, idx: int) -> dict:
        ci, t = self._rows[idx]
        z = self._feats[ci][t]
        y = bool(self._labels[ci][t])
        return {
            "z": torch.from_numpy(z.copy()).float(),
            "y": torch.tensor(1.0 if y else 0.0, dtype=torch.float32),
            "harp": self.harps[ci],
            "t": t,
        }

    def per_cube_arrays(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for ci, h in enumerate(self.harps):
            out[h] = {"feat": self._feats[ci], "label": self._labels[ci],
                      "valid": self._valids[ci]}
        return out


def write_split_record(out_dir: Path, train: list[str], val: list[str],
                       novel: list[str], cls: str, window_hr: int,
                       pos_weight: float) -> None:
    rec = {"train": train, "val": val, "novel_eval": novel,
           "cls": cls, "window_hr": window_hr, "pos_weight": pos_weight}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(rec, indent=2))
