"""Probe dataset: cached encoder features [T, D] + cached 1D wind target [T] per cube.

For each (cube, frame_idx) it returns (z [D], y_norm scalar, meta).
- Drops frames where feature row is NaN (Time==0 sentinel or all-padded pool).
- Drops frames where target is NaN (zero valid pixels).
- Applies log10(y+1) and z-score using globally fitted (μ_log, σ_log) — fit on
  TRAIN cubes only (no val leakage). Stats supplied by the trainer.

All cube paths are derived from `data/manifest.json`. Feature cache filename
follows `<harp>_feat_<feat_tag>.npy`; target cache from `wind_target.cache_path()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .wind_target import WIND_PROBE_CLIP, cache_path as wind_cache_path


@dataclass
class TargetStats:
    mu: float
    sigma: float

    def normalize(self, y: np.ndarray) -> np.ndarray:
        return ((np.log10(y + 1.0) - self.mu) / self.sigma).astype(np.float32)

    def invert(self, y_norm: np.ndarray) -> np.ndarray:
        return (np.power(10.0, y_norm * self.sigma + self.mu) - 1.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mu": float(self.mu), "sigma": float(self.sigma)}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetStats":
        return cls(mu=float(d["mu"]), sigma=float(d["sigma"]))


def _load_feats(data_dir: Path, harp: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    feats = np.load(data_dir / f"{harp}_feat_{tag}.npy")              # [T, D]
    valid = np.load(data_dir / f"{harp}_feat_{tag}_valid.npy").astype(bool)
    if feats.shape[0] != valid.shape[0]:
        raise ValueError(f"{harp}: feats T={feats.shape[0]} != valid T={valid.shape[0]}")
    return feats.astype(np.float32, copy=False), valid


def _load_target(data_dir: Path, harp: str, abs_value: bool, clip: float) -> np.ndarray:
    cube_path = data_dir / f"{harp}.zarr"
    p = wind_cache_path(cube_path, abs_value=abs_value, clip=clip)
    return np.load(p).astype(np.float32, copy=False)                  # [T]


def fit_target_stats(harps: list[str], data_dir: Path, abs_value: bool, clip: float) -> TargetStats:
    vals: list[np.ndarray] = []
    for h in harps:
        y = _load_target(data_dir, h, abs_value=abs_value, clip=clip)
        finite = y[np.isfinite(y) & (y > 0)]
        if finite.size:
            vals.append(np.log10(finite + 1.0))
    flat = np.concatenate(vals) if vals else np.array([0.0], dtype=np.float32)
    mu = float(np.mean(flat))
    sigma = float(np.std(flat))
    if sigma < 1e-6:
        sigma = 1.0
    return TargetStats(mu=mu, sigma=sigma)


def split_cubes_for_probe(manifest_path: Path, val_frac: float, seed: int,
                          allowlist: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Same logic as main_v5.split_cubes_by_harp; duplicated to avoid hard dep."""
    import random
    entries = json.loads(Path(manifest_path).read_text())["cubes"]
    ids = sorted(e["harp_id"] for e in entries)
    if allowlist is not None:
        keep = set(allowlist)
        ids = [i for i in ids if i in keep]
    if len(ids) < 2:
        raise ValueError(f"Need ≥2 cubes after allowlist; got {ids}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


class ProbeFrameDataset(Dataset):
    """Per-frame (z [D], y_norm scalar, harp_id, t) — drops NaN frames."""

    def __init__(self, harps: list[str], data_dir: str | Path, feat_tag: str,
                 target_abs: bool, target_clip: float, stats: TargetStats) -> None:
        super().__init__()
        self.harps = list(harps)
        self.data_dir = Path(data_dir)
        self.feat_tag = feat_tag
        self.target_abs = target_abs
        self.target_clip = target_clip
        self.stats = stats

        rows: list[tuple[int, int]] = []   # (cube_local_idx, frame_t)
        feats: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        for ci, h in enumerate(self.harps):
            f, v_feat = _load_feats(self.data_dir, h, feat_tag)
            y = _load_target(self.data_dir, h, abs_value=target_abs, clip=target_clip)
            T = min(f.shape[0], y.shape[0])
            f = f[:T]; y = y[:T]; v_feat = v_feat[:T]
            keep = v_feat & np.isfinite(y) & (y > 0) & np.all(np.isfinite(f), axis=1)
            feats.append(f); ys.append(y); valids.append(keep)
            for t in np.where(keep)[0]:
                rows.append((ci, int(t)))

        self._feats = feats          # list of [T_i, D]
        self._ys = ys                # list of [T_i]
        self._valids = valids        # list of [T_i] bool
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def dim(self) -> int:
        return int(self._feats[0].shape[1]) if self._feats else 0

    def __getitem__(self, idx: int) -> dict:
        ci, t = self._rows[idx]
        z = self._feats[ci][t]
        y = self._ys[ci][t]
        y_norm = self.stats.normalize(np.array([y], dtype=np.float32))[0]
        return {
            "z": torch.from_numpy(z.copy()).float(),
            "y_norm": torch.tensor(float(y_norm), dtype=torch.float32),
            "y": torch.tensor(float(y), dtype=torch.float32),
            "harp": self.harps[ci],
            "t": t,
        }

    def per_cube_arrays(self) -> dict[str, dict]:
        """Return dict harp_id → {'feat':[T,D], 'y':[T], 'valid':[T]} for eval."""
        out: dict[str, dict] = {}
        for ci, h in enumerate(self.harps):
            out[h] = {"feat": self._feats[ci], "y": self._ys[ci], "valid": self._valids[ci]}
        return out
