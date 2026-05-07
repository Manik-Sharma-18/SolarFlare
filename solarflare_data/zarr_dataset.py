"""ZarrCubeDataset + BucketedShapeSampler + D4 chiral augmentation.

Spec: docs/V5_JEPA/06_data.md §11.4 (chiral pseudoscalar D4) + §11.5 (loader items 1, 2, 3, 7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .zarr_loader import (
    CubeHandle,
    cube_norm_stats,
    iter_valid_starts,
    open_cube,
    read_window,
)


# (rot_k, flip_h, sign_flip) — full D4 with chiral correction per §11.4.
D4_CHIRAL_OPS: tuple[tuple[int, bool, bool], ...] = (
    (0, False, False),  # identity
    (2, False, False),  # 180°
    (0, True,  True),   # H-flip + negate
    (2, True,  True),   # V-flip = rot180 ∘ H-flip + negate
    (1, False, True),   # 90° + negate
    (3, False, True),   # 270° + negate
    (1, True,  False),  # 90° ∘ H-flip — diagonal mirror, sign preserved
    (3, True,  False),  # 270° ∘ H-flip — anti-diagonal mirror, sign preserved
)


@dataclass
class WindowIndex:
    cube_idx: int
    t_start: int
    aug_op: int  # index into D4_CHIRAL_OPS


class ZarrCubeDataset(Dataset):
    """Sliding-window dataset over zarr cubes with D4 chiral augmentation.

    Returns dict with:
        wind:        FloatTensor [t_in + t_out, 1, H, W]   z-scored, NaN→0
        valid_mask:  BoolTensor  [t_in + t_out, 1, H, W]
        meta:        dict (cube_idx, harp_id, t_start, aug_op, mu, sigma, shape)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        t_in: int,
        t_out: int,
        augment: bool = True,
        cube_subset: list[str] | None = None,
        cubes_dir_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.window = self.t_in + self.t_out
        self.augment = augment

        manifest = json.loads(Path(manifest_path).read_text())
        entries = manifest["cubes"]
        if cube_subset is not None:
            keep = set(cube_subset)
            entries = [e for e in entries if e["harp_id"] in keep]
        if cubes_dir_root is not None:
            root = Path(cubes_dir_root)
            for e in entries:
                e["path"] = str(root / Path(e["path"]).name)

        self.entries = entries
        self.cubes: list[CubeHandle] = [open_cube(e["path"]) for e in entries]
        self._norm_cache: dict[int, tuple[float, float]] = {}

        # Precompute per-cube valid window starts; build index uniformly per cube
        # (item 7 — long cubes shouldn't dominate). One index entry per (cube, start, aug).
        self.index: list[WindowIndex] = []
        ops = D4_CHIRAL_OPS if augment else D4_CHIRAL_OPS[:1]
        for ci, cube in enumerate(self.cubes):
            starts = list(iter_valid_starts(cube, self.window))
            for t in starts:
                for ai in range(len(ops)):
                    self.index.append(WindowIndex(ci, t, ai))

    def __len__(self) -> int:
        return len(self.index)

    def _get_norm(self, ci: int) -> tuple[float, float]:
        cached = self._norm_cache.get(ci)
        if cached is not None:
            return cached
        stats = cube_norm_stats(self.cubes[ci])
        self._norm_cache[ci] = stats
        return stats

    def __getitem__(self, idx: int) -> dict:
        item = self.index[idx]
        cube = self.cubes[item.cube_idx]
        wind, valid = read_window(cube, item.t_start, self.window)  # [H, W, T], [H, W, T]
        mu, sigma = self._get_norm(item.cube_idx)
        wind = (wind - mu) / sigma

        # [H, W, T] -> [T, 1, H, W]
        wind_t = torch.from_numpy(wind).permute(2, 0, 1).unsqueeze(1).contiguous()
        valid_t = torch.from_numpy(valid).permute(2, 0, 1).unsqueeze(1).contiguous()

        rot_k, flip_h, sign_flip = D4_CHIRAL_OPS[item.aug_op]
        if rot_k:
            wind_t = torch.rot90(wind_t, k=rot_k, dims=(-2, -1))
            valid_t = torch.rot90(valid_t, k=rot_k, dims=(-2, -1))
        if flip_h:
            wind_t = torch.flip(wind_t, dims=(-1,))
            valid_t = torch.flip(valid_t, dims=(-1,))
        if sign_flip:
            wind_t = -wind_t

        return {
            "wind": wind_t.float(),
            "valid_mask": valid_t.bool(),
            "meta": {
                "cube_idx": item.cube_idx,
                "harp_id": cube.harp_id,
                "t_start": item.t_start,
                "aug_op": item.aug_op,
                "mu": mu,
                "sigma": sigma,
                "shape_hw": (int(wind_t.shape[-2]), int(wind_t.shape[-1])),
            },
        }


class BucketedShapeSampler(Sampler[list[int]]):
    """Group dataset indices by post-augmentation (H, W) so each batch is shape-coherent.

    Cross-bucket diversity comes from grad accumulation in the trainer (item 1, §11.5).
    """

    def __init__(
        self,
        dataset: ZarrCubeDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        buckets: dict[tuple[int, int], list[int]] = {}
        for i, item in enumerate(dataset.index):
            cube = dataset.cubes[item.cube_idx]
            h, w, _ = cube.shape
            rot_k, _, _ = D4_CHIRAL_OPS[item.aug_op]
            shape = (w, h) if rot_k % 2 else (h, w)
            buckets.setdefault(shape, []).append(i)
        self.buckets = buckets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for shape, ids in self.buckets.items():
            ids = list(ids)
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                chunk = ids[i:i + self.batch_size]
                if chunk:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n = 0
        for ids in self.buckets.values():
            n += (len(ids) + self.batch_size - 1) // self.batch_size
        return n
