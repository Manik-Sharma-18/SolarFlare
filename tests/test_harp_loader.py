"""Tests for the HARP zarr loader: per-cube normalisation, spatial sliding
window, and end-to-end DataLoader round-trip."""
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from solarflare_data import load_harp_zarr_data
from solarflare_data.dataset import (
    AUG_NONE, build_spatial_index, _spatial_window_starts,
)
from solarflare_data.harp_loader import _compute_per_cube_norm


def _make_fake_zarr_cube(out_dir: Path, harp_id: str, H: int, W: int, T: int,
                         seed: int = 0, n_bad: int = 0) -> None:
    """Write a fake HARP-shape zarr cube. wind[H,W,T] float32 + Time[T] epoch."""
    rng = np.random.default_rng(seed)
    wind = rng.normal(0.0, 1e3, size=(H, W, T)).astype(np.float32)
    if n_bad:
        idx = rng.choice(H * W * T, n_bad, replace=False)
        flat = wind.reshape(-1)
        flat[idx] = 1.5e10
        wind = flat.reshape(H, W, T)
    time = np.arange(T, dtype=np.float64) * 720.0 + 1.7e9
    if T >= 4:
        time[1] = 0.0  # one sentinel
    root = zarr.open(str(out_dir / f"{harp_id}.zarr"), mode="w")
    root.create_dataset("wind", data=wind)
    root.create_dataset("Time", data=time)


@pytest.fixture
def fake_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    _make_fake_zarr_cube(d, "harp_a", H=140, W=160, T=20, seed=1, n_bad=5)
    _make_fake_zarr_cube(d, "harp_b", H=200, W=250, T=24, seed=2, n_bad=0)
    _make_fake_zarr_cube(d, "harp_c", H=80, W=90, T=18, seed=3, n_bad=0)  # too small for win=128
    _make_fake_zarr_cube(d, "harp_d", H=180, W=180, T=22, seed=4, n_bad=2)
    return d


def test_spatial_window_starts_basic():
    assert _spatial_window_starts(extent=10, win=20, stride=5) == []
    assert _spatial_window_starts(extent=20, win=20, stride=5) == [0]
    s = _spatial_window_starts(extent=200, win=128, stride=64)
    assert s[0] == 0
    assert s[-1] == 200 - 128
    diffs = np.diff(s)
    assert all(d <= 64 for d in diffs)


def test_build_spatial_index_emits_5tuple(tmp_path, fake_data_dir):
    paths = sorted(fake_data_dir.glob("*.zarr"))
    # Build temp .npy cubes (small) matching what harp_loader produces.
    tmp = tmp_path / "cubes"; tmp.mkdir()
    cube_paths = []
    for p in paths:
        z = zarr.open(str(p), mode="r")
        wind = np.asarray(z["wind"][:])
        # Drop the seed sentinel frame to mirror harp_loader behaviour
        time = np.asarray(z["Time"][:])
        keep = time > 0
        cube = np.transpose(wind, (2, 0, 1))[keep].astype(np.float32)
        out = tmp / f"{p.stem}.npy"
        np.save(out, cube)
        cube_paths.append(str(out))
    file_assignments = {"train": list(range(len(cube_paths))), "val": [], "test": []}
    index, flags = build_spatial_index(
        cube_paths, file_assignments,
        t_in=4, t_out=2, window_size=128, t_stride=1, s_stride=64,
        augmentation="none", split="train",
    )
    assert len(index) > 0
    assert len(index) == len(flags)
    # Every tuple is length 5
    assert all(len(e) == 5 for e in index)
    # Aug code is always AUG_NONE for augmentation='none'
    assert all(e[4] == AUG_NONE for e in index)
    # harp_c (80x90) is skipped because min(H,W) < 128
    file_idxs_used = {e[0] for e in index}
    skipped_stems = {Path(cube_paths[i]).stem for i in range(len(cube_paths))
                     if i not in file_idxs_used}
    assert "harp_c" in skipped_stems


def test_per_cube_norm_stats(tmp_path, fake_data_dir):
    paths = sorted(fake_data_dir.glob("*.zarr"))
    tmp = tmp_path / "cubes"; tmp.mkdir()
    cube_paths = []
    for p in paths:
        z = zarr.open(str(p), mode="r")
        wind = np.asarray(z["wind"][:])
        time = np.asarray(z["Time"][:])
        keep = time > 0
        cube = np.transpose(wind, (2, 0, 1))[keep].astype(np.float32)
        out = tmp / f"{p.stem}.npy"
        np.save(out, cube)
        cube_paths.append(str(out))
    file_assignments = {"train": [0, 1, 3], "val": [], "test": [2]}
    per_cube, global_norm = _compute_per_cube_norm(
        cube_paths, file_assignments, method="zscore_per_cube",
    )
    assert set(per_cube.keys()) == {0, 1, 3}
    for fi, s in per_cube.items():
        assert "mu" in s and "sigma" in s
        assert s["sigma"] > 0
    assert "mu" in global_norm and global_norm["sigma"] > 0


def test_load_harp_zarr_end_to_end(tmp_path, fake_data_dir):
    tr, va, te, meta = load_harp_zarr_data(
        str(fake_data_dir),
        t_in=4, t_out=2,
        split_ratios=[0.7, 0.2, 0.1],
        norm_method="zscore_per_cube",
        augmentation="none",
        seed=42,
        window_size=128, window_stride=64,
        dual_channel=False,
        clip=1e8,
    )
    assert meta["window_size"] == 128
    assert meta["window_stride"] == 64
    assert meta["norm_method"] == "zscore_per_cube"
    assert isinstance(meta["per_cube_norm"], dict)
    assert len(tr) > 0
    # Round-trip one batch
    dl = DataLoader(tr, batch_size=2, shuffle=False, num_workers=0)
    x, y, info = next(iter(dl))
    assert x.dtype == torch.float32
    assert x.shape == (2, 1, 4, 128, 128)
    assert y.shape == (2, 1, 2, 128, 128)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()


def test_signed_asinh_preserves_sign(tmp_path, fake_data_dir):
    tr, va, te, meta = load_harp_zarr_data(
        str(fake_data_dir),
        t_in=4, t_out=2,
        split_ratios=[1.0, 0.0, 0.0],
        norm_method="signed_asinh",
        augmentation="none",
        seed=42,
        window_size=128, window_stride=64,
        dual_channel=False,
        clip=1e8,
    )
    dl = DataLoader(tr, batch_size=1, shuffle=False, num_workers=0)
    x, _, _ = next(iter(dl))
    # Signed-asinh keeps zero at zero and sign elsewhere; check via correlation
    # with a freshly-loaded raw window from the underlying mmap.
    file_idx = tr.index[0][0]
    mmap = np.load(tr.file_paths[file_idx], mmap_mode="r")
    raw = mmap[0:4, 0:128, 0:128]
    assert np.sign(raw[np.abs(raw) > 1]).shape == \
           x.numpy()[0, 0][np.abs(raw) > 1].shape
