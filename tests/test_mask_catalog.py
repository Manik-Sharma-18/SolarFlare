"""Unit tests for solarflare_data/mask_catalog.py.

Verifies per-policy shape/dtype/ratio/structure, mixer category proportions,
determinism under fixed seed, and edge-case shapes.
"""
from __future__ import annotations

import math

import pytest
import torch

from solarflare_data.mask_catalog import (
    POLICIES,
    curriculum_mix,
    sample_mixed,
)

CPU = torch.device("cpu")


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _policy(name, B, T, Hp, Wp, **kwargs):
    seed = kwargs.pop("seed", 0)
    return sample_mixed(B, T, Hp, Wp, mix={name: 1.0},
                        generator=_gen(seed), device=CPU, **kwargs)


def _ratio(m):
    return float(m.float().mean().item())


# ---------- shape / dtype --------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
def test_policy_shape_and_dtype(policy):
    m = _policy(policy, 4, 6, 8, 10, t_out=2)
    assert m.shape == (4, 6, 8, 10)
    assert m.dtype == torch.bool


# ---------- per-policy ratio + structure ----------------------------------

def test_short_tube_ratio_avg():
    # tube branches 50/50 short_area=0.15 / long_area=0.40 → avg ~0.275.
    m = sample_mixed(2000, 4, 12, 12, mix={"tube": 1.0},
                     generator=_gen(0), device=CPU)
    expected = (0.15 + 0.40) / 2
    assert abs(_ratio(m) - expected) < 0.05


def test_future_ratio():
    m = _policy("future", 200, 10, 6, 6)
    assert abs(_ratio(m) - 0.30) < 0.01


def test_cross_time_ratio():
    m = _policy("cross_time", 200, 10, 6, 6)
    assert abs(_ratio(m) - 0.30) < 0.01


def test_tail_ratio():
    m = _policy("tail", 8, 6, 4, 4, t_out=2)
    assert abs(_ratio(m) - 2 / 6) < 1e-6


def test_tube_replicates_across_time():
    m = _policy("tube", 8, 5, 10, 10)
    for b in range(m.shape[0]):
        for t in range(1, m.shape[1]):
            assert torch.equal(m[b, 0], m[b, t])


def test_future_is_contiguous_tail():
    m = _policy("future", 4, 10, 4, 4)
    for b in range(m.shape[0]):
        per_frame = m[b].any(dim=(-1, -2))
        idx = torch.where(per_frame)[0].tolist()
        assert idx == list(range(idx[0], 10))


def test_cross_time_frames_fully_on_or_off():
    m = _policy("cross_time", 8, 10, 6, 6)
    for b in range(m.shape[0]):
        for t in range(m.shape[1]):
            f = m[b, t]
            assert f.all() or (~f).all()


def test_tail_is_deployment_aligned():
    m = _policy("tail", 2, 6, 4, 4, t_out=2)
    assert not m[:, :4].any().item()
    assert m[:, -2:].all().item()


# ---------- determinism ---------------------------------------------------

def test_same_seed_bit_identical():
    kw = dict(mix={"tube": 0.5, "future": 0.3, "cross_time": 0.2},
              device=CPU, t_out=2)
    a = sample_mixed(4, 6, 8, 8, generator=_gen(123), **kw)
    b = sample_mixed(4, 6, 8, 8, generator=_gen(123), **kw)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = sample_mixed(8, 6, 8, 8, mix={"tube": 1.0}, generator=_gen(0), device=CPU)
    b = sample_mixed(8, 6, 8, 8, mix={"tube": 1.0}, generator=_gen(1), device=CPU)
    assert not torch.equal(a, b)


# ---------- mixer ---------------------------------------------------------

def test_mixer_category_proportions():
    mix = {"tube": 0.5, "future": 0.3, "cross_time": 0.2}
    B = 4000
    m = sample_mixed(B, 8, 6, 6, mix=mix, generator=_gen(0), device=CPU)
    # Per-item fingerprint: tube → no full-frame masks; future/cross_time →
    # all full-frame masks.
    counts = {"tube": 0, "frame_level": 0}
    for b in range(B):
        per_frame = m[b].any(dim=(-1, -2))
        full_or_empty = (m[b].all(dim=(-1, -2)) | ~per_frame).all().item()
        if full_or_empty and per_frame.any():
            counts["frame_level"] += 1
        else:
            counts["tube"] += 1
    assert abs(counts["tube"] / B - 0.5) < 0.05
    assert abs(counts["frame_level"] / B - 0.5) < 0.05


# ---------- edge cases ----------------------------------------------------

def test_edge_T1():
    m = sample_mixed(4, 1, 4, 4,
                     mix={"tube": 0.5, "future": 0.3, "cross_time": 0.2},
                     generator=_gen(0), device=CPU, t_out=1)
    assert m.shape == (4, 1, 4, 4) and m.dtype == torch.bool


def test_edge_Hp_Wp_1():
    m = sample_mixed(2, 4, 1, 1, mix={"tube": 1.0},
                     generator=_gen(0), device=CPU)
    assert m.shape == (2, 4, 1, 1)
    assert m.all().item()


def test_invalid_policy_raises():
    with pytest.raises(ValueError, match="unknown policies"):
        sample_mixed(1, 2, 2, 2, mix={"bogus": 1.0},
                     generator=_gen(0), device=CPU)


def test_zero_weights_raises():
    with pytest.raises(ValueError, match="sum to 0"):
        sample_mixed(1, 2, 2, 2, mix={"tube": 0.0},
                     generator=_gen(0), device=CPU)


# ---------- curriculum ----------------------------------------------------

def test_curriculum_tail_only_phase():
    cfg = {"policy_mix": {"tube": 0.5, "future": 0.3, "cross_time": 0.2},
           "curriculum": {"tail_only_pct": 0.10, "warmup_pct": 0.20}}
    assert curriculum_mix(0, 50, cfg) == {"tail": 1.0}


def test_curriculum_full_mix_phase():
    cfg = {"policy_mix": {"tube": 0.5, "future": 0.3, "cross_time": 0.2},
           "curriculum": {"tail_only_pct": 0.10, "warmup_pct": 0.20}}
    assert curriculum_mix(25, 50, cfg) == {"tube": 0.5, "future": 0.3, "cross_time": 0.2}


def test_curriculum_warmup_blends():
    cfg = {"policy_mix": {"tube": 1.0},
           "curriculum": {"tail_only_pct": 0.10, "warmup_pct": 0.20}}
    out = curriculum_mix(10, 50, cfg)
    assert set(out.keys()) == {"tube", "tail"}
    assert math.isclose(out["tube"], 0.5, abs_tol=1e-9)
    assert math.isclose(out["tail"], 0.5, abs_tol=1e-9)
