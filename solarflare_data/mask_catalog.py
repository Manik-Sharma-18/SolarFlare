"""V5 JEPA mask catalog (Path B, Strategy B / MAE-style zero-token).

Per docs/V5_JEPA/03_masks_and_pathak.md: short tube / long tube / future block /
cross-time / tail. Strategy B: True == MASKED token (zeroed in pixel space pre-adapter).

API:
    sample_mixed(B, T, Hp, Wp, *, mix, ..., generator, device) -> [B,T,Hp,Wp] bool
    curriculum_mix(epoch, total_epochs, cfg) -> dict[str, float]

Sampling runs on CPU with the supplied generator (deterministic across devices),
then transferred to `device`. True == masked, False == visible. Tail policy uses
the t_out knob so it matches the deployment-aligned validation task.

`target_ratio` is informational in v1 — per-policy knobs (area / pct / t_out)
drive the actual mask ratio. Effective ratios at defaults:
    short_tube ~15%   long_tube ~40%   future ~30%   cross_time ~30%   tail t_out/T
"""
from __future__ import annotations

import math

import torch

POLICIES = ("tube", "future", "cross_time", "tail")


def _sample_tube_one(
    T: int, Hp: int, Wp: int, area: float, gen: torch.Generator
) -> torch.Tensor:
    """One rectangular spatial block, replicated across all T frames."""
    target = max(1, int(round(area * Hp * Wp)))
    h = max(1, int(round(math.sqrt(max(1, target * Hp) / max(1, Wp)))))
    h = min(h, Hp)
    w = max(1, int(round(target / h)))
    w = min(w, Wp)
    y0 = int(torch.randint(0, Hp - h + 1, (1,), generator=gen).item())
    x0 = int(torch.randint(0, Wp - w + 1, (1,), generator=gen).item())
    m = torch.zeros(T, Hp, Wp, dtype=torch.bool)
    m[:, y0:y0 + h, x0:x0 + w] = True
    return m


def _sample_future_one(
    T: int, Hp: int, Wp: int, frac: float, gen: torch.Generator
) -> torch.Tensor:
    """Last ceil(frac*T) frames fully masked."""
    k = max(1, min(T, int(math.ceil(frac * T))))
    m = torch.zeros(T, Hp, Wp, dtype=torch.bool)
    m[T - k:] = True
    return m


def _sample_cross_time_one(
    T: int, Hp: int, Wp: int, frac: float, gen: torch.Generator
) -> torch.Tensor:
    """ceil(frac*T) random non-contiguous frames fully masked."""
    k = max(1, min(T, int(math.ceil(frac * T))))
    perm = torch.randperm(T, generator=gen)
    pick = perm[:k]
    m = torch.zeros(T, Hp, Wp, dtype=torch.bool)
    m[pick] = True
    return m


def _sample_tail_one(T: int, Hp: int, Wp: int, t_out: int) -> torch.Tensor:
    """Last t_out frames fully masked. Deterministic; deployment-aligned."""
    k = max(1, min(T, int(t_out)))
    m = torch.zeros(T, Hp, Wp, dtype=torch.bool)
    m[T - k:] = True
    return m


def sample_mixed(
    B: int,
    T: int,
    Hp: int,
    Wp: int,
    *,
    mix: dict,
    target_ratio: float = 0.80,
    short_area: float = 0.15,
    long_area: float = 0.40,
    future_pct: float = 0.30,
    cross_time_pct: float = 0.30,
    t_out: int = 2,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """[B, T, Hp, Wp] bool. True == masked (target side), False == visible (context side).

    `mix` keys ⊆ {"tube", "future", "cross_time", "tail"}; values are probability
    weights (auto-normalized). `tube` branches 50/50 between short_area / long_area.
    """
    if B <= 0 or T <= 0 or Hp <= 0 or Wp <= 0:
        raise ValueError(f"shape must be positive: B={B} T={T} Hp={Hp} Wp={Wp}")
    if not mix:
        raise ValueError("mix is empty")
    bad = set(mix) - set(POLICIES)
    if bad:
        raise ValueError(f"unknown policies: {sorted(bad)} (allowed: {POLICIES})")
    cats = list(mix.keys())
    w = torch.tensor([float(mix[c]) for c in cats])
    if (w < 0).any():
        raise ValueError("mix weights must be ≥ 0")
    if float(w.sum()) <= 0:
        raise ValueError("mix weights sum to 0")
    w = w / w.sum()

    out = torch.zeros(B, T, Hp, Wp, dtype=torch.bool)
    pick = torch.multinomial(w, B, replacement=True, generator=generator)
    for b in range(B):
        cat = cats[int(pick[b].item())]
        if cat == "tube":
            r = float(torch.rand((), generator=generator).item())
            area = short_area if r < 0.5 else long_area
            out[b] = _sample_tube_one(T, Hp, Wp, area, generator)
        elif cat == "future":
            out[b] = _sample_future_one(T, Hp, Wp, future_pct, generator)
        elif cat == "cross_time":
            out[b] = _sample_cross_time_one(T, Hp, Wp, cross_time_pct, generator)
        elif cat == "tail":
            out[b] = _sample_tail_one(T, Hp, Wp, t_out)
    return out.to(device)


def curriculum_mix(epoch: int, total_epochs: int, cfg: dict) -> dict:
    """Curriculum-aware policy mix.

    frac = epoch / max(1, total_epochs).
        frac < tail_only_pct                    → {"tail": 1.0}
        frac < tail_only_pct + warmup_pct       → linear blend tail → policy_mix
        else                                    → policy_mix verbatim
    """
    cur = cfg.get("curriculum", {}) if cfg else {}
    tail_only = float(cur.get("tail_only_pct", 0.10))
    warmup = float(cur.get("warmup_pct", 0.20))
    full_mix = dict(cfg.get("policy_mix", {"tube": 0.5, "future": 0.3, "cross_time": 0.2}))
    if not full_mix:
        return {"tail": 1.0}
    frac = epoch / max(1, total_epochs)
    if frac < tail_only:
        return {"tail": 1.0}
    if frac < tail_only + warmup:
        alpha = (frac - tail_only) / max(1e-9, warmup)
        blended = {k: alpha * v for k, v in full_mix.items()}
        blended["tail"] = blended.get("tail", 0.0) + (1.0 - alpha)
        s = sum(blended.values())
        return {k: v / s for k, v in blended.items() if v > 0}
    return dict(full_mix)
