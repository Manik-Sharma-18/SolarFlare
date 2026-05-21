# Mask strategies + curriculum

V5 mask catalog. Adapted from V-JEPA-2-AC + Brain-JEPA cross-time/cross-ROI. Pathak random-region kept as ablation only.

**Code:** `solarflare_data/mask_catalog.py` (149 LOC). **Tests:** `tests/test_mask_catalog.py` (22 tests, ratios ±3% on N=2000).
**Strategy:** B — MAE-style zero-token (masked patches × 0 post-adapter). See `09_progress.md` for why-not-A (sparse-token rewrite scope).

---

## Catalog

| Mask | Shape | Scale | Purpose |
|---|---|---|---|
| Short tube | spatial block, all T | 15% area | local AR structure |
| Long tube | spatial block, all T | 40% area | global flux-sheet motion |
| Future block | full spatial, last K frames | 100%×K/T | force real future prediction (V-JEPA 2-AC causal) |
| Cross-time (Brain-JEPA) | full spatial, random T frames | 30% time | non-causal temporal infill |
| Tail (downstream) | full spatial, last `t_out` frames | 100%×t_out/T | aligns with deployment use |
| Random-region (Pathak) | irregular blob | 25% area | ablation only |

**Default mix:** `{tube: 0.5, future: 0.3, cross_time: 0.2}` (E09 CONFIRMED). Total token mask ratio: ~75–85%.

**Loss restriction:** INTERSECTION of `mask & valid_token & token_pad_mask`. Never grade on outlier-clipped pixels or padded tokens.

---

## Curriculum

`training/jepa_trainer.py` blends mix over epochs:

| Phase | Epoch range (slow / E09) | Mix |
|---|---|---|
| Tail-only | 0 → 25% | `{tail: 1.0}` |
| Warmup | 25% → 40% | linear blend tail → full mix |
| Full mix | 40% → end | `{tube:0.5, future:0.3, cross_time:0.2}` |

For 100-ep slow curriculum (`tail_only_pct=0.25`, `warmup_pct=0.40`): tail→ep25, full-mix→ep65.

### Fast vs slow

| Schedule | `tail_only_pct` / `warmup_pct` | Result |
|---|---|---|
| Fast (E05) | 0.10 / 0.20 (tail→ep5, full→ep15) | 18-ep plateau ep7–31; best 0.0689 ep49, not converged. |
| **Slow (E09) CONFIRMED** | **0.25 / 0.40** (tail→ep25, full→ep65) | Monotone descent; best **0.00831 ep98**. 8× better. |

Fast transition pushes model into full mix before tail-aligned features stabilize → long plateau as it re-adapts. Slow gives tail-aligned features time to consolidate before adding tube/future/cross_time. See F1 + F7 in `12_experiments_findings.md`.

---

## Evidence — when each strategy proven

- **F1 — slow curriculum CONFIRMED:** E09 sanity mask-ON 100ep MPS, val 0.00831 ep98.
- **F3 — tube-only collapses:** E12 sanity 100ep CUDA `{tube:1.0}`, best 0.0402 ep41 then val 4× to 0.172 by ep99. Future + cross_time doing real regularization work, not just curriculum pacing.
- **F4 — mask-OFF saturates ep24:** E06 sanity 50ep MPS, val 0.0125 ep38. Useful as early-epoch baseline only.
- **F5 — mask-ON/OFF losses not comparable:** mask-OFF grades all tokens; mask-ON grades only masked tokens (~80% ratio). Different denominators.

Per-arm policy isolation (E13 tube+future, E14 tube+cross, E15 uniform) running on 5060ti — will identify whether future OR cross_time is the regularizer, or whether any 2-of-3 suffices.

---

## Why not Pathak Context Encoder

Original `ARCHITECTURE_V5.md` (Pathak port) had: 3D conv encoder/decoder (252M CFC bottleneck), joint L2 + adversarial loss (λ_rec=0.999, λ_adv=0.001), 3D DCGAN discriminator, mean-fill masks with 7-px overlap × 10 weight.

| Pathak choice | Why obsolete |
|---|---|
| Adversarial loss | Modern FMs (MAE, JEPA) drop GAN. Training instability + minimal benefit at scale. |
| Channel-wise FC | 252 M params for FC over spatial × temporal. ViT self-attention same job, fraction of params. |
| Pixel-space L2 | Causes blur / persistence collapse — exactly V4's failure. JEPA latent loss avoids by construction. |
| 3D conv | Tubelet patch + ViT more param-efficient + benefits from RoPE. |
| Random region masks | V-JEPA multi-block 3D tubes outperform on motion tasks (+10–21 pts SSv2). |

Pathak (2016) was right for 2016. **2026 best practice: V-JEPA-2-AC + tubelet patch + masked latent loss.**
