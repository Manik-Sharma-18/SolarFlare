# SGNNET as MLP Replacement — Backlog

**Status:** BACKLOG. Blocked on path_a baseline convergence (need reference val loss before speculative arch swap).
**Source:** Dhiraj's `neuro_graph` project (SGNNET — Sparse Geometric Neural Network).
**Created:** 2026-05-12.

---

## TL;DR

SGNNET = recursive sparse graph net designed as FFN-replacement for VGG16 classifier.
Strong Pareto wins on Imagenette as **single FC-head** replacement. NOT validated as in-transformer FFN stack.
Worth a single-location Tier-0 ablation after baseline lands. Wholesale swap = high-risk, low-EV.

---

## SGNNET in one paragraph

Per-neuron: `W_pos ∈ R^D` + activation. Forward = K iterations of `Z ← normalize(gather_K_hh(Z))`. Readout = `C_ho · Z`, then dot with `W_pos`. Topology = Watts-Strogatz small-world, **frozen at init**. Anti-Hebbian suppression (AH) load-bearing on top.

Code: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/src/sgnnet/model_smallworld.py` (330 lines, would need 200-line split per IndraAstra cap).

---

## CONFIRMED wins (Imagenette, VGG16 FC-replacement, 5060 Ti)

| Metric | SGNNET K=1 distill | VGG_FC | Δ |
|---|---|---|---|
| Acc | 95.95% | ~96% | ≈parity |
| Params | 34,976 | 119.59M | 3419× fewer |
| FLOPs | 0.20M | 124M | 620× fewer |
| Wall-time (B=32) | 0.280ms | 1.570ms | 5.6× faster |
| Peak mem | 477 MiB | 2293 MiB | 4.8× less |

- D=16 ceiling no-KD: 97.30% (N=4096).
- CIFAR-10: 80.57% (−5.67pp vs Linear).
- **Audio: honest negative result** (paper scope).

---

## Load-bearing constraints (CONFIRMED, must port together)

- **AH mechanism = prerequisite.** Without AH: 73pp collapse (91.5% → 18.8%).
- **Topology static at inference.** Every dynamic-conn variant KILLED across step511–525 (destructive rewire, additive expansion, alternating schedule, teleport, Monte-Carlo refresh, edge-shift β).
- **F.normalize load-bearing.**
- **`C_ho` readout required** when activations on unit sphere. Mean-pool → 12% (random).
- **LayerNorm > L2 sphere** (+2.24pp at N=1024 D=16).
- **N=1024 winners do NOT transfer to N=2048** (twopop, curriculum all KILLED).
- **K_iter scaling = mechanism-specific.** ΔW proj prefers K=4 at N=4096 (+0.97pp); rotation prefers K=5.

---

## V5 JEPA — where MLPs live

1. **Patch embed:** Conv → linear (input adapter).
2. **ViT encoder FFN:** 12 blocks × 2-layer MLP. Majority of param count.
3. **Predictor FFN:** block-causal transformer, same structure.
4. **Predictor output projection:** final `nn.Linear(dim, dim)` back to embedding space.

---

## Risk table for drop-in swap

| Risk | Severity | Notes |
|---|---|---|
| Validated only as single FC head, not multi-block in-transformer FFN | HIGH | composition with backprop through stack untested |
| Best Pareto needs KD from teacher MLP; V5 JEPA from-scratch has no teacher | HIGH | no-KD ceiling 97.30% still strong but small dataset |
| Audio negative result; solar AR cubes structurally further from Imagenette than audio | HIGH | strong prior against direct transfer |
| FFN N_out ≈ hidden_dim (e.g. 384) regime not tested (Imagenette N_out=10) | HIGH | readout `C_ho` shape jumps 30×+ |
| V5 bottleneck = attention + data loading, not FFN | MEDIUM | speedup ceiling small in our regime |
| MPS path not benchmarked; we are MPS-primary on Mac Mini | MEDIUM | 5060 Ti wins may not transfer |
| 200-line cap; `model_smallworld.py` is 330 LOC | LOW | mechanical split |

---

## Recommended scout protocol (when unblocked)

Single Tier-0 ablation, **one location at a time**. Order:

1. **Predictor output projection.** Smallest dim, clearest target, lowest blast radius. Swap final `nn.Linear(dim, dim)` for `SGNNET_SmallWorld(N=512, D=16, K_iter=3, K_hh=2)` + AH + LayerNorm + C_ho readout.
   - Compare val JEPA loss at 20 ep on `v5_sanity.yaml`.
   - Pass: Δ within ±5% of baseline.
2. **Patch embed MLP.** Only if (1) neutral/positive.
3. **Inside ViT FFN.** Only if (1) and (2) both clear, AND path_a baseline converged.

Carry AH + small-world `conn_hh` + LayerNorm + `C_ho` readout verbatim. Do **NOT** try dynamic `conn_hh` (4 closed directions).

---

## Block

**Wait for:** path_a CUDA E07/E08 convergence (need reference val number). Don't burn slots on speculative arch before baseline lands.

**Then:** evaluate cost. If V5 already memory/wall-clock acceptable, deprioritize. SGNNET wins biggest when MLP is the bottleneck — V5's bottleneck is attention + zarr I/O.

---

## Pointers

- Source code: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/src/sgnnet/model_smallworld.py`
- Research brief: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/sparse_geometric_network_report.md`
- Latest design notes: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/LEARNINGS_design_2026_04_15.md`
- Belief audit: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/LEARNINGS_phase5_p15j_belief_update.md`
- Norm/diagnostics: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/LEARNINGS_phase5_p16.md`
