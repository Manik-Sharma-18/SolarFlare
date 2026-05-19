# 12-findings — V5 JEPA Hard Truths

Extracted from the run log. Each entry: claim, evidence, tag (CONFIRMED | HYPOTHESIS | STALE).
Source for each: `12_experiments.md` summary table + per-experiment section.

---

## F1. Slow curriculum CONFIRMED — `tail_only_pct=0.25, warmup_pct=0.40` is the default

- **Evidence:** E09 (sanity, mask-ON, 100ep, MPS) → best val **0.00831 ep98**, monotonic descent ep79→98, no plateau. E05 same config but fast curriculum (`tail_only_pct=0.10, warmup_pct=0.20`) → best 0.0689 ep49 with 18-ep plateau ep7–31.
- **Effect size:** 8× better than E05 fast curriculum; 1.5× better than E06 mask-OFF baseline on the same 4-cube setup despite harder task.
- **Tag:** CONFIRMED.

## F2. Wind flux clip 1e8 is essential — 1e5 destroyed signal

- **Evidence:** E03 vs prior. Switching `WIND_FLUX_CLIP` from 1e5 → 1e8 lifted val 0.202 → 0.0407 in 5 ep MPS sanity (5×). Justified physically: per-pixel max ~1e7; 1e5 clipped legitimate peaks.
- **Code:** `solarflare_data/zarr_loader.py`. Concept page: `concepts/wind_flux_clipping.md`.
- **Tag:** CONFIRMED.

## F3. Tube-only mask collapses past curriculum transition

- **Evidence:** E12 (CUDA 5060ti, sanity, `policy_mix={tube:1.0}`, 100ep slow): best **0.04017 ep41**, then val ballooned 4× to 0.172 by ep99 while train kept dropping (0.026). Curve flat-then-rises after full-mix transition ep65.
- **Implication:** future + cross_time policies are doing real work, not just curriculum pacing. Removing them costs 5× vs E09 full-mix (0.00831) AND destabilizes late training on 4 cubes.
- **Operational rule:** always load `best.pt`, not `last.pt`, for any diverged arm.
- **Tag:** CONFIRMED.

## F4. Mask-OFF saturates at sanity scale ~ep24

- **Evidence:** E06 (sanity, mask-OFF, 50ep, MPS) → val plateau ep24–49, best 0.0125 ep38. 3 training cubes memorized.
- **Implication:** mask-OFF informative only as early-epochs baseline on 4-cube sanity. Not a thesis-grade evaluation point.
- **Tag:** CONFIRMED.

## F5. Mask-ON vs Mask-OFF losses are NOT directly comparable

- **Why:** mask-OFF grades all tokens; mask-ON grades only masked tokens (~80% ratio). Different denominators.
- **Use:** compare within mask-ON arms or within mask-OFF arms; never cross.
- **Tag:** CONFIRMED.

## F6. MPS SDPA returns NaN under `attn_mask + no_grad`

- **Evidence:** train (with grad) finite 0.187; eval (no_grad) NaN at predictor step 0. CUDA unaffected.
- **Fix:** `models/v5/predictor.py` routes MPS through manual `(q@kᵀ)·scale → masked_fill → softmax → @v`. CUDA keeps SDPA.
- **Do not unify these paths without testing val_loss on MPS.**
- **Tag:** CONFIRMED.

## F7. 18-ep curriculum plateau (E05) is the failure mode slow curriculum fixes

- **Evidence:** E05 val flat ~0.105–0.127 ep7–31 then descent. Caused by `tail_only_pct=0.10 / warmup_pct=0.20` transitioning the model to tube+future+cross_time before it had adapted.
- **Tag:** CONFIRMED. Supersedes the original mask catalog landing assumption that fast warmup would work.

## F8. Sanity-mask floor: val 0.00530 (E15)

- **Current floor:** E15 uniform mix 0.00530 @ ep99 (supersedes E09 0.00831). See F11.
- **Implication:** any path_a-scale run must beat 0.00530 on harder full setup to count as progress.
- **Tag:** CONFIRMED at sanity scale only. Whether the floor holds at thesis scale is open (depends on E16 + future path_a runs).

## F9. Frozen JEPA features carry phase/temporal structure but not absolute scale — calibration-fixable

- **Evidence (E11, E09 features):** median per-cube Pearson r ≈ 0.73 on 17 novel cubes; absolute medAPE 46% novel vs 17.4% encoder-val. XGBoost lifts train R² 0.20→0.61 but novel aggregate flat → not capacity.
- **Evidence (E30-probe-cal, E30 v2 features, 2026-05-19):** raw novel R² −30 (harp_51) to −10 (harp_17) → +0.18 to +0.74 after per-cube affine y=a·ŷ+b (30% cal / 70% eval). **Median medAPE 22% (raw) → 9.6% (calibrated) across 8 cubes.** 4/5 novel cubes reach <17% medAPE — matches val cubes. F9 hypothesis fully validated.
- **Levers that work:** per-cube affine calibration (1 fit per cube, 30% holdout). Closes ~70% of medAPE gap.
- **Lever that does NOT work universally:** harp_245 fits a=3.7, b=−6.6e3 → calibration extrapolates wildly (220% medAPE). Persistence baseline beats encoder on this cube — structurally wrong, not scale-shifted. 1/5 novel cubes resist calibration.
- **Failed levers:** spatial-mean Δy probe (collapses inter-frame variance), richer pooling pending.
- **Tag:** CONFIRMED twice (E11 + E30-probe-cal). Artifacts: `outputs_probe/E30_eval/probe_calibration.md`.

## F10. Path A (LoRA on Surya) abandoned 2026-05-08 — architectural lock

- **Reason:** HelioSpectFormer hard-locked to `img_size=4096 / 60-min / 13ch`. AR cubes are variable HxW / 12-min / 1ch. No LoRA-only adapter bridges this.
- **Action:** dropped `transformers`, `peft`, `huggingface_hub` from requirements. Do not reintroduce.
- **Tag:** CONFIRMED (decision; not an experimental result).

## F11. Uniform mask mix > tube-heavy mix — new sanity floor 0.00530

- **Evidence:** E15 `{tube:0.34, future:0.33, cross_time:0.33}` 100ep slow curric → best **val 0.00530 @ ep99**, monotonic descent. Same arch / cubes / curriculum as E09 (only mix differs).
- **Effect size vs siblings:** 1.57× E09 `{0.5/0.3/0.2}`; 2.21× E14 tube+cross `{0.7/0.3}`; 3.42× E13 tube+future `{0.7/0.3}`; 7.58× E12 tube-only.
- **Implication:** E09's tube-heavy mix over-weighted tube. Cross_time + future jointly under-weighted. F3 (tube needs companions) was correct directionally but understated — *equal* weighting of all 3 beats *any* tube-dominant mix tested.
- **Action:** re-anchor mask-ratio sweep (E21–E24) on E15 mix, not E09's. Re-run wind probe on E15 ckpt — test whether 1.57× lower JEPA loss carries to probe medAPE.
- **Tag:** CONFIRMED. Sanity scale only — re-verify at path_a scale.

## F12. Cross-attn predictor must drop causal in masked pretext path

- **Evidence:** E31 (mini_mps, cross_attn=true, slow curric). Crashed ep20→21 transition at `assert ca_mask.any(dim=-1).all()` — query row with no valid context key.
- **Root cause:** old code built `ca_mask = (frame_idx[:,None] >= ctx_frame_idx[None,:]) & ctx_mask[None,:]`. When `cross_time` policy masked frame 0 fully (introduced ep20 by curriculum warmup blending tail→mix), no ctx tokens have frame_idx ≤ 0 → empty row for queries at frame 0. `BlockCausalPredictor` self-attn never hit it: K/V includes all tokens (incl. zeroed-masked), so frame-0 queries attend to their own zeroed selves.
- **Fix:** `models/v5/cross_attn_predictor.py` masked path now uses `ca_mask = ones(N, N_ctx)` (modulated only by `token_pad_mask`). JEPA masked pretext has no future-leakage concern — predicting masked from visible is not autoregressive. Causal stays only in rollout-mode (`ctx_mask=None`) self-attn fallback.
- **Action:** resumed E31 from `last.pt` ep21 without retraining. Strengthened diag asserts in `jepa_model.py` (shape mismatch pred vs target) also added.
- **Followup MPS sizing crash ep23:** `MPSNDArray dim > INT_MAX` at step 23430. Root cause: MPS softmax bf16→fp32 internal buffer size in bytes = `B*H*N*N_ctx*4`; with `N=14*tpf, N_ctx≈0.25N`, buffer exceeds INT_MAX when `tpf > ~1912`. Only `harp_11930` (tpf=2200) breaches in 19-cube allowlist. Dropped from allowlist; resumed from ep22 `last.pt` (best val 0.0520).
- **Tag:** CONFIRMED (bug + fix). Validation that cross-attn matches block-causal val curve at sanity scale still open — depends on E31 completion.

---

## How to add to this file

Promote a finding here only when:
- A run has completed (not in-progress) **and**
- A clean val curve or quantitative comparison exists **and**
- A control / baseline is present **or** the claim is a hard architectural lock.

Otherwise leave in `09_progress.md` narrative or `12_experiments.md` per-experiment section as HYPOTHESIS. The `docs-sync` skill flags untagged claims here.
