# 05 — Open questions + next steps

## 9. Open questions for senior review

1. ~~**Path A vs Path B?**~~ **RESOLVED 2026-05-08**: Path A KILLED — Surya HelioSpectFormer hard-locked to img_size=4096 / 60-min / 13ch; AR cubes variable HxW / 12-min / 1ch. No LoRA-only adapter bridges. Path B (JEPA-from-scratch) is active. See `01_path_a.md` + F10.
2. ~~**Forecast horizon: 4×12=48 min, not 24 h.**~~ **RESOLVED 2026-04-25**: horizon flexible, 48 min floor acceptable, t_in/t_out tunable.
3. **Should V5 jointly forecast pixels AND classify flares?** MC-JEPA shared-encoder pattern supports this. Might double the value of pretrain compute.
4. **Embedding-space loss vs pixel loss as primary?** V-JEPA 2-AC defaults to embedding-L1. Pixel-L1+SSIM is V4's failure. Recommend embedding-primary, pixel-secondary at 0.1×.
5. **Mask ratio: 75% or 90%?** V-JEPA defaults 90%. Brain-JEPA 75%. Solar autocorrelation higher than natural video. Empirical sweep needed.
6. **LeJEPA (no EMA) vs standard EMA?** LeJEPA fewer hyperparameters but newer (Nov 2025). Risk: less battle-tested.
7. **Surya is HMI+AIA pretrained. Our data is winding-flux derived from HMI vector field. Channel mismatch — how to project?**
8. **SuryaBench size for Path B**: how much do we use? Full corpus is 218 TB, intractable. Need subsetting strategy.
9. **Do we add SHARP scalars now (V5.2 multimodal) or defer?** Adds complexity but matches operational SOTA.
10. ~~**Curriculum boundaries?**~~ **RESOLVED 2026-05-11 (E09 empirical)**: slow curriculum `tail_only_pct=0.25 / warmup_pct=0.40` (tail→ep25 → full mix→ep65 on 100-ep run). Fast 0.10/0.20 caused 18-ep plateau in E05. See F1 + F7 in `12_experiments_findings.md`.
11. ~~**Sign convention of winding flux**~~ **RESOLVED 2026-05-04**: chiral pseudoscalar confirmed. Augmentation set: enable H/V flip + 90°/270° rotation **with explicit negation** `x' = -T(x)`; 180° rotation safe without negation. See `06_data.md` §11.4.
12. **NaN policy**: are NaNs reserved for instrument-bad-pixel only, or also off-disk / off-AR padding? Drives masking strategy (mask-out-NaN vs nan_to_num + valid_mask channel).

---

## 10. Concrete next steps

1. ~~**Decision: Path A vs Path B**~~ **DONE 2026-05-08**: Path B selected (architectural lock killed A).
2. **Active follow-ups** (see `INDEX.md` Active research): E12-E15 mask-policy ablation, E16 bigger-backbone arm, Stage-2 probe (per-cube affine calibration + richer pooling).
3. **Diagnostic baselines in parallel**: persistence numbers, SimVPv2 from scratch.
4. ~~**Re-audit forecast horizon with stakeholder.**~~ **DONE 2026-04-25**: 48 min floor accepted, t_in/t_out tunable.
5. **Send senior follow-up** covering `06_data.md` §11.1d Q1–Q3 (Time==0, zero-pixel, NaN policy).
