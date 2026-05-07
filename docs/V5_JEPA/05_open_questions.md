# 05 — Open questions + next steps

## 9. Open questions for senior review

1. **Path A vs Path B?** LoRA on Surya is highest-leverage. But Surya pretrains on full-disk 4096² SDO, our cubes are variable-size active-region patches (sample 627×877). Domain gap unknown. Adapter design matters.
2. ~~**Forecast horizon: 4×12=48 min, not 24 h.**~~ **RESOLVED 2026-04-25**: horizon flexible, 48 min floor acceptable, t_in/t_out tunable.
3. **Should V5 jointly forecast pixels AND classify flares?** MC-JEPA shared-encoder pattern supports this. Might double the value of pretrain compute.
4. **Embedding-space loss vs pixel loss as primary?** V-JEPA 2-AC defaults to embedding-L1. Pixel-L1+SSIM is V4's failure. Recommend embedding-primary, pixel-secondary at 0.1×.
5. **Mask ratio: 75% or 90%?** V-JEPA defaults 90%. Brain-JEPA 75%. Solar autocorrelation higher than natural video. Empirical sweep needed.
6. **LeJEPA (no EMA) vs standard EMA?** LeJEPA fewer hyperparameters but newer (Nov 2025). Risk: less battle-tested.
7. **Surya is HMI+AIA pretrained. Our data is winding-flux derived from HMI vector field. Channel mismatch — how to project?**
8. **SuryaBench size for Path B**: how much do we use? Full corpus is 218 TB, intractable. Need subsetting strategy.
9. **Do we add SHARP scalars now (V5.2 multimodal) or defer?** Adds complexity but matches operational SOTA.
10. **Curriculum: 1-step → 2-step → 4-step rollout. Boundaries at what % of epochs?** V-JEPA 2 doesn't curriculum, it does parallel teacher-forcing + 2-step rollout summed. Pick one.
11. ~~**Sign convention of winding flux**~~ **RESOLVED 2026-05-04**: chiral pseudoscalar confirmed. Augmentation set: enable H/V flip + 90°/270° rotation **with explicit negation** `x' = -T(x)`; 180° rotation safe without negation. See `06_data.md` §11.4.
12. **NaN policy**: are NaNs reserved for instrument-bad-pixel only, or also off-disk / off-AR padding? Drives masking strategy (mask-out-NaN vs nan_to_num + valid_mask channel).

---

## 10. Concrete next steps

1. **Decision required**: Path A (Surya LoRA) vs Path B (pretrain from scratch). Path A starts ~tomorrow; Path B is multi-week.
2. **If Path A:**
   - Pull Surya weights from HF (`nasa-ibm-ai4science/Surya-1.0`).
   - Build input adapter from `wind` array (zarr, signed fp32, NaN-aware) → Surya 13-channel input.
   - Build V-JEPA-2-AC predictor (12L × 768 × 12h, block-causal, RoPE-3D).
   - Build SegFormer-style pixel decoder.
   - Set up smooth-L1 embedding loss + curriculum.
   - First-pass training run on 1 cube, verify gradients flow, no collapse.
3. **Diagnostic baselines in parallel**: persistence numbers, SimVPv2 from scratch.
4. ~~**Re-audit forecast horizon with stakeholder.**~~ **DONE 2026-04-25**: 48 min floor accepted, t_in/t_out tunable.
5. **Send senior follow-up** covering `06_data.md` §11.1d Q1–Q3 (Time==0, zero-pixel, NaN policy).
