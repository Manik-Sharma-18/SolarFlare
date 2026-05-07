# 04 — Multimodal extension + diagnostic baselines

## 7. Multimodal extension (V5.2 — addresses pixel-only ceiling)

Pixel-only solar flare models cap around TSS 0.7. Hybrid CNN-TCN + SHARP magnetic parameters reaches TSS 0.85. Gap is 15+ points.

Recommendation if downstream is flare classification not pixel forecast:
- Two branches sharing same time index:
  - **Pixel branch**: V5.0 frozen encoder + attentive probe → flare-class logits
  - **Tabular branch**: SHARP scalar params (USFLUX, MEANGAM, MEANGBT, …) → small TCN → flare-class logits
- **Late fusion**: weighted sum of logits, learned scalar weight.
- Reproduces hybrid-SOTA pattern.

Out of scope for pure forecasting V5.0. Logged as V5.2 for review.

---

## 8. Diagnostic baselines (cheap, run alongside)

1. **Persistence**: predict frame_t for all t+1..t+4. Numerical floor. V4 collapsed to this.
2. **Linear probe** on pooled features → MSE. Floor of frozen encoder.
3. **Attentive probe (Efficient Probing variant, arXiv 2506.10178)** for binary "flare in next N hours."
4. **SimVPv2 end-to-end from scratch** on same data. Independent upper bound. **If SimVPv2 beats frozen-Surya recipe, encoder doesn't capture forecasting features and SSL objective is wrong.** Critical sanity check.
