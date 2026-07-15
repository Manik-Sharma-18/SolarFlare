# Walkthrough — procedure, findings, plots (spoken tour)

Aim: ~10–12 min. Order: **problem → data insight → method → result → honesty.**
Professor's ask: "explain the procedure, findings, and plots so we can converge."

## Procedure (what was done)
1. **The gap.** All prior winding-based forecasting integrates the winding *map*
   to a single scalar first. That cancels the sign-resolved, PIL-localised
   structure. This thesis keeps the 2D map and feeds its spatiotemporal structure
   to a deep model — no prior work does this. (Ch1 §1.4, Fig 1.4.)
2. **Data insight (your strongest moment).** The stored field `wind[H,W,T]` is
   near-white in time (frame-to-frame spatial corr ≈ 0.02) — not forecastable as
   is. Its time-integral, the accumulated **total** winding `W[t]=Σ_{τ≤t}wind[τ]`,
   is smooth and strongly persistent (lag-1 ≈ 0.98). So the forecastable object is
   the total. (Ch2 §2.1, Fig 2.5 autocorr; premise Figs 2.6–2.7.)
3. **Preprocessing + why.** Signed field, heavy tail to ~10⁷ → `asinh(w/10³)`
   compression + robust (99.5-pct) clip to [−1,1]; Gaussian low-pass + downsample
   to 88×132; **gap-aware** windowing (no window crosses a missing acquisition).
4. **Model.** ConvLSTM encoder–forecaster (3 layers, 96 hidden, 3.3 M params).
   Reads 10 frames (120 min), autoregressively predicts 4 (48 min). Predicts a
   **residual** on the last frame → a do-nothing net = persistence. (Ch2 §2.2–2.4.)
5. **Evaluation.** Strict **leave-one-active-region-out**: train on 26 regions,
   test on unseen HARP 11930. Score = skill over persistence
   `1 − MAE_model/MAE_persist`. Report the **final** model (no test-set checkpoint).

## Findings (six crisp claims)
1. **The forecastable object is the accumulated total**, derived from the data,
   not assumed. (Figs 2.5–2.7.)
2. **The model generalises across active regions** — +8.6% skill on an unseen
   region, 95% CI [+6.8, +10.3], P(skill>0)=1.000. (Table 3.1.)
3. **Skill grows with lead time** — model≈persistence at +12 min, pulls below and
   the gap widens to +48 min → genuine short-term dynamics. (Fig 3.3.)
4. **Errors are small and structured** — a few % of amplitude, localised at
   winding-lobe boundaries (sharp-gradient PILs); envelope reproduced. (Fig 3.2.)
5. **Skill is real but persistence-dominated** — the total is easy largely because
   it is autocorrelated; the honest measure is the margin over persistence. (Fig 3.1.)
6. **Loss reweighting can't beat plain Huber** (ablation) — the ceiling is
   physical, not a tuning failure; skill lives in the active windows. (Table 3.2, App A.)

## Plots (one line each)
- **Fig 2.5** autocorrelation — rate vs total → why we forecast the total.
- **Fig 2.6** spatial-mean rate ≈ 0 → the scalar cancels the signal (premise).
- **Fig 2.7** spatial-mean total = smooth drift → structure gone when collapsed.
- **Fig 3.1** training trace → convergence + persistence ceiling made visible.
- **Fig 3.2** Pred/Truth/Difference triptych (median window) → verification.
- **Fig 3.3** MAE vs lead time → skill grows with horizon.
- **Fig 3.4** gallery across days → qualitative robustness.
- **Fig 3.5 / App A** ablation → rigour, negative result.
- **Table 3.1** headline + scalar-mean baseline + bootstrap CI.

## One-sentence framing to open and close with
"This is a working spatiotemporal pipeline for 2D winding-flux maps and an honest
measurement of its ceiling; the scientific payoff — does winding *structure*
predict flares — is the classification task this sets up."
