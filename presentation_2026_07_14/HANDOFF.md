# Thesis Handoff — SolarFlare V1 ConvLSTM (2026-07-14)

Hand this to a fresh session to continue Manik Sharma's IIT-BHU IDD thesis.

## What the thesis is
Forecast **2-D magnetic winding-flux maps** of solar active regions with a
ConvLSTM encoder–forecaster — the literature gap (everyone integrates the winding
map to a 1-D scalar first, which cancels the sign-resolved PIL structure). It is a
**proof-of-concept pipeline + honest ceiling measurement**, not a flare-forecast
result; flare classification is named as the outlook.

## Location, build, state
- Repo `/Volumes/T9/IndraAstra/manik/SolarFlare`, branch `thesis/v1-convlstm-forecasting`.
- Source `thesis/` (`main.tex`, `thesis.cls`, `chapters/`, `figures/`, `bib/refs.bib`).
- Build: `cd thesis && bash build.sh` → `thesis/build/main.pdf` (TinyTeX; pdflatex +
  bibtex ×2). Currently **43 pages, clean** (0 undefined refs, 0 overfull boxes).
- Overleaf export: `thesis_overleaf.zip` (repo root, ~11.6 MB, source-only). Rebuild
  after edits: `cd thesis; rm -f ../thesis_overleaf.zip; zip -rq ../thesis_overleaf.zip . -x 'build/*' -x '._*' -x '*/._*' -x 'main.log' -x 'main.aux'`.
- ⚠️ **ALL session work is UNCOMMITTED** (last commit `9520691` predates it). Commit when asked.

## Structure (chapters, eqs, floats)
- **Ch1 Intro**: flares/space-weather, forecasting landscape (SHARP→CNN/LSTM), winding
  gap. Eqs 1.1 helicity flux, 1.2 winding flux (σ=sgn Bz), 1.3 winding-flux density
  (= our data). Figs: goes_scale, taxonomy, winding_map.
- **Ch2 Architecture**: data+preprocess — Eq 2.1 asinh(s=1e3), 2.2 cumsum→total,
  2.3 Gaussian low-pass, 2.4 PSD P(k) + wavenumber k=2π/λ, 2.5 **end-to-end composed
  pipeline**. ConvLSTM cell — Eq 2.6 gates(i,f,g,o), 2.7 cell update. Encoder–forecaster
  — Eq 2.8 stacked-encoder recurrence, 2.9 residual x̂=x+Δ (W_out 1×1). Training §.
  Figs: ar_gallery(+limb HARP892), dataset(bar), cadence, normalization,
  **spatialmean_rate(2.6)**, **spatialmean_total(2.7)**, autocorr, psd, pipeline,
  convlstm_cell, encoder_forecaster, **forward(2.12, tensor shapes)**.
  Tables: **dataset(2.1, gridded, holdout separate)**, clip(2.2), hparams(2.3).
- **Ch3 Results**: setup, Eq 3.1 skill. Convergence Fig 3.1 (losscurve). Headline +
  Table 3.1 (scalar-mean/persist/model + bootstrap CI). Triptych 3.2 (median window),
  horizon 3.3, gallery 3.4. §3.3 ablation (Table 3.2 + Fig 3.5 ablation bars). Discussion.
- **App A Methods**: Adam, cosine, GroupNorm, Huber, grad-clip, bootstrap (Eqs A.1–A.6).
- **App B**: ablation training curves (Fig B.1).
- **bib**: 16 entries. Physics: bobra2015, nishizuka2018, huang2018, liu2019,
  mactaggart2021, williams2026(ApJ 999:87), mactaggart2020(winding), prior2019(helicity
  flux). ML: shi2015, hochreiter1997, kingma2015, loshchilov2017, wu2018, huber1964,
  pascanu2013, efron1979.

## Headline results (all in thesis)
- Holdout **HARP 11930**, leave-one-AR-out, 46 gap-aware windows, **final model** (no
  test-set checkpoint selection). Model **MAE 0.0105** vs persistence **0.0115** =
  **skill +8.6%**, bootstrap 95% CI **[+6.8, +10.3]**, P(skill>0)=1.000.
- Per-horizon skill grows with lead: ~[+3, +8, +10, +9]% at +12/+24/+36/+48 min.
  Active-window +0.104 ≫ quiet +0.048. Scalar-mean baseline MAE **0.398** (spatial mean
  ≈0 → structure carries the signal).
- **Ablation** (35-ep quick config): plain Huber +6.8%/+8.7% (winner); horizon
  +6.3%/+8.1% (wash); delta −44.7%/−29.3% (toxic — downweights easy pixels).
- **Clip sensitivity**: [−1,1]/99.5→+0.040, [−1.5,1.5]/99.5→+0.041 (clip limit
  irrelevant), [−1,1]/99.9→+0.006 (scale-pct matters, interacts w/ Huber elbow).

## Model & data facts
- zarr `wind[H,W,T]` = winding **rate** (near-white, lag-1 ~0.02); `cumsum` over time =
  accumulated **total** (smooth, lag-1 ~0.98) — the forecast target.
- Config: grid 88×132, **separate 3-layer encoder + 3-layer decoder** ConvLSTM, hidden
  96, kernel 3, **3.33M params**, t_in=10/t_out=4 (120→48 min), residual (persistence
  fallback), group-norm(8,96), forget-bias +1. Train: Adam 1e-3, cosine 130 ep, bs8,
  grad-clip 1.0, Huber β=1, stride-4 gap-aware windows. Preproc: cumsum→area-downsample→
  asinh(1e3)→Gaussian σ1.5→robust(99.5%)→clip[−1,1]. Exclude harp_1149.

## Infra / how to regenerate
- **5060ti** (all training): `ssh 5060ti`; interp `/home/indra/solarflare/venv/bin/python`
  (torch+cuda+zarr); code `/home/indra/solarflare/scratchpad/forecast_loo_wl.py` (has
  `--lossmode plain|delta|horizon`, `--clip`, `--cscale`). Launch in `tmux`; monitor
  output FILES not logs (ssh flaky). Checkpoints `loo_phase2_plain*.pt`.
- **Local** (figs/build): `PYTHONPATH=/Volumes/T9/IndraAstra/.venv/lib/python3.14/site-packages
  /opt/homebrew/bin/python3.14` (torch-mps, zarr, pymupdf, pypdf). Scratch dir
  `/Users/indra/.claude/jobs/a2ba8286/tmp/` holds every figure script:
  `verify_phase2_final.py` (triptych+horizon), `spatial_mean_figs.py` (2.6/2.7),
  `loss_curve.py`, `ablation_fig.py`, `ablation_curves.py`, `fig_forward.py`,
  `cheap_wins.py` (bootstrap+scalar+median triptych), `clip_probe.py`. Phase-2 artifacts
  in `tmp/phase2/`. Fig scripts write PDFs directly to `thesis/figures/`.
- Render PDF pages to check: pymupdf `fitz.open(...).get_pixmap(dpi=95).save(...)`.

## Float placement (recent fix)
All figures/tables use **`[!htbp]`** — near-reference, no whitespace, no far-deferral.
Do NOT revert to `[t]` (piles up) or `[H]` (whitespace before big figs). Relies on
`float` package + float-fraction tuning in `main.tex` preamble (both in the zip).

## Open items / not done
1. **N=1 test region** (biggest) — only HARP 11930 reported. Run 3–4 more leave-one-out
   folds (~4 h each full / ~35 min quick) → mean ± std skill. Kills the sharpest critique.
2. Early stopping honestly = needs a separate **validation region** (not the test) →
   re-run; currently report final model (chosen for integrity).
3. Offered-but-not-added: a "Backward pass / gradient flow" paragraph in §2.2
   (∂c_t/∂c_{t-1}=f highway); a concat→conv→split cell diagram.
4. Minor: Fig 2.2 (dataset bar) title still says "27 active regions"; symbol W (total
   map) vs L (winding functional) not unified.

## User (Manik) preferences — FOLLOW THESE
- **Minimal citations in main text** (appendix method-citations OK).
- Present the **total** as the intended target; do NOT frame rate-vs-total as a mistake.
- **Report the final model** — never select a checkpoint on the held-out region (prof
  flagged "not manipulated"). Integrity matters throughout.
- **Full transparency**: tensor shapes, matrices, end-to-end pipeline.
- **Gridded tables**, holdout separated at bottom.
- Figures **near their references**.
- Viva audience = **physicists who know the physics, not the ML** (see `VIVA_PREP/`).
- Deadline was a **19:30 IST talk on 2026-07-14** (Google Meet walkthrough of
  procedure/findings/plots to the professor, Dr. Abhishekh Kumar Srivastava).

## Companion docs (this folder, presentation_2026_07_14/)
`SUMMARY.md` (numbers+talk arc), `METRICS.md` (every metric justified),
`VIVA_PREP/{00_walkthrough,01_critiques,02_qa,03_unstated_assumptions}.md`,
`logs/` (phase2_train + 3 ablation arms + clip_sensitivity), figure PDFs/PNGs, and the
phase-2 checkpoint `loo_phase2_plain.pt` (+ `.json` config). Findings note:
`.planning/findings_M6_loss_ablation.md`. Memory: `~/.claude/.../memory/` (group_id manik).
