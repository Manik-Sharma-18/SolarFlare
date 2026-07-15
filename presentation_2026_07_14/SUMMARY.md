# Presentation pack — 2026-07-14

Winding-flux **accumulated-total** forecasting, ConvLSTM encoder–forecaster,
strict **leave-one-active-region-out** (holdout HARP 11930, `harp_1149` excluded).

## Headline (Phase-2 long run: grid 88×132, 3 layers, 130 epochs)
- Persistence MAE **0.0115** → ConvLSTM MAE **0.0105** (final) / **0.0103** (best ckpt)
- **Skill over persistence +8.6%** (final) / **+10.4%** (best) — reproduces thesis (+9.3%)
- Per-horizon skill **[+2.3%, +8.1%, +10.4%, +9.7%]** at t+12/+24/+36/+48 min
  → **skill grows with lead time**, model below persistence at every step
- **Active-window skill +10.4%** ≫ quiet-window +4.8% → edge is where the field moves

## Figures
| file | shows |
|---|---|
| `fig_triptych.png/pdf` | Prediction / Truth / Difference per lead time (best window). Pred≈Truth; error small-scale (±0.03, 1/10 field scale), localized at winding-lobe boundaries, grows +12→+48 min. |
| `fig_skill_horizon.png/pdf` | MAE vs lead time, model vs persistence (SEM bars). Model below persistence every step, gap widening. |
| `fig_ablation.png/pdf` | Loss-reweighting ablation (skill by arm, all vs active windows). |
| `loo_phase2_plain.png/pdf` | Input history → 4 forecast frames vs observed, UTC-stamped. |

## Ablation (Phase-1 quick, 35 epochs, all arms identical config)
Tested two forecast-loss reweightings vs plain Huber:

| arm | skill (all) | skill (active) | verdict |
|---|---|---|---|
| **plain** (Huber) | **+0.068** | **+0.087** | winner |
| horizon (lead-time ramp) | +0.063 | +0.081 | tied — traded early skill for late, no net gain |
| delta (weight by \|target−persist\|) | −0.447 | −0.293 | toxic — wrecked persistence-easy pixels |

**Message:** on a persistence-dominated *total*, plain Huber already sits at the
thin achievable margin; reweighting only moves error around. Skill is real but
localized to active windows. Full detail: `.planning/findings_M6_loss_ablation.md`.

## Talk arc (suggested)
1. Problem — forecast 2D winding-flux *maps*, not the 1D scalar (lit gap).
2. Data — rate vs accumulated total; total is the forecastable object (cumsum).
3. Model — ConvLSTM encoder–forecaster, residual/persistence bias, leave-one-AR-out.
4. Result — triptych + horizon curve (+10% skill, grows with lead, active-window).
5. Rigor — loss ablation (negative result, honest ceiling), persistence baseline throughout.

## Provenance
- Script `scratchpad/forecast_loo_wl.py` (5060ti). Config in `loo_phase2_plain.json`.
- Checkpoint `loo_phase2_plain.pt` (best, MAE 0.0103). Rebuild via config dict inside.
