# Metrics — what each is, and why (viva justification)

All fields are in **normalised units**: after `asinh(w/10³)` + robust ($99.5$-pct)
scaling + clip to $[-1,1]$. So every error below is dimensionless and comparable
across active regions. This is a **field-regression** task (forecast a 2D map), so
the metrics are regression/forecast metrics — not event-classification scores
(TSS, CSI, AUC), which belong to the separate flare-classification task (outlook).

---

## A. Core forecast metrics

### 1. Mean Absolute Error (MAE)
**Def:** mean of $|{\hat{x}} - x|$ over all pixels, frames, windows.
**Why MAE, not MSE/RMSE:** the residual field is sparse and heavy-tailed —
error concentrates in a few specks at the winding-lobe boundaries. MSE squares
those, so a handful of boundary pixels would dominate the score and hide the
bulk behaviour. MAE weights every pixel linearly → reports typical error, robust
to the rare extremes. It is also in the field's own units, so it is directly
interpretable (e.g. $0.0105$ on a $[-1,1]$ field $\approx 0.5\%$ of full scale).
**Caveat:** MAE alone says nothing about *skill* — a strongly autocorrelated
field has low MAE for free (see §C).

### 2. Persistence baseline
**Def:** the forecast "copy the last observed frame forward."
**Why:** for a smoothly evolving, strongly autocorrelated field, persistence is
the natural **zero-skill reference** — the forecast you get for free with no
model. The model's residual parameterisation ($\hat{x}=x_t+\Delta_\theta$) is
literally built around it, so it is the fair thing to beat. Quoting it beside
every result is what keeps the account honest.

### 3. Skill over persistence
**Def:** $\text{skill}=1-\mathrm{MAE_{model}}/\mathrm{MAE_{persistence}}$.
**Why:** a standard forecasting **skill score**. It divides out the "free"
predictability persistence already captures and reports only what the **model
adds**. $+8.6\%$ means the model's error is $8.6\%$ below the trivial forecast.
This is the number that actually measures learning, which is why it is the
headline, not raw MAE.
**Reads as:** $0$ = no better than copying; $>0$ = genuine dynamics learned;
$<0$ = worse than doing nothing.

### 4. Per-horizon (per-lead-time) skill
**Def:** skill computed separately at each lead $+12,+24,+36,+48$ min.
**Why:** a single averaged number hides *how* skill behaves with forecast
distance. Splitting by lead time shows the model and persistence nearly coincide
at $+12$ min (one step, field barely moves) and the model pulls further below
persistence as lead grows — evidence it has learned **short-term dynamics**, not
just a static correction. Its advantage is largest exactly where persistence
decays fastest.

### 5. Active-vs-quiet window skill
**Def:** skill restricted to windows whose persistence error is above (active) /
below (quiet) the median.
**Why:** on a persistence-dominated target most windows are quiet — the field
barely changes, and nothing can beat copying it. Averaging over them **dilutes**
any real skill. Separating the active windows (where the field actually moves)
localises where the model helps: skill is uniformly larger there, confirming the
model's edge is in the dynamic windows, not a global artefact.

---

## B. Uncertainty and significance

### 6. Bootstrap 95% confidence interval + $P(\text{skill}>0)$
**Def:** resample the $46$ held-out windows with replacement $B=20{,}000$ times,
recompute skill each time, take the $2.5$–$97.5$ percentile range.
**Why:** the skill is one number from a finite sample of windows; without an
interval one cannot tell signal from fluctuation. The bootstrap is
**distribution-free** (no Gaussian assumption) and resamples **whole windows**,
which respects that pixels within a window are correlated. Result
$[+6.8\%,+10.3\%]$, entirely above zero, $P(\text{skill}>0)=1.000$ → the
improvement is real, not luck.
**Caveat (be ready for it):** windows overlap at stride $4$ and are themselves
autocorrelated, so the effective sample size is $<46$; the interval is a
window-level bootstrap, not an independent-sample one. Still directional and
honest.

---

## C. Baseline for the thesis premise

### 7. Scalar-mean baseline MAE
**Def:** replace each frame by its single spatial-average value (a flat map),
then score MAE.
**Why:** this is not a competitor — it is a **decomposition** that measures how
much predictable signal lives in the spatial *structure* versus the spatial
*mean*. Because the winding field is bipolar, its spatial mean nearly cancels, so
the flat forecast is catastrophic ($0.398$, ~$35\times$ persistence). That
number *is* the thesis premise made quantitative: almost all of the field's
content is in the structure the scalar integral throws away.
**How to say it:** "the honest baseline is persistence (which keeps structure);
the scalar row shows what you lose the moment you collapse to a number."

---

## D. Diagnostic / data-justification metrics

### 8. Spatial autocorrelation vs time lag
**Def:** correlation between winding frames separated by a time lag.
**Why:** justifies the **target choice**. The raw rate is near-white (lag-1
$\approx0.02$) → unforecastable; its time-integral (the total) is smooth (lag-1
$\approx0.98$) → forecastable. This is the evidence that we forecast the
accumulated total, not the rate.

### 9. Radially-averaged power spectrum (PSD)
**Def:** field energy as a function of spatial frequency.
**Why:** justifies the **low-pass**. Energy exists at every scale, but the
fine scales are the temporally incoherent ones (from §8) — energetic but
unpredictable. The PSD + autocorrelation together show the low-pass discards
energy that could not have been forecast anyway.

---

## E. Training objective (not a reporting metric)

### 10. Smooth-$L_1$ / Huber loss
**Def:** $L_2$ near zero, $L_1$ in the tails (elbow at $\beta=1$).
**Why:** the **optimisation** signal, distinct from the eval metric. Huber gives
stable, $L_2$-like gradients for small residuals but $L_1$-like robustness to the
rare high-amplitude frames, so training is not hijacked by outliers. We
deliberately **report** unweighted MAE (§1), not the loss, so the yardstick is
fixed while the training signal can be engineered — that separation is what makes
the loss ablation (Table 3.2) a fair comparison.

---

## What we deliberately do NOT use here, and why
- **RMSE/MSE** — outlier-dominated on this sparse-residual field (§1).
- **SSIM / perceptual scores** — designed for natural images; no physical meaning
  for a signed pseudoscalar field.
- **Pixel correlation** — scale-free, insensitive to bias, and inflated by the
  field's strong autocorrelation; would flatter the model.
- **TSS / CSI / AUC** — event-classification scores; this chapter forecasts a
  field, it does not classify flares. Those are the metrics for the outlook task.
