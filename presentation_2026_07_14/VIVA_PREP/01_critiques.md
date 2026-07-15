# Critiques — the tough examiner's view (be ready)

Ranked by how likely he presses it. For each: the objection, and your answer.

## 1. The test is N = 1 (biggest)
You call it leave-one-AR-out but report a single held-out region (HARP 11930).
One region has no error bar of its own — it could be a lucky fold.
**Answer:** "The window-level bootstrap gives [+6.8, +10.3] on this region, so
the number is stable *within* it. The honest extension is the skill *distribution*
across all folds; that is the immediate next run." (Optionally: launch 3–4 more
folds and quote mean ± std.)

## 2. Did you manufacture your own forecastability?
Integrating a near-white rate gives a random walk, which is trivially
autocorrelated. "You made an unforecastable signal forecastable by summing it."
**Answer:** "The accumulated winding is a genuine physical quantity — the net
non-potential winding injected — not a smoothing trick. And I never claim the
total is hard to forecast; I quote persistence everywhere and report only the
+8.6% the model adds *beyond* it."

## 3. The method blurs away the very structure you sell
Premise: the map's fine PIL structure is the signal. Method: low-pass + downsample
it away, forecast the envelope. So does the 2D model beat a scalar?
**Answer:** "The scalar-mean baseline (0.398, ~35× persistence) shows the spatial
structure carries essentially all the signal; the low-pass keeps the *large-scale*
structure and discards only the temporally-incoherent fine detail (PSD + autocorr,
Figs 2.5/2.x), which is unpredictable anyway." *Weak spot — own it: a per-PIL
resolved model is future work.*

## 4. What is the scientific payoff?
Forecasting maps 48 min ahead is a pipeline result, not a flare-physics result.
**Answer:** "Correct. The contribution is the pipeline + an honest ceiling, and it
sets up the real question — does winding *structure* classify flare-productive
regions. I have preliminary evidence spatial winding stats beat the scalar for
flare TSS; that is the outlook." Don't imply the forecast *is* the science.

## 5. Plot-level
- **Triptych** now shows the **median** window (fixed) — say so, pre-empts
  cherry-picking.
- **Horizon error bars** are optimistic: windows overlap (stride 4) →
  effective N < 46. State it.
- **Delta ablation** could look like a strawman (floor 0.2). Say: "the floor was
  untuned; the point is that naive reweighting is fragile, not worthless."

## 6. Data quality
Limb effect shown (HARP 892) but handling not stated; if limb-degraded frames sit
in the training pool they inject noise. One sentence on your policy closes it.

## 7. Missing baselines (besides persistence)
No AR/linear extrapolation of the total, no SHARP-parameter comparison. The
scalar-mean row helps but is a decomposition, not a competitor model.

---

## Improvement pointers, ranked for the remaining hours
1. **Done:** bootstrap CI; scalar-mean baseline; median triptych.
2. **High value, ~4 h:** run 3–4 more leave-one-out folds → mean ± std skill (kills #1).
3. **Free, now:** add the caveat sentences (limb handling, error-bar autocorrelation,
   delta floor) into the thesis discussion.
4. **Framing, free:** open/close on honest scope (pipeline + ceiling → classification).
