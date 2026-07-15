# Unstated assumptions & gaps to flesh out

Places the thesis assumes the reader already knows something, or asserts without
citation/derivation. Grouped by severity. Each: **where → what's missing → fix.**

## A. Physics / data provenance (a physicist WILL ask)

1. **How winding flux is computed from the magnetogram — MISSING METHOD.**
   Ch1 §1.3 and Ch2 §2.1 use `wind` as given ("derived from SDO/HMI"). The actual
   algorithm (field-line winding integral / ARTop; Prior & MacTaggart) is never
   stated or cited. *This is the #1 gap.*
   **Fix:** one paragraph in Ch2 §2.1 defining winding flux as the rate of winding
   injection and citing the computation method (Prior & Yeates 2014; MacTaggart &
   Prior 2019 / the ARTop paper).

2. **Rate-vs-total physical identity is glossed.** Ch2 Eq. 2.2 calls `wind` the
   "per-frame winding" and cumsum the "accumulated total." But "winding *flux*"
   physically means a *rate* (dL/dt); the cumsum recovers the winding L(t). A
   physicist will ask if the persistence is physical or an integration artefact.
   **Fix:** state plainly that `wind` is the winding-flux *rate* and `W[t]` its
   time-integral, and that persistence is quoted precisely to separate physics
   from the integral's built-in smoothness. (You already argue this in §3.3 —
   surface it once in §2.1.)

3. **Units of winding flux never given.** "per-pixel |w| ≈ 10⁷" (CLAUDE.md) appears
   nowhere with units. Field called a "signed pseudoscalar" only.
   **Fix:** state the physical units once in Ch2 §2.1.

4. **AR tracking / co-registration assumed.** The model assumes frames of a cube
   are spatially aligned over multi-day sequences (HARP de-rotates/tracks). Never
   stated; differential rotation and foreshortening not discussed beyond the limb
   effect image.
   **Fix:** one sentence that HARP patches are tracked/registered, and that
   near-limb passages are the residual geometric limitation.

5. **Limb-effect handling not stated.** Ch2 §2.1 *shows* the limb effect (HARP 892,
   Fig 2.1) but never says how limb-degraded frames are treated in training.
   **Fix:** state the policy (kept as-is / masked / near-limb excluded).

6. **`harp_1149` exclusion unquantified.** "one cube … pathological, noise-dominated
   … excluded" — the pathology (values to ~10¹⁰) is not quantified.
   **Fix:** one clause giving the criterion.

## B. ML methods used but not cited

7. **Adam** (Ch2 §2.5) — cite Kingma & Ba (2015).
8. **Cosine learning-rate decay** — cite Loshchilov & Hutter (SGDR, 2017).
9. **Group normalisation** (Ch2 §2.3) — cite Wu & He (2018).
10. **Huber / smooth-L1 loss** — cite Huber (1964) (minor).
11. **Gradient clipping** — cite Pascanu et al. (2013) (minor).
12. **Bootstrap CI** (Ch3 §3.2) — cite Efron (1979) (minor, but stats-aware
    examiners like it).
   **Fix:** add these to `bib/refs.bib` and `\citep` at first use. Cheap, closes
   "where does this come from?" for every method.

## C. Choices stated as fact, not justified

13. **asinh softening s = 10³** — "fixed by prior probing of the amplitude
    distribution," but the probing is neither shown nor cited.
    **Fix:** one line on the criterion, or an appendix panel of the distribution.
14. **Downsample 88×132 and Gaussian σ = 1.5** — specific numbers, no rationale.
    **Fix:** justify (memory/compute vs envelope scale) in one clause.
15. **t_in = 10, t_out = 4** — history/horizon lengths asserted (Ch2 §2.3).
    **Fix:** one sentence on why (120-min context, 48-min horizon as a
    proof-of-concept, not a flare lead time).

## D. Logical leaps (assertions doing heavy lifting)

16. **"A model that can anticipate the field's evolution has, in effect, learned
    the structure a downstream forecaster would need."** (Ch1 §1.4, closing.)
    This bridge *justifies the entire forecasting task* but is asserted with no
    support. A referee will target it.
    **Fix:** soften to a hypothesis ("we take field-forecasting as a *proxy* task
    that a flare classifier could build on"), and name the classification follow-up
    as the actual test of the claim.

17. **"winding … is concentrated along polarity inversion lines, where flares
    originate."** (Ch1 §1.4.) Asserted; physics audience accepts, but uncited.
    **Fix:** cite the PIL–flare association (e.g. Schrijver 2007) — one ref.

18. **N = 1 test region generalisation.** Reporting one held-out fold implicitly
    assumes HARP 11930 is representative of "generalises across regions."
    **Fix:** report multiple folds, or explicitly scope the claim to this region.

## Priority for tonight (cheap, high credibility)
- B7–B12 + D17: add ~6 citations. Minutes. Removes the "uncited method" smell wholesale.
- A1 + A2: the winding-computation citation + rate/total sentence — the two a
  physicist is most likely to probe.
- D16: soften the proxy-task claim — protects the thesis's logic.
