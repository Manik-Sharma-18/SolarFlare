# Thesis Review — Issue Tracker

Source: 5 parallel professor-style subagents, 2026-05-20.
Status: `[ ]` open · `[x]` fixed · `[!]` blocked (needs author input) · `[~]` partial.

Build status (2026-05-20): `main.tex` compiles clean — 71 pp, 0 warnings, 0 undefined refs/cites. `extended_abstract/abstract.tex` compiles clean — 2 pp.

---

## 1. Critical (must fix before submission)

- [x] **thesis.cls** — supervisor name, 12pt, twoside, 1.5in inner margin, 1.5× spacing, `\the...` aliases, fancyhdr `headheight=15pt`.
- [x] **frontmatter/*.tex** — `\@` macros replaced with class-level `\the...` aliases.
- [x] **`natbib` + `bibtopic`** — verified clean build with all five `btSect` blocks.
- [ ] **techreports.bib:21-27 vjepa21_2026** — arXiv ID `2603.14482` looks fabricated; verify or remove.
- [x] **6_probe.tex** — wrong F11 cross-ref removed.
- [x] **6_probe_flare.tex** — AUC-vs-TSS → AUC-vs-AUC.
- [x] **5_experiments_findings.tex:23 F9** — claim "four of five novel cubes" verified against Table 6.2 (all five novel cubes listed with persistence medAPE; four beat persistence, harp_245 the documented exception).
- [x] **3_data.tex / acknowledgement.tex** — "senior collaborator" removed; cites SDO/HMI/SHARP/Prior-MacTaggart pipeline.
- [x] **thesis.cls:32** — `% per friend's template` comment scrubbed.
- [x] **3_data.tex:86** — broken `app:clip` ref → `app:clipping`.

## 2. First-Person Speech

- [x] **7_summary.tex:174** — "known to the author" → "reported in the published literature to date".

## 3. Vague Attributions

- [x] **acknowledgement.tex** — generic "peers" sentence removed.
- [x] **c_clipping.tex:62-64** — "deferred work in the project log" → "deferred to future work (Chapter 7)".

## 4. Missing / Wrong Citations

### Missing
- [x] **1_intro.tex:8-11** — flare energy budget → `aschwanden2005physics, priest2014mhd`.
- [ ] **1_intro.tex:11-14** — space-weather operational impacts → NRC 2008 or Pulkkinen 2007 (not in bib).
- [~] **1_intro.tex:32-38** — TSS≈0.7/0.85 — wrong cites removed, claim weakened. Add Nishizuka 2018 / Liu 2019 if author has them.
- [ ] **1_intro.tex:62-69** — persistence-collapse failure mode → Mathieu 2016 or Ravuri 2021 (not in bib).
- [ ] **2_background.tex:36** — Raphaldini six-to-seven hours AR 11318 → page/figure cite.
- [ ] **2_background.tex:58** — upstream pipeline $10^5$–$10^7$ legitimate range → forward-ref to data chapter.
- [x] **3_data.tex** — `pesnell2012sdo`, `hoeksema2014hmi`, `scherrer2012hmi` (plate scale), `prior2020helicity, priormactaggart2024winding` (chirality).
- [ ] **4_architecture.tex:43-46** — Surya backbone channel budget → add cite to `surya2025`.
- [x] **4_architecture.tex** — ViT (`dosovitskiy2020vit`), RoPE (`su2021rope`), smooth-$L_1$ (`girshick2015fastrcnn`).
- [ ] **4_architecture.tex:198** — gradient checkpointing → Chen et al. 2016 (not in bib).
- [x] **5_experiments.tex:15** — cosine + AdamW → `loshchilov2017sgdr, loshchilov2019adamw`.
- [ ] **6_probe.tex:32** — smooth-$L_1$ → cite as above.
- [ ] **6_probe_flare.tex:55** — lag-one persistence baseline → Barnes / Leka (not in bib).
- [ ] **7_summary.tex:97** — pixel-only TSS≈0.7 ceiling → cite.

### Wrong / Mismatched
- [x] **6_probe_flare.tex:30** — wrong cite rephrased.
- [x] **7_summary.tex:162-163** — wrong cite rephrased (climax removed).
- [x] **1_intro.tex:38-41** — wrong climax/aurora cite removed.
- [ ] **1_intro.tex:14-16** — forecast-horizon claim cites `bobra2014sharp` (SHARP params paper, not horizons).

## 5. Style / Consistency

- [ ] Numeric precision drift — $0.040$ vs $0.04017$ across chapters.
- [ ] Unit-format drift — "12-min" vs "twelve-minute" vs "12 min".
- [ ] Model name — "V-JEPA-2-AC" vs "V-JEPA~2" vs `\textsc{ac}`. Unify.
- [ ] Identifier styling — `\texttt` vs `\mathtt` vs `\mathit` for config keys.
- [ ] Symbol collision — $r$ = mask ratio (Ch5) vs $r$ = Pearson (Ch6).
- [x] **4_architecture.tex:18** — "an target" → "a target".
- [x] **5_experiments.tex:20** — `k^\top` inside `\texttt{}` moved to math mode.
- [x] **2_background.tex:99** — "MaskViT subsequently scale" sentence reflowed.
- [x] **2_background.tex:147 vs 1_intro.tex:71** — "NASA-IMPACT/IBM" → "NASA--IBM" (unified).
- [ ] Abstract uses author-year cites with no bibliography; convert to bibtex or footnotes.
- [x] Acronyms first-use — HMI/AIA/SDO/SHARP/GOES/HARP/EMA/LoRA/medAPE/MAE/MLP/AUC/CSI/HSS/ViT expanded at first occurrence.
- [x] **1_intro.tex:159** — "median ... medAPE" double-count removed.

## 6. Figures / Diagrams

- [ ] **Ch4** — no architecture/block diagram. Add JEPA schematic.
- [ ] **Ch3** — no data-pipeline figure.
- [ ] **Ch3** — no example data frame.
- [ ] **6_probe_flare.tex** — no ROC curve.
- [x] All figures `[h]` → `[htbp]`.
- [ ] **d_curves.tex** captions — no axis labels / legend / colour convention.
- [ ] probe_*_bars.pdf — colour-only encoding; add hatch/marker fallback.
- [ ] **b_masks.tex:71-72** — add subfigure panels (a-e).

## 7. Tables

- [x] All tables `[h]` → `[htbp]`.
- [ ] Numeric columns left-aligned → right-aligned for decimal compare.
- [ ] Single-seed reporting — disclose explicitly or replicate.
- [ ] Mask mixture inconsistency: `a_hyperparams.tex` 3-family vs `b_masks.tex`/`7_summary.tex` 5-family. Reconcile.
- [ ] **5_experiments.tex** — no E25–E28 mask-ratio sweep table.

## 8. Result-Reporting

- [ ] **6_probe.tex:181** — "CONFIRMED at thesis scale" — no random-init control. Downgrade to HYPOTHESIS or add control.
- [ ] **6_probe_flare.tex** — AUC=1.000 / TSS=1.000 on n=174 without bootstrap CI overstates.
- [ ] **5_experiments.tex** — E28 terminated at epoch 36; concave claim on 3 surviving points.
- [ ] **6_probe.tex:73-75** — aggregate $R^2/r$ over concatenated cubes; report per-cube also.
- [ ] **6_probe.tex** — $\log y = a \log \hat y + b$ on signed $y$ needs $\log|y|$ or sign-preserving definition.

## 9. Math / Notation

- [ ] **6_probe.tex:16** — $h_p, w_p$ undefined locally.
- [ ] **5_experiments.tex:27** — $M$ used as both Boolean and $\{0,1\}$ multiplier.
- [x] **3_data.tex** — winding-flux units stated in priors list.

## 10. Grammar / Phrasing

- [ ] **3_data.tex:25-26** — "will ever be supplied" — already removed in rewrite.
- [ ] **3_data.tex:71** — "this guard was found to destroy legitimate signal" — passive evasive.
- [x] **3_data.tex** — "instrument-bad" → "pixels flagged as instrument-invalid".
- [ ] **4_architecture.tex:22-24** — semicolon-joined loose clauses.
- [ ] **4_architecture.tex:106** — "that is, frame $t$" comma splice (acceptable; low priority).
- [ ] **4_architecture.tex:168-171** — narrative bug erratum; move or rephrase.
- [x] **4_architecture.tex** — `${\sim}\,12\,\mathrm{GB}$` → `approximately 12\,GB`.
- [x] **5_experiments.tex** — "missing the submission window by seven days" → "exceeded the available time budget".
- [ ] **5_experiments.tex:175** — figure width inconsistency.
- [x] **6_probe.tex** — "four-hundred-thousand-parameter" → `$\sim\!4 \times 10^{5}$-parameter`.
- [x] **7_summary.tex** — "Finding~F9 CONFIRMED" → "Finding~F9, confirmed".
- [ ] **7_summary.tex:184-188** — closing run-on (140+ words); split.

## 11. Bibliography

- [ ] **theses.bib** — `khan2026thesis` not cited; verify or remove.
- [ ] **techreports.bib:21-27 vjepa21_2026** — verify arXiv `2603.14482`.
- [ ] All `@techreport` preprints — replace `author = {Name and others}` with full lists.
- [ ] Books bib — add ISBN/DOI; brace proper nouns (`{Sun}`).
- [ ] **techreports.bib** — Meta / Meta AI / Meta FAIR institution naming inconsistent.
- [ ] Cite-key conventions mixed (`tong2022videomae` vs `surya2025`); normalise.
- [ ] **techreports.bib wmae2023** — institution = "arXiv" meaningless; use real affiliation.

## 12. Frontmatter / Guidelines Compliance

- [x] **certificate.tex** — "our supervision" neutralised; candidate-pronoun neutralised.
- [x] **acknowledgement.tex** — "Their" → "His".
- [x] **abstract.tex** — Keywords line added.
- [x] **abstract.tex** — "one-degree-of-freedom" → "two-parameter (one $(a,b)$ pair)".
- [x] **abstract.tex** — "earlier launch" → "earlier sanity-scale configuration".
- [x] **abstract.tex** — `$\tau \downarrow$-better` jargon → "monotone response in which a smaller $\tau$ yields lower validation loss".
- [x] **main.tex** — "References and Bibliography" → "Bibliography".
- [ ] **copyright.tex** — guidelines may require Head-of-Department countersignature line.
- [ ] **frontmatter symbols.tex:29** — French apostrophe `d'unit\'es`; cosmetic.

---

## Build instructions

```bash
source /Volumes/T9/IndraAstra/.venv/bin/activate
export PATH=/Users/indra/.pytinytex/bin/universal-darwin:$PATH
export BIBINPUTS="/Volumes/T9/IndraAstra/manik/SolarFlare/thesis:/Volumes/T9/IndraAstra/manik/SolarFlare/thesis/bib:"
cd /Volumes/T9/IndraAstra/manik/SolarFlare/thesis
mkdir -p build
pdflatex -interaction=nonstopmode -output-directory=build main.tex
for i in 1 2 3 4 5; do (cd build && bibtex main$i); done
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
# Output: build/main.pdf

cd extended_abstract && mkdir -p build
pdflatex -interaction=nonstopmode -output-directory=build abstract.tex
# Output: extended_abstract/build/abstract.pdf
```

---

## Progress log

- 2026-05-20 — review created. Supervisor: Dr. Abhishekh Kumar Srivastava. Cube provider: no person attribution.
- 2026-05-20 — class-file rewrite, frontmatter refactor, critical-bug fixes, acronyms, citation additions (`pesnell2012sdo`, `hoeksema2014hmi`, `priormactaggart2024winding`, `dosovitskiy2020vit`, `su2021rope`, `girshick2015fastrcnn`, `loshchilov2019adamw`, `loshchilov2017sgdr`), wrong-citation removal (climax/aurora/bobra-for-TSS).
- 2026-05-20 — TinyTeX installed via `pytinytex` (variation=2, SSL_CERT_FILE via certifi). `bibtopic` installed via tlmgr. Clean build: 71 pp, 0 warnings, 0 undefined refs. Additional fixes from build: broken `app:clip` ref, fancyhdr headheight=15pt, `\texorpdfstring` on math-in-section-title for `M$+$/24~h` and `C$+$/12~h`, all `[h]` → `[htbp]`, `\sim` jargon → words, "submission window" / "known to the author" / "Finding F9 CONFIRMED" / "NASA-IMPACT" / "MaskViT scale" prose cleanup, wind-flux units note in priors list, generic-peers thanks and "project log" deferral pointer cleaned.
