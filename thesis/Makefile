.PHONY: all clean figures abstract watch

all: main.pdf

main.pdf: main.tex thesis.cls $(wildcard chapters/*.tex) $(wildcard frontmatter/*.tex) $(wildcard appendices/*.tex) bib/references.bib
	latexmk -pdf -interaction=nonstopmode main.tex

abstract:
	cd extended_abstract && latexmk -pdf -interaction=nonstopmode abstract.tex

figures:
	cd .. && python3 scripts/thesis_figs/val_curves.py
	cd .. && python3 scripts/thesis_figs/render_masks.py
	cd .. && python3 scripts/thesis_figs/wind_examples.py
	cd .. && python3 scripts/thesis_figs/clip_audit.py
	cd .. && python3 scripts/thesis_figs/probe_scatter.py

watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls *.synctex.gz
