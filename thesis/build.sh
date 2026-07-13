#!/bin/bash
cd "$(dirname "$0")"/build
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
export TEXINPUTS=..:; export BIBINPUTS=..:; export BSTINPUTS=..:
pdflatex -interaction=nonstopmode ../main.tex >/dev/null 2>&1
for a in main1 main2 main3 main4; do [ -f $a.aux ] && bibtex $a >/dev/null 2>&1; done
pdflatex -interaction=nonstopmode ../main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode ../main.tex >/dev/null 2>&1
