#!/bin/sh
# Build docs/ariadne_II.tex in a throwaway texlive container (scratch copy, docs untouched).
set -e
cp -r /src/. /build/ ; true
cd /build
tlmgr update --self >/dev/null 2>&1 || true; tlmgr install txfonts sttools natbib float caption lineno helvetic times psnfss >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode ariadne_II.tex >/dev/null 2>&1 || true
bibtex ariadne_II > bibtex.out 2>&1 || true
pdflatex -interaction=nonstopmode ariadne_II.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode ariadne_II.tex >/dev/null 2>&1 || true
# Run from the repo root:
#   docker run --rm -v "$PWD/docs":/src:ro -v /tmp/ariadne2_build:/build \
#     -v "$PWD/docs/build_paper.sh":/b.sh:ro texlive/texlive:latest-small sh /b.sh
# The PDF lands in /tmp/ariadne2_build/ariadne_II.pdf.
