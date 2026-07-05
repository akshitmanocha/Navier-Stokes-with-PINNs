# Building `paper.tex` → PDF

The paper is a single self-contained LaTeX file (`report/paper.tex`) with the
bibliography embedded — no `.bib` file needed. It pulls two figures from
`../figures/` (`headline_germany.png`, `synthetic_recovery.png`).

## Easiest: Overleaf (no install, recommended)
1. Go to <https://www.overleaf.com> → **New Project → Upload Project**.
2. Zip and upload these, preserving the folder layout:
   ```
   report/paper.tex
   figures/headline_germany.png
   figures/synthetic_recovery.png
   ```
   (Or upload `paper.tex` and the two PNGs, then fix the image paths — see note.)
3. Set the main document to `paper.tex` and click **Recompile**.

**Path note:** `paper.tex` uses `\graphicspath{{../figures/}}`, which assumes the
file sits in `report/` with `figures/` as a sibling. On Overleaf, easiest is to
upload the whole repo so that structure is preserved. If you instead put
everything in one flat folder, change the line near the top to
`\graphicspath{{./}}`.

## Local (if you install TeX)
- macOS: install MacTeX (`brew install --cask mactex`) or BasicTeX, then:
  ```bash
  cd report
  pdflatex paper.tex && pdflatex paper.tex   # run twice so references resolve
  ```
- Or use Tectonic (single binary, auto-downloads packages):
  ```bash
  cd report && tectonic paper.tex
  ```

Run `pdflatex` **twice** — the first pass writes the cross-references and
citations, the second resolves them.

## Output
A 2–3 page two-column paper: abstract, intro, SEIR background, method, results
(with the hero + synthetic figures and two result tables), limitations,
conclusion, and 11 references.
