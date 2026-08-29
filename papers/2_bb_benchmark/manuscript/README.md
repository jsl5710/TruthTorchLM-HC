# Paper 2 — Pure black-box UQ benchmark (G × D → M → V)

Self-contained ACL-format source. **Compiles as-is on Overleaf** (acl.sty + acl_natbib.bst bundled).
Result data backing the tables is in the sibling `../results/` folder.

## Files
- `main.tex` — the paper
- `references.bib` — bibliography
- `acl.sty`, `acl_natbib.bst` — official ACL 2023+ style (bundled)

## Compile
Overleaf: upload the zip, set the main document to `main.tex`, compile (pdfLaTeX).
Local: `pdflatex main ; bibtex main ; pdflatex main ; pdflatex main`

## Before submission
- De-anonymize the author block.
- Verify the DisAAD / DALD citation details in `references.bib`.

Code + data: https://github.com/jsl5710/TruthTorchLM-HC
