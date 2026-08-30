# Paper 3 — CoI-Verbalized

*"CoI-Verbalized: Discriminative Self-Consistency Verification Beats Verbalized Confidence
on Long-Form Factuality."* Result data (summary JSONs behind every table) is in the sibling
`../results/` folder.

Self-contained ACL-format source. **Compiles as-is on Overleaf** (acl.sty + acl_natbib.bst bundled).

## Files
- `main.tex` — the paper
- `references.bib` — bibliography
- `acl.sty`, `acl_natbib.bst` — official ACL 2023+ style (bundled)

## Compile
Overleaf: upload the zip, set the main document to `main.tex`, compile (pdfLaTeX).
Local: `pdflatex main ; bibtex main ; pdflatex main ; pdflatex main`

## Result provenance (all numbers regenerable from disk)
- Master grid (6 targets x up to 12 datasets, rungs n=1/3/6/7): `scripts/coi_master_table.py`
  reads results_coi_extend / _verify / _longfact / _bigtargets(_health) / _gen / _closed.
- 8B suite: `results_coi_extend` + `results_coi_verify` + `results_coi_longfact` + `results_coi_gen`
  (gsm8k/hotpot). Large open (32B/70B): `results_coi_bigtargets(_health)` (seed 0, 4-bit).
  Closed (GPT-4o/Claude via gateway): `results_coi_closed` (`scripts/coi_closed.py`).
- Multi-sample baselines (DSE / LexSim / EigV, N=5, 3-seed): `results_full/aggregated.json` (trivia)
  and `results_full_health/stage_cd_<gen>_seed*.json` (medlfqa).
- Aggregation-search appendix: `scripts/coi_aggsearch_offline.py`.
- Method: `src/TruthTorchLM/truth_methods/coi_verbalized.py` (rungs n=1..7; n=6 decision-verify,
  n=7 self-consistency). Generation prep: `scripts/coi_gen_prep.py`.

## Known gaps / before submission
- De-anonymize the author block.
- Single-seed CoI; baselines are 3-seed means. Add CoI seed variance for camera-ready.
- Large-open (32B/70B) and closed targets cover 6 cached datasets, NOT the full 12 — no LongFact
  or multi-hop cell on those columns. The "overconfidence" claim would be strengthened by a
  long-form-general closed cell (needs LongFact generation on closed/large targets).
- BioASQ excluded as degenerate (near-single-class); noted in-paper.
- Verify the semantic-entropy / spectral citation details in `references.bib`.

Code + data: https://github.com/jsl5710/TruthTorchLM-HC

## Figures
Figures are drawn inline with `pgfplots` (no external image files), so the
source is self-contained. Overleaf includes pgfplots by default; a local build needs
TeX Live's `pgf`/`pgfplots`. `\pgfplotsset{compat=1.17}` is set in the preamble.
