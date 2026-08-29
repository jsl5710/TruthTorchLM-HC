# Papers

Three papers from this pure black-box UQ project. Each folder holds the manuscript and the
result data behind its tables, side by side:

```
papers/<paper>/
  manuscript/   ACL-format source (main.tex, references.bib, acl.sty, acl_natbib.bst, README)
  results/      per-seed summary JSONs behind the tables (+ provenance README)
```

| # | folder | paper |
|---|--------|-------|
| 1 | `1_distilled_proxy/` | *Read-Out, Not Objective* — latency-first benchmark of distilled-proxy black-box UQ |
| 2 | `2_bb_benchmark/` | Pure black-box UQ benchmark (G × D → M → V; 4 frontier targets, ~70 methods) |
| 3 | `3_coi_verbalized/` | *CoI-Verbalized* — discriminative self-consistency verification vs. verbalized confidence |

Each manuscript compiles as-is on Overleaf (style files bundled). Result folders carry only
summary JSONs; raw generations / parquet caches are regenerable from `../hc_benchmark/` +
`../scripts/`.
