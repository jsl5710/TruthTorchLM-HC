# CoI-Verbalized result data

Summary JSONs behind every table in the paper (`../manuscript/main.tex`). One file per
`(target, seed)`; each holds per-dataset cells with the AUROC/ECE/latency for rungs
n=1..7. Regenerate the tables with `scripts/coi_master_table.py` (and the per-study
`coi_*_compare.py`). Large per-item raw chain logs (`rawlog_*.jsonl`) are omitted to keep
the repo lean; they are regenerable from the scripts.

| dir | targets | what |
|-----|---------|------|
| `results_coi/` | llama-3.1-8b, qwen3-8b (seed1) | original n=1..5 ablation + `aggsearch_offline` (App. G) |
| `results_coi_verify/` | 8B (seed1) | decision-framed (n6) + self-consistency (n7) rungs |
| `results_coi_longfact/` | 8B (seed1) | LongFact ablation |
| `results_coi_extend/` | 8B (seed1) | 7 extra cached datasets (NaturalQA, PopQA, MCQ, health) |
| `results_coi_gen/` | 8B (seed1) | GSM8K + HotpotQA (generated + gold-judged) |
| `results_coi_bigtargets{,_health}/` | qwen3-32b, llama3.3-70b (seed0, 4-bit) | large-open grid |
| `results_coi_closed/` | jhu-gpt-4o, jhu-claude-haiku-4.5 (seed0) | frontier-closed via gateway |
| `cache_coi_longfact/` | — | LongFact factuality fractions + binarized labels (70B judge) |

Method: `src/TruthTorchLM/truth_methods/coi_verbalized.py` (rungs 1–7).
