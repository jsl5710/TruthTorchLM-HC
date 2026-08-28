# Result artifacts (backing the paper tables)

Aggregated scored metrics (AUROC/AUPRC/PRR/ECE/ACE/MCE/Brier + latency) for every
method x dataset x target. Raw generation caches (`.parquet`) and model weights are NOT
included (large; regenerable) — these JSONs are the numbers behind the tables.

| dir | what | regenerate / read with |
|---|---|---|
| `results_mt/` | direct pure-BB + grey-box methods, 2 open targets, full N-sweep, 8 metrics | `scripts/stage_cd_open.py` |
| `results_closed/` | direct methods + verbalized on GPT-4o / Claude-Haiku | `scripts/stage_cd_api.py`, `scripts/verbalized_closed.py` |
| `results_mt_estimators/` | 9 read-outs (incl. Perplexity) x proxy x dataset + multi-metric | `scripts/mt_estimators.py` |
| `results_mt_score/` | proxy native scoring (DALD/DisAAD/Ours), all students | `scripts/mt_score.slurm` |
| `mt_stack/` | 5-signal LODO stacker per-item dumps + reports | `scripts/mt_stack_fit.py` |
| `results_rq5/` | single-target uncertainty-aware oracle sweep | `scripts/rq5_score.py` |

Table regenerators: `mt_proxy_fair.py`, `mt_variant_ppl.py`, `mt_est_bb_vs_gb.py`,
`mt_est_metrics_report.py`. Narrative: `docs/multitarget_findings.md`.
