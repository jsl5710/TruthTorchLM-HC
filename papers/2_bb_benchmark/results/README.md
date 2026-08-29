# Benchmark paper — result data

Result artifacts behind the tables in `../manuscript/main.tex` (the pure black-box UQ
benchmark, G × D → M → V). Per-seed / per-target JSON summaries (AUROC/AUPRC/PRR + measured
latency); raw generations and parquet caches are omitted (regenerable from the harness).

| folder | backs (paper section) |
|--------|-----------------------|
| `results_full/` | open-target grid (Qwen3, Llama; 7 domains), incl. `aggregated.json` |
| `results_full_health/` | health-domain grid (per-seed `stage_cd_*`) |
| `results_closed/`, `results_closed_full/` | closed targets (GPT-4o, Claude-Haiku via gateway) |
| `results_mt/` | large open targets (Qwen3-32B, Llama-3.3-70B) |
| `results_gshake/` | consistency/spectral methods at N-sweep |
| `results_cudasync/`, `results_tau/` | CUDA-sync latency measurement / threshold calibration |
| `results_cat2/`, `results_validate/` | category-2 methods / judge validation |

Full pipeline (Stage A→D, generators registry, method readiness): `hc_benchmark/` + `scripts/`.
Every (G,D) generation is cached and content-hashed so methods compare on identical inputs.
