# Cluster test plan — small run → full run

A staged ladder for validating the benchmark on the cluster. Each tier costs more and
proves more; a failure at an early tier is cheap to catch, so **run them in order** and
only advance when the current tier is green.

| Tier | Proves | Cost | Time |
| :-- | :-- | :-- | :-- |
| 0 · Install & offline tests | the code imports and its logic is correct | free | minutes |
| 1 · Tiny live smoke | the live pipeline (API → Stage A–D) produces real numbers | ~cents | minutes |
| 2 · Small real run | one full config end to end at small scale | ~a dollar | ~10–30 min |
| 3 · Full run | the real grid (all methods, full datasets, seeds, N-sweep) | scales up | hours |

---

## Tier 0 — install & offline tests (no API cost)

```bash
# get the code
git clone https://github.com/jsl5710/TruthTorchLM-HC.git   # or: cd repo && git pull
cd TruthTorchLM-HC
git submodule update --init --recursive --depth 1

# environment (install the CUDA build of torch first if the cluster needs a specific one)
conda create -y --name ttlm-hc python=3.10 && conda activate ttlm-hc
pip install -r requirements.txt
pip install -e .

# the offline suite — the logic layers, no GPU/API needed
python -m pytest -q                       # PASS ≈ 264 passed, a few skipped

# what's supported / what's ready
python -m hc_benchmark.capabilities
python -m hc_benchmark.readiness          # 8/10 ready; DisAAD + OOD gate need prep (expected)
```

**Green =** the test suite passes and the two reports print. If this fails, stop here — it's
an install/import problem, not a benchmark problem.

---

## Tier 1 — tiny live smoke (the acceptance test)

```bash
export OPENAI_API_KEY=sk-...              # target + LLM judge

# confirm the target is a pure text-only black box (one cheap call)
python scripts/verify_provider.py openai  # PASS ... TEXT-ONLY (black-box OK)

# the end-to-end smoke run: Stage A→D + a serial-vs-concurrent latency check
python scripts/smoke_run.py
```

First run downloads `microsoft/deberta-large-mnli` (~1.5 GB) for the NLI clustering — one
time. Knobs (env): `SMOKE_GENERATOR`, `SMOKE_DEVICE` (auto cuda/cpu), `SMOKE_DATASETS`,
`SMOKE_SIZE`.

**Green =** every `[D]` row has numbers (AUROC/PRR/ECE/MCE/Brier per method × N), both label
sets exist, `VerbalizedConfidence` aux-ms ≈ 0, the §5 concurrency check ran, and
`hc_benchmark/results/smoke_run.json` was written. AUROC ≈ 0.5 on a handful of items is
expected — this proves the machinery, not a research claim. Full checklist +
troubleshooting: `docs/SMOKE_RUN_RUNBOOK.md`.

---

## Tier 2 — small real run (one config, end to end)

The `hc_benchmark.run` driver runs a full config: Stage A→D, multi-seed, both correctness
criteria, writes a results bundle. Start with the smoke config (already small):

```bash
python -m hc_benchmark.run --config hc_benchmark/configs/smoke.yaml
# -> hc_benchmark/results/results_<dataset>_<generator>_<hash>.json
```

Then a slightly larger real slice — copy a config and bump `size_of_data` / add a dataset:

```bash
cp hc_benchmark/configs/health_primary.yaml hc_benchmark/configs/small.yaml
# edit small.yaml: size_of_data: 0.05, seeds: [0], n_sweep: [1, 5]
python -m hc_benchmark.run --config hc_benchmark/configs/small.yaml
```

**Green =** a `results_*.json` bundle is written with per-method, per-N, per-criterion
metrics and no exception. Inspect it — every method should have real AUROC/PRR/calibration
numbers.

> The default method set (`run.build_methods`) is the inference methods: VerbalizedConfidence
> + DiscreteSemanticEntropy + EigV + NumSemanticSet. **DisAAD** (needs a trained proxy) and
> the **OOD gate** (needs a KB fit) are not in this set — add them once prepared (see below).

---

## Tier 3 — full run (the grid)

Scale the config to the real experiment: full datasets, all seeds, the full N-sweep,
serial **and** concurrent, multiple generators.

```bash
# health biomedical primary, full
python -m hc_benchmark.run --config hc_benchmark/configs/health_primary.yaml

# sweep providers by pointing at each config
python -m hc_benchmark.run --config hc_benchmark/configs/anthropic_claude.yaml   # ANTHROPIC_API_KEY
python -m hc_benchmark.run --config hc_benchmark/configs/gemini.yaml             # GEMINI_API_KEY
python -m hc_benchmark.run --config hc_benchmark/configs/open_llama.yaml         # GPU host
```

Multi-seed mean ± std and bootstrap CIs come out of Stage D; the accuracy-vs-marginal-ms
frontier plot is produced by `hc_benchmark/plots/frontier.py` from the results.

### The two methods that need a prep step first

- **DisAAD** (proxy) — train it on the cluster, then it's usable:
  ```bash
  # edit hc_benchmark/configs/disaad.yaml (teacher = your target, student = small open model)
  python -c "import yaml; from hc_benchmark.disaad_train import DisAADTrainingConfig, train_proxy; \
             train_proxy(DisAADTrainingConfig(**yaml.safe_load(open('hc_benchmark/configs/disaad.yaml'))))"
  python -m hc_benchmark.readiness --proxy-path hc_benchmark/disaad/proxy   # flips to READY
  ```
  Then add `DisAAD.from_pretrained('hc_benchmark/disaad/proxy')` to `run.build_methods`.
- **OOD-PCA gate** — fit once on your KB: `PCAGate(embed_fn=...).fit(kb_documents)`.

---

## Rule of thumb

Never jump tiers. If Tier 1 is green but Tier 2 errors, it's a config/scale issue, not a
code issue — the pipeline already worked at Tier 1. Keep each run's `results_*.json`; the
config hash in the filename ties every number back to the exact conditions that produced it.
