# hc_benchmark — Stage A→D harness

The orchestration layer for the pure black-box UQ benchmark
([`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)). It sits **on top of** the
`TruthTorchLM` library and wires the fork's additions — latency instrumentation,
calibration/safety metrics, health datasets — into four cached stages that mirror the
DisAAD repo skeleton.

```
Stage A  generate   primary answer + n_max samples  →  Parquet cache   (stage_a_generate.py)
Stage B  label      correctness via BLEURT and LLM-judge, both kept    (stage_b_label.py)
Stage C  score      each method reads the cache, sweeps N∈{1,3,5,10,20} (stage_c_score.py)
Stage D  evaluate   §4 metrics + §5 frontier, per-dataset, mean±std     (stage_d_evaluate.py)
```

## The one idea that matters

**The same generations feed every method and every N.** Stage A draws them once and
caches them; Stage C serves each N in the sweep by *truncating* that cached list, and
hands it to each `TruthMethod` as a pre-built `sampled_generations_dict` so the target is
never called again at score time. This is protocol §6's central control: it removes
generation cost and generation variance as confounds between methods, and turns the
N-sweep into a cache read rather than N separate runs.

Every artifact is keyed by `BenchmarkConfig.content_hash()`, so a change to decoding
params yields a new cache rather than silently reusing an incompatible one (protocol §5
measurement hygiene).

## Run it

```bash
git submodule update --init --recursive --depth 1
pip install -r requirements.txt && pip install -e .

export OPENAI_API_KEY=...          # for an API target + LLM judge
python -m hc_benchmark.run --config hc_benchmark/configs/smoke.yaml
```

`smoke.yaml` is the round-one acceptance test (≈50 items, one API generator, two
black-box methods, N∈{1,5}). `health_primary.yaml` is a representative research config.

## What is deliberately left to the launch script

- **Method construction** (`run.build_methods`) is code, not config: a `TruthMethod` may
  need a loaded entailment model or a proxy checkpoint — an object, not a string. Round
  one ships the black-box pair `VerbalizedConfidence` + `NumSemanticSetUncertainty`
  (the text-only cluster-count consistency method; `SemanticEntropy` is grey-box here and
  excluded by the §1 filter).
- **Local (HuggingFace) targets** need a model + tokenizer handed in; `stage_a_generate`
  wires the API path and points you at the local one.
- **DisAAD and the other round-two methods** are not in `build_methods` yet — see the
  round-two section of the plan and `THIRD_PARTY_NOTICES.md`.
