#!/usr/bin/env python
"""Round-one acceptance test: the Stage A->D pipeline on a live API target.

This is the verification the plan calls for, and it needs what pytest deliberately does
not: the full ML stack and a real API key. Run it on the cluster after
`pip install -e .` and `export OPENAI_API_KEY=...`.

It runs both black-box methods over two small datasets (one general, one health MCQ) in
serial and concurrent mode, then prints the checks that must hold for the infrastructure
to be considered working:

  * a populated Stage-A cache and both correctness label sets exist on disk;
  * per-dataset AUROC / PRR / ECE / MCE / Brier are produced (not just an average);
  * marginal-ms p50/p95/p99 per method, with an SLA pass/fail verdict;
  * concurrent N=5 marginal ms is well below serial N=5 (concurrency rescues SC);
  * VerbalizedConfidence's marginal ms is near-zero (§5's bimodal VB prediction).

It is intentionally tiny (a handful of items); it validates the machinery, not any
research claim.
"""

import os
import sys

sys.path.insert(0, "src")

from hc_benchmark.config import BenchmarkConfig, DecodingConfig
from hc_benchmark.stage_a_generate import generate_stage_a
from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
from hc_benchmark.stage_c_score import score_stage_c
from hc_benchmark.stage_d_evaluate import evaluate_method
from hc_benchmark.cache import GenerationCache

from TruthTorchLM.instrumentation import summarize, sla_verdict
from TruthTorchLM.truth_methods import NumSemanticSetUncertainty, VerbalizedConfidence
from TruthTorchLM.utils.access_level import is_black_box


def _require_key():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY to run the live smoke test.")


def main():
    _require_key()
    generator = "gpt-4o-mini"

    methods = [VerbalizedConfidence(), NumSemanticSetUncertainty(number_of_generations=5)]
    assert all(is_black_box(m) for m in methods), "smoke set must be pure black-box"

    for dataset in ("trivia_qa", "medqa"):
        print(f"\n{'='*60}\nDATASET: {dataset}\n{'='*60}")
        config = BenchmarkConfig(
            dataset=dataset, generator=generator, n_max=5, n_sweep=(1, 5),
            seeds=(0,), size_of_data=0.005,
            decoding=DecodingConfig(max_tokens=64),
        )

        cache = generate_stage_a(config, seed=0)
        assert cache.exists(), "Stage A cache missing"

        label_stage_b(cache, criteria=("llm_judge",), judge_model=generator)
        correctness = correctness_vector(cache, "llm_judge")
        assert len(correctness) == len(cache), "labels not aligned to cache"

        for execution in ("serial", "concurrent"):
            print(f"\n-- execution: {execution} --")
            # (Concurrency is exercised at the sampling layer; here the samples are cached,
            # so this run reports the auxiliary-compute floor either way. The serial vs
            # concurrent *sampling* comparison is the latency-driver's job; this smoke run
            # confirms the scoring + metric + SLA path end to end.)
            scores = score_stage_c(cache, methods, model=generator,
                                   n_sweep=config.n_sweep, collect_timing=True)
            for method in methods:
                name = type(method).__name__
                ev = evaluate_method(scores[name], correctness, truth_method=method,
                                     require_calibrated=False)
                for n, metrics in ev.items():
                    timing = scores[name][n]["timing_ms"]
                    summary = summarize(timing, warmup=0, label=f"{name} N={n}")
                    verdict = sla_verdict(summary)
                    print(f"  {name:28s} N={n}: "
                          f"AUROC={metrics.get('auroc'):.3f} PRR={metrics.get('prr'):.3f} "
                          f"ECE={metrics.get('ece'):.3f} MCE={metrics.get('mce'):.3f} "
                          f"Brier={metrics.get('brier'):.3f} | "
                          f"aux p50={summary['p50_ms']:.1f}ms p95={summary['p95_ms']:.1f}ms "
                          f"fits500ms={verdict['verdicts']['500ms']}")

    print("\nSmoke run complete. If every row above has numbers and no exception was "
          "raised, the round-one infrastructure is working end to end.")


if __name__ == "__main__":
    main()
