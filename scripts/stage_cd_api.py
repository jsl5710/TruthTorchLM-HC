#!/usr/bin/env python
"""Stage A->D for CLOSED models via the JHU gateway (OpenAI-compatible proxy).

Mirrors stage_cd_open.py but Stage A calls the gateway instead of a local model. Two-phase
so it works even if GPU nodes lack outbound network:

  --generate-only  : (run where there IS network, e.g. login) call the gateway for the
                     primary + N samples per item, measure generator latency, write the
                     Stage-A cache. No GPU.
  (default)        : (GPU node) cache-hit Stage A, then local Qwen3-4B judge (Stage B) +
                     consistency methods (Stage C) + AUROC/latency (Stage D).

Only the consistency methods run (they score cached samples -- no live target call), so the
scoring node needs no gateway access. Credential from GATEWAY_KEY env (never hard-coded).

    # phase 1, login node:
    GATEWAY_KEY=... python scripts/stage_cd_api.py --model openai/gpt-4o-mini \
        --generator-key jhu-gpt-4o-mini --datasets trivia_qa --max-items 40 --n-max 5 \
        --generate-only --cache-root ~/JasonLucas/outputs/cache_closed
    # phase 2, GPU (via scripts/stage_cd_api.slurm)
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

GATEWAY_BASE = "https://gateway.engineering.jhu.edu/gateway"
SYSTEM_PROMPT = "You are a helpful assistant. Answer the question concisely."


def _gateway_call(model, question, temperature, max_tokens, timeout=180, retries=4):
    """One gateway chat completion -> (text, latency_ms), with retry.

    The gateway intermittently returns a non-JSON body or a transient 5xx/list error; retry
    with backoff. After exhausting retries, return ("", latency) so one bad call doesn't
    kill the whole generation -- an empty answer is labeled not-attempted downstream.
    """
    import requests
    key = os.environ["GATEWAY_KEY"]
    t0 = time.perf_counter()
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(
                GATEWAY_BASE + "/compat/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": question}],
                      "max_completion_tokens": max_tokens, "temperature": temperature},
                timeout=timeout,
            )
            j = r.json()  # may raise JSONDecodeError on an empty/HTML body
            if isinstance(j, list):
                raise RuntimeError(f"gateway error list: {str(j)[:150]}")
            txt = ((j.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
            return txt, (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:100]}"
            time.sleep(1.5 * (attempt + 1))
    print(f"[gateway] giving up after {retries} tries ({last}) -> empty answer")
    return "", (time.perf_counter() - t0) * 1000.0


def generate_stage_a_gateway(config, model, seed, cache_root, max_tokens,
                             sample_temperature, max_items):
    """Stage A via the gateway: primary (greedy) + n_max samples per item, cached."""
    from tqdm import tqdm
    from hc_benchmark.cache import GenerationCache
    from TruthTorchLM.utils.dataset_utils import get_dataset

    cache = GenerationCache(cache_root, config, seed)
    if cache.exists():
        print(f"[A/gateway] cache hit: {cache.path.name} -- skipping generation.")
        return cache

    dataset = get_dataset(config.dataset, size_of_data=config.size_of_data, seed=seed)
    if max_items:
        dataset = dataset[:max_items]

    items, gen_ms = [], []
    for i, data in enumerate(tqdm(dataset, desc=f"[A/gateway] {config.dataset}/{model}")):
        q = data["question"]
        primary, plat = _gateway_call(model, q, 0.0, max_tokens)
        gen_ms.append(plat)
        samples = [_gateway_call(model, q, sample_temperature, max_tokens)[0]
                   for _ in range(config.n_max)]
        items.append({
            "item_id": data.get("item_id", i), "question": q,
            "context": data.get("context", ""), "ground_truths": data["ground_truths"],
            "primary_answer": primary, "samples": samples,
            "stratum": data.get("stratum"), "outcome_type": data.get("outcome_type", "factual_error"),
            "generator_ms": plat,   # API round-trip = generator time (§5 denominator)
        })
    cache.write(items)
    med = sorted(gen_ms)[len(gen_ms) // 2] if gen_ms else 0
    print(f"[A/gateway] wrote {len(items)} items -> {cache.path.name} | generator p50={med:.0f}ms")
    return cache


def _make_config(dataset, generator_key, n_max, size, seed):
    from hc_benchmark.config import BenchmarkConfig
    return BenchmarkConfig(dataset=dataset, generator=generator_key,
                           generator_backend="jhu_gateway", n_max=n_max,
                           n_sweep=(1,), seeds=(seed,), size_of_data=size,
                           correctness_criteria=("llm_judge",))


def main():
    ap = argparse.ArgumentParser(description="Stage A->D for closed models via JHU gateway.")
    ap.add_argument("--model", required=True, help="gateway model name, e.g. openai/gpt-4o-mini")
    ap.add_argument("--generator-key", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--datasets", nargs="+", default=["trivia_qa"])
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--max-items", type=int, default=40)
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument("--n-sweep", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Reasoning models (gemini-2.5-pro) burn tokens thinking before "
                         "answering; 2048 avoids empty/truncated answers. Non-reasoning "
                         "models (gpt-4o-mini) stop early and ignore the headroom.")
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--generate-only", action="store_true",
                    help="Only call the gateway + cache (run where there is network; no GPU).")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_closed"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_closed"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)

    # -- Phase 1: gateway generation (+cache) for every dataset --------------
    caches = {}
    for ds in args.datasets:
        cfg = _make_config(ds, args.generator_key, args.n_max, args.size, args.seed)
        caches[ds] = generate_stage_a_gateway(
            cfg, args.model, args.seed, args.cache_root, args.max_tokens,
            args.sample_temperature, args.max_items)
    if args.generate_only:
        print("\n[generate-only] caches written; run again on a GPU node to score.")
        return

    # -- Phase 2: judge (Stage B) + consistency methods (C) + eval (D) -------
    import torch
    from stage2_open import load_judge, _mcq_fn, MCQ_DATASETS
    from stage_cd_open import build_bb_methods, _filter_valid
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import summarize, sla_verdict

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    judge_fn, _jm, _jt = load_judge(args.judge, device, args.dtype)
    methods = build_bb_methods(args.n_max, device, include_verbalized=False)  # no live target
    n_sweep = tuple(n for n in args.n_sweep if n <= args.n_max) or (1,)

    cells = []
    for ds in args.datasets:
        cache = caches[ds]
        is_mcq = ds in MCQ_DATASETS
        label_fn = _mcq_fn() if is_mcq else judge_fn
        label_stage_b(cache, criteria=("llm_judge",), _judge_fn=label_fn,
                      judge_model=("mcq_match(letter)" if is_mcq else args.generator_key))
        correctness = correctness_vector(cache, "llm_judge")
        scores = score_stage_c(cache, methods, model="gateway", n_sweep=n_sweep, collect_timing=True)
        fcorr, fscores = _filter_valid(correctness, scores)
        rows = {}
        for method in methods:
            name = type(method).__name__
            ev = evaluate_method(fscores[name], fcorr, truth_method=method, require_calibrated=False)
            for n, metrics in ev.items():
                tm = fscores[name][n]["timing_ms"]
                row = {m: metrics.get(m) for m in
                       ("auroc", "auprc", "auarc", "prr", "ece", "ace", "mce", "brier")}
                if tm:
                    row["p50_ms"] = round(summarize(tm, warmup=0, label=name)["p50_ms"], 3)
                rows[f"{name}_N{n}"] = row
        cells.append({"dataset": ds, "n_items": len(fcorr), "n_positive": sum(fcorr), "rows": rows})
        print(f"[{ds}] {len(fcorr)} items, {sum(fcorr)} correct")

    out = os.path.join(args.results_root, f"stage_cd_{args.generator_key}_seed{args.seed}.json")
    json.dump({"generator_key": args.generator_key, "model": args.model, "seed": args.seed,
               "cells": cells}, open(out, "w"), indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
