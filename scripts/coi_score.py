#!/usr/bin/env python
"""Score CoI-Verbalized (all 5 ablation rungs) + the verbalized-confidence baseline over a target's
cached generations, reusing the benchmark harness. Judge for the chains = the target model itself
(configurable). Reports AUROC/AUPRC/AUARC/PRR + ECE/MCE/Brier + per-instance latency (relative to
generator time) per rung, so Section-6's accuracy-vs-latency ladder falls out directly.

    python scripts/coi_score.py --generator llama-3.1-8b --model meta-llama/Llama-3.1-8B-Instruct \
        --datasets trivia_qa medlfqa --chain-counts 1 2 3 4 5 --seeds 1 \
        --qa-cache-root ~/JasonLucas/outputs/cache_full \
        --health-cache-root ~/JasonLucas/outputs/cache_full_health
"""
import argparse, glob, json, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)

HEALTH = ["medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, help="generator key in cache filenames (e.g. llama-3.1-8b)")
    ap.add_argument("--model", required=True, help="HF repo of the target/judge model to load")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--chain-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--judge-label", default="qwen3-4b-instruct-2507")
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--mt-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_mt"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_coi"))
    ap.add_argument("--max-items", type=int, default=0, help="0=all; small for smoke.")
    ap.add_argument("--max-new-tokens", type=int, default=6144)  # long-form multi-claim JSON needs room
    ap.add_argument("--labels-file", default="", help="precomputed {item_id: 0/1} correctness (LongFact has no gold); skips the judge.")
    ap.add_argument("--device", default=None); ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit NF4 target (for 32B/70B on one GPU)")
    args = ap.parse_args()
    os.makedirs(args.results_root, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from stage2_open import load_judge, _make_config, _mcq_fn, MCQ_DATASETS
    from stage_cd_open import _filter_valid
    from hc_benchmark.cache import GenerationCache
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import summarize
    from TruthTorchLM.truth_methods import CoIVerbalized
    from pathlib import Path as _Path

    class _GlobCache(GenerationCache):
        def __init__(self, root, config, seed, explicit):
            super().__init__(root, config, seed); self._explicit = _Path(explicit)
        @property
        def path(self): return self._explicit

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[coi] loading target/judge model {args.model} (4bit={args.load_4bit}) ...", flush=True)
    load_kw = dict(torch_dtype=dt, local_files_only=True, device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tgt = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).eval()
    tgt_tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tgt_tok.pad_token is None: tgt_tok.pad_token = tgt_tok.eos_token
    judge_fn = None
    if not args.labels_file:
        judge_fn, _jm, _jt = load_judge(args.judge, device, args.dtype)   # correctness judge

    def cache_root_for(ds):
        if ds in HEALTH: return args.health_cache_root
        if os.path.exists(os.path.join(args.mt_cache_root, "")) and \
           glob.glob(os.path.join(args.mt_cache_root, f"stageA_{ds}_{args.generator}_*.parquet")):
            return args.mt_cache_root
        return args.qa_cache_root

    for seed in args.seeds:
        cells = []
        for ds in args.datasets:
            root = cache_root_for(ds)
            cfg = _make_config(ds, args.generator, n_max=1, size=1.0, seed=seed)
            cache = GenerationCache(root, cfg, seed)
            if not cache.exists():
                hits = glob.glob(os.path.join(root, f"stageA_{ds}_{args.generator}_*_seed{seed}.parquet"))
                if hits: cache = _GlobCache(root, cfg, seed, hits[0])
            if not cache.exists():
                print(f"[{ds} s{seed}] no cache in {root} -- skip"); cells.append({"dataset": ds, "error": "no_cache"}); continue

            if args.labels_file:
                lab = json.load(open(args.labels_file))
                correctness = [int(lab.get(str(it["item_id"]), -1)) for it in cache.read()]
            else:
                is_mcq = ds in MCQ_DATASETS
                label_stage_b(cache, criteria=("llm_judge",),
                              _judge_fn=(_mcq_fn() if is_mcq else judge_fn),
                              judge_model=("mcq_match(letter)" if is_mcq else args.judge_label))
                correctness = correctness_vector(cache, "llm_judge")

            rows = {}
            for n in args.chain_counts:
                raw_log = os.path.join(args.results_root, f"rawlog_{args.generator}_{ds}_n{n}_seed{seed}.jsonl")
                open(raw_log, "w").close()  # fresh per run
                method = CoIVerbalized(chain_count=n, max_new_tokens=args.max_new_tokens, log_path=raw_log)
                scores = score_stage_c(cache, [method], model=tgt, tokenizer=tgt_tok,
                                       n_sweep=(1,), collect_timing=True)
                fcorr, fscores = _filter_valid(correctness, scores)
                ev = evaluate_method(fscores["CoIVerbalized"], fcorr, truth_method=method,
                                     require_calibrated=False)
                metrics = ev.get(1, {})
                row = {k: metrics.get(k) for k in ("auroc", "auprc", "auarc", "prr", "ece", "ace", "mce", "brier")}
                tm = fscores["CoIVerbalized"][1].get("timing_ms")
                if tm: row["p50_ms"] = round(summarize(tm, warmup=0, label=f"CoI-n{n}")["p50_ms"], 2)
                rows[f"CoIVerbalized_n{n}"] = row
                print(f"[{ds} s{seed} n={n}] AUROC={row.get('auroc')} ECE={row.get('ece')} p50={row.get('p50_ms')}ms", flush=True)
            n_pos = sum(1 for c in correctness if c == 1); n_it = sum(1 for c in correctness if c in (0, 1))
            cells.append({"dataset": ds, "n_items": n_it, "n_positive": n_pos, "rows": rows})

        out = os.path.join(args.results_root, f"coi_{args.generator}_seed{seed}.json")
        json.dump({"generator": args.generator, "model": args.model, "judge": args.judge_label,
                   "chain_counts": args.chain_counts, "seed": seed, "cells": cells},
                  open(out, "w"), indent=2, default=str)
        print(f"[coi] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
