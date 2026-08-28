#!/usr/bin/env python
"""Stage C->D for the DisAAD proxy method -- place the trained proxy on the M-axis.

DisAAD (Cui et al., ACL 2026) is a per-target method: a small student proxy is distilled
to mimic ONE target's output distribution, then its evidential uncertainty scores that
target's responses. Our proxy was distilled from teacher **qwen3-8b**, so this runner
scores the *already-cached qwen3-8b generations* (the same Stage-A parquet every other
method reads -- no regeneration) and reports AUROC + latency, directly comparable to the
consistency/verbalized methods computed on qwen3-8b.

The proxy forward pass is timed as AUXILIARY_COMPUTE: one small, target-decoupled pass
(protocol s5 "Pob via proxy"), so its latency is a near-constant floor independent of how
slow the target is -- the whole point of DisAAD.

    # after the proxy is trained (manifest present):
    python scripts/stage_cd_disaad.py \
        --proxy-path ~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b \
        --generator-key qwen3-8b --seeds 0 1 2 \
        --qa-cache-root ~/JasonLucas/outputs/cache_full \
        --health-cache-root ~/JasonLucas/outputs/cache_full_health \
        --results-root ~/JasonLucas/outputs/results_disaad
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

QA_DATASETS = ["trivia_qa", "natural_qa", "pop_qa", "truthful_qa"]
HEALTH_DATASETS = ["medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"]


def main():
    ap = argparse.ArgumentParser(description="Stage C->D for the DisAAD proxy method.")
    ap.add_argument("--proxy-path",
                    default=os.path.expanduser("~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged"),
                    help="the MERGED full model (LoRA adapter merged into base); training saves a "
                         "LoRA adapter under .../qwen3-0.6b/logs/saved_models/best_model, merged to "
                         "'.../merged' for plain AutoModelForCausalLM loading.")
    ap.add_argument("--generator-key", default="qwen3-8b",
                    help="target whose cached generations to score (must match the proxy's teacher).")
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--judge-label", default="qwen3-4b-instruct-2507")
    ap.add_argument("--datasets", nargs="+", default=QA_DATASETS + HEALTH_DATASETS)
    ap.add_argument("--modes", nargs="+", default=["au", "eu"],
                    help="evidential measures to report as separate M-axis entries.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-max", type=int, default=10, help="must match the cached generation's n_max.")
    ap.add_argument("--size", type=float, default=1.0, help="must match the cached generation's size.")
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_disaad"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="load a proxy without the training-ready manifest (use with care).")
    ap.add_argument("--dump-items", default="", help="if set, write per-item {dataset,item_id,correct,au,eu} JSON here (for the stacker).")
    ap.add_argument("--dump-tag", default="proxy", help="key prefix for the dumped per-item file basename.")
    args = ap.parse_args()

    os.makedirs(args.results_root, exist_ok=True)

    import torch
    from stage2_open import load_judge, _make_config, _mcq_fn, MCQ_DATASETS
    from stage_cd_open import _filter_valid
    from hc_benchmark.cache import GenerationCache
    from pathlib import Path as _Path

    class _GlobCache(GenerationCache):
        """GenerationCache pinned to an explicit parquet found by glob -- for caches whose config
        hash differs from how they were written (e.g. the gateway/stage_cd_api closed cache)."""
        def __init__(self, root, config, seed, explicit):
            super().__init__(root, config, seed)
            self._explicit = _Path(explicit)

        @property
        def path(self):
            return self._explicit
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import summarize
    from TruthTorchLM.truth_methods import DisAAD

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Judge once; proxy once (shared across modes -- the forward pass is identical, only the
    # logit->uncertainty aggregation differs, so each mode is a thin wrapper on one model).
    judge_fn, _jm, _jt = load_judge(args.judge, device, args.dtype)
    print(f"[disaad] loading proxy from {args.proxy_path} (top_k={args.top_k}) ...")
    base = DisAAD.from_pretrained(args.proxy_path, mode=args.modes[0], top_k=args.top_k,
                                  device=device, allow_unverified=args.allow_unverified)
    methods = {m: DisAAD(proxy_model=base.proxy_model, proxy_tokenizer=base.proxy_tokenizer,
                         mode=m, top_k=args.top_k, device=device) for m in args.modes}
    print(f"[disaad] proxy ready; modes={args.modes}")

    for seed in args.seeds:
        cells = []
        dump_rows = []
        for ds in args.datasets:
            cache_root = args.health_cache_root if ds in HEALTH_DATASETS else args.qa_cache_root
            cfg = _make_config(ds, args.generator_key, n_max=args.n_max, size=args.size, seed=seed)
            cache = GenerationCache(cache_root, cfg, seed)
            if not cache.exists():
                # content_hash may differ from how the cache was written (e.g. the gateway path in
                # stage_cd_api uses a different config) -> fall back to globbing the parquet by
                # (dataset, generator, seed), like rq5_score does. label_stage_b re-judges anyway.
                import glob as _glob
                hits = _glob.glob(os.path.join(cache_root, f"stageA_{ds}_{args.generator_key}_*_seed{seed}.parquet"))
                if hits:
                    cache = _GlobCache(cache_root, cfg, seed, hits[0])   # pin to the real file
            if not cache.exists():
                print(f"[{ds} s{seed}] no cache at {cache.path} -- skipping")
                cells.append({"dataset": ds, "error": "no_cache"})
                continue

            is_mcq = ds in MCQ_DATASETS
            label_fn = _mcq_fn() if is_mcq else judge_fn
            label_stage_b(cache, criteria=("llm_judge",), _judge_fn=label_fn,
                          judge_model=("mcq_match(letter)" if is_mcq else args.judge_label))
            correctness = correctness_vector(cache, "llm_judge")

            rows = {}
            mode_tv = {}
            fcorr_keep = None
            for mode, method in methods.items():
                # model="proxy" (a str) -> TruthMethod dispatches to forward_api, which builds
                # the prompt from messages(question) and scores primary_answer with the proxy.
                scores = score_stage_c(cache, [method], model="proxy", n_sweep=(1,),
                                       collect_timing=True)
                fcorr, fscores = _filter_valid(correctness, scores)
                fcorr_keep = fcorr
                mode_tv[mode] = fscores["DisAAD"][1].get("truth_values") or []
                ev = evaluate_method(fscores["DisAAD"], fcorr, truth_method=method,
                                     require_calibrated=False)
                metrics = ev.get(1, {})
                row = {k: metrics.get(k) for k in
                       ("auroc", "auprc", "auarc", "prr", "ece", "ace", "mce", "brier")}
                tm = fscores["DisAAD"][1]["timing_ms"]
                if tm:
                    row["p50_ms"] = round(summarize(tm, warmup=0, label=f"DisAAD-{mode}")["p50_ms"], 3)
                rows[f"DisAAD-{mode}_N1"] = row
            if args.dump_items and fcorr_keep is not None:
                items = cache.read()
                keep = [i for i, c in enumerate(correctness) if c in (0, 1)]
                iids = [items[i].get("item_id") for i in keep]
                au = mode_tv.get("au", [])
                eu = mode_tv.get("eu", [])
                for j, (iid, c) in enumerate(zip(iids, fcorr_keep)):
                    dump_rows.append({"dataset": ds, "item_id": iid, "correct": int(c),
                                      f"{args.dump_tag}_au": (au[j] if j < len(au) else None),
                                      f"{args.dump_tag}_eu": (eu[j] if j < len(eu) else None)})
            n_pos = sum(1 for c in correctness if c == 1)
            n_it = sum(1 for c in correctness if c in (0, 1))
            cells.append({"dataset": ds, "n_items": n_it, "n_positive": n_pos, "rows": rows})
            print(f"[{ds} s{seed}] {n_it} items, {n_pos} correct | "
                  + " ".join(f"{m}:AUROC={cells[-1]['rows'].get(f'DisAAD-{m}_N1',{}).get('auroc')}"
                             for m in args.modes))

        out = os.path.join(args.results_root, f"stage_cd_disaad-{args.generator_key}_seed{seed}.json")
        json.dump({"generator_key": args.generator_key, "method": "DisAAD",
                   "proxy_path": args.proxy_path, "modes": args.modes, "seed": seed,
                   "cells": cells}, open(out, "w"), indent=2, default=str)
        print(f"[disaad] wrote {out}")
        if args.dump_items:
            dp = args.dump_items.replace("SEED", str(seed))
            os.makedirs(os.path.dirname(dp) or ".", exist_ok=True)
            json.dump(dump_rows, open(dp, "w"), indent=1, default=str)
            print(f"[disaad] wrote {len(dump_rows)} per-item rows -> {dp}")


if __name__ == "__main__":
    main()
