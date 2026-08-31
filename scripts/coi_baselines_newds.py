#!/usr/bin/env python
"""Run the multi-sample baselines (semantic entropy / lexical similarity / spectral) on the
datasets we generated ourselves (LongFact, GSM8K, HotpotQA), which were cached with a single
answer per item and therefore had no baseline numbers.

The paper currently argues these baselines are "inapplicable" to long-form because they would
resample a 180-word answer N times. That is a cost argument, not an impossibility argument, so
we pay the cost once and report what they actually achieve. Phase 1 draws N-1 extra samples per
prompt at temperature and rewrites the cache with a full `samples` list; phase 2 scores the
standard consistency/graph methods against the existing correctness labels.

    python scripts/coi_baselines_newds.py --generator llama-3.1-8b \
        --model meta-llama/Llama-3.1-8B-Instruct --datasets longfact gsm8k hotpot_qa
"""
import argparse, glob, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)

GEN_SYS = {
    "longfact": "You are a knowledgeable assistant. Answer the question thoroughly and factually in a short paragraph.",
    "default": "You are a helpful assistant. Answer the question correctly and concisely.",
}
CACHE = {"longfact": "cache_coi_longfact", "gsm8k": "cache_coi_gen", "hotpot_qa": "cache_coi_gen"}
MAXNEW = {"longfact": 320, "gsm8k": 512, "hotpot_qa": 512}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["longfact", "gsm8k", "hotpot_qa"])
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_coi_baselines"))
    args = ap.parse_args()
    os.makedirs(args.results_root, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hc_benchmark.cache import GenerationCache
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from stage2_open import _make_config
    from stage_cd_open import build_bb_methods, _filter_valid
    from pathlib import Path
    from tqdm import tqdm

    W = os.path.expanduser("~/JasonLucas/outputs")

    class _V(GenerationCache):
        def __init__(self, root, cfg, seed, explicit):
            super().__init__(root, cfg, seed); self._e = Path(explicit)
        @property
        def path(self): return self._e

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bl] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               local_files_only=True, device_map="auto").eval()

    def apply(chat):
        try:
            return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    cells = []
    for ds in args.datasets:
        root = f"{W}/{CACHE[ds]}"
        hits = glob.glob(f"{root}/stageA_{ds}_{args.generator}_*_seed{args.seed}.parquet")
        if not hits:
            print(f"[bl] {ds}: no cache -- skip"); continue
        cfg = _make_config(ds, args.generator, n_max=1, size=1.0, seed=args.seed)
        cache = _V(root, cfg, args.seed, hits[0])
        items = cache.read()
        # labels: longfact/gsm8k/hotpot were labelled into a sidecar json by the prep scripts
        lf = (glob.glob(f"{root}/labels_{ds}_{args.generator}_seed{args.seed}.json") or
              glob.glob(f"{root}/labels_{ds}*_{args.generator}_seed{args.seed}.json"))
        if not lf:
            print(f"[bl] {ds}: no labels -- skip"); continue
        labs = json.load(open(lf[0]))
        correctness = [int(labs.get(str(it["item_id"]), -1)) for it in items]

        # ---- phase 1: draw N samples per prompt (the cost the paper claimed was prohibitive)
        sysmsg = GEN_SYS.get(ds, GEN_SYS["default"])
        need = args.n_samples
        print(f"[bl] {ds}: sampling {need} generations x {len(items)} items ...", flush=True)
        for it in tqdm(items, desc=f"[bl/{ds}] sample", leave=False):
            p = apply([{"role": "system", "content": sysmsg},
                       {"role": "user", "content": it["question"]}])
            enc = tok(p, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=MAXNEW.get(ds, 512), do_sample=True,
                                   temperature=args.temperature, top_p=0.95,
                                   num_return_sequences=need,
                                   pad_token_id=(tok.pad_token_id or tok.eos_token_id))
            gen = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                   for o in out]
            it["samples"] = gen
        aug_root = f"{W}/cache_coi_baselines"; os.makedirs(aug_root, exist_ok=True)
        acfg = _make_config(ds, args.generator, n_max=need, size=1.0, seed=args.seed)
        acache = GenerationCache(aug_root, acfg, args.seed); acache.write(items)
        print(f"[bl] {ds}: wrote {acache.path}", flush=True)

        # ---- phase 2: score the consistency/graph baselines
        methods = build_bb_methods(need, dev, include_verbalized=False)
        scores = score_stage_c(acache, methods, model=mdl, tokenizer=tok,
                               n_sweep=(5, need), collect_timing=True)
        rows = {}
        for m in methods:
            nm = type(m).__name__
            if nm not in scores: continue
            for N in (5, need):
                if N not in scores[nm]: continue
                fc, fs = _filter_valid(correctness, {nm: {N: scores[nm][N]}})
                ev = evaluate_method(fs[nm][N], fc, truth_method=m, require_calibrated=False)
                a = (ev.get(N) or ev.get(1) or {}).get("auroc")
                if a is not None: rows[f"{nm}_N{N}"] = {"auroc": round(float(a), 4)}
        npos = sum(1 for c in correctness if c == 1); nit = sum(1 for c in correctness if c in (0, 1))
        cells.append({"dataset": ds, "n_items": nit, "n_positive": npos, "rows": rows})
        print(f"[bl] {ds}: " + ", ".join(f"{k}={v['auroc']}" for k, v in sorted(rows.items())), flush=True)
        dst = f"{args.results_root}/baselines_{args.generator}_seed{args.seed}.json"
        json.dump({"generator": args.generator, "cells": cells}, open(dst, "w"), indent=2)
    print(f"[bl] wrote {args.results_root}/baselines_{args.generator}_seed{args.seed}.json")


if __name__ == "__main__":
    main()
