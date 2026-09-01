#!/usr/bin/env python
"""CoI-Verbalized on CLOSED targets via the JHU gateway (pure black-box: the target is also the
chain-judge). Reuses cached closed generations (the fixed answer y) + their Stage-B labels, and
drives the SAME CoIVerbalized chain logic through gateway calls instead of a local model. Scores
rungs n=1/3/6/7 and reports AUROC per (target, dataset). Login-node (network only, no GPU).

    source ~/JasonLucas/.gateway_key
    python scripts/coi_closed.py --model openai/gpt-4o --key jhu-gpt-4o
    python scripts/coi_closed.py --model anthropic/claude-haiku-4.5 --key jhu-claude-haiku-4.5
"""
import argparse, glob, json, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)

import numpy as np
GATEWAY_BASE = "https://gateway.engineering.jhu.edu/gateway"
DEFAULT_DS = ["trivia_qa", "truthful_qa", "medqa", "bioasq", "medlfqa", "gsm8k"]


def _gw_call(model, sys_prompt, user_prompt, max_tokens, temperature, timeout=180, retries=4):
    import requests
    key = os.environ["GATEWAY_KEY"]
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(GATEWAY_BASE + "/compat/chat/completions",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model,
                                    "messages": [{"role": "system", "content": sys_prompt},
                                                 {"role": "user", "content": user_prompt}],
                                    "max_completion_tokens": max_tokens, "temperature": temperature},
                              timeout=timeout)
            j = r.json()
            if isinstance(j, list):
                raise RuntimeError(str(j)[:120])
            return ((j.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:80]}"; time.sleep(1.5 * (attempt + 1))
    print(f"[coi-closed] give up ({last})", flush=True)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gateway model id, e.g. openai/gpt-4o")
    ap.add_argument("--key", required=True, help="cache key, e.g. jhu-gpt-4o")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DS)
    ap.add_argument("--chain-counts", type=int, nargs="+", default=[1, 3, 6, 7])
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_closed"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_coi_closed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--overwrite-logs", action="store_true",
                    help="ignore existing per-item logs instead of resuming from them")
    args = ap.parse_args()
    if "GATEWAY_KEY" not in os.environ:
        sys.exit("GATEWAY_KEY not set (source ~/JasonLucas/.gateway_key)")
    os.makedirs(args.results_root, exist_ok=True)

    from sklearn.metrics import roc_auc_score
    from calibrate_eval import _CacheView, _labels_for
    from TruthTorchLM.truth_methods.coi_verbalized import CoIVerbalized, SYSTEM, _extract_json
    import re

    def make_gen(model):
        def gen(user, do_sample=False, temperature=None, max_new=None):
            t = (temperature if (do_sample and temperature) else 0.0)
            txt = _gw_call(model, SYSTEM, user, max_tokens=(max_new or 1024), temperature=t)
            return {"generated_text_skip_specials": txt}
        return gen
    gen = make_gen(args.model)

    def score_item(method, q, y):
        n = method.chain_count
        if n == 1:
            raw = gen(method._prompt(q, y), max_new=64)["generated_text_skip_specials"]
            obj = _extract_json(raw) or {}
            m = re.search(r'-?\d+(?:\.\d+)?', str(obj.get("confidence", "")))
            c = float(m.group()) if m else 50.0
            c = c / 100.0 if c > 1.0 else c
            return max(0.0, min(1.0, c))
        if n in (6, 7):
            return method._decision_verify(q, y, gen)["truth_value"]
        raw = gen(method._prompt(q, y), max_new=2048)["generated_text_skip_specials"]
        return method._phi(_extract_json(raw) or {})["aggregate_confidence"]

    cells = []
    for ds in args.datasets:
        pqs = glob.glob(os.path.join(args.cache_root, f"stageA_{ds}_{args.key}_*_seed{args.seed}.parquet"))
        if not pqs:
            print(f"[{ds}] no cache -- skip"); continue
        labels = _labels_for(pqs[0])
        if labels is None:
            print(f"[{ds}] no labels -- skip"); continue
        items = _CacheView(pqs[0]).read()
        if args.max_items:
            items = items[:args.max_items]
        rows = {}
        for n in args.chain_counts:
            method = CoIVerbalized(chain_count=n)
            tv, yy = [], []
            from tqdm import tqdm
            # per-item log, same schema as the open-model rung logs so that
            # scripts/coi_verify_from_raw.py can recompute these AUROCs from raw data too.
            raw_log = os.path.join(args.results_root,
                                   f"rawlog_{args.key}_{ds}_n{n}_seed{args.seed}.jsonl")
            done = {}
            if os.path.exists(raw_log) and not args.overwrite_logs:
                for line in open(raw_log):
                    if line.strip():
                        r = json.loads(line)
                        if r.get("item_id") is not None:
                            done[str(r["item_id"])] = r["tv"]
                if done:
                    print(f"[{args.key}/{ds} n={n}] resuming: {len(done)} items already scored",
                          flush=True)
            logf = open(raw_log, "a")
            for it in tqdm(items, desc=f"[{args.key}/{ds} n{n}]", leave=False):
                c = labels.get(str(it["item_id"]), {}).get("correct_llm_judge")
                if c not in (0, 1):
                    continue
                iid = str(it["item_id"])
                if iid in done:
                    tv.append(done[iid]); yy.append(c); continue
                v = score_item(method, it["question"], it.get("primary_answer", ""))
                if v is None:
                    continue
                logf.write(json.dumps({"item_id": iid, "q": str(it["question"])[:160],
                                       "n": n, "tv": v, "label": int(c)}) + "\n")
                logf.flush()
                tv.append(v); yy.append(c)
            logf.close()
            if len(set(yy)) < 2:
                rows[f"CoIVerbalized_n{n}"] = {"auroc": None, "n": len(yy)}; continue
            au = roc_auc_score(yy, tv)  # higher confidence -> correct
            rows[f"CoIVerbalized_n{n}"] = {"auroc": round(float(au), 4), "n": len(yy),
                                           "n_positive": int(sum(yy))}
            print(f"[{args.key}/{ds} n={n}] AUROC={au:.3f} (n={len(yy)}, pos={int(sum(yy))})", flush=True)
        cells.append({"dataset": ds, "rows": rows})
        json.dump({"target": args.model, "key": args.key, "seed": args.seed, "cells": cells},
                  open(os.path.join(args.results_root, f"coi_{args.key}_seed{args.seed}.json"), "w"),
                  indent=2, default=str)
    print(f"[coi-closed] wrote results for {args.key}", flush=True)


if __name__ == "__main__":
    main()
