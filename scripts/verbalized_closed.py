#!/usr/bin/env python
"""Verbalized-confidence on CLOSED targets (gateway), the one pure-BB direct method the closed arm
skipped (stage_cd_api uses include_verbalized=False -- "no live target"). For each cached
(question, target-answer) it elicits a 0-100 confidence via ONE gateway call using the SAME
VC prompt as the open VerbalizedConfidence method, so the numbers are comparable. Reads the existing
cache_closed generations + labels; writes results_closed/verbalized_<key>_seed0.json with a
VerbalizedConfidence_N1 row per dataset (auroc/auprc/ece/mce/brier), mergeable into the leaderboard.

Login node (gateway). ~150 calls/dataset. Cost is one short completion per item.
    source ~/JasonLucas/.gateway_key
    python scripts/verbalized_closed.py --model openai/gpt-4o --key jhu-gpt-4o
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

QA = ["trivia_qa", "natural_qa", "pop_qa", "truthful_qa"]
HEALTH = ["medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"]
DEFAULT_DS = ["trivia_qa", "bioasq", "medqa", "medlfqa", "gsm8k", "truthful_qa", "wikipedia_factual"]
GATEWAY_BASE = "https://gateway.engineering.jhu.edu/gateway"


def _extract_conf(text):
    s = "".join(c for c in (text or "").strip() if c.isdigit() or c == ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100.0 if v > 1.0 else v          # accept 0-100 or 0-1


def _vc_call(model, sys_prompt, user_prompt, max_tokens=32, timeout=120, retries=4):
    import requests
    key = os.environ["GATEWAY_KEY"]
    for attempt in range(retries):
        try:
            r = requests.post(
                GATEWAY_BASE + "/compat/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": user_prompt}],
                      "max_completion_tokens": max_tokens, "temperature": 0.0},
                timeout=timeout)
            j = r.json()
            if isinstance(j, list):
                raise RuntimeError(str(j)[:120])
            return ((j.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
        except Exception as exc:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
    print(f"[vc] give up ({last})")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gateway model, e.g. openai/gpt-4o")
    ap.add_argument("--key", required=True, help="generator key, e.g. jhu-gpt-4o (names cache + output)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DS)
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_closed"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_closed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=0, help="0=all; small for a smoke test.")
    args = ap.parse_args()
    if "GATEWAY_KEY" not in os.environ:
        sys.exit("GATEWAY_KEY not set (source ~/JasonLucas/.gateway_key)")

    from sklearn.metrics import roc_auc_score, average_precision_score
    from calibrate_eval import _CacheView, _labels_for, cross_fit_calibrate, _calib_metrics
    from TruthTorchLM.templates import VC_SYSTEM_PROMPT, VC_USER_PROMPT
    from tqdm import tqdm

    os.makedirs(args.results_root, exist_ok=True)
    cells = []
    for ds in args.datasets:
        pqs = glob.glob(os.path.join(args.cache_root, f"stageA_{ds}_{args.key}_*_seed{args.seed}.parquet"))
        if not pqs:
            print(f"[{ds}] no cache -- skip"); continue
        pq = pqs[0]
        labels = _labels_for(pq)
        if labels is None:
            print(f"[{ds}] no labels -- skip"); continue
        items = _CacheView(pq).read()
        if args.max_items:
            items = items[:args.max_items]
        u, y = [], []
        for it in tqdm(items, desc=f"[vc/{args.key}] {ds}", leave=False):
            c = labels.get(str(it["item_id"]), {}).get("correct_llm_judge")
            if c not in (0, 1):
                continue
            up = VC_USER_PROMPT.format(question=it["question"], generated_text=it.get("primary_answer", ""))
            conf = _extract_conf(_vc_call(args.model, VC_SYSTEM_PROMPT, up))
            if conf is None:
                continue
            u.append(1.0 - conf); y.append(c)          # uncertainty = 1 - confidence
        if len(set(y)) < 2:
            print(f"[{ds}] <2 classes -- skip"); continue
        u = np.array(u, float); yy = np.array(y); inc = 1 - yy
        row = {"auroc": round(float(roc_auc_score(inc, u)), 4),
               "auprc": round(float(average_precision_score(inc, u)), 4)}
        try:
            probs, _ = cross_fit_calibrate(1.0 - u, yy, seed=args.seed)   # confidence -> P(correct)
            row.update(_calib_metrics(probs, yy))
        except Exception:
            pass
        cells.append({"dataset": ds, "n_items": len(y), "n_positive": int(yy.sum()),
                      "rows": {"VerbalizedConfidence_N1": row}})
        print(f"[{ds}] n={len(y)} AUROC={row['auroc']} AUPRC={row['auprc']} ECE={row.get('ece')}")

    out = os.path.join(args.results_root, f"verbalized_{args.key}_seed{args.seed}.json")
    json.dump({"generator_key": args.key, "model": args.model, "method": "VerbalizedConfidence",
               "seed": args.seed, "cells": cells}, open(out, "w"), indent=2, default=str)
    print(f"\n[vc] wrote {out}")


if __name__ == "__main__":
    main()
