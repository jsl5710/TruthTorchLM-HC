#!/usr/bin/env python
"""Spot-check Stage-B judge labels against the real answers they were computed on.

After a labeling run, this joins each Stage-A cache (question / primary_answer /
ground_truths) with its .labels.json (correct_llm_judge) so you can eyeball whether the
judge's correct/incorrect calls are sane on ACTUAL target output -- the check the
construct validation can't give you (it used synthetic answers).

Login-node safe: reads parquet + json only, loads no model.

    python scripts/spotcheck_labels.py --generator qwen3-1.7b \
        --datasets trivia_qa natural_qa pop_qa simple_qa web_questions wikipedia_factual \
        --n 6
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path


def _labels_for(parquet_path):
    p = Path(parquet_path).with_suffix(".labels.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt(x, n):
    s = str(x).replace("\n", " ")
    return s[:n]


VERDICT = {1: "CORRECT", 0: "INCORRECT", -1: "not_attempted/unparsed"}


def main():
    ap = argparse.ArgumentParser(description="Spot-check judge labels vs real answers.")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache"))
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=6, help="sampled items to print per dataset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", choices=["sample", "errors", "all"], default="sample",
                    help="sample: random items; errors: only judge==0/-1; all: every item.")
    args = ap.parse_args()

    import pandas as pd

    rng = random.Random(args.seed)
    for ds in args.datasets:
        pattern = os.path.join(args.cache_root, f"stageA_{ds}_{args.generator}_*.parquet")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"\n=== {ds}: no cache found ({pattern}) ===")
            continue
        parquet = matches[-1]
        labels = _labels_for(parquet)
        if labels is None:
            print(f"\n=== {ds}: cache present but no .labels.json (not labeled yet) ===")
            continue

        df = pd.read_parquet(parquet)
        lab = labels["labels"]
        rows = []
        dist = {1: 0, 0: 0, -1: 0}
        for _, r in df.iterrows():
            entry = lab.get(str(r["item_id"]), {})
            y = entry.get("correct_llm_judge")
            if y is None:
                continue
            dist[y] = dist.get(y, 0) + 1
            rows.append((r, y, entry.get("llm_judge_raw", "")))

        n = len(rows)
        acc = (dist[1] / n) if n else float("nan")
        print(f"\n=== {ds} ({n} labeled | judge_model={labels.get('judge_model')}) ===")
        print(f"    CORRECT={dist[1]}  INCORRECT={dist[0]}  not_attempted/unparsed={dist[-1]}  "
              f"| judged-correct rate={acc:.3f}")

        if args.show == "errors":
            shown = [x for x in rows if x[1] != 1]
        elif args.show == "all":
            shown = rows
        else:
            shown = rows[:]
            rng.shuffle(shown)
            shown = shown[:args.n]

        for r, y, raw in shown[: (None if args.show == "all" else args.n)]:
            # ground_truths is JSON-encoded in the parquet (cache.write does json.dumps);
            # cache.read() decodes it for the judge, so decode it here too for display.
            gts = r["ground_truths"]
            if isinstance(gts, str):
                try:
                    gts = json.loads(gts)
                except (json.JSONDecodeError, ValueError):
                    gts = [gts]
            gts = list(gts)
            print(f"  [{VERDICT[y]:22s}] Q: {_fmt(r['question'], 75)}")
            print(f"     answer: {_fmt(r['primary_answer'], 90)}")
            print(f"     gold:   {_fmt(gts[0] if gts else '', 75)}"
                  + (f"  (+{len(gts)-1} more)" if len(gts) > 1 else ""))
            if raw:
                print(f"     judge_raw (unparsed): {_fmt(raw, 60)}")


if __name__ == "__main__":
    main()
