#!/usr/bin/env python
"""Fill the missing calibration for the VERBALIZED methods (VerbalizedConfidence, PTrue).

The main calibration pass (calibrate_eval.py) runs with include_verbalized=False, because these
methods elicit a confidence answer from the LIVE target (a string generator key can't do that).
This loads the actual target model and scores the two verbalized methods over the cached items,
then cross-fit calibrates -> ECE, using the identical protocol as the direct/proxy arms. Same
cross_fit_calibrate/_calib_metrics as calibrate_eval, so ECE is comparable.

    python scripts/calibrate_verbalized.py --generator qwen3-8b --target Qwen/Qwen3-8B --seed 0
"""

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from calibrate_eval import _CacheView, _labels_for, cross_fit_calibrate, _calib_metrics  # noqa: E402

HEALTH = {"medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"}


def main():
    ap = argparse.ArgumentParser(description="Calibration for the verbalized methods (live target).")
    ap.add_argument("--generator", default="qwen3-8b")
    ap.add_argument("--target", default="Qwen/Qwen3-8B", help="HF id / path of the LIVE target.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="verbalized is single/few-pass; N-invariant.")
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--qa-results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_full"))
    ap.add_argument("--health-results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_full_health"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import roc_auc_score
    from hc_benchmark.stage_c_score import score_stage_c
    from TruthTorchLM.truth_methods import VerbalizedConfidence, PTrue

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[calib-verb] loading target {args.target} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dt,
                                                 local_files_only=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    methods = [VerbalizedConfidence(), PTrue()]

    for cache_root, results_root, datasets_are_health in (
            (args.qa_cache_root, args.qa_results_root, False),
            (args.health_cache_root, args.health_results_root, True)):
        pqs = sorted(glob.glob(os.path.join(cache_root, f"stageA_*_{args.generator}_*_seed{args.seed}.parquet")))
        cells = []
        for pq in pqs:
            ds = os.path.basename(pq).split("stageA_", 1)[1].split(f"_{args.generator}_", 1)[0]
            labels = _labels_for(pq)
            if labels is None:
                continue
            cache = _CacheView(pq)
            corr = [labels.get(str(it["item_id"]), {}).get("correct_llm_judge") for it in cache.read()]
            keep = [i for i, c in enumerate(corr) if c in (0, 1)]
            y = [corr[i] for i in keep]
            if len(set(y)) < 2:
                continue
            scores = score_stage_c(cache, methods, model=model, tokenizer=tok,
                                   n_sweep=(args.n,), collect_timing=False)
            rows = {}
            for m in methods:
                name = type(m).__name__
                if name not in scores or args.n not in scores[name]:
                    continue
                tv = [scores[name][args.n]["truth_values"][i] for i in keep]
                probs, honest = cross_fit_calibrate(tv, y, seed=args.seed)
                cm = _calib_metrics(probs, y)
                try:
                    au = round(float(roc_auc_score(y, tv)), 4)
                except Exception:
                    au = None
                rows[f"{name}_N{args.n}"] = {**cm, "auroc": au, "cross_fitted": honest}
            cells.append({"dataset": ds, "n_items": len(keep), "n_positive": sum(y), "rows": rows})
            print(f"[{ds} s{args.seed}] " + " ".join(f"{k}:ECE={v['ece']}" for k, v in rows.items()))
        out = os.path.join(results_root, f"calibration_verbalized_{args.generator}_seed{args.seed}.json")
        json.dump({"generator": args.generator, "seed": args.seed, "cells": cells},
                  open(out, "w"), indent=2, default=str)
        print(f"[calib-verb] wrote {out}")


if __name__ == "__main__":
    main()
