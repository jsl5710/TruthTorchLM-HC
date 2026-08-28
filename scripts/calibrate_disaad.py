#!/usr/bin/env python
"""RQ3 calibration for the DisAAD proxy -- the same cross-fitted normalizer the direct
methods get (scripts/calibrate_eval.py), so proxy vs. direct is an apples-to-apples
quality comparison (ECE/MCE/Brier + AUROC) and not raw-vs-calibrated.

DisAAD is a per-target proxy (teacher qwen3-8b), so it scores the qwen3-8b caches across
both ID groups -- QA (cache_full) and Health (cache_full_health). One proxy forward pass
per item; no NLI, no regeneration. Reuses calibrate_eval's cross_fit_calibrate/_calib_metrics
/_CacheView/_labels_for verbatim so the calibration protocol is identical to the direct arm.

    python scripts/calibrate_disaad.py --seed 0 \
        --proxy-path ~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged \
        --results-root ~/JasonLucas/outputs/results_disaad
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


def main():
    ap = argparse.ArgumentParser(description="RQ3 cross-fit calibration for the DisAAD proxy.")
    ap.add_argument("--proxy-path",
                    default=os.path.expanduser("~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged"))
    ap.add_argument("--generator", default="qwen3-8b", help="proxy's teacher = target to score")
    ap.add_argument("--modes", nargs="+", default=["au", "eu"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_disaad"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    from hc_benchmark.stage_c_score import score_stage_c
    from TruthTorchLM.truth_methods import DisAAD

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_root, exist_ok=True)

    base = DisAAD.from_pretrained(args.proxy_path, mode=args.modes[0], top_k=args.top_k, device=device)
    methods = {m: DisAAD(proxy_model=base.proxy_model, proxy_tokenizer=base.proxy_tokenizer,
                         mode=m, top_k=args.top_k, device=device) for m in args.modes}
    print(f"[calib-disaad] proxy {args.proxy_path} loaded; modes={args.modes}")

    parquets = []
    for root in (args.qa_cache_root, args.health_cache_root):
        parquets += sorted(glob.glob(os.path.join(root, f"stageA_*_{args.generator}_*_seed{args.seed}.parquet")))

    cells = []
    for pq in parquets:
        ds = os.path.basename(pq).split("_")[1]
        labels = _labels_for(pq)
        if labels is None:
            print(f"[{ds}] no labels sidecar -- skip"); continue
        cache = _CacheView(pq)
        items = cache.read()
        corr = [labels.get(str(it["item_id"]), {}).get("correct_llm_judge") for it in items]
        keep = [i for i, c in enumerate(corr) if c in (0, 1)]
        y = [corr[i] for i in keep]
        if len(set(y)) < 2:
            print(f"[{ds}] single-class labels -- skip"); continue
        rows = {}
        for mode, method in methods.items():
            scores = score_stage_c(cache, [method], model="proxy", n_sweep=(1,), collect_timing=False)
            tv = [scores["DisAAD"][1]["truth_values"][i] for i in keep]
            probs, honest = cross_fit_calibrate(tv, y, seed=args.seed)
            cm = _calib_metrics(probs, y)
            try:
                au = round(float(roc_auc_score(y, tv)), 4)
            except Exception:
                au = None
            rows[f"DisAAD-{mode}_N1"] = {**cm, "auroc": au, "cross_fitted": honest}
            print(f"[{ds}] DisAAD-{mode}: ECE={cm['ece']} MCE={cm['mce']} Brier={cm['brier']} "
                  f"AUROC={au} {'(oof)' if honest else '(in-sample!)'}")
        cells.append({"dataset": ds, "n_items": len(keep), "n_positive": sum(y), "rows": rows})

    out = os.path.join(args.results_root, f"calibration_disaad-{args.generator}_seed{args.seed}.json")
    json.dump({"generator": args.generator, "method": "DisAAD", "proxy_path": args.proxy_path,
               "seed": args.seed, "cells": cells}, open(out, "w"), indent=2, default=str)
    print(f"\n[calib-disaad] wrote {out}")


if __name__ == "__main__":
    main()
