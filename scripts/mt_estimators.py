#!/usr/bin/env python
"""Multi-target RQ2 (generalized): hold ONE trained proxy fixed and apply ALL NINE read-outs to its
logits -- EDL-AU/EU, MSP, Entropy, Energy, LogTokU, Perplexity, MaxNLL, LogitMargin -- over the cached
eval data. Reports AUROC per read-out (in `rows`, scalar, for backward-compat) PLUS a full per-read-out
metric suite (in `metrics`: auroc, auprc, and cross-fit-calibrated ece/mce/brier), so the read-out
comparison can be shown on ranking AND calibration, not just AUROC.

    python scripts/mt_estimators.py --variant-dir <proxy dir> --base <student repo> \
        --teacher-key qwen3-32b --seeds 0

Writes results_mt_estimators/est_<tag>_seed<seed>.json  ({cells:[{dataset, rows{ro:auroc},
metrics{ro:{auroc,auprc,ece,mce,brier}}}]}).
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

ESTS = ["au", "eu", "msp", "entropy", "energy", "logtoku", "ppl", "maxnll", "margin"]
NAMES = {"au": "EDL-AU", "eu": "EDL-EU", "msp": "MSP", "entropy": "Entropy",
         "energy": "Energy", "logtoku": "LogTokU",
         "ppl": "Perplexity", "maxnll": "MaxNLL", "margin": "LogitMargin"}


def main():
    ap = argparse.ArgumentParser(description="All-estimators-on-one-proxy (generalized RQ2).")
    ap.add_argument("--variant-dir", default="", help="a trained proxy dir (holds the adapter). "
                    "Omit with --no-adapter to read a plain model's logits (grey-box target arm).")
    ap.add_argument("--base", required=True, help="model HF repo id: the proxy base (black-box) "
                    "or the TARGET repo itself with --no-adapter (grey-box).")
    ap.add_argument("--no-adapter", action="store_true",
                    help="score --base directly with no LoRA merge -- use to read the TARGET's own "
                         "logits (GREY-BOX ceiling): same read-outs, but on target logprobs, not the proxy.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="load --base 4-bit (nf4, device_map=auto) -- for a 32B/70B target that "
                         "won't fit bf16 on one GPU.")
    ap.add_argument("--tag", default="", help="override the output tag (e.g. 'greybox_llama3.3-70b').")
    ap.add_argument("--teacher-key", required=True, help="generator key in the stageA cache filenames.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cache-roots", nargs="+",
                    default=[os.path.expanduser("~/JasonLucas/outputs/cache_mt")])
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_mt_estimators"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sklearn.metrics import roc_auc_score, average_precision_score
    from calibrate_eval import _CacheView, _labels_for, cross_fit_calibrate, _calib_metrics
    from rq1_preservation import evidential_all

    os.makedirs(args.results_root, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.no_adapter:
        # GREY-BOX arm: read the TARGET's own logits (no proxy). 4-bit for 32B/70B.
        tag = args.tag or ("greybox_" + os.path.basename(args.base.rstrip("/")))
        print(f"[mt-est] {tag}: loading TARGET {args.base} directly (grey-box) 4bit={args.load_in_4bit}", flush=True)
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            proxy = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                         device_map="auto", local_files_only=True).eval()
        else:
            proxy = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dt,
                                                         local_files_only=True).to(device).eval()
    else:
        tag = args.tag or os.path.basename(args.variant_dir.rstrip("/")).replace("proxy_mt_", "").replace("proxy_", "")
        hits = glob.glob(os.path.join(args.variant_dir, "*", "logs", "saved_models", "best_model"))
        if not hits:
            sys.exit(f"[mt-est] no adapter under {args.variant_dir}")
        adapter = hits[0]
        print(f"[mt-est] {tag}: merging {adapter}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dt, local_files_only=True)
        proxy = PeftModel.from_pretrained(base, adapter).merge_and_unload().to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)

    tk = args.teacher_key
    for seed in args.seeds:
        cells = []
        pqs = []
        for root in args.cache_roots:
            pqs += sorted(glob.glob(os.path.join(root, f"stageA_*_{tk}_*_seed{seed}.parquet")))
        for pq in pqs:
            ds = os.path.basename(pq).split("stageA_", 1)[1].split(f"_{tk}_", 1)[0]
            labels = _labels_for(pq)
            if labels is None:
                continue
            items = _CacheView(pq).read()
            U = {e: [] for e in ESTS}
            y = []
            for it in items:
                c = labels.get(str(it["item_id"]), {}).get("correct_llm_judge")
                if c not in (0, 1):
                    continue
                est = evidential_all(proxy, tok, it["question"], it.get("primary_answer", ""),
                                     device, top_k=args.top_k)
                for e in ESTS:
                    U[e].append(est[e])
                y.append(c)
            if len(set(y)) < 2:
                continue
            yy = np.array(y)
            inc = 1 - yy                                       # 1 = incorrect answer (what we flag)
            row = {}          # scalar AUROC per read-out (backward-compatible with analysis scripts)
            metrics = {}      # full per-read-out metrics: auroc, auprc, + calibrated ece/mce/brier
            for e in ESTS:
                u = np.array(U[e], float)
                ok = np.isfinite(u)
                if ok.sum() >= 2 and len(set(inc[ok])) == 2:
                    au = float(roc_auc_score(inc[ok], u[ok]))
                    row[NAMES[e]] = round(au, 4)
                    m = {"auroc": round(au, 4),
                         "auprc": round(float(average_precision_score(inc[ok], u[ok])), 4)}
                    # calibrated ECE/Brier: cross-fit isotonic on confidence(-u)->P(correct)
                    try:
                        probs, _cal = cross_fit_calibrate(-u[ok], yy[ok], seed=seed)
                        m.update(_calib_metrics(probs, yy[ok]))
                        m["calibrated"] = bool(_cal)
                    except Exception:
                        pass
                    metrics[NAMES[e]] = m
            cells.append({"dataset": ds, "n_items": len(y), "n_positive": int(sum(y)),
                          "rows": row, "metrics": metrics})
            print(f"[{ds} s{seed}] " + " ".join(f"{NAMES[e]}={row.get(NAMES[e],'-')}" for e in ESTS), flush=True)
        out = os.path.join(args.results_root, f"est_{tag}_seed{seed}.json")
        import json
        json.dump({"proxy": tag, "base": args.base, "teacher_key": tk, "seed": seed, "cells": cells},
                  open(out, "w"), indent=2)
        print(f"[mt-est] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
