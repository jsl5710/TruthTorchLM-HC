#!/usr/bin/env python
"""RQ5 Stage 4 — score an uncertainty-aware proxy variant (head OR edl) as a UQ method.

Merges the variant's LoRA adapter in-memory, then reads its uncertainty per response over the
cached qwen3-8b generations (QA + health) and reports AUROC / AUPR / cross-fit ECE / latency —
same footing as the DisAAD baseline (stage_cd_disaad.py), so the RQ5 comparison is apples-to-apples.

  * head : forward(prompt+response, hidden_states) -> mean-pooled hidden -> trained MLP head -> u
  * edl  : forward -> evidential EPISTEMIC uncertainty (EU) over the response span -> u
           (EU is the measure the edl variants supervise; matches DisAAD's read-out)

    python scripts/rq5_score.py --variant-dir ~/JasonLucas/outputs/disaad/proxy_rq5_EccentricityUncertainty_edl --mode edl
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
sys.path.insert(0, os.path.join(_REPO, "third_party", "DisAAD", "scripts"))  # rq5_uncertainty

from calibrate_eval import _CacheView, _labels_for, cross_fit_calibrate, _calib_metrics  # noqa: E402

HEALTH = {"medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"}
TEACHER = "qwen3-8b"
SYS = "You are a helpful assistant. Answer the question concisely."


def main():
    ap = argparse.ArgumentParser(description="Score an RQ5 uncertainty-aware proxy variant.")
    ap.add_argument("--variant-dir", required=True, help="proxy_rq5_<oracle>_<mode> directory.")
    ap.add_argument("--mode", required=True, choices=["head", "edl", "both"])
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--teacher-key", default=TEACHER,
                    help="generator key in the stageA cache filenames (multi-target: qwen3-32b / llama3.3-70b).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_rq5"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--dump-items", default="", help="if set, write per-item {dataset,item_id,correct,u} JSON here (for the stacker).")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sklearn.metrics import roc_auc_score, average_precision_score
    from rq5_uncertainty import UncertaintyHead, _masked_mean

    os.makedirs(args.results_root, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tag = os.path.basename(args.variant_dir.rstrip("/")).replace("proxy_", "")   # rq5_<oracle>_<mode>
    # adapter lives at <variant>/<scoring_model_name>/logs/saved_models/best_model
    hits = glob.glob(os.path.join(args.variant_dir, "*", "logs", "saved_models", "best_model"))
    if not hits:
        sys.exit(f"[rq5-score] no adapter (best_model) under {args.variant_dir}")
    adapter = hits[0]

    print(f"[rq5-score] merging {adapter} ...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dt, local_files_only=True)
    proxy = PeftModel.from_pretrained(base, adapter).merge_and_unload().to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    head = None
    if args.mode in ("head", "both"):
        hp = os.path.join(os.path.dirname(adapter), "uncertainty_head.pt")
        ck = torch.load(hp, map_location=device)
        head = UncertaintyHead(ck["hidden_size"]).to(device).float()
        head.load_state_dict(ck["state_dict"]); head.eval()
        print(f"[rq5-score] loaded head ({ck.get('oracle')})")

    def score_item(prompt, response):
        text = tok.apply_chat_template([{"role": "system", "content": SYS},
                                        {"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False, enable_thinking=False)
        enc = tok(text + (response or ""), return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = proxy(**enc, output_hidden_states=(args.mode in ("head","both")))
            if args.mode in ("head", "both"):
                pooled = _masked_mean(out.hidden_states[-1], enc["attention_mask"])
                u = float(head(pooled.float())[0])
            else:
                plen = tok(text, return_tensors="pt")["input_ids"].shape[1]
                lg = out.logits[0][plen - 1:-1]
                if lg.shape[0] == 0:
                    lg = out.logits[0][-1:]
                top = lg.topk(args.top_k, dim=-1).values
                a = torch.clamp(top, min=0.0)
                u = float((args.top_k / (a.sum(-1) + args.top_k)).mean())
        if device == "cuda":
            torch.cuda.synchronize()
        return u, (time.perf_counter() - t0) * 1000.0

    teacher_key = args.teacher_key
    for seed in args.seeds:
        cells = []
        dump_rows = []
        pqs = (sorted(glob.glob(os.path.join(args.qa_cache_root, f"stageA_*_{teacher_key}_*_seed{seed}.parquet"))) +
               sorted(glob.glob(os.path.join(args.health_cache_root, f"stageA_*_{teacher_key}_*_seed{seed}.parquet"))))
        for pq in pqs:
            ds = os.path.basename(pq).split("stageA_", 1)[1].split(f"_{teacher_key}_", 1)[0]
            labels = _labels_for(pq)
            if labels is None:
                continue
            items = _CacheView(pq).read()
            u, ms, y = [], [], []
            for it in items:
                c = labels.get(str(it["item_id"]), {}).get("correct_llm_judge")
                if c not in (0, 1):
                    continue
                ui, mi = score_item(it["question"], it.get("primary_answer", ""))
                u.append(ui); ms.append(mi); y.append(c)
                if args.dump_items:
                    dump_rows.append({"dataset": ds, "item_id": it["item_id"],
                                      "correct": int(c), "ours_u": float(ui)})
            if len(set(y)) < 2:
                continue
            u = np.array(u); yy = np.array(y); inc = 1 - yy
            probs, _ = cross_fit_calibrate(-u, yy, seed=seed)
            row = {"auroc": round(float(roc_auc_score(inc, u)), 4),
                   "aupr": round(float(average_precision_score(inc, u)), 4),
                   "ece": _calib_metrics(probs, yy)["ece"],
                   "p50_ms": round(float(np.median(ms)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3)}
            cells.append({"dataset": ds, "n_items": len(y), "n_positive": int(yy.sum()),
                          "rows": {f"{tag}_N1": row}})
            print(f"[{ds} s{seed}] AUROC={row['auroc']} AUPR={row['aupr']} ECE={row['ece']} p50={row['p50_ms']}ms")
        out = os.path.join(args.results_root, f"rq5_{tag}_seed{seed}.json")
        json.dump({"variant": tag, "mode": args.mode, "generator_key": teacher_key, "seed": seed,
                   "cells": cells}, open(out, "w"), indent=2, default=str)
        print(f"[rq5-score] wrote {out}")
        if args.dump_items:
            dp = args.dump_items.replace("SEED", str(seed))
            os.makedirs(os.path.dirname(dp) or ".", exist_ok=True)
            json.dump(dump_rows, open(dp, "w"), indent=1, default=str)
            print(f"[rq5-score] wrote {len(dump_rows)} per-item rows -> {dp}")


if __name__ == "__main__":
    main()
