#!/usr/bin/env python
"""RQ1 — Proxy Uncertainty Preservation: does the distilled proxy preserve the TEACHER's
uncertainty behavior, ID vs OOD?

DisAAD distills the 0.6B proxy to mimic the qwen3-8b teacher's *outputs*; it then assumes the
proxy also inherits the teacher's *uncertainty*. Nothing in the loss enforces that. This script
measures it directly: for each cached qwen3-8b response, compute the SAME four logit-based
uncertainty estimators on BOTH models' logits over (prompt + response) --

    AU, EU (evidential / Dirichlet)  ·  MSP (1 - max softmax)  ·  Entropy (predictive)

-- one forward pass per model, mean-aggregated over the response span (matching DisAAD's
aggregation). Writes per-item records so rq1_report.py can measure teacher<->proxy agreement,
calibration, and ranking, split ID (TriviaQA = the distillation domain) vs OOD (everything else).

    python scripts/rq1_preservation.py --seed 0 \
        --proxy-path ~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from calibrate_eval import _CacheView, _labels_for  # noqa: E402

QA_ROOT = os.path.expanduser("~/JasonLucas/outputs/cache_full")
HEALTH_ROOT = os.path.expanduser("~/JasonLucas/outputs/cache_full_health")
TARGET = "qwen3-8b"


def _energy(v):
    """Energy score = -logsumexp(logits). Confident (peaked) logits -> very negative; flat/uncertain
    -> higher. Oriented higher = more uncertain."""
    from scipy.special import logsumexp
    return float(-logsumexp(v))


def _logtoku(v):
    """LogTokU-style logit uncertainty: treat ReLU(logits) as full-vocabulary Dirichlet evidence and
    read epistemic (evidence-scarcity) uncertainty V/(Σ ReLU(logit) + V). Distinct from the top-k EDL
    estimators (full vocab, not top-k). Best-effort implementation of Ma et al. 2025 — 'where
    applicable', documented as such in the RQ2 report. Higher = more uncertain."""
    ev = np.maximum(0.0, v)
    V = len(v)
    return float(V / (ev.sum() + V))


def evidential_all(model, tok, prompt, response, device, top_k=10):
    """SIX uncertainty estimators from ONE forward pass over prompt+response, mean over the response
    span, all oriented higher = more uncertain: EDL-AU, EDL-EU, MSP, Entropy, Energy, LogTokU.
    (RQ1 uses AU/EU/MSP/Entropy teacher-vs-proxy; RQ2 uses all six on the proxy logits.)"""
    import torch
    from TruthTorchLM.truth_methods.disaad import (
        evidential_aleatoric, evidential_epistemic, max_softmax_probability, softmax_entropy)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    full_ids = tok(prompt + (response or ""), return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(full_ids).logits[0]                 # (seq, vocab)
    start = prompt_ids.shape[1] - 1
    resp_t = logits[start:-1].float()                      # (n_resp, vocab), positions predicting resp tokens
    # the ACTUAL response token at each position (for likelihood-based read-outs) -- these use which
    # token was emitted, not just the distribution shape, so they carry signal the six shape-estimators miss.
    tgt = full_ids[0, start + 1:]                          # aligned to resp_t rows
    n = min(resp_t.shape[0], tgt.shape[0])
    if n > 0:
        logp = torch.log_softmax(resp_t[:n], dim=-1)
        nll = (-logp[torch.arange(n), tgt[:n]]).cpu().numpy()          # -log P(actual tok), higher=uncertain
        srt = torch.sort(resp_t[:n], dim=-1, descending=True).values
        margin = (srt[:, 0] - srt[:, 1]).cpu().numpy()                  # top1-top2 logit gap (confidence)
    else:
        nll, margin = np.array([0.0]), np.array([0.0])
    resp = logits[start:-1].float().cpu().numpy()
    if len(resp) == 0:
        resp = logits[-1:].float().cpu().numpy()
    return {
        "au": float(np.mean([evidential_aleatoric(v, top_k) for v in resp])),
        "eu": float(np.mean([evidential_epistemic(v, top_k) for v in resp])),
        "msp": float(np.mean([1.0 - max_softmax_probability(v) for v in resp])),   # -> uncertainty
        "entropy": float(np.mean([softmax_entropy(v) for v in resp])),
        "energy": float(np.mean([_energy(v) for v in resp])),
        "logtoku": float(np.mean([_logtoku(v) for v in resp])),
        "ppl": float(np.mean(nll)),                        # mean NLL (length-normalized) -> higher=uncertain
        "maxnll": float(np.max(nll)),                      # weakest-link token
        "margin": float(-np.mean(margin)),                 # -(top1-top2): higher=uncertain
        "n_tok": len(resp),
    }


def main():
    ap = argparse.ArgumentParser(description="RQ1 teacher-vs-proxy uncertainty extraction.")
    ap.add_argument("--proxy-path",
                    default=os.path.expanduser("~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged"))
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all; >0 caps items/dataset (smoke).")
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_rq1"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_root, exist_ok=True)
    dt = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[rq1] loading teacher {args.teacher} + proxy {args.proxy_path} on {device} ...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dt,
                                                   local_files_only=True).to(device).eval()
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    proxy = AutoModelForCausalLM.from_pretrained(args.proxy_path, torch_dtype=dt,
                                                 local_files_only=True).to(device).eval()
    proxy_tok = AutoTokenizer.from_pretrained(args.proxy_path, local_files_only=True)

    parquets = (sorted(glob.glob(os.path.join(QA_ROOT, f"stageA_*_{TARGET}_*_seed{args.seed}.parquet"))) +
                sorted(glob.glob(os.path.join(HEALTH_ROOT, f"stageA_*_{TARGET}_*_seed{args.seed}.parquet"))))
    print(f"[rq1] {len(parquets)} caches for seed {args.seed}")

    records = []
    from tqdm import tqdm
    for pq in parquets:
        # filename = stageA_<dataset>_<TARGET>_<hash>_seed<n>.parquet; dataset itself contains
        # underscores (trivia_qa, mmlu_med), so split on the generator, not the first "_".
        ds = os.path.basename(pq).split("stageA_", 1)[1].split(f"_{TARGET}_", 1)[0]
        labels = _labels_for(pq)
        if labels is None:
            print(f"[{ds}] no labels -- skip"); continue
        items = _CacheView(pq).read()
        if args.max_items:
            items = items[:args.max_items]
        for it in tqdm(items, desc=f"[rq1] {ds} s{args.seed}", leave=False):
            corr = labels.get(str(it["item_id"]), {}).get("correct_llm_judge")
            if corr not in (0, 1):
                continue
            q, ans = it["question"], it.get("primary_answer", "")
            t = evidential_all(teacher, teacher_tok, q, ans, device, args.top_k)
            p = evidential_all(proxy, proxy_tok, q, ans, device, args.top_k)
            records.append({"dataset": ds, "item_id": it["item_id"], "correct": corr,
                            "teacher": t, "proxy": p})
        print(f"[{ds}] {sum(1 for r in records if r['dataset']==ds)} items scored")

    out = os.path.join(args.results_root, f"rq1_unc_{TARGET}_seed{args.seed}.json")
    json.dump({"target": TARGET, "teacher": args.teacher, "proxy_path": args.proxy_path,
               "seed": args.seed, "top_k": args.top_k, "records": records},
              open(out, "w"), indent=1, default=str)
    print(f"[rq1] wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
