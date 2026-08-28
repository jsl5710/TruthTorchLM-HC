#!/usr/bin/env python
"""RQ5 <-> RQ1 loop-closure — did uncertainty-aware distillation improve UNCERTAINTY PRESERVATION,
not just error-discrimination?

Re-runs RQ1's teacher-vs-proxy agreement, but on a UA variant's DEPLOYED uncertainty (EDL-EU for
edl variants; the trained head output for head variants) vs. the teacher's EU, split ID (TriviaQA)
vs OOD (rest). Compares against the DisAAD baseline (from results_rq1). If OOD agreement rises, the
explicit supervision closed the RQ1 EU-preservation gap.

    python scripts/rq5_preservation.py --variant-dir .../proxy_rq5_EccentricityUncertainty_edl --mode edl
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
sys.path.insert(0, os.path.join(_REPO, "third_party", "DisAAD", "scripts"))

from calibrate_eval import _CacheView, _labels_for  # noqa: E402
from rq1_preservation import evidential_all           # 6-estimator teacher/proxy uncertainty

QA_ROOT = os.path.expanduser("~/JasonLucas/outputs/cache_full")
HEALTH_ROOT = os.path.expanduser("~/JasonLucas/outputs/cache_full_health")
HEALTH = {"medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"}
TEACHER = "qwen3-8b"
ID = {"trivia_qa"}
SYS = "You are a helpful assistant. Answer the question concisely."


def _spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return None
    return spearmanr(a[m], b[m]).correlation


def _auroc(unc, y):
    from sklearn.metrics import roc_auc_score
    unc, y = np.asarray(unc, float), np.asarray(y, float)
    m = np.isfinite(unc)
    if len(set(y[m].tolist())) < 2:
        return None
    return roc_auc_score(1 - y[m], unc[m])


def baseline_agreement():
    """DisAAD baseline: teacher-EU vs proxy-EU Spearman + proxy AUROC, ID/OOD, from results_rq1."""
    recs = []
    for f in glob.glob(os.path.expanduser("~/JasonLucas/outputs/results_rq1/rq1_unc_*_seed0.json")):
        recs += json.load(open(f)).get("records", [])
    out = {}
    for split, keep in (("ID", lambda ds: ds in ID), ("OOD", lambda ds: ds not in ID)):
        r = [x for x in recs if keep(x["dataset"])]
        te = [x["teacher"]["eu"] for x in r]; pe = [x["proxy"]["eu"] for x in r]; y = [x["correct"] for x in r]
        out[split] = {"agree": _spear(te, pe), "auroc": _auroc(pe, y), "n": len(r)}
    return out


def main():
    ap = argparse.ArgumentParser(description="RQ5<->RQ1 preservation loop-closure for a UA variant.")
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["head", "edl"])
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_rq5_pres"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rq5_uncertainty import UncertaintyHead, _masked_mean

    os.makedirs(args.results_root, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tag = os.path.basename(args.variant_dir.rstrip("/")).replace("proxy_", "")

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dt,
                                                   local_files_only=True).to(device).eval()
    tt = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    adapter = glob.glob(os.path.join(args.variant_dir, "*", "logs", "saved_models", "best_model"))[0]
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dt, local_files_only=True)
    proxy = PeftModel.from_pretrained(base, adapter).merge_and_unload().to(device).eval()
    pt = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    head = None
    if args.mode == "head":
        ck = torch.load(os.path.join(os.path.dirname(adapter), "uncertainty_head.pt"), map_location=device)
        head = UncertaintyHead(ck["hidden_size"]).to(device).float(); head.load_state_dict(ck["state_dict"]); head.eval()
    print(f"[rq5-pres] {tag} merged; teacher + proxy loaded")

    def proxy_u(prompt, response):
        if args.mode == "edl":
            return evidential_all(proxy, pt, prompt, response, device, args.top_k)["eu"]
        text = pt.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                                      add_generation_prompt=True, tokenize=False, enable_thinking=False)
        enc = pt(text + (response or ""), return_tensors="pt").to(device)
        with torch.no_grad():
            out = proxy(**enc, output_hidden_states=True)
            return float(head(_masked_mean(out.hidden_states[-1], enc["attention_mask"]).float())[0])

    from tqdm import tqdm
    recs = []
    pqs = (sorted(glob.glob(os.path.join(QA_ROOT, f"stageA_*_{TEACHER}_*_seed{args.seed}.parquet"))) +
           sorted(glob.glob(os.path.join(HEALTH_ROOT, f"stageA_*_{TEACHER}_*_seed{args.seed}.parquet"))))
    for pq in pqs:
        ds = os.path.basename(pq).split("stageA_", 1)[1].split(f"_{TEACHER}_", 1)[0]
        labels = _labels_for(pq)
        if labels is None:
            continue
        for it in tqdm(_CacheView(pq).read(), desc=f"[rq5-pres] {ds}", leave=False):
            c = labels.get(str(it["item_id"]), {}).get("correct_llm_judge")
            if c not in (0, 1):
                continue
            q, a = it["question"], it.get("primary_answer", "")
            recs.append({"dataset": ds, "correct": c,
                         "teacher_eu": evidential_all(teacher, tt, q, a, device, args.top_k)["eu"],
                         "proxy_u": proxy_u(q, a)})

    res = {"variant": tag, "mode": args.mode, "splits": {}}
    for split, keep in (("ID", lambda d: d in ID), ("OOD", lambda d: d not in ID)):
        r = [x for x in recs if keep(x["dataset"])]
        te = [x["teacher_eu"] for x in r]; pu = [x["proxy_u"] for x in r]; y = [x["correct"] for x in r]
        res["splits"][split] = {"agree_teacher": _spear(te, pu), "auroc": _auroc(pu, y), "n": len(r)}
    json.dump({**res, "records": recs}, open(os.path.join(args.results_root, f"rq5pres_{tag}_seed{args.seed}.json"), "w"),
              indent=1, default=str)

    base_ag = baseline_agreement()
    print(f"\n=== RQ5<->RQ1 preservation: {tag} (mode={args.mode}) vs DisAAD baseline ===")
    print(f"{'split':5s} {'':14s} {'agree(teacher-EU vs proxy)':>28} {'proxy AUROC':>12}")
    for sp in ("ID", "OOD"):
        b = base_ag[sp]; v = res["splits"][sp]
        f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        print(f"{sp:5s} baseline(DisAAD-EU) {f(b['agree']):>16} {f(b['auroc']):>18}")
        print(f"{sp:5s} {tag[:18]:18s} {f(v['agree_teacher']):>12} {f(v['auroc']):>18}")


if __name__ == "__main__":
    main()
