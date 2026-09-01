#!/usr/bin/env python
"""Independent recomputation of every CoI AUROC from RAW per-item data.

The result JSONs contain AUROCs produced at scoring time. This script ignores them: it reads the
per-item truth values from the rung raw logs and the per-item correctness labels from the Stage-B
sidecars (or the precomputed labels files), recomputes AUROC itself, and diffs against what the
JSONs report. Any disagreement indicates a bug in the scoring path, not a rounding difference.
"""
import glob, json, os, re
import numpy as np
from sklearn.metrics import roc_auc_score

W = os.path.expanduser("~/JasonLucas/outputs")
HEALTH = {"medqa","mmlu_med","kqa","medlfqa","bioasq"}
LBL_DIRS = ["cache_full","cache_full_health","cache_mt","cache_closed","cache_closed_full"]

def labels_for(gen, ds, seed):
    # 1) Stage-B sidecar next to the Stage-A cache
    for root in LBL_DIRS:
        for f in glob.glob(f"{W}/{root}/stageA_{ds}_{gen}_*_seed{seed}.labels.json"):
            labs = json.load(open(f))["labels"]
            return [int(labs[k]["correct_llm_judge"]) for k in sorted(labs, key=lambda x: int(x))]
    # 2) precomputed labels files (LongFact / generated datasets)
    for pat in (f"{W}/cache_coi_longfact/labels_{ds}_{gen}_seed{seed}.json",
                f"{W}/cache_coi_gen/labels_{ds}_{gen}_seed{seed}.json",
                f"{W}/cache_coi_gen/labels_merged_{gen}_seed{seed}.json"):
        if os.path.exists(pat):
            lab = json.load(open(pat))
            def key(k):
                m = re.search(r"(\d+)$", k); return int(m.group(1)) if m else 0
            ks = [k for k in lab if k.startswith(ds[:3])] or list(lab)
            return [int(lab[k]) for k in sorted(ks, key=key)]
    return None

def reported(gen, ds, n, seed):
    for d in ["results_coi_closed","results_coi_rungs2/lf","results_coi_rungs2/gen","results_coi_rungs","results_coi_extend",
              "results_coi_verify","results_coi_longfact","results_coi_gen","results_coi_bigtargets_health",
              "results_coi_bigtargets","results_coi"]:
        p = f"{W}/{d}/coi_{gen}_seed{seed}.json"
        if not os.path.exists(p): continue
        for c in json.load(open(p)).get("cells", []):
            if c.get("dataset") == ds:
                r = (c.get("rows") or {}).get(f"CoIVerbalized_n{n}")
                if r and r.get("auroc") is not None: return r["auroc"]
    return None

def main():
    files = sorted(set(glob.glob(f"{W}/results_coi*/rawlog_*_seed*.jsonl")
                       + glob.glob(f"{W}/results_coi_rungs2/*/rawlog_*_seed*.jsonl")))
    ok = mism = noref = nolab = 0; worst = []
    for f in files:
        m = re.search(r"rawlog_(.+)_n(\d+)_seed(\d+)\.jsonl$", os.path.basename(f))
        if not m: continue
        stem, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
        # split "<gen>_<dataset>" using the known dataset suffixes
        ds = next((d for d in ["trivia_qa","natural_qa","pop_qa","longfact","truthful_qa","medqa",
                               "mmlu_med","bioasq","kqa","medlfqa","gsm8k","hotpot_qa"]
                   if stem.endswith("_"+d)), None)
        if not ds: continue
        gen = stem[:-(len(ds)+1)]
        y = labels_for(gen, ds, seed)
        if y is None: nolab += 1; continue
        tv = [json.loads(l)["tv"] for l in open(f) if l.strip()]
        k = min(len(y), len(tv))
        yy = [y[i] for i in range(k) if y[i] in (0,1)]
        ss = [tv[i] for i in range(k) if y[i] in (0,1)]
        if len(set(yy)) < 2: continue
        mine = roc_auc_score(np.array(yy), np.array(ss, float))
        rep = reported(gen, ds, n, seed)
        if rep is None: noref += 1; continue
        d = abs(mine - rep)
        if d < 5e-4: ok += 1
        else:
            mism += 1; worst.append((d, gen, ds, n, mine, rep))
    print(f"recomputed from raw per-item data:")
    print(f"  MATCH     {ok}")
    print(f"  MISMATCH  {mism}")
    print(f"  no label / no reported value: {nolab} / {noref}")
    for d, gen, ds, n, a, b in sorted(worst, reverse=True)[:12]:
        print(f"    {gen:20s}/{ds:11s} n{n}: raw={a:.4f} json={b:.4f} diff={d:.4f}")

if __name__ == "__main__": main()
