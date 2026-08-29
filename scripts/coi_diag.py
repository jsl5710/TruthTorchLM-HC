#!/usr/bin/env python
"""Diagnose the medlfqa n=3 collapse: is the fixed-prior phi (v0) the culprit, or is the model's
per-claim RISK signal genuinely uninformative on medical? Labels correctness (judge) for the cached
(x,y), aligns to the EXISTING raw logs (n1 verbalized conf, n3 phi, per-claim risks), and reports
AUROC of each signal + risk distribution for correct vs incorrect. No CoI re-generation needed."""
import argparse, glob, json, os, sys, re
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.dirname(_HERE)
sys.path.insert(0,os.path.join(_REPO,"src")); sys.path.insert(0,_REPO); sys.path.insert(0,_HERE)
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--generator",default="llama-3.1-8b"); ap.add_argument("--dataset",default="medlfqa")
    ap.add_argument("--judge",default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--cache-root",default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--rawlog-root",default=os.path.expanduser("~/JasonLucas/outputs/results_coi"))
    ap.add_argument("--seed",type=int,default=1)
    args=ap.parse_args()
    import torch
    from sklearn.metrics import roc_auc_score
    from stage2_open import load_judge, _make_config, _mcq_fn, MCQ_DATASETS
    from hc_benchmark.cache import GenerationCache
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from calibrate_eval import _CacheView
    from pathlib import Path as _Path
    class _GlobCache(GenerationCache):
        def __init__(s,root,cfg,seed,exp): super().__init__(root,cfg,seed); s._e=_Path(exp)
        @property
        def path(s): return s._e
    device="cuda" if torch.cuda.is_available() else "cpu"
    # correctness
    cfg=_make_config(args.dataset,args.generator,n_max=1,size=1.0,seed=args.seed)
    hits=glob.glob(os.path.join(args.cache_root,f"stageA_{args.dataset}_{args.generator}_*_seed{args.seed}.parquet"))
    cache=_GlobCache(args.cache_root,cfg,args.seed,hits[0])
    is_mcq=args.dataset in MCQ_DATASETS
    jf=_mcq_fn() if is_mcq else load_judge(args.judge,device,"bf16")[0]
    label_stage_b(cache,criteria=("llm_judge",),_judge_fn=jf,judge_model=("mcq" if is_mcq else "judge"))
    corr=correctness_vector(cache,"llm_judge")
    items=_CacheView(hits[0]).read()
    q2c={ (it["question"][:160]): c for it,c in zip(items,corr) }   # align by question prefix
    # raw logs
    def load_raw(n):
        f=f"{args.rawlog_root}/rawlog_{args.generator}_{args.dataset}_n{n}_seed{args.seed}.jsonl"
        return {r["q"]:r for r in (json.loads(l) for l in open(f) if l.strip())} if os.path.exists(f) else {}
    r1=load_raw(1); r3=load_raw(3)
    rows=[]
    for q,c in q2c.items():
        if c not in (0,1) or q not in r3: continue
        rec3=r3[q]; risks=[float(x) for x in re.findall(r'"risk"\s*:\s*([0-9]*\.?[0-9]+)', rec3.get("raw",""))]
        rows.append({"inc":1-c, "n1":r1.get(q,{}).get("tv"), "n3":rec3["tv"],
                     "meanrisk":(np.mean(risks) if risks else np.nan),
                     "maxrisk":(max(risks) if risks else np.nan)})
    inc=np.array([r["inc"] for r in rows])
    def auroc(key, sign=1):
        v=np.array([r[key] for r in rows],float); ok=np.isfinite(v)
        if ok.sum()<2 or len(set(inc[ok]))<2: return float("nan")
        return roc_auc_score(inc[ok], sign*v[ok])
    print(f"\n### {args.generator} / {args.dataset}: n={len(rows)}, incorrect={int(inc.sum())} ###")
    print(f"  AUROC(correctness, n1 verbalized conf) = {auroc('n1',-1):.3f}   [holistic self-rating]")
    print(f"  AUROC(correctness, n3 phi)             = {auroc('n3',-1):.3f}   [deterministic phi v0]")
    print(f"  AUROC(correctness, mean per-claim risk)= {auroc('meanrisk',1):.3f}   [raw risk signal]")
    print(f"  AUROC(correctness, max per-claim risk) = {auroc('maxrisk',1):.3f}   [weakest-claim risk]")
    cr=[r["meanrisk"] for r in rows if r["inc"]==0]; ir=[r["meanrisk"] for r in rows if r["inc"]==1]
    print(f"  mean risk | correct={np.nanmean(cr):.3f}  incorrect={np.nanmean(ir):.3f}  (separation={np.nanmean(ir)-np.nanmean(cr):+.3f})")
    print("  VERDICT:", "raw risk has NO signal on medical -> no phi can fix it" if abs(auroc('meanrisk',1)-0.5)<0.04
          else "raw risk HAS residual signal -> a fitted phi (v1) could exploit it")

if __name__=="__main__": main()
