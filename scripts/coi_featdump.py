#!/usr/bin/env python
"""Dump per-item {correctness, CoI features} for the aggregation search: can a better math than the
fixed-prior phi (v0) correlate confidence with correctness? Features come from the existing n=3 raw
logs (grounding, risk, contradictions, n_claims); correctness from the judge (no target model needed).
Writes results_coi/features_<gen>_<ds>_seed1.json for offline fitting (scripts/coi_aggsearch.py)."""
import argparse, glob, json, os, sys, re
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.dirname(_HERE)
sys.path.insert(0,os.path.join(_REPO,"src")); sys.path.insert(0,_REPO); sys.path.insert(0,_HERE)
import numpy as np

CELLS=[("llama-3.1-8b","trivia_qa","cache_full"),("llama-3.1-8b","medlfqa","cache_full_health"),
       ("qwen3-8b","trivia_qa","cache_full"),("qwen3-8b","medlfqa","cache_full_health")]

def feats(raw):
    from TruthTorchLM.truth_methods.coi_verbalized import _extract_json
    o=_extract_json(raw) or {}
    claims=(o.get("chain_1_decompose") or {}).get("claims") or []
    ids=[c.get("id",f"c{i+1}") for i,c in enumerate(claims)] or ["c1"]
    per={c.get("id"):c for c in (o.get("chain_2_evidence_critique") or {}).get("claims") or []}
    contr=(o.get("chain_3_consistency") or {}).get("contradictions") or []
    risks=[]; grounds=[]
    for cid in ids:
        c=per.get(cid,{}); 
        try: risks.append(float(c.get("risk",0.5)))
        except: risks.append(0.5)
        grounds.append((c.get("grounding") or "parametric").split("|")[0].strip().lower())
    risks=risks or [0.5]
    aq=(o.get("chain_3_consistency") or {}).get("answers_query",True)
    return {"n_claims":len(ids),"mean_risk":float(np.mean(risks)),"min_risk":float(min(risks)),
            "max_risk":float(max(risks)),"std_risk":float(np.std(risks)),
            "frac_contextual":sum(g=="contextual" for g in grounds)/len(grounds),
            "frac_parametric":sum(g=="parametric" for g in grounds)/len(grounds),
            "frac_inferred":sum(g=="inferred" for g in grounds)/len(grounds),
            "has_contradiction":int(len(contr)>0),"answers_query":int(bool(aq)),
            "prod_1mrisk":float(np.prod([1-r for r in risks]))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--judge",default="Qwen/Qwen3-4B-Instruct-2507"); ap.add_argument("--seed",type=int,default=1)
    args=ap.parse_args()
    import torch
    from stage2_open import load_judge,_make_config,_mcq_fn,MCQ_DATASETS
    from hc_benchmark.cache import GenerationCache
    from hc_benchmark.stage_b_label import label_stage_b,correctness_vector
    from calibrate_eval import _CacheView
    from pathlib import Path as _Path
    class _GC(GenerationCache):
        def __init__(s,r,c,se,e): super().__init__(r,c,se); s._e=_Path(e)
        @property
        def path(s): return s._e
    dev="cuda" if torch.cuda.is_available() else "cpu"
    jf=load_judge(args.judge,dev,"bf16")[0]
    WORK=os.path.expanduser("~/JasonLucas")
    for gen,ds,croot in CELLS:
        root=f"{WORK}/outputs/{croot}"
        hits=glob.glob(f"{root}/stageA_{ds}_{gen}_*_seed{args.seed}.parquet")
        if not hits: print(f"[{gen}/{ds}] no cache"); continue
        cfg=_make_config(ds,gen,n_max=1,size=1.0,seed=args.seed); cache=_GC(root,cfg,args.seed,hits[0])
        is_mcq=ds in MCQ_DATASETS
        label_stage_b(cache,criteria=("llm_judge",),_judge_fn=(_mcq_fn() if is_mcq else jf),judge_model="j")
        corr=correctness_vector(cache,"llm_judge")
        items=_CacheView(hits[0]).read()
        q2c={it["question"][:160]:c for it,c in zip(items,corr)}
        rl=f"{WORK}/outputs/results_coi/rawlog_{gen}_{ds}_n3_seed{args.seed}.jsonl"
        recs={r["q"]:r for r in (json.loads(l) for l in open(rl) if l.strip())} if os.path.exists(rl) else {}
        out=[]
        for q,c in q2c.items():
            if c not in (0,1) or q not in recs: continue
            f=feats(recs[q].get("raw","")); f["correct"]=int(c); f["phi_v0"]=recs[q]["tv"]
            out.append(f)
        p=f"{WORK}/outputs/results_coi/features_{gen}_{ds}_seed{args.seed}.json"
        json.dump(out,open(p,"w")); print(f"[{gen}/{ds}] {len(out)} items -> {p}")
if __name__=="__main__": main()
