#!/usr/bin/env python
"""Canonical single-answer vs multi-sample comparison for paper 3.

Single-answer  = our rungs n1..n7: score ONE fixed answer (1 target generation, already paid).
Multi-sample   = consistency/graph baselines at their best N: need N extra target generations.
P(True) and VerbalizedConfidence are excluded from the baseline side (grey-box / our own n1).
Emits the grouped comparison table + a per-cell table, both as LaTeX.
"""
import json, glob, os, re
from collections import defaultdict
W = os.path.expanduser("~/JasonLucas/outputs")
OUT = f"{W}/results_coi_final"; os.makedirs(OUT, exist_ok=True)
EXCLUDE = ("PTrue", "VerbalizedConfidence")
DIRS = ["results_coi_rungs2/lf","results_coi_rungs2/gen","results_coi_rungs","results_coi_extend",
        "results_coi_verify","results_coi_longfact","results_coi_gen",
        "results_coi_bigtargets_health","results_coi_bigtargets","results_coi_closed","results_coi"]
MODELS = [("llama-3.1-8b","Llama-8B"),("qwen3-8b","Qwen-8B"),("qwen3-32b","Qwen-32B"),
          ("llama3.3-70b","Llama-70B"),("jhu-gpt-4o","GPT-4o"),("jhu-claude-haiku-4.5","Claude-H")]
DS = [("trivia_qa","TriviaQA","general short"),("natural_qa","NaturalQA","general short"),
      ("pop_qa","PopQA","general short"),("longfact","LongFact","general long"),
      ("truthful_qa","TruthfulQA","adversarial"),("medqa","MedQA","health MCQ"),
      ("mmlu_med","MMLU-Med","health MCQ"),("bioasq","BioASQ","health free"),
      ("kqa","K-QA","health free"),("medlfqa","MedLFQA","health long"),
      ("gsm8k","GSM8K","math"),("hotpot_qa","HotpotQA","multi-hop")]

def load():
    rung, bal, base = {}, {}, {}
    for d in DIRS:
        for f in glob.glob(f"{W}/{d}/coi_*_seed*.json"):
            g = re.sub(r"^coi_|_seed\d+\.json$","",os.path.basename(f))
            for c in json.load(open(f)).get("cells",[]):
                ds = c.get("dataset")
                if not ds: continue
                for k,v in (c.get("rows") or {}).items():
                    m = re.search(r"_n(\d+)$",k)
                    if not m or v.get("auroc") is None: continue
                    rung.setdefault((g,ds),{}).setdefault(int(m.group(1)), v["auroc"])
                    N,P = c.get("n_items"), c.get("n_positive")
                    if N is None: N,P = v.get("n"), v.get("n_positive")
                    if N is not None and P is not None: bal.setdefault((g,ds),(N,P))
    def add(g,ds,k,a):
        if any(k.startswith(x) for x in EXCLUDE): return
        base.setdefault((g,ds),{}).setdefault(k,a)
    for f in glob.glob(f"{W}/results_coi_baselines/baselines_*_seed*.json"):
        js=json.load(open(f))
        for c in js.get("cells",[]):
            for k,v in (c.get("rows") or {}).items():
                if v.get("auroc") is not None: add(js["generator"],c["dataset"],k,v["auroc"])
    p=f"{W}/results_full/aggregated.json"
    if os.path.exists(p):
        for g,dss in json.load(open(p)).items():
            for ds,ms in dss.items():
                for k,v in ms.items():
                    if isinstance(v,dict) and isinstance(v.get("auroc"),dict): add(g,ds,k,v["auroc"]["mean"])
    for g in ["llama-3.1-8b","qwen3-8b"]:
        for f in glob.glob(f"{W}/results_full_health/stage_cd_{g}_seed*.json"):
            for c in json.load(open(f)).get("cells",[]):
                for k,v in (c.get("rows") or {}).items():
                    if isinstance(v,dict) and v.get("auroc") is not None: add(g,c["dataset"],k,v["auroc"])
    for g in ["qwen3-32b","llama3.3-70b"]:
        f=f"{W}/results_mt/stage_cd_{g}_seed0.json"
        if os.path.exists(f):
            for c in json.load(open(f)).get("cells",[]):
                for k,v in (c.get("rows") or {}).items():
                    if isinstance(v,dict) and v.get("auroc") is not None: add(g,c["dataset"],k,v["auroc"])
    return rung, bal, base

def main():
    rung, bal, base = load()
    rows=[]
    for ds,dl,grp in DS:
        for g,gl in MODELS:
            r=rung.get((g,ds))
            if not r: continue
            N,P = bal.get((g,ds),(None,None))
            if N is None or P is None or min(P,N-P)<15: continue
            sa_k=max(r,key=r.get); sa=r[sa_k]
            bm=base.get((g,ds),{})
            ms_k,ms=(max(((a,k) for k,a in bm.items()))[::-1] if bm else (None,None))
            rows.append({"dataset":ds,"dataset_label":dl,"group":grp,"model":g,"model_label":gl,
                         "n":N,"neg":min(P,N-P),
                         "single_best":sa,"single_best_rung":f"n{sa_k}",
                         "multi_best":ms,"multi_best_method":ms_k,
                         "single_wins":(None if ms is None else bool(sa>ms)),
                         "delta":(None if ms is None else round(sa-ms,4))})
    json.dump(rows, open(f"{OUT}/sa_vs_ms.json","w"), indent=1)
    # grouped summary
    agg=defaultdict(lambda:{"n":0,"win":0,"d":[]})
    for r in rows:
        if r["multi_best"] is None: continue
        a=agg[r["group"]]; a["n"]+=1; a["win"]+=r["single_wins"]; a["d"].append(r["delta"])
    order=["adversarial","health MCQ","health long","health free","general short","general long","math","multi-hop"]
    lines=[]
    print(f"{'data type':<16}{'cells':>6}{'single-answer wins':>20}{'median delta':>14}")
    print("-"*58)
    tw=tn=0
    for grp in order:
        if grp not in agg: continue
        a=agg[grp]; ds_=sorted(a["d"]); med=ds_[len(ds_)//2]
        print(f"{grp:<16}{a['n']:>6}{f'{a[chr(119)+chr(105)+chr(110)]}/{a[chr(110)]}':>20}{med:>+14.3f}")
        lines.append(f"{grp.title()} & {a['n']} & {a['win']}/{a['n']} & ${med:+.3f}$ \\\\")
        tw+=a["win"]; tn+=a["n"]
    print("-"*58); print(f"{'TOTAL':<16}{tn:>6}{f'{tw}/{tn}':>20}")
    lines.append(f"\\midrule\n\\textbf{{Total}} & {tn} & \\textbf{{{tw}/{tn}}} & \\\\")
    open(f"{OUT}/table_sa_vs_ms.tex","w").write("\n".join(lines)+"\n")
    print(f"\nwrote {OUT}/sa_vs_ms.json and table_sa_vs_ms.tex")

if __name__=="__main__": main()
