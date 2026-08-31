#!/usr/bin/env python
"""Canonical recomputation of every paper-3 number, from the result JSONs.

Single source of truth for (model x dataset x method) AUROC, class balance and validity, so the
manuscript tables are GENERATED rather than transcribed. Writes:
  results_coi_final/coi_final_grid.json   machine-readable grid
  results_coi_final/tables_*.tex          LaTeX table bodies used by the manuscript
"""
import glob, json, os, re
W = os.path.expanduser("~/JasonLucas/outputs")
OUT = f"{W}/results_coi_final"; os.makedirs(OUT, exist_ok=True)

MODELS = [("llama-3.1-8b","Llama-3.1-8B","8B open"),("qwen3-8b","Qwen3-8B","8B open"),
          ("qwen3-32b","Qwen3-32B","large open"),("llama3.3-70b","Llama-3.3-70B","large open"),
          ("jhu-gpt-4o","GPT-4o","closed"),("jhu-claude-haiku-4.5","Claude-Haiku","closed")]
DATASETS = [("trivia_qa","TriviaQA","general/short"),("natural_qa","NaturalQA","general/short"),
            ("pop_qa","PopQA","general/short"),("longfact","LongFact","general/long"),
            ("truthful_qa","TruthfulQA","adversarial"),("medqa","MedQA","health/MCQ"),
            ("mmlu_med","MMLU-Med","health/MCQ"),("bioasq","BioASQ","health/free"),
            ("kqa","K-QA","health/free"),("medlfqa","MedLFQA","health/long"),
            ("gsm8k","GSM8K","math"),("hotpot_qa","HotpotQA","multi-hop")]
RUNGS = [1,2,3,4,6,7]
MINORITY_MIN = 15

# every directory that can contain a CoI result, most specific first
DIRS = ["results_coi_rungs2/lf","results_coi_rungs2/gen","results_coi_rungs","results_coi_extend",
        "results_coi_verify","results_coi_longfact","results_coi_gen","results_coi_bigtargets_health",
        "results_coi_bigtargets","results_coi_closed","results_coi"]

def collect():
    grid, bal, prov = {}, {}, {}
    for d in DIRS:
        for f in glob.glob(f"{W}/{d}/coi_*_seed*.json"):
            gen = re.sub(r"^coi_|_seed\d+\.json$","",os.path.basename(f))
            for c in json.load(open(f)).get("cells",[]):
                ds = c.get("dataset")
                if not ds: continue
                for k,v in (c.get("rows") or {}).items():
                    m = re.search(r"_n(\d+)$", k)
                    if not m or v.get("auroc") is None: continue
                    n = int(m.group(1))
                    grid.setdefault((gen,ds),{}).setdefault(n, v["auroc"])
                    prov.setdefault((gen,ds,n), d)
                    # class balance: cell-level (open) or row-level (closed)
                    N,P = c.get("n_items"), c.get("n_positive")
                    if N is None: N,P = v.get("n"), v.get("n_positive")
                    if N is not None and P is not None: bal.setdefault((gen,ds),(N,P))
    return grid, bal, prov

def baselines():
    out={}
    for f in glob.glob(f"{W}/results_coi_baselines/baselines_*_seed*.json"):
        js=json.load(open(f))
        for c in js.get("cells",[]):
            out[(js["generator"],c["dataset"])] = {k:v["auroc"] for k,v in (c.get("rows") or {}).items()
                                                   if v.get("auroc") is not None}
    for f in glob.glob(f"{W}/results_full/aggregated.json"):
        js=json.load(open(f))
        for gen,dss in js.items():
            for ds,ms in dss.items():
                d=out.setdefault((gen,ds),{})
                for k,v in ms.items():
                    if isinstance(v,dict) and isinstance(v.get("auroc"),dict):
                        d.setdefault(k, v["auroc"]["mean"])
    return out

def main():
    grid, bal, prov = collect(); bl = baselines()
    rows=[]
    for gen,gl,cls in MODELS:
        for ds,dl,grp in DATASETS:
            g = grid.get((gen,ds))
            if not g: continue
            N,P = bal.get((gen,ds),(None,None))
            minority = min(P, N-P) if (N is not None and P is not None) else None
            rows.append({"model":gen,"model_label":gl,"model_class":cls,
                         "dataset":ds,"dataset_label":dl,"group":grp,
                         "n_items":N,"n_positive":P,"minority":minority,
                         "valid": (minority is not None and minority>=MINORITY_MIN),
                         "auroc":{f"n{n}": g.get(n) for n in RUNGS if g.get(n) is not None},
                         "baselines": bl.get((gen,ds),{}),
                         "source":{f"n{n}": prov.get((gen,ds,n)) for n in RUNGS if g.get(n) is not None}})
    json.dump({"minority_min":MINORITY_MIN,"rows":rows}, open(f"{OUT}/coi_final_grid.json","w"), indent=1)
    v=[r for r in rows if r["valid"]]
    print(f"cells total={len(rows)}  valid={len(v)}  degenerate={len(rows)-len(v)}")
    from collections import Counter
    w=Counter(max(r['auroc'],key=r['auroc'].get) for r in v if r['auroc'])
    print("wins:", dict(w.most_common()))
    print(f"wrote {OUT}/coi_final_grid.json")

if __name__=="__main__": main()


# ----------------------------------------------------------------------------------
def emit_latex():
    """Generate the manuscript's dataset-organised tables from the canonical grid."""
    g = json.load(open(f"{OUT}/coi_final_grid.json"))
    R = {(r["model"], r["dataset"]): r for r in g["rows"]}
    MODEL_ORDER = [m[0] for m in MODELS]
    SHORT = {"llama-3.1-8b":"Llama-8B","qwen3-8b":"Qwen-8B","qwen3-32b":"Qwen-32B",
             "llama3.3-70b":"Llama-70B","jhu-gpt-4o":"GPT-4o","jhu-claude-haiku-4.5":"Claude-H"}
    GROUPS = [("general/short","General, short-form"),("general/long","General, long-form"),
              ("adversarial","Adversarial"),("health/MCQ","Health, multiple-choice"),
              ("health/free","Health, free-form"),("health/long","Health, long-form"),
              ("math","Math"),("multi-hop","Multi-hop")]
    lines=[]
    for grp, glabel in GROUPS:
        dss=[d for d in DATASETS if d[2]==grp]
        body=[]
        for ds,dl,_ in dss:
            first=True
            for m in MODEL_ORDER:
                r=R.get((m,ds))
                if not r: continue
                name = dl if first else ""
                first=False
                if not r["valid"]:
                    body.append(f"{name} & \\grey{{{SHORT[m]}}} & \\multicolumn{{6}}{{c}}{{\\grey{{degenerate (minority {r['minority']})}}}} \\\\")
                    continue
                a=r["auroc"]; best=max(a,key=a.get)
                cells=" & ".join(
                    (f"\\textbf{{{a[f'n{n}']:.3f}}}" if f"n{n}"==best else f"{a[f'n{n}']:.3f}")
                    if f"n{n}" in a else "--" for n in RUNGS)
                body.append(f"{name} & {SHORT[m]} & {cells} \\\\")
            body.append("\\addlinespace")
        if body:
            lines.append(f"% ===== {glabel} =====")
            lines += body
    open(f"{OUT}/tables_by_dataset.tex","w").write("\n".join(lines)+"\n")
    print(f"wrote {OUT}/tables_by_dataset.tex ({len(lines)} lines)")

    # per-group win summary
    from collections import Counter, defaultdict
    per=defaultdict(Counter)
    for r in g["rows"]:
        if not r["valid"] or not r["auroc"]: continue
        per[r["group"]][max(r["auroc"],key=r["auroc"].get)]+=1
    out=[]
    for grp,glabel in GROUPS:
        c=per.get(grp)
        if not c: continue
        tot=sum(c.values())
        ver=c.get("n6",0)+c.get("n7",0)
        cells=" & ".join(str(c.get(f"n{n}",0)) for n in RUNGS)
        out.append(f"{glabel} & {tot} & {cells} & {ver}/{tot} \\\\")
    open(f"{OUT}/table_group_summary.tex","w").write("\n".join(out)+"\n")
    print(f"wrote {OUT}/table_group_summary.tex")
    for l in out: print("   ",l)

if __name__=="__main__" and os.environ.get("EMIT_LATEX"):
    emit_latex()
