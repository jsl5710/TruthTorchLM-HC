#!/usr/bin/env python
"""AUROC vs measured latency per method, for paper 3's efficiency table.

Two cost components, kept separate because they are not interchangeable:
  * scoring latency  -- the method's own compute, measured p50 per item (batch 1, GPU warm).
  * target calls     -- generations of the ANSWER that the method forces. Our single-answer
                        rungs force zero (they score the answer already produced); a
                        multi-sample method at N forces N-1 extra full generations.
Reported over valid cells only (minority >= 15) and only on the open targets, which are the
ones where both families have measured timings.
"""
import json, glob, os, re, statistics
W = os.path.expanduser("~/JasonLucas/outputs")
DIRS = ["results_coi_rungs2/lf","results_coi_rungs2/gen","results_coi_rungs","results_coi_extend",
        "results_coi_verify","results_coi_longfact","results_coi_gen",
        "results_coi_bigtargets_health","results_coi_bigtargets","results_coi"]
OPEN = ["llama-3.1-8b","qwen3-8b","qwen3-32b","llama3.3-70b"]

def valid_cells():
    g = json.load(open(f"{W}/results_coi_final/coi_final_grid.json"))["rows"]
    return {(r["model"], r["dataset"]) for r in g if r["valid"]}

def main():
    ok = valid_cells()
    auc, lat = {}, {}
    for d in DIRS:
        for f in glob.glob(f"{W}/{d}/coi_*_seed*.json"):
            gen = re.sub(r"^coi_|_seed\d+\.json$", "", os.path.basename(f))
            if gen not in OPEN: continue
            for c in json.load(open(f)).get("cells", []):
                if (gen, c.get("dataset")) not in ok: continue
                for k, v in (c.get("rows") or {}).items():
                    m = re.search(r"_n(\d+)$", k)
                    if not m: continue
                    key = f"n{m.group(1)}"
                    if v.get("auroc") is not None: auc.setdefault(key, []).append(v["auroc"])
                    if v.get("p50_ms"): lat.setdefault(key, []).append(v["p50_ms"])
    for gen in ["qwen3-32b","llama3.3-70b","llama-3.1-8b","qwen3-8b"]:
        for f in glob.glob(f"{W}/results_mt/stage_cd_{gen}_seed0.json") + \
                 glob.glob(f"{W}/results_full_health/stage_cd_{gen}_seed*.json"):
            for c in json.load(open(f)).get("cells", []):
                if (gen, c.get("dataset")) not in ok: continue
                for k, v in (c.get("rows") or {}).items():
                    if not isinstance(v, dict): continue
                    if k.startswith(("PTrue","VerbalizedConfidence")): continue  # grey-box / single-answer
                    if not (k.endswith("_N5") or k.endswith("_N10")): continue
                    if v.get("auroc") is not None: auc.setdefault(k, []).append(v["auroc"])
                    if v.get("p50_ms"): lat.setdefault(k, []).append(v["p50_ms"])
    NAME = {"n1":"verbalized (global self-rating)","n2":"per-claim self-rated confidence",
            "n3":"$\\phi$ over self-rated risk","n4":"split evidence/critique chains",
            "n5":"$n_4$ + logged self-rating","n6":"decision-framed verify","n7":"self-consistency verify ($K{=}5$)"}
    print(f"{'method':<40}{'cells':>6}{'meanAUROC':>11}{'p50 scoring ms':>16}{'forced gens':>13}")
    print("-"*88)
    rows=[]
    for k in ["n1","n2","n3","n4","n5","n6","n7"]:
        if k not in auc: continue
        rows.append((NAME[k], len(auc[k]), statistics.fmean(auc[k]),
                     statistics.median(lat[k]) if lat.get(k) else None, 0))
    fam = {}
    for k in auc:
        if k.startswith("n") and len(k) <= 2: continue
        base, N = k.rsplit("_N", 1)
        fam.setdefault(int(N), []).append(k)
    for N in sorted(fam):
        best = max(fam[N], key=lambda k: statistics.fmean(auc[k]))
        rows.append((f"best multi-sample @ $N{{=}}{N}$ ({best.rsplit('_N',1)[0]})",
                     len(auc[best]), statistics.fmean(auc[best]),
                     statistics.median(lat[best]) if lat.get(best) else None, N))
    for nm, n, a, l, gens in rows:
        ls = f"{l:.0f}" if l else "--"
        print(f"{nm[:40]:<40}{n:>6}{a:>11.3f}{ls:>16}{gens:>13}")
    json.dump([{"method":nm,"cells":n,"mean_auroc":round(a,4),
                "p50_scoring_ms":(round(l) if l else None),"forced_generations":g}
               for nm,n,a,l,g in rows],
              open(f"{W}/results_coi_final/latency_auc.json","w"), indent=1)
    print(f"\nwrote {W}/results_coi_final/latency_auc.json")

if __name__ == "__main__": main()
