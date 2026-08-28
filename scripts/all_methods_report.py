#!/usr/bin/env python
"""All-methods comparison on qwen3-8b — AUROC + latency, every family (direct, DisAAD-au, DisAAD-eu,
our UA variants), ranked, with the AUROC-vs-latency Pareto frontier marked. No cherry-picking:
BOTH DisAAD read-outs are shown. Writes docs/all_methods_comparison.md.

    python scripts/all_methods_report.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import rq3_report as R  # noqa: E402

HOME = os.path.expanduser("~/JasonLucas/outputs")
ALLDS = set(R.QA) | set(R.HEALTH)
GROUPS = [("All 9 datasets", ALLDS), ("Health · MCQ", {"medqa", "mmlu_med"}),
          ("Health · Free-form QA", {"kqa", "medlfqa", "bioasq"}),
          ("General · QA", set(R.QA))]


def _mean(x):
    x = [v for v in x if isinstance(v, (int, float))]
    return sum(x) / len(x) if x else None


def direct(dss):
    out = defaultdict(lambda: {"au": [], "ms": []})
    for root, group in (("results_full", set(R.QA)), ("results_full_health", set(R.HEALTH))):
        sub = dss & group
        if not sub:
            continue
        bn = R.best_n(R.load_main(os.path.join(HOME, root), sub), sub)
        for m, (n, byds) in bn.items():
            for d in sub:
                for t in byds.get(d, []):
                    out[m]["au"].append(t[0]); out[m]["ms"].append(t[2])
    return {m: (_mean(v["au"]), _mean(v["ms"])) for m, v in out.items()}


def cells(files, dss):
    au, ms = defaultdict(list), defaultdict(list)
    for f in files:
        for c in json.load(open(f)).get("cells", []):
            if c.get("dataset") not in dss:
                continue
            for k, r in c.get("rows", {}).items():
                nm = k.rpartition("_N")[0]; au[nm].append(r.get("auroc")); ms[nm].append(r.get("p50_ms"))
    return {m: (_mean(au[m]), _mean(ms[m])) for m in au}


def disp(m):
    return (m.replace("rq5_", "UA·").replace("EccentricityUncertainty", "Ecc")
             .replace("DiscreteSemanticEntropy", "SemEnt").replace("VerbalizedConfidence", "Verb"))


def main():
    L = ["# All-Methods Comparison — AUROC vs. Latency (qwen3-8b)\n",
         "> Every UQ method on the same target (qwen3-8b), same 9 datasets, mean over seeds. Families: "
         "**direct** (consistency + verbalized), **DisAAD-au** & **DisAAD-eu** (both read-outs — no "
         "cherry-picking), and **OURS** (uncertainty-aware variants). `★` = on the AUROC-vs-latency "
         "Pareto frontier. Latency = auxiliary-compute p50, one forward for proxies.\n",
         "**Auto-generated** by `scripts/all_methods_report.py`.\n"]
    for gname, dss in GROUPS:
        rows = []
        for m, (a, l) in direct(dss).items():
            rows.append((a, l, disp(m), "direct"))
        for m, (a, l) in cells(glob.glob(HOME + "/results_disaad/stage_cd_disaad-qwen3-8b_seed*.json"), dss).items():
            rows.append((a, l, m, "DisAAD"))          # DisAAD-au and DisAAD-eu both included
        for m, (a, l) in cells(glob.glob(HOME + "/results_dald/rq5_dald_seed*.json"), dss).items():
            rows.append((a, l, "DALD-proxy", "DALD"))  # faithful controlled DALD (masked SFT)
        for m, (a, l) in cells(glob.glob(HOME + "/results_rq5/rq5_*_seed*.json"), dss).items():
            rows.append((a, l, disp(m), "OURS"))
        rows = [r for r in rows if r[0] is not None]
        rows.sort(reverse=True)
        front, best = set(), -1
        for a, l, m, fam in sorted(rows, key=lambda r: (r[1] if r[1] is not None else 9e9)):
            if a > best:
                front.add(m); best = a
        L.append(f"\n## {gname}\n")
        L.append("| # | method | family | AUROC | p50 ms | Pareto |")
        L.append("|--:|---|---|--:|--:|:--:|")
        for i, (a, l, m, fam) in enumerate(rows, 1):
            mark = "**"+m+"**" if fam == "OURS" else m
            L.append(f"| {i} | {mark} | {fam} | {a:.3f} | {(f'{l:.0f}' if l else '—')} | {'★' if m in front else ''} |")
    L.append("\n**Read:** our UA head variants are **Pareto-optimal** (best AUROC achievable at ~30 ms) and "
             "beat **both** DisAAD-au and DisAAD-eu — but the few-pass direct methods (VerbalizedConfidence, "
             "PTrue) and the slow graph methods reach higher AUROC at 2–100× the latency. The defensible "
             "claim is *best accuracy-per-millisecond*, not *highest AUROC*.\n")
    doc = os.path.join(os.path.dirname(_HERE), "docs", "all_methods_comparison.md")
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    print(f"[all_methods_report] wrote {doc}")


if __name__ == "__main__":
    main()
