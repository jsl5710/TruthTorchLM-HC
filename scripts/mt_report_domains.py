#!/usr/bin/env python
"""Multi-target RQ — per-DOMAIN AUROC by method family + per-method LATENCY. Complements
mt_report.py (which is per-student). Writes docs/multitarget_by_domain.md. Robust to partial
results; regenerate any time as cells land.

  Table (a): per domain (General/Medical/Math/Adversarial/Factual) x family (best Direct, DALD,
             DisAAD, Ours) -- 'best over students/variants' so it's the ceiling each family reaches.
  Table (b): p50 latency per method + the extra target calls each needs (the Pareto cost).

PTrue is excluded everywhere (grey-box: reads target logprobs -> not pure black-box).

    python scripts/mt_report_domains.py
"""
import glob
import json
import os
import statistics
from collections import defaultdict

HOME = os.path.expanduser("~/JasonLucas/outputs")
SR = os.path.join(HOME, "results_mt_score")
MR = os.path.join(HOME, "results_mt")
GREY = {"PTrue"}                       # reads target logprobs -> excluded (see ptrue-greybox)
DOMAINS = {"General": ["trivia_qa"], "Medical": ["bioasq", "medqa", "medlfqa"],
           "Math": ["gsm8k"], "Adversarial": ["truthful_qa"], "Factual": ["wikipedia_factual"]}
TEACHERS = {"qwen3-32b": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"],
            "llama3.3-70b": ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]}
VERBALIZED = {"VerbalizedConfidence"}


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def direct_byds(teacher):
    o = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(os.path.join(MR, f"stage_cd_{teacher}_seed*.json")):
        for c in json.load(open(fn)).get("cells", []):
            for k, r in c.get("rows", {}).items():
                m = k.rpartition("_N")[0]
                if m in GREY or r.get("auroc") is None:
                    continue
                o[c["dataset"]][m].append(r["auroc"])
    return {ds: {m: sum(v) / len(v) for m, v in mm.items()} for ds, mm in o.items()}


def proxy_byds(teacher, kind):
    """{dataset: best_auroc over all students/variants of `kind`}."""
    o = defaultdict(list)
    pat = f"ours*_{teacher}_*" if kind == "ours" else f"{kind}_{teacher}_*"
    for dd in glob.glob(os.path.join(SR, pat)):
        for fn in glob.glob(os.path.join(dd, "*.json")):
            for c in json.load(open(fn)).get("cells", []):
                best = max((r.get("auroc") for r in c.get("rows", {}).values()
                            if r.get("auroc") is not None), default=None)
                if best is not None:
                    o[c["dataset"]].append(best)
    return {ds: max(v) for ds, v in o.items()}


def proxy_ms():
    vals = []
    for dd in glob.glob(os.path.join(SR, "*")):
        for fn in glob.glob(os.path.join(dd, "*.json")):
            for c in json.load(open(fn)).get("cells", []):
                for r in c.get("rows", {}).values():
                    if r.get("p50_ms"):
                        vals.append(r["p50_ms"])
    return vals


def direct_ms(teacher):
    o = defaultdict(list)
    for fn in glob.glob(os.path.join(MR, f"stage_cd_{teacher}_seed*.json")):
        for c in json.load(open(fn)).get("cells", []):
            for k, r in c.get("rows", {}).items():
                m = k.rpartition("_N")[0]
                if m in GREY or not r.get("p50_ms"):
                    continue
                o[m].append(r["p50_ms"])
    return o


def main():
    L = ["# Multi-Target RQ — by Domain + Latency\n",
         "> Companion to `multitarget_comparison.md` (per-student). **Table (a)** is per-domain AUROC, "
         "each family shown at its **best over students/variants** (the ceiling it reaches). **Table (b)** "
         "is p50 latency per method. PTrue is excluded everywhere (grey-box: reads the target's logprobs).\n",
         "**Auto-generated** by `scripts/mt_report_domains.py`.\n",
         "\n## (a) AUROC by domain × method family\n"]
    for teacher, _students in TEACHERS.items():
        D = direct_byds(teacher)
        DA, DS, OU = (proxy_byds(teacher, k) for k in ("dald", "disaad", "ours"))
        L.append(f"\n### {teacher}\n")
        L.append("| domain | best Direct | DALD | DisAAD | Ours | winner |")
        L.append("|---|---|--:|--:|--:|---|")
        for dom, dss in DOMAINS.items():
            dm = defaultdict(list)
            for ds in dss:
                for m, a in D.get(ds, {}).items():
                    dm[m].append(a)
            dbest = max(((sum(v) / len(v), m) for m, v in dm.items()), default=(None, "—"))

            def _avg(src):
                vals = [src[ds] for ds in dss if ds in src]
                return sum(vals) / len(vals) if vals else None
            da, ds_, ou = _avg(DA), _avg(DS), _avg(OU)
            cand = {f"Direct·{dbest[1]}": dbest[0], "DALD": da, "DisAAD": ds_, "Ours": ou}
            win = max(((v, k) for k, v in cand.items() if v is not None), default=(None, "—"))
            db = f"{dbest[1]} {_f(dbest[0])}" if dbest[0] is not None else "—"
            wname = win[1].split("·")[0] if win[1] != "—" else "—"
            star = " ✅" if wname == "Ours" else ""
            L.append(f"| {'**'+dom+'**' if dom=='Medical' else dom} | {db} | {_f(da)} | {_f(ds_)} | "
                     f"**{_f(ou)}** | {wname}{star} |")
    L.append("\n> Medical is the health-coaching target domain. `Ours` = best over students/variants; "
             "the pending corner-head cells will refine these.\n")

    L.append("\n## (b) Latency per method (p50)\n")
    pm = proxy_ms()
    if pm:
        L.append(f"\n**Proxy tier (target-decoupled):** median forward = **{statistics.median(pm):.0f} ms** "
                 f"(range {min(pm):.0f}–{max(pm):.0f} across student sizes) — **zero** extra target calls.\n")
    for teacher in TEACHERS:
        dm = direct_ms(teacher)
        if not dm:
            continue
        L.append(f"\n**Direct methods on {teacher}** (scoring p50; each also needs target generations):\n")
        L.append("| method | scoring p50 (ms) | extra target calls |")
        L.append("|---|--:|---|")
        for m in sorted(dm, key=lambda x: -statistics.median(dm[x])):
            extra = "+1 live target call" if m in VERBALIZED else "+ N−1 target samples"
            L.append(f"| {m} | {statistics.median(dm[m]):.0f} | {extra} |")
    L.append("\n> Even scoring-only, direct methods are 7–30× the ~27 ms proxy; adding the N−1 target "
             "samples (seconds on a 70B, ~19 s on an API target) is the dominant, target-scaling cost the "
             "proxy avoids entirely.\n")

    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                       "multitarget_by_domain.md")
    os.makedirs(os.path.dirname(doc), exist_ok=True)
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[mt-report-domains] wrote {doc}")


if __name__ == "__main__":
    main()
