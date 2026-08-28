#!/usr/bin/env python
"""Report: is the distilled proxy (qwen3-0.6b student) as ACCURATE as the teacher (qwen3-8b) on the
tasks, across datasets? Aggregates proxy_task_accuracy.py outputs, grouped Domain -> Task type.

    python scripts/proxy_accuracy_report.py
"""
import glob
import json
import os
from collections import defaultdict

HOME = os.path.expanduser("~/JasonLucas/outputs")
GROUPS = [("Health", "MCQ", ["medqa", "mmlu_med"]),
          ("Health", "Free-form QA", ["kqa", "medlfqa", "bioasq"]),
          ("General", "QA", ["trivia_qa", "natural_qa", "pop_qa", "truthful_qa"])]
LAB = {"medqa": "MedQA", "mmlu_med": "MMLU-Med", "kqa": "K-QA", "medlfqa": "MedLFQA",
       "bioasq": "BioASQ", "trivia_qa": "TriviaQA", "natural_qa": "NaturalQA",
       "pop_qa": "PopQA", "truthful_qa": "TruthfulQA"}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def load():
    acc = defaultdict(lambda: {"p": [], "t": []})
    for f in sorted(glob.glob(os.path.join(HOME, "results_proxy_acc", "proxy_acc_seed*.json"))):
        for r in json.load(open(f)).get("rows", []):
            acc[r["dataset"]]["p"].append(r.get("proxy_acc"))
            acc[r["dataset"]]["t"].append(r.get("teacher_acc"))
    return acc


def fmt(x, pct=True):
    if not isinstance(x, (int, float)):
        return "—"
    return f"{100*x:.1f}%" if pct else f"{x:+.1f}"


def main():
    acc = load()
    if not acc:
        print("No results_proxy_acc/*.json yet — run proxy_task_accuracy.py first."); return
    L = ["# RQ4 — Proxy Task Accuracy (companion to RQ1)\n",
         "> **RQ4.** Is the distilled **proxy** (student `qwen3-0.6b`) as **accurate on the tasks** as "
         "the **teacher** (`qwen3-8b`) it was distilled from — across QA, MCQ, and health datasets?\n",
         "**Linked to RQ1:** RQ1 asks whether the proxy preserves the teacher's *uncertainty*; RQ4 is "
         "the counterpart — whether it preserves the teacher's *task accuracy*. Read together they "
         "separate 'good uncertainty proxy' from 'good answerer'.\n",
         "Proxy run as a generator (greedy, thinking off), judged identically to the teacher "
         "(MCQMatch for MCQ, Qwen3-4B judge otherwise). Mean over 3 seeds. **Auto-generated** by "
         "`scripts/proxy_accuracy_report.py`.\n"]
    all_d = []
    for domain, task, dss in GROUPS:
        L.append(f"\n## {domain} — {task}\n")
        L.append("| Dataset | Proxy acc | Teacher acc | Δ (proxy−teacher, pts) |")
        L.append("|---|--:|--:|--:|")
        for ds in dss:
            p, t = _mean(acc[ds]["p"]), _mean(acc[ds]["t"])
            d = (100 * (p - t)) if (p is not None and t is not None) else None
            if d is not None:
                all_d.append(p - t)
            L.append(f"| {LAB.get(ds, ds)} | {fmt(p)} | {fmt(t)} | {fmt(d, pct=False)} |")

    md = _mean(all_d)
    # per-group means
    L.append("\n## Summary\n")
    for domain, task, dss in GROUPS:
        gp = _mean([v for ds in dss for v in acc[ds]["p"]])
        gt = _mean([v for ds in dss for v in acc[ds]["t"]])
        gd = (100 * (gp - gt)) if (gp is not None and gt is not None) else None
        L.append(f"- **{domain} · {task}:** proxy {fmt(gp)} vs teacher {fmt(gt)} "
                 f"(Δ {fmt(gd, pct=False)} pts)\n")
    if md is not None:
        verdict = ("**yes, essentially** — within ±2 pts on average" if abs(md) <= 0.02 else
                   f"**no** — the proxy is {abs(100*md):.1f} pts "
                   f"{'below' if md < 0 else 'above'} the teacher on average")
        L.append(f"\n**Overall:** mean proxy−teacher accuracy gap = **{100*md:+.1f} pts**. "
                 f"Is the 0.6B proxy as accurate as the 8B teacher? {verdict}. "
                 "Distillation transfers *outputs* on the distillation domain (TriviaQA), but this "
                 "measures whether task *accuracy* holds across all task types — the larger the gap, "
                 "the more the proxy is a good *uncertainty* proxy without being a good *answerer*.\n")

    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                       "rq4_proxy_task_accuracy.md")
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[proxy-acc-report] wrote {doc}")


if __name__ == "__main__":
    main()
