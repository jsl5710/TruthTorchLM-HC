#!/usr/bin/env python
"""RQ2 report — Uncertainty Estimator Evaluation: does EDL beat other logit-based estimators when
all are applied to the SAME proxy logits?

The DisAAD paper compares EDL-on-proxy-logits against estimators computed on the *target's* logits,
which confounds the estimator with the input representation. RQ2 removes that confound: hold the
input fixed (the distilled 0.6B **proxy** logits, from rq1_preservation.py) and vary only the
estimator --

    EDL-AU, EDL-EU (evidential) · MSP · Predictive Entropy · Energy · LogTokU

-- then compare discrimination (AUROC/AUPR at flagging wrong answers) and calibration (cross-fit
ECE). This isolates the estimator's contribution. Writes docs/rq2_estimator_evaluation.md.

    python scripts/rq2_report.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from calibrate_eval import cross_fit_calibrate, _calib_metrics  # noqa: E402

HOME = os.path.expanduser("~/JasonLucas/outputs")
ESTIMATORS = [("au", "EDL-AU", "evidential"), ("eu", "EDL-EU", "evidential"),
              ("msp", "MSP", "softmax"), ("entropy", "Entropy", "softmax"),
              ("energy", "Energy", "energy"), ("logtoku", "LogTokU", "evidential*")]
GROUPS = [("Health · MCQ", {"medqa", "mmlu_med"}),
          ("Health · Free-form QA", {"kqa", "medlfqa", "bioasq"}),
          ("General · QA", {"trivia_qa", "natural_qa", "pop_qa", "truthful_qa"})]


def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(HOME, "results_rq1", "rq1_unc_*_seed*.json"))):
        recs += json.load(open(f)).get("records", [])
    return recs


def _finite(*a):
    m = np.ones(len(a[0]), bool)
    for x in a:
        m &= np.isfinite(x)
    return [x[m] for x in a]


def metrics(recs, est):
    from sklearn.metrics import roc_auc_score, average_precision_score
    unc = np.array([r["proxy"][est] for r in recs], float)
    y = np.array([r["correct"] for r in recs], float)
    unc, y = _finite(unc, y)
    if len(set(y.tolist())) < 2:
        return None
    inc = 1 - y
    probs, _ = cross_fit_calibrate(-unc, y, seed=0)
    return {"auroc": roc_auc_score(inc, unc), "aupr": average_precision_score(inc, unc),
            "ece": _calib_metrics(probs, y)["ece"], "n": len(y), "npos": int(inc.sum())}


def fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def table(L, recs, title):
    L.append(f"\n### {title}\n")
    L.append("| Estimator | Family | AUROC | AUPR | ECE |")
    L.append("|---|---|--:|--:|--:|")
    rows = []
    for est, lab, fam in ESTIMATORS:
        m = metrics(recs, est)
        if m:
            rows.append((m["auroc"], lab, fam, m))
    rows.sort(reverse=True)
    for au, lab, fam, m in rows:
        star = " ✦" if fam == "evidential" else ""
        L.append(f"| {lab}{star} | {fam} | {fmt(m['auroc'])} | {fmt(m['aupr'])} | {fmt(m['ece'])} |")
    return {lab: m for _a, lab, _f, m in rows}


def main():
    recs = load_records()
    if not recs:
        print("No RQ1 records in results_rq1/ — run rq1_preservation.py first (RQ2 reuses them)."); return

    L = []
    L.append("# RQ2 — Uncertainty Estimator Evaluation\n")
    L.append("> **RQ2.** Does Evidential Deep Learning (EDL) provide superior uncertainty estimation "
             "compared with established logit-based estimators when applied to the **same proxy logits**?\n")
    L.append("**Auto-generated** by `scripts/rq2_report.py` (reuses the proxy-side logits from "
             "`rq1_preservation.py`).\n")
    L.append("## Setup — isolate the estimator, hold the input fixed\n")
    L.append("- **Fixed input:** the distilled **qwen3-0.6b proxy** logits over each cached "
             "(prompt + response) — identical for every estimator, so any difference is the "
             "*estimator's* contribution, not the representation.\n"
             "- **Estimators:** EDL-AU, EDL-EU (evidential, top-k Dirichlet) · MSP (1−max softmax) · "
             "predictive Entropy · Energy (−logsumexp) · LogTokU (full-vocab evidential*). All oriented "
             "higher = more uncertain, mean-aggregated over the response span.\n"
             "- **Quality:** AUROC & AUPR at flagging wrong answers (task correctness = LLM-judge / MCQ "
             "label); calibration = cross-fitted isotonic ECE. `✦` marks the EDL family.\n"
             "- *LogTokU is a best-effort implementation of Ma et al. 2025 (full-vocabulary evidential "
             "epistemic); flagged `*` as 'where applicable'.\n"
             f"- Pooled over 3 seeds; {len(recs)} proxy responses.\n")

    overall = table(L, recs, "All datasets (proxy logits)")
    for gname, dss in GROUPS:
        sub = [r for r in recs if r["dataset"] in dss]
        if sub:
            table(L, sub, gname)

    # headline
    L.append("\n## Finding — is EDL superior?\n")
    if overall:
        edl = [overall[k]["auroc"] for k in ("EDL-AU", "EDL-EU") if k in overall]
        others = {k: overall[k]["auroc"] for k in overall if k not in ("EDL-AU", "EDL-EU")}
        best_edl = max(edl) if edl else None
        best_other = max(others.values()) if others else None
        best_other_name = max(others, key=others.get) if others else "—"
        if best_edl is not None and best_other is not None:
            if best_edl >= best_other:
                L.append(f"- On the same proxy logits, the best EDL measure (AUROC **{best_edl:.3f}**) "
                         f"**≥** the best non-EDL estimator ({best_other_name} {best_other:.3f}) overall "
                         "— EDL's evidential view adds signal beyond softmax/energy.\n")
            else:
                L.append(f"- On the same proxy logits, the best non-EDL estimator (**{best_other_name} "
                         f"{best_other:.3f}**) **beats** the best EDL measure ({best_edl:.3f}) overall "
                         "— EDL is **not** superior once the input representation is controlled; its "
                         "reported edge in the paper may be an input-confound, not an estimator win.\n")
        L.append("- Compare AU vs EU: they read different failure modes (data-ambiguity vs "
                 "evidence-scarcity); the winner shifts by task type (see the per-group tables).\n")
    L.append("- ECE columns show which estimator is best *calibrated* on the proxy logits, independent "
             "of ranking ability.\n")

    doc = os.path.join(os.path.dirname(_HERE), "docs", "rq2_estimator_evaluation.md")
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[rq2_report] wrote {doc}")


if __name__ == "__main__":
    main()
