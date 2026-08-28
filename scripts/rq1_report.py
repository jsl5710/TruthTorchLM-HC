#!/usr/bin/env python
"""RQ1 report — does the proxy preserve the teacher's uncertainty behavior, ID vs OOD?

Consumes the per-item teacher/proxy uncertainty records from rq1_preservation.py and measures,
for each estimator (AU/EU/MSP/Entropy) and each split:

  * Agreement   — Spearman rho, Pearson r, Kendall tau between teacher and proxy uncertainty
                  (does the proxy's uncertainty *track* the teacher's, response by response?)
  * Discrimination — AUROC of teacher vs proxy uncertainty at flagging wrong answers, and the
                  proxy-minus-teacher gap (does the proxy discriminate worse, and more so OOD?)
  * Calibration — cross-fitted ECE of teacher vs proxy (does the proxy's calibration collapse OOD?)

Split: **ID = TriviaQA** (the distillation domain) vs **OOD = everything else** (other QA + all
health). Writes docs/rq1_uncertainty_preservation.md.

    python scripts/rq1_report.py
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
ID_DATASETS = {"trivia_qa"}
ESTIMATORS = ["au", "eu", "msp", "entropy"]
EST_LABEL = {"au": "EDL-AU", "eu": "EDL-EU", "msp": "MSP", "entropy": "Entropy"}


def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(HOME, "results_rq1", "rq1_unc_*_seed*.json"))):
        recs += json.load(open(f)).get("records", [])
    return recs


def split_of(ds):
    return "ID (TriviaQA)" if ds in ID_DATASETS else "OOD (rest)"


def vectors(recs, est):
    t = np.array([r["teacher"][est] for r in recs], float)
    p = np.array([r["proxy"][est] for r in recs], float)
    y = np.array([r["correct"] for r in recs], float)
    return t, p, y


def _finite(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= np.isfinite(a)
    return [a[m] for a in arrs]


def agreement(t, p):
    from scipy.stats import spearmanr, pearsonr, kendalltau
    t, p = _finite(t, p)
    if len(t) < 3 or np.std(t) == 0 or np.std(p) == 0:
        return None
    return {"spearman": spearmanr(t, p).correlation,
            "pearson": pearsonr(t, p)[0],
            "kendall": kendalltau(t, p).correlation}


def auroc(unc, y):
    from sklearn.metrics import roc_auc_score
    unc, y = _finite(unc, y)
    if len(set(y.tolist())) < 2:
        return None
    return roc_auc_score(1 - y, unc)   # uncertainty flags INCORRECT (y=1 correct -> 1-y=incorrect)


def ece(unc, y, seed=0):
    unc, y = _finite(unc, y)
    if len(set(y.tolist())) < 2:
        return None
    probs, _ = cross_fit_calibrate(-unc, y, seed=seed)   # -uncertainty = confidence -> P(correct)
    return _calib_metrics(probs, y)["ece"]


def fmt(x, p=3):
    return f"{x:+.{p}f}" if isinstance(x, (int, float)) and p == 3 and False else \
        (f"{x:.{p}f}" if isinstance(x, (int, float)) else "—")


def main():
    recs = load_records()
    if not recs:
        print("No RQ1 records in results_rq1/ — run rq1_preservation.py first."); return
    by_split = defaultdict(list)
    for r in recs:
        by_split[split_of(r["dataset"])].append(r)
    splits = ["ID (TriviaQA)", "OOD (rest)"]
    nseeds = len({(r.get("item_id"), r["dataset"]) for r in recs})

    L = []
    L.append("# RQ1 — Proxy Uncertainty Preservation\n")
    L.append("> **RQ1.** Does the distilled proxy preserve the uncertainty behavior of the target "
             "black-box LLM under both in-distribution (ID) and out-of-distribution (OOD) conditions?\n")
    L.append("**Auto-generated** by `scripts/rq1_report.py`.\n")
    L.append("## Setup\n")
    L.append("- **Teacher** = `qwen3-8b` (the distillation target); **proxy** = the distilled "
             "`qwen3-0.6b` (merged). The **same four logit-based estimators** are computed on **both** "
             "models' logits over each cached (prompt + response): **EDL-AU, EDL-EU, MSP, Entropy** "
             "(one forward pass each, mean over the response span, higher = more uncertain).\n"
             "- **Split:** ID = **TriviaQA** (the distillation domain, `tqa`); OOD = **everything else** "
             "(NaturalQA/PopQA/TruthfulQA + all health). If distillation transferred *uncertainty* "
             "(not just outputs), teacher↔proxy agreement should hold ID and — the hypothesis under "
             "test — **degrade OOD**.\n"
             f"- Pooled over 3 seeds; {len(recs)} teacher/proxy response pairs total "
             f"({len(by_split[splits[0]])} ID, {len(by_split[splits[1]])} OOD).\n")

    # --- agreement ---
    L.append("\n## Agreement — does the proxy's uncertainty track the teacher's?\n")
    L.append("| Estimator | Split | Spearman ρ | Pearson r | Kendall τ |")
    L.append("|---|---|--:|--:|--:|")
    for est in ESTIMATORS:
        for sp in splits:
            t, p, _ = vectors(by_split[sp], est)
            a = agreement(t, p)
            if a:
                L.append(f"| {EST_LABEL[est]} | {sp} | {fmt(a['spearman'])} | {fmt(a['pearson'])} | {fmt(a['kendall'])} |")

    # --- discrimination + calibration ---
    L.append("\n## Discrimination & calibration — does the proxy degrade vs the teacher?\n")
    L.append("| Estimator | Split | Teacher AUROC | Proxy AUROC | Δ(proxy−teacher) | Teacher ECE | Proxy ECE |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    for est in ESTIMATORS:
        for sp in splits:
            t, p, y = vectors(by_split[sp], est)
            ta, pa = auroc(t, y), auroc(p, y)
            d = (pa - ta) if (ta is not None and pa is not None) else None
            L.append(f"| {EST_LABEL[est]} | {sp} | {fmt(ta)} | {fmt(pa)} | "
                     f"{('%+.3f' % d) if isinstance(d,(int,float)) else '—'} | "
                     f"{fmt(ece(t,y))} | {fmt(ece(p,y))} |")

    # --- per-OOD-dataset agreement (which shifts break preservation) ---
    L.append("\n## OOD detail — teacher↔proxy agreement by dataset (EDL-AU)\n")
    by_ds = defaultdict(list)
    for r in recs:
        by_ds[r["dataset"]].append(r)
    L.append("| Dataset | Split | n | Spearman ρ (AU) | Proxy AUROC (AU) | Teacher AUROC (AU) |")
    L.append("|---|---|--:|--:|--:|--:|")
    for ds in sorted(by_ds, key=lambda d: (d not in ID_DATASETS, d)):
        t, p, y = vectors(by_ds[ds], "au")
        a = agreement(t, p)
        L.append(f"| {ds} | {'ID' if ds in ID_DATASETS else 'OOD'} | {len(by_ds[ds])} | "
                 f"{fmt(a['spearman']) if a else '—'} | {fmt(auroc(p,y))} | {fmt(auroc(t,y))} |")

    # --- headline ---
    def sp_mean(sp, metric):
        vals = []
        for est in ESTIMATORS:
            t, p, _ = vectors(by_split[sp], est)
            a = agreement(t, p)
            if a:
                vals.append(a[metric])
        return float(np.mean(vals)) if vals else None
    id_sp, ood_sp = sp_mean(splits[0], "spearman"), sp_mean(splits[1], "spearman")
    L.append("\n## Finding — is uncertainty preserved?\n")
    if id_sp is not None and ood_sp is not None:
        drop = id_sp - ood_sp
        L.append(f"- **Mean teacher↔proxy Spearman agreement:** ID = **{id_sp:.3f}**, OOD = **{ood_sp:.3f}** "
                 f"(Δ = {drop:+.3f}). "
                 + ("Agreement **holds ID and collapses OOD** — the proxy preserves the teacher's "
                    "*outputs* but not its *uncertainty* off-distribution, confirming the RQ1 concern."
                    if drop > 0.1 else
                    "Agreement is **similar ID and OOD** — no evidence of an OOD preservation collapse "
                    "on this axis." if abs(drop) <= 0.1 else
                    "Agreement is *higher* OOD than ID — unexpected; inspect per-dataset detail.") + "\n")
    L.append("- **Read the Δ(proxy−teacher) AUROC column**: a large negative Δ that worsens OOD is the "
             "'silent proxy degradation' RQ1 predicts — the proxy's uncertainty discriminates errors "
             "well ID but loses the teacher's signal OOD.\n")

    doc = os.path.join(os.path.dirname(_HERE), "docs", "rq1_uncertainty_preservation.md")
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[rq1_report] wrote {doc}")


if __name__ == "__main__":
    main()
