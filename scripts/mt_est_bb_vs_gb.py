#!/usr/bin/env python
"""Compare fast logit read-outs BLACK-BOX (our proxy) vs GREY-BOX (target logits) + PTrue.

Reads results_mt_estimators/est_ours_<teacher>_<student>_seed0.json (proxy, 9 read-outs) and
est_greybox_<teacher>_seed0.json (target, 9 read-outs), plus PTrue from results_mt. Prints, per
teacher: each read-out's mean AUROC (proxy vs target), the fair best-per-dataset read-out, and the
grey-box lift -- i.e. what target-logit access buys over our proxy.
"""
import glob
import json
import os

import numpy as np

WORK = os.path.expanduser("~/JasonLucas")
EST = f"{WORK}/outputs/results_mt_estimators"
ORDER = ["trivia_qa", "bioasq", "medqa", "medlfqa", "gsm8k", "truthful_qa", "wikipedia_factual"]
READOUTS = ["EDL-AU", "EDL-EU", "MSP", "Entropy", "Energy", "LogTokU", "Perplexity", "MaxNLL", "LogitMargin"]
NEW = {"Perplexity", "MaxNLL", "LogitMargin"}
# (teacher, base-proxy tag, best-variant-proxy tag)
CELLS = [
    ("qwen3-32b", "ours_qwen3-32b_qwen3-4b", "ours_head_dse_lam5_qwen3-32b_qwen3-4b"),
    ("llama3.3-70b", "ours_llama3.3-70b_llama3.2-3b", "ours_ecc_lam10_llama3.3-70b_llama3.2-3b"),
]


def load_est(tagfile):
    fs = glob.glob(f"{EST}/est_{tagfile}_seed0.json")
    if not fs:
        return None
    d = json.load(open(fs[0]))
    out = {}  # dataset -> {readout: auroc}
    for c in d.get("cells", []):
        if c.get("dataset"):
            out[c["dataset"]] = c.get("rows", {})
    return out


def ptrue(teacher):
    f = f"{WORK}/outputs/results_mt/stage_cd_{teacher}_seed0.json"
    out = {}
    if os.path.exists(f):
        for c in json.load(open(f))["cells"]:
            r = c.get("rows", {}).get("PTrue_N1", {})
            if r.get("auroc") is not None:
                out[c["dataset"]] = r["auroc"]
    return out


def mean_over_ds(per_ds, readout):
    vals = [per_ds[ds].get(readout) for ds in per_ds if per_ds[ds].get(readout) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def fair_best(per_ds):
    """mean over datasets of the best single read-out per dataset (oracle read-out ceiling)."""
    vals = []
    for ds in per_ds:
        cand = [per_ds[ds].get(r) for r in READOUTS if per_ds[ds].get(r) is not None]
        if cand:
            vals.append(max(cand))
    return float(np.mean(vals)) if vals else float("nan")


for teacher, base_cell, best_cell in CELLS:
    bb = load_est(base_cell)              # black-box: Ours-base proxy
    bv = load_est(best_cell)              # black-box: best-performing Ours variant
    gb = load_est(f"greybox_{teacher}")   # grey-box: target logits
    pt = ptrue(teacher)
    print(f"\n===== {teacher} =====")
    print(f"  BB-base = {base_cell}")
    print(f"  BB-best = {best_cell}")
    for lbl, x in [("BB-base", bb), ("BB-best", bv), ("GB-target", gb)]:
        if x is None:
            print(f"  !! no results yet for {lbl}")
    print(f"\n{'read-out':<14}{'BB-base':>10}{'BB-best':>10}{'GB-target':>11}   {'new?':<4}")
    rows = []
    for r in READOUTS:
        b = mean_over_ds(bb, r) if bb else float("nan")
        v = mean_over_ds(bv, r) if bv else float("nan")
        g = mean_over_ds(gb, r) if gb else float("nan")
        rows.append((r, b, v, g))
    for r, b, v, g in rows:
        print(f"{r:<14}{b:>10.3f}{v:>10.3f}{g:>11.3f}   {'NEW' if r in NEW else '':<4}")
    print("-" * 49)

    def summarize(name, per_ds, col):
        if not per_ds:
            return
        star = max(rows, key=lambda t: (t[col] if not np.isnan(t[col]) else -1))
        tail = "  <-- NEW likelihood read-out wins!" if star[0] in NEW else ""
        print(f"{name:<10} best single: {star[0]} = {star[col]:.3f}{tail}   |  fair best-per-ds: {fair_best(per_ds):.3f}")
    summarize("BB-base", bb, 1)
    summarize("BB-best", bv, 2)
    summarize("GB-target", gb, 3)
    if pt:
        print(f"{'PTrue':<10} (grey-box): {np.mean(list(pt.values())):.3f}")
    if bb and gb:
        print(f"grey-box lift (fair, vs BB-base): {fair_best(gb) - fair_best(bb):+.3f}")
