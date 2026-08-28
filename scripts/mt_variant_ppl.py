#!/usr/bin/env python
"""Rank every Ours variant by the Perplexity read-out (and its own fair best-per-ds), per teacher.
Answers: under the best read-out (Perplexity), which Ours config is actually best? Reads all
est_ours_*_<teacher>_<student>_seed0.json in results_mt_estimators (run mt_est_variants.sh first)."""
import glob
import json
import os

import numpy as np

WORK = os.path.expanduser("~/JasonLucas")
EST = f"{WORK}/outputs/results_mt_estimators"
ORDER = ["trivia_qa", "bioasq", "medqa", "medlfqa", "gsm8k", "truthful_qa", "wikipedia_factual"]
MEDADV = {"bioasq", "medqa", "medlfqa", "truthful_qa"}
READOUTS = ["EDL-AU", "EDL-EU", "MSP", "Entropy", "Energy", "LogTokU", "Perplexity", "MaxNLL", "LogitMargin"]
CELLS = [("qwen3-32b", "qwen3-4b"), ("llama3.3-70b", "llama3.2-3b")]


def variant(tag, teacher, student):
    core = tag[len("ours_"):]
    suf = f"{teacher}_{student}"
    if core == suf:
        return "base(ecc_lam5)"
    return core[:-(len(suf) + 1)] if core.endswith("_" + suf) else core


def load(f):
    return {c["dataset"]: c.get("rows", {}) for c in json.load(open(f))["cells"]}


def mean(per_ds, readout, keys=None):
    ks = keys if keys is not None else ORDER
    v = [per_ds[ds].get(readout) for ds in ks if ds in per_ds and isinstance(per_ds[ds].get(readout), (int, float))]
    return float(np.mean(v)) if v else float("nan")


def fair_mean(per_ds, keys=None):
    ks = keys if keys is not None else ORDER
    vals = []
    for ds in ks:
        if ds not in per_ds:
            continue
        c = [per_ds[ds].get(r) for r in READOUTS if isinstance(per_ds[ds].get(r), (int, float))]
        if c:
            vals.append(max(c))
    return float(np.mean(vals)) if vals else float("nan")


for teacher, student in CELLS:
    rows = []
    for f in glob.glob(f"{EST}/est_ours*_{teacher}_{student}_seed0.json"):
        tag = os.path.basename(f)[len("est_"):-len("_seed0.json")]
        per = load(f)
        if not any("Perplexity" in r for r in per.values()):
            continue  # stale file (no Perplexity) -> skip
        v = variant(tag, teacher, student)
        rows.append((mean(per, "Perplexity"), mean(per, "Perplexity", MEDADV), fair_mean(per), v, per))
    rows.sort(reverse=True)
    print(f"\n===== {teacher} -> {student} : Ours variants ranked by Perplexity (mean-7) =====")
    print(f"{'variant':<20}{'Ppl(7)':>9}{'Ppl Med/Adv':>13}{'fair-best':>11}")
    for p7, pma, fair, v, _ in rows:
        star = "  <== BEST" if (p7, pma, fair, v, _) == rows[0] else (" [BASE]" if v.startswith("base") else "")
        print(f"{v:<20}{p7:>9.3f}{pma:>13.3f}{fair:>11.3f}{star}")
    if rows:
        best = rows[0]
        print(f"\nbest variant under Perplexity: {best[3]}  (Ppl-7 {best[0]:.3f}, Med/Adv {best[1]:.3f})")
        print(f"{'per-dataset Perplexity ->':<20}" + "".join(f"{d[:9]:>10}" for d in ORDER))
        per = best[4]
        print(f"{best[3][:20]:<20}" + "".join(
            (f"{per[d].get('Perplexity'):>10.3f}" if d in per and isinstance(per[d].get('Perplexity'), (int, float)) else f"{'—':>10}")
            for d in ORDER))
