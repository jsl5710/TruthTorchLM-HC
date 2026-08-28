#!/usr/bin/env python
"""FAIR proxy comparison: every proxy (Ours / DALD / DisAAD) read with the SAME best read-out
(Perplexity), plus each proxy's own fair best-per-dataset read-out, vs VerbalizedConfidence (direct).
Answers: is Ours a better *proxy*, or was the earlier gap just a read-out artifact (Ours got Perplexity,
DALD/DisAAD their weak native LogTokU-au)?

Needs est_{ours,dald,disaad}_<teacher>_<student>_seed0.json in results_mt_estimators (run
mt_est_bbgb.sh + mt_est_proxies.sh first).
"""
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


def load(tag):
    fs = glob.glob(f"{EST}/est_{tag}_seed0.json")
    if not fs:
        return None
    return {c["dataset"]: c.get("rows", {}) for c in json.load(open(fs[0]))["cells"]}


def vc(teacher):
    f = f"{WORK}/outputs/results_mt/stage_cd_{teacher}_seed0.json"
    return {c["dataset"]: c.get("rows", {}).get("VerbalizedConfidence_N1", {}).get("auroc")
            for c in json.load(open(f))["cells"]}


def col(per_ds, readout):    # dataset -> auroc for one read-out
    return {ds: per_ds[ds].get(readout) for ds in per_ds} if per_ds else {}


def fair_col(per_ds):        # dataset -> best single read-out (that proxy's fair best-per-ds)
    out = {}
    if not per_ds:
        return out
    for ds in per_ds:
        c = [per_ds[ds].get(r) for r in READOUTS if isinstance(per_ds[ds].get(r), (int, float))]
        out[ds] = max(c) if c else None
    return out


def mean(d, keys=None):
    ks = keys if keys is not None else list(d)
    v = [d[k] for k in ks if isinstance(d.get(k), (int, float))]
    return float(np.mean(v)) if v else float("nan")


for teacher, student in CELLS:
    ours = load(f"ours_{teacher}_{student}")
    dald = load(f"dald_{teacher}_{student}")
    disa = load(f"disaad_{teacher}_{student}")
    verb = vc(teacher)
    print(f"\n===== {teacher} -> {student}  (all proxies read via Perplexity; 'fair' = each proxy's best-per-ds) =====")
    for lbl, x in [("dald", dald), ("disaad", disa)]:
        if x is None:
            print(f"  !! missing est_{lbl}_{teacher}_{student} -- run mt_est_proxies.sh")
    cols = {
        "Ours·Ppl": col(ours, "Perplexity"), "DALD·Ppl": col(dald, "Perplexity"),
        "DisAAD·Ppl": col(disa, "Perplexity"),
        "Ours·fair": fair_col(ours), "DALD·fair": fair_col(dald), "DisAAD·fair": fair_col(disa),
        "Verbal": verb,
    }
    names = list(cols)
    hdr = f"{'dataset':<19}{'dom':<5}" + "".join(f"{n:>11}" for n in names)
    print(hdr)
    for ds in ORDER:
        dom = "M/A" if ds in MEDADV else "rest"
        cells = "".join(f"{(cols[n].get(ds)):>11.3f}" if isinstance(cols[n].get(ds), (int, float))
                        else f"{'—':>11}" for n in names)
        print(f"{ds:<19}{dom:<5}{cells}")
    print("-" * len(hdr))
    print(f"{'MEAN(7)':<24}" + "".join(f"{mean(cols[n]):>11.3f}" for n in names))
    print(f"{'MEAN Med/Adv':<24}" + "".join(f"{mean(cols[n], MEDADV):>11.3f}" for n in names))
