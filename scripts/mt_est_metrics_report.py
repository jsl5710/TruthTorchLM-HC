#!/usr/bin/env python
"""Read the multi-metric `metrics` field (from the re-run mt_estimators sweep) and report the read-out
comparison on ranking AND calibration -- AUROC, AUPRC, and cross-fit-calibrated ECE/Brier -- for the
deployable proxies. Answers: does the Perplexity read-out calibrate better than the EDL/LogTokU default?
Run scripts/mt_est_metrics_rerun.sh first. Read-outs with no `metrics` field are skipped (stale files)."""
import glob
import json
import os

import numpy as np

WORK = os.path.expanduser("~/JasonLucas")
EST = f"{WORK}/outputs/results_mt_estimators"
KEY_READOUTS = ["EDL-AU", "LogTokU", "Entropy", "Perplexity", "MaxNLL"]
METS = ["auroc", "auprc", "ece", "brier"]
CELLS = [("qwen3-32b", "qwen3-4b"), ("llama3.3-70b", "llama3.2-3b")]


def load_metrics(tag):
    fs = glob.glob(f"{EST}/est_{tag}_seed0.json")
    if not fs:
        return None
    d = json.load(open(fs[0]))
    if not any("metrics" in c for c in d.get("cells", [])):
        return None  # stale (AUROC-only) file
    return [c for c in d["cells"] if c.get("metrics")]


def mean_metric(cells, readout, metric):
    vals = [c["metrics"].get(readout, {}).get(metric) for c in cells]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return np.mean(vals) if vals else float("nan")


any_found = False
for teacher, student in CELLS:
    for meth in ["ours", "dald", "disaad"]:
        cells = load_metrics(f"{meth}_{teacher}_{student}")
        if cells is None:
            print(f"[{meth}/{teacher}] no multi-metric file yet (run mt_est_metrics_rerun.sh)")
            continue
        any_found = True
        print(f"\n===== {teacher} :: {meth.upper()}  (mean over {len(cells)} datasets) =====")
        print(f"{'read-out':<13}{'AUROC':>8}{'AUPRC':>8}{'ECE':>8}{'Brier':>8}")
        for r in KEY_READOUTS:
            vals = [mean_metric(cells, r, m) for m in METS]
            if all(np.isnan(v) for v in vals):
                continue
            print(f"{r:<13}" + "".join(f"{v:8.3f}" if not np.isnan(v) else f"{'—':>8}" for v in vals))
if not any_found:
    print("\nNo multi-metric files found. Run:  scripts/mt_est_metrics_rerun.sh")
