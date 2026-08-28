#!/usr/bin/env python
"""Post-completion calibration pass: fit a normalizer and compute MEANINGFUL ECE/MCE/ACE/Brier.

The main runs save discrimination metrics (AUROC/AUPRC/PRR) on RAW uncertainty scores, which
are not probabilities -- so calibration metrics on them are meaningless. This re-scores each
cached cell, fits an isotonic normalizer mapping raw score -> P(correct), and -- crucially --
does it with **k-fold cross-fitting** (fit on train folds, predict the held-out fold) so ECE
is measured out-of-fold, not on the data the normalizer saw. Run AFTER all datasets/models
finish. Generations are cached, so this re-scores (NLI) but never re-generates.

    python scripts/calibrate_eval.py --cache-root ~/JasonLucas/outputs/cache_full \
        --results-root ~/JasonLucas/outputs/results_full --generator qwen3-8b --seed 0
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from stage_cd_open import build_bb_methods  # noqa: E402


class _CacheView:
    """Read a Stage-A parquet directly (json-decoding samples/ground_truths) so score_stage_c
    can consume it without reconstructing the BenchmarkConfig."""

    def __init__(self, parquet):
        import pandas as pd
        df = pd.read_parquet(parquet)
        self.path = Path(parquet)
        self._items = []
        for _, r in df.iterrows():
            it = dict(r)
            for f in ("ground_truths", "samples"):
                if isinstance(it.get(f), str):
                    it[f] = json.loads(it[f])
            self._items.append(it)

    def read(self, n=None):
        if n is None:
            return self._items
        return [{**it, "samples": list(it["samples"])[:n]} for it in self._items]

    def exists(self):
        return True

    def __len__(self):
        return len(self._items)


def _labels_for(parquet):
    p = Path(parquet).with_suffix(".labels.json")
    return json.loads(p.read_text())["labels"] if p.exists() else None


def cross_fit_calibrate(truth_values, correctness, k=5, seed=0):
    """Out-of-fold isotonic calibration: raw score -> P(correct), never predicting a point
    the fold was fit on. Returns calibrated probs aligned to the inputs."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    tv = np.asarray(truth_values, float)
    y = np.asarray(correctness, float)
    oof = np.full(len(tv), np.nan)
    n_splits = min(k, int(y.sum()), int((1 - y).sum()))  # need both classes per fold
    if n_splits < 2:
        # too few of one class to cross-fit; fit on all (optimistic, flagged by caller)
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(tv, y)
        return ir.predict(tv), False
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(tv, y):
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(tv[tr], y[tr])
        oof[te] = ir.predict(tv[te])
    return oof, True


def _calib_metrics(probs, y, bins=10):
    probs = np.asarray(probs, float); y = np.asarray(y, float)
    brier = float(np.mean((probs - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = mce = 0.0
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1.0)
        if not m.any():
            continue
        gap = abs(probs[m].mean() - y[m].mean())
        ece += (m.mean()) * gap
        mce = max(mce, gap)
    return {"ece": round(float(ece), 4), "mce": round(float(mce), 4), "brier": round(brier, 4)}


def main():
    ap = argparse.ArgumentParser(description="Calibration re-eval with cross-fitted normalizer.")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--generator", required=True, help="generator key (matches cache filenames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sweep", type=int, nargs="+", default=[10])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from hc_benchmark.stage_c_score import score_stage_c
    from sklearn.metrics import roc_auc_score
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_root, exist_ok=True)

    parquets = sorted(glob.glob(os.path.join(
        args.cache_root, f"stageA_*_{args.generator}_*_seed{args.seed}.parquet")))
    if not parquets:
        print(f"No caches for {args.generator} in {args.cache_root}"); return

    # methods sized to the max samples available (all cells share n_max here)
    n_max = max(args.n_sweep)
    methods = build_bb_methods(n_max, device, include_verbalized=False)
    n_sweep = tuple(sorted(set(args.n_sweep)))

    cells = []
    for pq in parquets:
        ds = os.path.basename(pq).split("_")[1] if "_" in os.path.basename(pq) else "?"
        cache = _CacheView(pq)
        labels = _labels_for(pq)
        if labels is None:
            print(f"[{ds}] no labels -- skip"); continue
        items = cache.read()
        corr = [labels.get(str(it["item_id"]), {}).get("correct_llm_judge") for it in items]
        keep = [i for i, c in enumerate(corr) if c in (0, 1)]
        y = [corr[i] for i in keep]
        print(f"\n[{ds}] {len(keep)} scorable ({sum(y)} correct) -- re-scoring for calibration ...")
        scores = score_stage_c(cache, methods, model=args.generator, n_sweep=n_sweep,
                               collect_timing=False)
        rows = {}
        for method in methods:
            name = type(method).__name__
            for n in n_sweep:
                tv = [scores[name][n]["truth_values"][i] for i in keep]
                if len(set(y)) < 2:
                    continue
                probs, honest = cross_fit_calibrate(tv, y, seed=args.seed)
                cm = _calib_metrics(probs, y)
                try:
                    au = round(float(roc_auc_score(y, tv)), 4)
                except Exception:
                    au = None
                rows[f"{name}_N{n}"] = {**cm, "auroc": au, "cross_fitted": honest}
                if n == n_max:
                    print(f"  {name:26s} N={n}: ECE={cm['ece']} MCE={cm['mce']} "
                          f"Brier={cm['brier']} {'(oof)' if honest else '(in-sample!)'}")
        cells.append({"dataset": ds, "n_items": len(keep), "n_positive": sum(y), "rows": rows})

    out = os.path.join(args.results_root, f"calibration_{args.generator}_seed{args.seed}.json")
    json.dump({"generator": args.generator, "seed": args.seed, "cells": cells},
              open(out, "w"), indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
