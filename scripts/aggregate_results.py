#!/usr/bin/env python
"""Aggregate Stage C->D result files across seeds and generators into the final table.

Reads every stage_cd_<generator>_seed<s>.json in a results dir (each = one generator, one
seed), aggregates AUROC / PRR / latency across seeds (mean +/- std), and prints the G-axis
comparison: for a dataset + method + N, AUROC per generator. Writes aggregated.json.

    python scripts/aggregate_results.py --results-root ~/JasonLucas/outputs/results_full \
        --dataset trivia_qa --n 10
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def main():
    ap = argparse.ArgumentParser(description="Aggregate G-axis results across seeds.")
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_full"))
    ap.add_argument("--dataset", default=None, help="print the per-generator table for this dataset")
    ap.add_argument("--n", type=int, default=10, help="N (sample budget) for the printed table")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc", "prr", "p50_ms"])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_root, "stage_cd_*_seed*.json")))
    if not files:
        print(f"No result files in {args.results_root}"); return

    # acc[(gen, dataset, method_N)][metric] -> list over seeds
    acc = defaultdict(lambda: defaultdict(list))
    meta = defaultdict(dict)  # (gen,dataset) -> {n_items, n_positive}
    seeds_seen = defaultdict(set)
    for f in files:
        d = json.load(open(f))
        gen = d["generator_key"]; seed = d.get("seed")
        for c in d.get("cells", []):
            ds = c.get("dataset")
            if "error" in c:
                continue
            meta[(gen, ds)] = {"n_items": c.get("n_items"), "n_positive": c.get("n_positive")}
            seeds_seen[(gen, ds)].add(seed)
            for mkey, row in c.get("rows", {}).items():
                for met in ("auroc", "auprc", "prr", "p50_ms"):
                    if row.get(met) is not None:
                        acc[(gen, ds, mkey)][met].append(row[met])

    # aggregated json
    agg = {}
    for (gen, ds, mkey), mets in acc.items():
        agg.setdefault(gen, {}).setdefault(ds, {})[mkey] = {
            met: dict(zip(("mean", "std", "n_seeds"), _mean_std(vals)))
            for met, vals in mets.items()
        }
    outp = os.path.join(args.results_root, "aggregated.json")
    json.dump(agg, open(outp, "w"), indent=2)
    print(f"Wrote {outp}\n")

    # summary: which (gen, dataset) cells exist and how many seeds
    print("=== coverage (generator x dataset : seeds, items, positives) ===")
    for (gen, ds) in sorted(meta):
        m = meta[(gen, ds)]
        print(f"  {gen:16s} {ds:14s} seeds={sorted(seeds_seen[(gen,ds)])} "
              f"n={m['n_items']} pos={m['n_positive']}")

    # G-axis table for one dataset + N
    if args.dataset:
        print(f"\n=== {args.metric.upper()} on {args.dataset} at N={args.n} "
              f"(mean +/- std over seeds) ===")
        gens = sorted({g for (g, ds) in meta if ds == args.dataset})
        methods = sorted({mkey.rsplit("_N", 1)[0]
                          for (g, ds, mkey) in acc if ds == args.dataset
                          and mkey.endswith(f"_N{args.n}")})
        header = "method".ljust(28) + "".join(g[:14].ljust(16) for g in gens)
        print(header)
        for meth in methods:
            row = meth.ljust(28)
            for g in gens:
                cell = acc.get((g, args.dataset, f"{meth}_N{args.n}"), {})
                m, sd, ns = _mean_std(cell.get(args.metric, []))
                row += (f"{m:.3f}±{sd:.2f}" if m is not None else "  -  ").ljust(16)
            print(row)


if __name__ == "__main__":
    main()
