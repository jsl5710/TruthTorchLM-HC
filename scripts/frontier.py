#!/usr/bin/env python
"""AUROC-vs-latency frontier -- the benchmark's headline (accuracy vs cost).

For each method, averages AUROC and per-item latency (p50 ms) across all result cells
(generators x seeds x datasets) at each N in the sweep, so you see the accuracy you buy
for the milliseconds you spend. Prints the frontier table and, if matplotlib is present,
saves a scatter/line plot (one curve per method across N).

    python scripts/frontier.py --results-root ~/JasonLucas/outputs/results_full
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def collect(results_root, dataset=None):
    # acc[method][N] -> {"auroc": [...], "ms": [...]}
    acc = defaultdict(lambda: defaultdict(lambda: {"auroc": [], "ms": []}))
    for f in glob.glob(os.path.join(results_root, "stage_cd_*_seed*.json")):
        d = json.load(open(f))
        for c in d.get("cells", []):
            if "error" in c or (dataset and c.get("dataset") != dataset):
                continue
            for mkey, row in c.get("rows", {}).items():
                name, _, n = mkey.rpartition("_N")
                if not n.isdigit():
                    continue
                acc[name][int(n)]["auroc"].append(row.get("auroc"))
                acc[name][int(n)]["ms"].append(row.get("p50_ms"))
    return acc


def main():
    ap = argparse.ArgumentParser(description="AUROC-vs-latency frontier.")
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_full"))
    ap.add_argument("--dataset", default=None, help="restrict to one dataset (default: all)")
    ap.add_argument("--plot", default=os.path.expanduser("~/JasonLucas/outputs/frontier.png"))
    args = ap.parse_args()

    acc = collect(args.results_root, args.dataset)
    if not acc:
        print(f"No results in {args.results_root}"); return
    Ns = sorted({n for m in acc.values() for n in m})

    scope = args.dataset or "all datasets"
    print(f"\n=== AUROC vs latency frontier ({scope}, averaged over generators x seeds) ===")
    header = "method".ljust(26) + "".join(f"N={n} (AUROC/ms)".ljust(20) for n in Ns)
    print(header)
    rows = {}
    for name in sorted(acc):
        line = name.ljust(26)
        rows[name] = {}
        for n in Ns:
            au = _mean(acc[name][n]["auroc"]); ms = _mean(acc[name][n]["ms"])
            rows[name][n] = (au, ms)
            line += (f"{au:.3f}/{ms:.0f}ms" if au is not None and ms is not None else "-").ljust(20)
        print(line)

    # best accuracy-per-cost at the top N
    topN = Ns[-1]
    print(f"\n=== ranked by AUROC at N={topN} (with its latency) ===")
    ranked = sorted(rows.items(), key=lambda kv: (kv[1][topN][0] or 0), reverse=True)
    for name, byn in ranked:
        au, ms = byn[topN]
        fit = "  <=500ms" if (ms is not None and ms <= 500) else "  >500ms"
        if au is not None:
            print(f"  {name:26s} AUROC={au:.3f}  latency={ms:.0f}ms{fit}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        for name in sorted(acc):
            xs = [rows[name][n][1] for n in Ns if rows[name][n][1] is not None]
            ys = [rows[name][n][0] for n in Ns if rows[name][n][0] is not None]
            if xs:
                ax.plot(xs, ys, "-o", label=name, alpha=0.8)
        ax.axvline(500, color="grey", ls="--", alpha=0.5)
        ax.text(500, ax.get_ylim()[0], " 500ms SLA", color="grey", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("per-item latency p50 (ms, log)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"UQ accuracy vs cost frontier ({scope})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"\nPlot -> {args.plot}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(plot skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
