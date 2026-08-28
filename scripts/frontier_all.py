#!/usr/bin/env python
"""Combined AUROC-vs-latency frontier across every regime + the DisAAD proxy.

Overlays the settings the benchmark covers on one accuracy-vs-cost plane:
  * QA-open     -- local open generators            (results_full,        N=5)
  * QA-closed   -- gpt-4o-mini + claude-haiku (API)  (results_closed_full, N=5)
  * Health      -- medqa/mmlu_med/kqa/medlfqa/bioasq (results_full_health, N=5)
  * agentic-tau -- tau-bench precomputed rollouts    (results_tau,         N=4)
and, as a distinct star series, the trained **DisAAD** proxy (results_disaad, N=1):
one decoupled proxy forward pass per item -- single-pass, ~constant ms -- scored on the
qwen3-8b caches (its distillation teacher), split into its QA and Health points.

Each point is one UQ method's mean AUROC and mean auxiliary-compute latency (p50 ms),
averaged over every cell in that regime. Latency is auxiliary compute only (sample/trial
generation excluded), so regimes are directly comparable. Prints a combined table (each
regime's Pareto frontier starred) and saves the scatter.

Caveat printed alongside: the 4 regimes average over all generators; DisAAD is qwen3-8b-only
(a per-target proxy), so compare it against the qwen3-8b column, not the multi-gen mean.

    python scripts/frontier_all.py
"""

import argparse
import glob
import json
import os
from collections import defaultdict

QA = {"trivia_qa", "natural_qa", "pop_qa", "truthful_qa"}
HEALTH = {"medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def collect(files, n, dataset_filter=None):
    """{method: (mean_auroc, mean_p50_ms)} at pass count `n`, over cells in `files`.

    `dataset_filter`: a set of dataset names to include (None = all).
    """
    au, ms = defaultdict(list), defaultdict(list)
    for f in files:
        d = json.load(open(f))
        for c in d.get("cells", []):
            if "error" in c or (dataset_filter and c.get("dataset") not in dataset_filter):
                continue
            for mkey, row in c.get("rows", {}).items():
                name, _, nn = mkey.rpartition("_N")
                if nn != str(n):
                    continue
                au[name].append(row.get("auroc"))
                ms[name].append(row.get("p50_ms"))
    out = {}
    for m in au:
        a, l = _mean(au[m]), _mean(ms[m])
        if a is not None and l is not None:
            out[m] = (a, l)
    return out


def pareto(points):
    """Method names on the accuracy-vs-latency Pareto frontier (low ms, high AUROC)."""
    items = sorted(points.items(), key=lambda kv: kv[1][1])  # by latency asc
    best, front = -1.0, []
    for name, (a, _l) in items:
        if a > best:
            front.append(name); best = a
    return front


def _print_table(label, pts):
    if not pts:
        print(f"\n=== {label}: no data ==="); return
    front = set(pareto(pts))
    print(f"\n=== {label}  (auxiliary-compute latency; * = on this regime's Pareto frontier) ===")
    print(f"    {'method':28s} {'AUROC':>7} {'p50_ms':>9}  {'<=500ms':>8}")
    for m, (a, l) in sorted(pts.items(), key=lambda kv: kv[1][0], reverse=True):
        star = "*" if m in front else " "
        print(f"  {star} {m:28s} {a:>7.3f} {l:>8.0f} {('yes' if l <= 500 else 'no'):>9}")


def main():
    home = os.path.expanduser("~/JasonLucas/outputs")
    ap = argparse.ArgumentParser(description="Combined AUROC-vs-latency frontier + DisAAD.")
    ap.add_argument("--open-root", default=os.path.join(home, "results_full"))
    ap.add_argument("--closed-root", default=os.path.join(home, "results_closed_full"))
    ap.add_argument("--health-root", default=os.path.join(home, "results_full_health"))
    ap.add_argument("--tau-file", default=os.path.join(home, "results_tau", "stage_cd_tau.json"))
    ap.add_argument("--disaad-root", default=os.path.join(home, "results_disaad"))
    ap.add_argument("--plot", default=os.path.join(home, "frontier_all.png"))
    args = ap.parse_args()

    open_f = glob.glob(os.path.join(args.open_root, "stage_cd_*_seed*.json"))
    closed_f = glob.glob(os.path.join(args.closed_root, "stage_cd_*_seed*.json"))
    health_f = glob.glob(os.path.join(args.health_root, "stage_cd_*_seed*.json"))
    tau_f = [args.tau_file] if os.path.exists(args.tau_file) else []
    disaad_f = glob.glob(os.path.join(args.disaad_root, "stage_cd_disaad-*_seed*.json"))

    regimes = {
        "QA-open (N=5)":     collect(open_f, 5, QA),
        "QA-closed (N=5)":   collect(closed_f, 5, QA),
        "Health (N=5)":      collect(health_f, 5, HEALTH),
        "agentic-tau (N=4)": collect(tau_f, 4),
    }
    # DisAAD proxy: single-pass (N=1), qwen3-8b only, split into QA vs Health points.
    disaad = {}
    for grp, filt in (("QA", QA), ("Health", HEALTH)):
        for m, v in collect(disaad_f, 1, filt).items():
            disaad[f"{m} ({grp})"] = v

    for label, pts in regimes.items():
        _print_table(label, pts)
    if disaad:
        _print_table("DisAAD proxy (N=1, qwen3-8b only -- compare vs the qwen3-8b column)", disaad)
    else:
        print("\n=== DisAAD proxy: no results yet (results_disaad empty -- scoring not finished) ===")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        styles = {"QA-open (N=5)": ("#1f77b4", "o"),
                  "QA-closed (N=5)": ("#d62728", "s"),
                  "Health (N=5)": ("#9467bd", "D"),
                  "agentic-tau (N=4)": ("#2ca02c", "^")}
        fig, ax = plt.subplots(figsize=(10.5, 7))
        for label, pts in regimes.items():
            if not pts:
                continue
            color, marker = styles[label]
            ax.scatter([l for _a, l in pts.values()], [a for a, _l in pts.values()],
                       c=color, marker=marker, s=55, alpha=0.8, label=label,
                       edgecolors="k", linewidths=0.4)
            fp = sorted([(pts[m][1], pts[m][0], m) for m in pareto(pts)])
            ax.plot([l for l, _a, _m in fp], [a for _l, a, _m in fp], color=color, lw=1.3, alpha=0.6)
            for l, a, m in fp:
                ax.annotate(m, (l, a), fontsize=6, color=color, xytext=(3, 3),
                            textcoords="offset points")
        # DisAAD overlay: gold stars, always labeled (it's the headline proxy method)
        if disaad:
            ax.scatter([l for _a, l in disaad.values()], [a for a, _l in disaad.values()],
                       c="#ff7f0e", marker="*", s=240, label="DisAAD proxy (qwen3-8b, N=1)",
                       edgecolors="k", linewidths=0.6, zorder=5)
            for m, (a, l) in disaad.items():
                ax.annotate(m, (l, a), fontsize=7, color="#b3560a", fontweight="bold",
                            xytext=(4, 4), textcoords="offset points", zorder=6)
        ax.axvline(500, color="grey", ls="--", alpha=0.5)
        ax.text(500, ax.get_ylim()[0], " 500ms SLA", color="grey", va="bottom", fontsize=8)
        ax.axhline(0.5, color="grey", ls=":", alpha=0.4)
        ax.text(ax.get_xlim()[0], 0.5, " chance (0.5)", color="grey", va="bottom", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("auxiliary-compute latency p50 (ms, log)")
        ax.set_ylabel("mean AUROC")
        ax.set_title("UQ accuracy-vs-cost frontier: QA-open / QA-closed / Health / tau + DisAAD proxy")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=140)
        print(f"\nPlot -> {args.plot}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(plot skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
