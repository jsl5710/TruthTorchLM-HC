#!/usr/bin/env python
"""Confirmation: does our BEST variant beat ALL other methods (DALD, DisAAD, PTrue, VerbalizedConf,
every direct method) on EACH of the 9 datasets, on qwen3-8b? Per-dataset AUROC, winner flagged.

    python scripts/rq5_confirm.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import rq3_report as R  # noqa: E402

HOME = os.path.expanduser("~/JasonLucas/outputs")
DATASETS = R.QA + R.HEALTH


def _mean(x):
    x = [v for v in x if isinstance(v, (int, float))]
    return sum(x) / len(x) if x else None


def direct_per_ds():
    """{method: {ds: auroc}} at each method's best-N (per its group)."""
    out = defaultdict(dict)
    for root, group in (("results_full", set(R.QA)), ("results_full_health", set(R.HEALTH))):
        bn = R.best_n(R.load_main(os.path.join(HOME, root), group), group)
        for m, (n, byds) in bn.items():
            for ds in group:
                if ds in byds:
                    out[m][ds] = _mean([t[0] for t in byds[ds]])
    return out


def cells_per_ds(files):
    au = defaultdict(lambda: defaultdict(list))
    for f in files:
        for c in json.load(open(f)).get("cells", []):
            for k, r in c.get("rows", {}).items():
                au[k.rpartition("_N")[0]][c["dataset"]].append(r.get("auroc"))
    return {m: {ds: _mean(v) for ds, v in byds.items()} for m, byds in au.items()}


def main():
    per = {}                                    # method -> {ds: auroc}, family tag in name
    fam = {}
    for m, byds in direct_per_ds().items():
        per[m] = byds; fam[m] = "direct"
    for m, byds in cells_per_ds(glob.glob(HOME + "/results_disaad/stage_cd_disaad-qwen3-8b_seed*.json")).items():
        per[m] = byds; fam[m] = "DisAAD"
    for m, byds in cells_per_ds(glob.glob(HOME + "/results_dald/rq5_dald_seed*.json")).items():
        per["DALD"] = byds; fam["DALD"] = "DALD"
    for m, byds in cells_per_ds(glob.glob(HOME + "/results_rq5/rq5_*_seed*.json")).items():
        per[m] = byds; fam[m] = "OURS"

    ours = [m for m in per if fam[m] == "OURS"]
    print(f"Loaded: {sum(f=='OURS' for f in fam.values())} OURS, "
          f"{'DALD' in per and 'DALD-yes' or 'DALD-NO'}, "
          f"{sum(f=='direct' for f in fam.values())} direct, {sum(f=='DisAAD' for f in fam.values())} DisAAD\n")

    def au(m, ds):
        return per.get(m, {}).get(ds)

    print(f"{'dataset':12s} {'OUR best':>22} {'beats-all?':>10} {'DALD':>6} {'PTrue':>6} {'Verb':>6} {'overall best':>26}")
    print("-" * 96)
    wins = 0
    for ds in DATASETS:
        ob = max(((au(m, ds) or -1, m) for m in per if au(m, ds) is not None), default=(None, "—"))
        our = max(((au(m, ds) or -1, m) for m in ours if au(m, ds) is not None), default=(None, "—"))
        beats = (our[0] is not None and ob[1] in ours)
        wins += beats
        nm = lambda m: m.replace("rq5_", "").replace("EccentricityUncertainty", "Ecc").replace("DiscreteSemanticEntropy", "SemEnt").replace("VerbalizedConfidence", "Verb")
        f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        print(f"{ds:12s} {f(our[0])+' '+nm(our[1])[:14]:>22} {('YES' if beats else 'no'):>10} "
              f"{f(au('DALD',ds)):>6} {f(au('PTrue',ds)):>6} {f(au('VerbalizedConfidence',ds)):>6} "
              f"{f(ob[0])+' '+nm(ob[1])[:16]:>26}")
    print(f"\nOUR variant is the single best method on {wins}/{len(DATASETS)} datasets.")
    print("(For the claim 'beats all', we need YES on every dataset — otherwise the honest claim is "
          "Pareto/latency-dominance, not raw-AUROC dominance.)")


if __name__ == "__main__":
    main()
