#!/usr/bin/env python
"""Compare the medlfqa signal-recovery rungs against the baselines, per (generator, dataset):
  n=1  verbalized floor (self-rated global confidence)
  n=3  deterministic phi over self-rated per-claim risk        [current default]
  n=6  decision-framed YES/NO verify (fresh greedy)            [method #2]
  n=7  self-consistency: K sampled YES/NO -> vote fraction     [method #1]

n=6/7 come from results_coi_verify; n=1/n=3 from whichever dir has them (verify dir first,
else the original results_coi). Reports AUROC + ECE + p50 latency so the accuracy-vs-latency
trade of the K-sample rung is explicit."""
import glob, json, os

WORK = os.path.expanduser("~/JasonLucas")
VER = f"{WORK}/outputs/results_coi_verify"
BASE = f"{WORK}/outputs/results_coi"
CELLS = [("llama-3.1-8b", "trivia_qa"), ("llama-3.1-8b", "medlfqa"),
         ("qwen3-8b", "trivia_qa"), ("qwen3-8b", "medlfqa")]
SEED = 1


def load(gen):
    out = {}
    for d in (VER, BASE):
        p = f"{d}/coi_{gen}_seed{SEED}.json"
        if os.path.exists(p):
            out[d] = json.load(open(p))
    return out


def row_for(js, ds, n):
    if not js:
        return None
    for cell in js.get("cells", []):
        if cell.get("dataset") == ds:
            return (cell.get("rows") or {}).get(f"CoIVerbalized_n{n}")
    return None


def get(gen, ds, n):
    """Prefer the verify-dir file for n; fall back to the baseline dir."""
    js = load(gen)
    r = row_for(js.get(VER), ds, n)
    if r is None:
        r = row_for(js.get(BASE), ds, n)
    return r


def fmt(r, k):
    if not r or r.get(k) is None:
        return "  -  "
    return f"{r[k]:.3f}" if k != "p50_ms" else f"{r[k]:.0f}"


def main():
    print("AUROC (higher=better) | ECE | p50_ms  per rung\n")
    hdr = f"{'cell':<24} {'metric':<7}{'n1_verb':>9}{'n3_phi':>9}{'n6_YN':>9}{'n7_selfC':>10}   winner"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for gen, ds in CELLS:
        rr = {n: get(gen, ds, n) for n in (1, 3, 6, 7)}
        au = {n: (rr[n] or {}).get("auroc") for n in (1, 3, 6, 7)}
        best = max((n for n in au if au[n] is not None), key=lambda n: au[n], default=None)
        for metric, k in (("AUROC", "auroc"), ("ECE", "ece"), ("p50ms", "p50_ms")):
            tag = f"{gen}/{ds}" if metric == "AUROC" else ""
            win = ""
            if metric == "AUROC" and best is not None:
                win = f"n{best} ({au[best]:.3f})"
            print(f"{tag:<24} {metric:<7}{fmt(rr[1],k):>9}{fmt(rr[3],k):>9}{fmt(rr[6],k):>9}"
                  f"{fmt(rr[7],k):>10}   {win}")
        # delta of best-new (max n6,n7) over the better baseline (max n1,n3)
        newbest = max((au[n] for n in (6, 7) if au[n] is not None), default=None)
        basebest = max((au[n] for n in (1, 3) if au[n] is not None), default=None)
        if newbest is not None and basebest is not None:
            summary[f"{gen}/{ds}"] = round(newbest - basebest, 3)
        print()
    print("=== best-new (n6/n7) minus best-baseline (n1/n3), AUROC ===")
    for k, v in summary.items():
        print(f"  {k:<24} {'+' if v >= 0 else ''}{v}")


if __name__ == "__main__":
    main()
