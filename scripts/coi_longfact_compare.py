#!/usr/bin/env python
"""LongFact (long-form GENERAL knowledge) ablation readout -- the length-vs-domain disambiguator.
If CoI (n=3 phi) / verify rungs (n6/n7) beat the verbalized floor (n1) here, the medlfqa collapse
was DOMAIN (health/subjective), not answer LENGTH. Reads results_coi_longfact/coi_<gen>_seed1.json."""
import json, os

R = os.path.expanduser("~/JasonLucas/outputs/results_coi_longfact")
GENS = ["llama-3.1-8b", "qwen3-8b"]
SEED = 1


def rows(gen):
    p = f"{R}/coi_{gen}_seed{SEED}.json"
    if not os.path.exists(p):
        return None, None
    js = json.load(open(p))
    for cell in js.get("cells", []):
        if cell.get("dataset") == "longfact":
            return cell, (cell.get("rows") or {})
    return None, None


def g(rw, n, k):
    r = (rw or {}).get(f"CoIVerbalized_n{n}")
    v = (r or {}).get(k)
    return v


def fmt(v, k):
    if v is None:
        return "  -  "
    return f"{v:.3f}" if k != "p50_ms" else f"{v:.0f}"


def main():
    print("LongFact ablation -- long-form GENERAL knowledge (median ~180/110 words)\n")
    hdr = f"{'generator':<16}{'metric':<8}{'n1_verb':>9}{'n3_phi':>9}{'n6_YN':>9}{'n7_selfC':>10}   winner"
    print(hdr); print("-" * len(hdr))
    verdicts = {}
    for gen in GENS:
        cell, rw = rows(gen)
        if not rw:
            print(f"{gen:<16} (no results yet)"); continue
        au = {n: g(rw, n, "auroc") for n in (1, 3, 6, 7)}
        best = max((n for n in au if au[n] is not None), key=lambda n: au[n], default=None)
        n_items = cell.get("n_items"); n_pos = cell.get("n_positive")
        for metric, k in (("AUROC", "auroc"), ("ECE", "ece"), ("p50ms", "p50_ms")):
            tag = f"{gen}" if metric == "AUROC" else (f"  ({n_items}it {n_pos}+)" if metric == "ECE" else "")
            win = f"n{best} ({au[best]:.3f})" if (metric == "AUROC" and best is not None) else ""
            print(f"{tag:<16}{metric:<8}{fmt(g(rw,1,k),k):>9}{fmt(g(rw,3,k),k):>9}"
                  f"{fmt(g(rw,6,k),k):>9}{fmt(g(rw,7,k),k):>10}   {win}")
        # does any CoI variant beat the verbalized floor?
        if au[1] is not None:
            beats = [f"n{n}(+{au[n]-au[1]:.3f})" for n in (3, 6, 7) if au[n] is not None and au[n] > au[1]]
            verdicts[gen] = beats
        print()
    print("=== does CoI beat the verbalized floor (n1) on long-form GENERAL knowledge? ===")
    for gen, beats in verdicts.items():
        print(f"  {gen:<16} {'YES -> ' + ', '.join(beats) if beats else 'NO -- verbalized floor wins (as on medlfqa)'}")
    print("\nInterpretation: CoI beating n1 here (but not on medlfqa) => medlfqa collapse is DOMAIN, not length.")


if __name__ == "__main__":
    main()
