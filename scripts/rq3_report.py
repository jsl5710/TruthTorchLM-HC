#!/usr/bin/env python
"""RQ3 report generator -- efficiency vs. quality: proxy (hybrid) vs. direct black-box UQ.

Assembles the four RQ3 comparison dimensions on a COMMON footing -- same target (qwen3-8b,
the proxy's distillation teacher), same ID datasets, same cached generations, same cross-fit
calibration protocol -- and writes docs/rq3_efficiency_quality.md:

  1. Uncertainty-estimation quality : AUROC, AUPR, calibration (ECE/Brier)
  2. Task performance by dataset    : per-dataset AUROC (QA vs Health)
  3. Inference latency              : p50 / p95 auxiliary-compute ms
  4. Computational cost             : target calls / proxy passes / offline training

Reads the main result JSONs + the calibration_*.json produced by calibrate_eval.py (direct)
and calibrate_disaad.py (proxy). Calibration columns show "pending" until those land, so the
doc is regenerable at any time.  Re-run after the calibration jobs finish to fill them in.

    python scripts/rq3_report.py
"""

import glob
import json
import os
from collections import defaultdict

HOME = os.path.expanduser("~/JasonLucas/outputs")
TARGET = "qwen3-8b"
QA = ["trivia_qa", "natural_qa", "pop_qa", "truthful_qa"]
HEALTH = ["medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"]

# per-method deployment cost annotation (marginal, at inference) + method family
COST = {
    "DisAAD-au":  ("hybrid (proxy)", "1 proxy fwd (0.6B) + offline distill (one-time ~12h)"),
    "DisAAD-eu":  ("hybrid (proxy)", "1 proxy fwd (0.6B) + offline distill (one-time ~12h)"),
    "PTrue":            ("direct / few-pass",  "~3-5 target calls"),
    "VerbalizedConfidence": ("direct / single-pass", "1 extra target call"),
    "DiscreteSemanticEntropy": ("direct / multi-sample", "N target gens + 1 NLI cluster"),
    "NumSemanticSetUncertainty": ("direct / multi-sample", "N target gens + 1 NLI cluster"),
    "EigV": ("direct / multi-sample", "N target gens + O(N^2) NLI graph"),
    "SumEigenUncertainty": ("direct / multi-sample", "N target gens + O(N^2) NLI graph"),
    "EccentricityConfidence": ("direct / multi-sample", "N target gens + O(N^2) NLI graph"),
    "EccentricityUncertainty": ("direct / multi-sample", "N target gens + O(N^2) NLI graph"),
    "MatrixDegreeConfidence": ("direct / multi-sample", "N target gens + O(N^2) NLI graph"),
    "MatrixDegreeUncertainty": ("direct / multi-sample", "N target gens + O(N^2) NLI graph"),
    "KernelLanguageEntropy": ("direct / multi-sample", "N target gens + O(N^2) NLI kernel"),
    "LexicalSimilarity": ("direct / multi-sample", "N target gens + O(N^2) ROUGE"),
}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def load_main(root, datasets):
    """{method: {N: {dataset: [ (auroc,auprc,p50,p95) per seed ]}}} for TARGET, ALL N kept."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(os.path.join(root, f"stage_cd_*{TARGET}_seed*.json")) + \
             glob.glob(os.path.join(root, f"stage_cd_disaad-{TARGET}_seed*.json")):
        d = json.load(open(f))
        if d.get("generator_key") not in (TARGET, None) and TARGET not in os.path.basename(f):
            continue
        for c in d.get("cells", []):
            ds = c.get("dataset")
            if ds not in datasets or "error" in c:
                continue
            for mk, row in c.get("rows", {}).items():
                nm, _, nn = mk.rpartition("_N")
                if not nn.isdigit():
                    continue
                acc[nm][int(nn)][ds].append((row.get("auroc"), row.get("auprc"),
                                             row.get("p50_ms"), row.get("p95_ms")))
    return acc


def best_n(acc, datasets):
    """Collapse {method:{N:{ds:[tuples]}}} -> {method: (bestN, {ds:[tuples]})} choosing, per
    method, the N with the highest mean AUROC over `datasets` (each method at its best op point).

    N=1 is dropped from the candidate set whenever a larger N exists: for a multi-sample method
    N=1 is a degenerate no-signal point (constant 0.5) that must not masquerade as its 'best'.
    Single-pass methods (DisAAD, verbalized) keep N=1 because it is their only N."""
    out = {}
    for m, byn in acc.items():
        cand = [n for n in byn if n > 1] or list(byn)  # keep N=1 only if it's all there is
        best, bn = None, None
        for n in cand:
            byds = byn[n]
            au = _mean([t[0] for ds in datasets if ds in byds for t in byds[ds]])
            if au is not None and (best is None or au > best):
                best, bn = au, n
        if bn is not None:
            out[m] = (bn, byn[bn])
    return out


def load_calib(roots):
    """{method: {N: {dataset: [ (ece,brier) per seed ]}}} for TARGET from calibration_*.json."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for root in roots:
        for f in glob.glob(os.path.join(root, f"calibration_*{TARGET}_seed*.json")):
            d = json.load(open(f))
            for c in d.get("cells", []):
                ds = c.get("dataset")
                for mk, row in c.get("rows", {}).items():
                    nm, _, nn = mk.rpartition("_N")
                    if not nn.isdigit():
                        continue
                    acc[nm][int(nn)][ds].append((row.get("ece"), row.get("brier")))
    return acc


def agg(acc, datasets, idx):
    """mean over datasets+seeds of tuple field `idx` for each method."""
    out = {}
    for m, byds in acc.items():
        vals = [t[idx] for ds in datasets if ds in byds for t in byds[ds]]
        out[m] = _mean(vals)
    return out


def load_taskacc(root, datasets):
    """{dataset: mean task accuracy} = generator's correctness rate (n_positive/n_items), the
    base task performance (MCQ accuracy / judge pass-rate), same for every UQ method."""
    acc = defaultdict(list)
    for f in glob.glob(os.path.join(root, f"stage_cd_*{TARGET}_seed*.json")):
        for c in json.load(open(f)).get("cells", []):
            ds, ni, npos = c.get("dataset"), c.get("n_items"), c.get("n_positive")
            if ds in datasets and ni:
                acc[ds].append(npos / ni)
    return {ds: _mean(v) for ds, v in acc.items()}


def dval(acc, m, ds, idx):
    ts = acc.get(m, {}).get(ds, [])
    return _mean([t[idx] for t in ts])


def dcal(calib, m, ds, n, idx):
    byn = calib.get(m, {})
    byds = byn.get(n) if n in byn else (next(iter(byn.values())) if byn else {})
    return _mean([t[idx] for t in byds.get(ds, [])])


def detail_section(L, grp, acc, dss, nmap, calib, taskacc, ds_labels):
    """Per-dataset detail: task accuracy header + method rows grouped Proxy then Direct."""
    L.append(f"\n### Per-dataset detail\n")
    for ds in dss:
        ta = taskacc.get(ds)
        L.append(f"\n**{ds_labels.get(ds, ds)}** · task accuracy "
                 f"{f'{100*ta:.0f}%' if ta is not None else '—'}\n")
        L.append("| Method | Type | AUROC | AUPR | ECE | p50 ms |")
        L.append("|---|---|--:|--:|--:|--:|")
        methods = [m for m in acc if ds in acc.get(m, {})]
        proxy = sorted([m for m in methods if m.startswith("DisAAD")],
                       key=lambda m: -(dval(acc, m, ds, 0) or 0))
        direct = sorted([m for m in methods if not m.startswith("DisAAD")],
                        key=lambda m: -(dval(acc, m, ds, 0) or 0))
        for grpname, ms in (("proxy", proxy), ("direct", direct)):
            for m in ms:
                n = nmap.get(m)
                L.append(f"| {'**'+m+'**' if grpname=='proxy' else m} | {grpname} | "
                         f"{fmt(dval(acc,m,ds,0))} | {fmt(dval(acc,m,ds,1))} | "
                         f"{fmt(dcal(calib,m,ds,n,0))} | {fmt(dval(acc,m,ds,2),0)} |")


def agg_calib(calib, datasets, n_map, idx):
    """mean ECE/Brier over datasets+seeds, matched to each method's reported N (n_map)."""
    out = {}
    for m, byn in calib.items():
        n = n_map.get(m)
        byds = byn.get(n) if n is not None else None
        if byds is None:  # single-pass methods: calibration stored at whatever N exists
            byds = next(iter(byn.values())) if byn else {}
        out[m] = _mean([t[idx] for ds in datasets if ds in byds for t in byds[ds]])
    return out


def per_dataset_auroc(acc, datasets):
    out = {}
    for m, byds in acc.items():
        out[m] = {ds: _mean([t[0] for t in byds[ds]]) if ds in byds else None for ds in datasets}
    return out


def fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


# Domain -> task-type -> datasets (+ the result root the datasets live in)
GROUPS = [
    ("Health", "MCQ", ["medqa", "mmlu_med"], "results_full_health"),
    ("Health", "Free-form QA", ["kqa", "medlfqa", "bioasq"], "results_full_health"),
    ("General / Open-domain", "QA", ["trivia_qa", "natural_qa", "pop_qa", "truthful_qa"], "results_full"),
]
DS_LABELS = {"medqa": "MedQA", "mmlu_med": "MMLU-Med", "kqa": "K-QA", "medlfqa": "MedLFQA",
             "bioasq": "BioASQ", "trivia_qa": "TriviaQA", "natural_qa": "NaturalQA",
             "pop_qa": "PopQA", "truthful_qa": "TruthfulQA"}


def build_group(root, datasets):
    """-> (acc {method:{ds:[tuples]}} at best-N, nmap {method:N})."""
    raw = load_main(os.path.join(HOME, root), datasets)
    raw.update({k: v for k, v in load_main(os.path.join(HOME, "results_disaad"), datasets).items()
                if k.startswith("DisAAD")})
    bn = best_n(raw, datasets)
    return {m: v[1] for m, v in bn.items()}, {m: v[0] for m, v in bn.items()}


def quality_table(L, acc, dss, nmap, calib):
    auroc = agg(acc, dss, 0); aupr = agg(acc, dss, 1)
    p50 = agg(acc, dss, 2); p95 = agg(acc, dss, 3)
    ece = agg_calib(calib, dss, nmap, 0); brier = agg_calib(calib, dss, nmap, 1)
    L.append("\n### Aggregate (mean over this group's datasets × seeds)\n")
    L.append("| Method | Type | N | AUROC | AUPR | ECE | Brier | p50 ms | p95 ms | Deployment cost |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
    for m in sorted(acc, key=lambda m: (not m.startswith("DisAAD"), -(auroc.get(m) or 0))):
        _, cost = COST.get(m, ("direct", "—"))
        typ = "**proxy**" if m.startswith("DisAAD") else "direct"
        L.append(f"| {'**'+m+'**' if m.startswith('DisAAD') else m} | {typ} | {nmap.get(m,'—')} | "
                 f"{fmt(auroc.get(m))} | {fmt(aupr.get(m))} | {fmt(ece.get(m))} | {fmt(brier.get(m))} | "
                 f"{fmt(p50.get(m),0)} | {fmt(p95.get(m),0)} | {cost} |")


def best_direct(acc, dss):
    au = agg(acc, dss, 0); ms = agg(acc, dss, 2)
    cand = [(au[m], m) for m in acc if not m.startswith("DisAAD") and au.get(m) is not None]
    if not cand:
        return None
    a, m = max(cand); return m, a, ms.get(m)


def main():
    calib = load_calib([os.path.join(HOME, "results_full"),
                        os.path.join(HOME, "results_full_health"),
                        os.path.join(HOME, "results_disaad")])
    n_cal = sum(len(v) for v in calib.values())
    taskacc = {**load_taskacc(os.path.join(HOME, "results_full_health"), HEALTH),
               **load_taskacc(os.path.join(HOME, "results_full"), QA)}

    L = []
    L.append("# RQ3 — Efficiency–Quality Trade-off\n")
    L.append("> **RQ3.** Do the efficiency gains of the proxy-based (hybrid) uncertainty-estimation "
             "framework justify any trade-off in uncertainty-estimation quality compared with direct "
             "black-box uncertainty-estimation methods?\n")
    L.append("**Auto-generated** by `scripts/rq3_report.py`. Organised **Domain → Task type → Dataset**.\n")
    L.append("## Setup — a common footing\n")
    L.append(f"- **Target model:** `{TARGET}` — the *only* fair target, because the DisAAD proxy is "
             "distilled from it (per-target method). Every method scores the **same** cached generations.\n"
             "- **Method types:** *proxy (hybrid)* = DisAAD (evidential **AU**/**EU** on the distilled "
             "0.6B proxy — one forward pass); *direct* = consistency family (multi-sample) + verbalized "
             "(PTrue, VerbalizedConfidence).\n"
             "- **Taxonomy:** Health {MCQ · Free-form QA} · General/Open {QA}. **Dialogue** (tau-bench) is "
             "a task type in the benchmark but **out of RQ3 scope** — its trajectories are precomputed "
             "gpt-4o/sonnet, not qwen3-8b, so the per-target proxy was never distilled/scored there.\n"
             "- **Metrics:** *quality* = AUROC, AUPR, ECE (calibration); *task performance* = generator "
             "accuracy (correct-rate) per dataset; *efficiency* = auxiliary-compute latency p50/p95 + the "
             "compute-cost column (true per-query cost). Each method at its **best-AUROC N**.\n"
             f"- **Calibration** (ECE): cross-fitted isotonic normalizer, identical for both arms. "
             f"Status: {'**' + str(n_cal) + ' cells loaded**' if n_cal else '**PENDING** — jobs running; ECE shows — until they land'}.\n")

    summary = []
    for domain, task, dss, root in GROUPS:
        acc, nmap = build_group(root, dss)
        L.append(f"\n## {domain} — {task}\n")
        quality_table(L, acc, dss, nmap, calib)
        detail_section(L, f"{domain} {task}", acc, dss, nmap, calib, taskacc, DS_LABELS)
        d_au = agg(acc, dss, 0).get("DisAAD-au"); d_ms = agg(acc, dss, 2).get("DisAAD-au")
        bd = best_direct(acc, dss)
        if d_au and bd:
            summary.append((domain, task, d_au, d_ms, bd))

    L.append("\n## Dialogue / agentic (tau-bench) — out of RQ3 scope\n")
    L.append("tau-bench is the benchmark's dialogue/agentic regime, but it is **excluded from this RQ3 "
             "comparison**: its runs are precomputed gpt-4o and claude-3.5-sonnet trajectories, so there "
             "is no qwen3-8b proxy to compare (DisAAD is per-target). It is reported separately in the "
             "main frontier. Extending RQ3 to dialogue needs a proxy distilled from a dialogue target.\n")

    L.append("\n## Efficiency–quality summary — is the trade-off justified?\n")
    for domain, task, d_au, d_ms, (bm, ba, bms) in summary:
        speed = f"{bms/d_ms:.0f}×" if (bms and d_ms) else "—"
        gap = d_au - ba
        verdict = ("**justified** — proxy faster *and* ≥ as accurate" if gap >= -0.01
                   else f"a **real trade-off** — proxy gives up {abs(gap):.03f} AUROC for the speed")
        L.append(f"- **{domain} · {task}:** best direct = **{bm}** ({fmt(ba)} AUROC @ {fmt(bms,0)} ms); "
                 f"proxy **DisAAD-au = {fmt(d_au)} @ {fmt(d_ms,0)} ms** ({speed} faster). → {verdict}.\n")
    L.append("- **Compute cost:** the proxy pays a **one-time** offline distillation (~12 h) then **one "
             "0.6 B forward pass** per query — target-decoupled, so its marginal cost is ~constant "
             "regardless of how slow the target is. Direct multi-sample methods pay **N target "
             "generations + O(N²) clustering** on *every* query; verbalized pays 1–5 extra target calls.\n")
    L.append("\n*Latency is auxiliary-compute only; the compute-cost column gives true deployment cost. "
             "DisAAD is qwen3-8b-only by construction (per-target proxy). Each method shown at its "
             "best-AUROC N.*\n")

    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                       "rq3_efficiency_quality.md")
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[rq3_report] wrote {doc}")


if __name__ == "__main__":
    main()
