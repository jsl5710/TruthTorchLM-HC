#!/usr/bin/env python
"""Multi-target RQ report: does the proxy's AUROC+latency advantage hold across targets and
WIDEN on the larger/slower one? Aggregates, per teacher x student:

  * direct methods on that TARGET       (results_mt/stage_cd_<teacher>_seed*.json, from mt_eval)
  * DALD-au/eu, DisAAD-au/eu proxies    (results_mt_score/<method>_<teacher>_<student>/stage_cd_disaad-*.json)
  * Ours (uncertainty-aware, EDL)       (results_mt_score/ours[_<oracle>_lam<l>]_<teacher>_<student>/rq5_*.json)
    -- the main table shows the BEST Ours config per cell; a sweep-detail table lists every
    (oracle, lambda) variant.

Writes docs/multitarget_comparison.md. Robust to partial results (reports what exists).

    python scripts/mt_report.py
"""
import glob
import json
import os
from collections import defaultdict

HOME = os.path.expanduser("~/JasonLucas/outputs")
DIRECT_ROOT = os.path.join(HOME, "results_mt")
SCORE_ROOT = os.path.join(HOME, "results_mt_score")
TEACHERS = ["qwen3-32b", "llama3.3-70b"]
STUDENTS = {"qwen3-32b": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"],
            "llama3.3-70b": ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


# PTrue reads the target's logprobs (P("true") token) -> it is GREY-BOX, not pure black-box, so it
# is excluded from this text-only benchmark. See src/TruthTorchLM/truth_methods/p_true.py.
GREY_BOX = {"PTrue"}


def _cells_auroc_ms(files):
    """{method: (mean_auroc, mean_p50ms)} over all datasets/seeds in the given result files."""
    au, ms = defaultdict(list), defaultdict(list)
    for f in files:
        for c in json.load(open(f)).get("cells", []):
            for k, r in c.get("rows", {}).items():
                m = k.rpartition("_N")[0]
                if m in GREY_BOX:
                    continue
                au[m].append(r.get("auroc")); ms[m].append(r.get("p50_ms"))
    return {m: (_mean(au[m]), _mean(ms[m])) for m in au}


def direct_for(teacher):
    return _cells_auroc_ms(glob.glob(os.path.join(DIRECT_ROOT, f"stage_cd_{teacher}_seed*.json")))


def proxy_for(teacher, student):
    """Symmetric proxy read-outs + every Ours sweep variant.
    Returns (fixed, ours_variants) where fixed = {DALD-au, DALD-eu, DisAAD-au, DisAAD-eu: (au,ms)}
    and ours_variants = {tag: (au,ms)} (tag e.g. 'ours', 'ours_ecc_lam2')."""
    fixed = {}
    for method in ("dald", "disaad"):
        d = os.path.join(SCORE_ROOT, f"{method}_{teacher}_{student}")
        label = "DALD" if method == "dald" else "DisAAD"
        for m, v in _cells_auroc_ms(glob.glob(os.path.join(d, "stage_cd_disaad-*.json"))).items():
            ro = m.rsplit("-", 1)[-1]                       # 'au' / 'eu'
            fixed[f"{label}-{ro}"] = v
    ours = {}
    suffix = f"_{teacher}_{student}"
    for dd in glob.glob(os.path.join(SCORE_ROOT, f"ours*{suffix}")):
        base = os.path.basename(dd)
        if not base.endswith(suffix):
            continue
        tag = base[: -len(suffix)]                          # 'ours' or 'ours_ecc_lam2'
        got = _cells_auroc_ms(glob.glob(os.path.join(dd, "rq5_*.json")))
        for _m, v in got.items():
            ours[tag] = v
    return fixed, ours


def _best_ours(ours):
    cand = [(v[0], t, v[1]) for t, v in ours.items() if v[0] is not None]
    return max(cand) if cand else (None, "—", None)


def _ours_au(ours, *tags):
    """First matching Ours variant's AUROC (base cell 'ours' == edl-ecc-λ5)."""
    for t in tags:
        if t in ours and ours[t][0] is not None:
            return ours[t][0]
    return None


def main():
    L = ["# Multi-Target Proxy RQ — AUROC + Latency Across Targets\n",
         "> Does the proxy advantage hold across target models and **widen on the larger/slower "
         "target**? Two teachers (Qwen3-32B, Llama-3.3-70B), each with a 3-student sweep. Proxy "
         "read-outs (DALD-au/eu, DisAAD-au/eu, Ours-EDL best config) are target-decoupled (~30 ms); "
         "direct methods run on the target (~750 ms). AUROC = mean over the 7-dataset subset.\n",
         "**Auto-generated** by `scripts/mt_report.py`.\n"]
    sweep_rows = []
    for teacher in TEACHERS:
        direct = direct_for(teacher)
        best_direct = max(((a or -1, m) for m, (a, l) in direct.items()), default=(None, "—"))
        L.append(f"\n## Teacher: {teacher}\n")
        if best_direct[0] and best_direct[0] > 0:
            dms = _mean([l for _, l in direct.values() if l])
            L.append(f"- best direct method: **{best_direct[1]}** AUROC={best_direct[0]:.3f}; "
                     f"mean direct latency p50 ≈ {dms:.0f} ms\n")
        else:
            L.append("_direct-method results not present yet (mt_eval still running)._\n")
        L.append("\n| student | DALD-au | DALD-eu | DisAAD-au | DisAAD-eu | edl·ecc λ1 | edl·ecc λ5 | edl·ecc λ10 | **Ours (best)** | best direct | proxy p50 ms |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for student in STUDENTS[teacher]:
            fixed, ours = proxy_for(teacher, student)
            bo_au, bo_tag, bo_ms = _best_ours(ours)
            g = lambda k: (f"{fixed[k][0]:.3f}" if k in fixed and fixed[k][0] is not None else "—")
            f3 = lambda x: (f"{x:.3f}" if isinstance(x, (int, float)) else "—")
            l1 = _ours_au(ours, "ours_ecc_lam1")
            l5 = _ours_au(ours, "ours_ecc_lam5", "ours")        # base cell == edl-ecc-λ5
            l10 = _ours_au(ours, "ours_ecc_lam10")
            pm = _mean([v[1] for v in list(fixed.values()) + list(ours.values()) if v[1]])
            bd = f"{best_direct[0]:.3f} {best_direct[1]}" if best_direct[0] else "—"
            ocol = f"**{bo_au:.3f}** ({bo_tag.replace('ours_','').replace('ours','edl·ecc·λ5')})" if bo_au is not None else "**—**"
            L.append(f"| {student} | {g('DALD-au')} | {g('DALD-eu')} | {g('DisAAD-au')} | {g('DisAAD-eu')} "
                     f"| {f3(l1)} | {f3(l5)} | {f3(l10)} | {ocol} | {bd} | {(f'{pm:.0f}' if pm else '—')} |")
            for tag, v in sorted(ours.items()):
                if v[0] is not None:
                    sweep_rows.append((teacher, student, tag.replace("ours_", "").replace("ours", "edl(default)"), v[0]))
    # --- sweep detail ---
    if sweep_rows:
        L.append("\n## Ours λ/oracle sweep — AUROC by config\n")
        L.append("| teacher | student | config (oracle_lamλ) | AUROC |")
        L.append("|---|---|---|--:|")
        for t, s, cfg, au in sweep_rows:
            L.append(f"| {t} | {s} | {cfg} | {au:.3f} |")
    L.append("\n**Read:** claim (a) = Ours-EDL (best config) ≥ DALD/DisAAD on each cell; claim (b) = the "
             "proxy's *latency* edge over direct methods grows with target size (a ~30 ms proxy forward "
             "is target-decoupled; direct methods pay the target's per-sample cost). If (a) fails even at "
             "the swept optimum, the honest claim is Pareto/latency dominance, not raw-AUROC dominance.\n")
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "multitarget_comparison.md")
    os.makedirs(os.path.dirname(doc), exist_ok=True)
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[mt-report] wrote {doc}")


if __name__ == "__main__":
    main()
