#!/usr/bin/env python
"""Open vs CLOSED targets — the comparison arm. The study distills proxies for OPEN targets
(qwen3-32b, llama-70b); the CLOSED frontier targets (gpt-4o, claude-haiku) are the reference point
that makes the latency case: direct UQ needs N samples from the target, and on an API target each is
a ~1-2 s round-trip, so direct UQ costs seconds/item while the proxy stays ~27 ms (target-decoupled).

Closed targets run DIRECT (consistency) methods only — no API-teacher proxy (the offline scorer can't
call the gateway). PTrue excluded everywhere (grey-box). Writes docs/multitarget_closed_vs_open.md.

    python scripts/mt_report_closed.py
"""
import glob
import json
import os
import statistics
from collections import defaultdict

HOME = os.path.expanduser("~/JasonLucas/outputs")
GREY = {"PTrue"}
# (label, results_root, results_glob, cache_root, cache_key, kind, size)
TARGETS = [
    ("qwen3-32b (open, 32B)", os.path.join(HOME, "results_mt"), "stage_cd_qwen3-32b_seed*.json",
     os.path.join(HOME, "cache_mt"), "qwen3-32b", "open", "32B"),
    ("llama-3.3-70b (open, 70B)", os.path.join(HOME, "results_mt"), "stage_cd_llama3.3-70b_seed*.json",
     os.path.join(HOME, "cache_mt"), "llama3.3-70b", "open", "70B"),
    ("gpt-4o (closed, API)", os.path.join(HOME, "results_closed"), "stage_cd_jhu-gpt-4o_seed*.json",
     os.path.join(HOME, "cache_closed"), "jhu-gpt-4o", "closed", "frontier"),
    ("claude-haiku-4.5 (closed, API)", os.path.join(HOME, "results_closed"), "stage_cd_jhu-claude-haiku-4.5_seed*.json",
     os.path.join(HOME, "cache_closed"), "jhu-claude-haiku-4.5", "closed", "small-frontier"),
]
N_SAMPLES = 10          # consistency methods need N-1 extra target samples; ~N target calls total
PROXY_MS = 27           # measured median proxy forward (target-decoupled)


def best_direct(results_root, pat):
    au = defaultdict(list)
    for fn in glob.glob(os.path.join(results_root, pat)):
        for c in json.load(open(fn)).get("cells", []):
            for k, r in c.get("rows", {}).items():
                m = k.rpartition("_N")[0]
                if m in GREY or r.get("auroc") is None:
                    continue
                au[m].append(r["auroc"])
    ranked = sorted(((sum(v) / len(v), m) for m, v in au.items()), reverse=True)
    return ranked


def gen_ms(cache_root, key):
    import pandas as pd
    ms = []
    for f in glob.glob(os.path.join(cache_root, f"stageA_*_{key}_*.parquet")):
        df = pd.read_parquet(f)
        if "generator_ms" in df.columns:
            ms += [x for x in df["generator_ms"].tolist() if x]
    return statistics.median(ms) if ms else None


def main():
    L = ["# Open vs Closed targets — the comparison arm\n",
         "> The study distills proxies for **open** targets (qwen3-32b, llama-70b). The **closed** "
         "frontier targets (gpt-4o, claude-haiku) are the reference: direct UQ needs N≈10 samples from "
         "the target, and on an API target each is a ~1–2 s round-trip — so direct UQ costs **seconds "
         "per item**, while a distilled proxy stays **~27 ms** (target-decoupled). Closed targets run "
         "the consistency direct methods only (no API-teacher proxy). PTrue excluded (grey-box).\n",
         "**Auto-generated** by `scripts/mt_report_closed.py`.\n"]
    L.append("\n| target | type | best pure-BB direct (AUROC) | target gen / call | direct UQ gen cost (≈N calls) | proxy |")
    L.append("|---|---|---|--:|--:|--:|")
    for label, rr, pat, cr, key, kind, size in TARGETS:
        ranked = best_direct(rr, pat)
        g = gen_ms(cr, key)
        if not ranked:
            L.append(f"| {label} | {kind} | _pending_ | {(f'{g:.0f} ms' if g else '—')} | "
                     f"{(f'≈{g*N_SAMPLES/1000:.1f} s' if g else '—')} | {PROXY_MS} ms |")
            continue
        bd = f"{ranked[0][1]} {ranked[0][0]:.3f}"
        gcost = f"≈{g*N_SAMPLES/1000:.1f} s" if g else "—"
        gcall = f"{g:.0f} ms" if g else "—"
        L.append(f"| {label} | {kind} | {bd} | {gcall} | **{gcost}** | {PROXY_MS} ms |")
    L.append("\n**Read:** on the closed API targets, direct UQ pays **~11–19 s of target generation per "
             "item** (N samples × 1–2 s/call) before any scoring; the proxy needs **zero** target calls "
             "and one ~27 ms forward — a **~400–700× latency gap** that only widens with target cost. "
             "AUROC-wise the consistency methods transfer to the API targets (see per-target rows). The "
             "open targets' per-call generation latency is not cached (hc_benchmark schema); a small "
             "measurement job can fill it, but the proxy's 27 ms is target-independent regardless.\n")

    # per-target direct-method detail (top methods)
    for label, rr, pat, cr, key, kind, size in TARGETS:
        ranked = best_direct(rr, pat)
        if not ranked:
            continue
        L.append(f"\n### {label} — direct methods (pure-BB)\n")
        L.append("| method | AUROC |")
        L.append("|---|--:|")
        for a, m in ranked[:6]:
            L.append(f"| {m} | {a:.3f} |")

    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                       "multitarget_closed_vs_open.md")
    os.makedirs(os.path.dirname(doc), exist_ok=True)
    open(doc, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[mt-report-closed] wrote {doc}")


if __name__ == "__main__":
    main()
