#!/usr/bin/env python
"""Dashboard for a benchmark run — Slurm job states + result coverage + failures.

Works for the open sweep and the closed run alike; point it at a results dir + job prefix.

    python scripts/monitor_runs.py                       # open run (full-*, results_full)
    python scripts/monitor_runs.py --job-prefix closed- --results-root ~/JasonLucas/outputs/results_closed
    watch -n 30 python scripts/monitor_runs.py           # live

Shows: how many jobs are running/pending, which (generator, seed) result cells have landed
and their item/positive counts (or ERROR), and any job logs containing a traceback.
"""

import argparse
import glob
import json
import os
import subprocess
from collections import Counter, defaultdict


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def slurm_now(prefix):
    """Running/pending counts + a short list, from squeue."""
    out = _run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T|%M|%R"])
    by_state, rows = Counter(), []
    for line in out.strip().splitlines():
        name, _, rest = line.partition("|")
        if not name.startswith(prefix):
            continue
        state, _, rest = rest.partition("|")
        elapsed, _, reason = rest.partition("|")
        by_state[state] += 1
        rows.append((name, state, elapsed, reason))
    return by_state, rows


def slurm_finished(prefix):
    """COMPLETED/FAILED/etc. for finished jobs today, from sacct."""
    out = _run(["sacct", "-n", "-X", "--starttime", "today",
                "-o", "JobName%40,State"])
    by_state = Counter()
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        if name.startswith(prefix):
            by_state[state] += 1
    return by_state


def coverage(results_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_root, "stage_cd_*_seed*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cells = []
        for c in d.get("cells", []):
            if "error" in c:
                cells.append((c.get("dataset"), "ERR"))
            else:
                cells.append((c.get("dataset"), f"{c.get('n_positive')}/{c.get('n_items')}"))
        rows.append((d.get("generator_key"), d.get("seed"), d.get("wall_seconds"), cells))
    return rows


def failures(logs_dir, prefix):
    bad = []
    for e in glob.glob(os.path.join(logs_dir, f"{prefix}*.err")):
        try:
            txt = open(e, errors="ignore").read()
        except Exception:
            continue
        if "Traceback" in txt or "CUDA out of memory" in txt or "Error" in txt:
            # last meaningful line
            last = [l for l in txt.strip().splitlines() if l.strip()][-1:] or [""]
            bad.append((os.path.basename(e), last[0][:110]))
    return bad


def main():
    ap = argparse.ArgumentParser(description="Monitor a benchmark run.")
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_full"))
    ap.add_argument("--job-prefix", default="full-")
    ap.add_argument("--logs-dir", default=os.path.expanduser("~/JasonLucas/logs"))
    args = ap.parse_args()

    print(f"===== RUN MONITOR  (prefix '{args.job_prefix}', {args.results_root}) =====")

    live, rows = slurm_now(args.job_prefix)
    fin = slurm_finished(args.job_prefix)
    print(f"\n[slurm now]      " + (", ".join(f"{k}={v}" for k, v in live.items()) or "no active jobs"))
    print(f"[slurm finished] " + (", ".join(f"{k}={v}" for k, v in fin.items()) or "none today"))
    for name, state, elapsed, reason in rows[:12]:
        print(f"    {name:26s} {state:3s} {elapsed:>8s}  {reason}")

    cov = coverage(args.results_root)
    print(f"\n[results] {len(cov)} result file(s) landed:")
    done_cells = 0
    for gen, seed, wall, cells in cov:
        summ = "  ".join(f"{ds}:{v}" for ds, v in cells)
        done_cells += sum(1 for _, v in cells if v != "ERR")
        print(f"    {str(gen):16s} seed{seed} ({wall}s)  {summ}")
    print(f"    -> {done_cells} good cells so far")

    bad = failures(args.logs_dir, args.job_prefix)
    if bad:
        print(f"\n[FAILURES] {len(bad)} log(s) with errors:")
        for fn, last in bad:
            print(f"    {fn}: {last}")
    else:
        print("\n[failures] none detected")


if __name__ == "__main__":
    main()
