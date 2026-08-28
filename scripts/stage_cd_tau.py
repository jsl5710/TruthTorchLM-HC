#!/usr/bin/env python
"""Stage C->D on tau-bench agentic trials (dataset #17).

tau-bench historical trajectories are pre-computed (gpt-4o / sonnet-3.5 on airline/retail),
each with a full dialogue and a reward. So this run BYPASSES Stage A (no generation) and
Stage B (reward IS the label -- no judge): it groups a task's K trials into a Stage-C cache
(each trial reduced to final message + tool-action signature = the sample set), injects the
reward as the correctness label, and runs the consistency methods + Stage D.

The question it answers: does black-box trial-consistency predict agentic task success?

    python scripts/stage_cd_tau.py --data-dir ~/JasonLucas/data/tau_bench \
        --results-root ~/JasonLucas/outputs/results_tau

Needs a GPU (the NLI methods). Never on the login node.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from stage_cd_open import build_bb_methods, _filter_valid  # noqa: E402


def _reduce_traj(traj):
    """Compact, comparable representation of one trajectory: final agent message + the
    sequence of tool-call names (the actions that determine the reward). Bounded length so
    it fits the NLI model's window."""
    calls, final = [], ""
    for m in traj or []:
        if m.get("role") != "assistant":
            continue
        if m.get("content"):
            final = m["content"]
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {}).get("name") or tc.get("name")
            if fn:
                calls.append(fn)
    txt = (final or "").strip().replace("\n", " ")[:400]
    return (txt + " || actions: " + ",".join(calls)).strip()


def build_cache(path, cache_root, seed, label_thresh=0.5):
    """Group a file's trials by task -> a Stage-A-shaped cache + a reward label file.

    primary_answer = trial 0's reduction; samples = all K trials' reductions; label = 1 if
    trial 0's reward >= thresh (pass^1). Returns (cache, K_trials_per_task).
    """
    from hc_benchmark.config import BenchmarkConfig
    from hc_benchmark.cache import GenerationCache
    from hc_benchmark.stage_b_label import _labels_path

    data = json.load(open(path))
    by_task = defaultdict(list)
    for e in data:
        by_task[e.get("task_id")].append(e)

    K = max(len(v) for v in by_task.values())
    items, labels = [], {}
    for tid, entries in by_task.items():
        if len(entries) < K:
            continue  # keep uniform trial count so the N-sweep is well-defined
        entries = sorted(entries, key=lambda e: e.get("trial", 0))
        reductions = [_reduce_traj(e.get("traj")) for e in entries]
        reward0 = float(entries[0].get("reward") or 0.0)
        items.append({
            "item_id": tid,
            "question": f"tau task {tid}",
            "context": "",
            "ground_truths": [],
            "primary_answer": reductions[0],
            "samples": reductions,
            "stratum": os.path.basename(path).replace(".json", ""),
            "outcome_type": "agentic_task",
        })
        labels[str(tid)] = {"correct_llm_judge": int(reward0 >= label_thresh)}

    name = os.path.basename(path).replace(".json", "").replace("-", "_").replace(".", "_")
    config = BenchmarkConfig(
        dataset=f"tau_{name}", generator=name, generator_backend="precomputed",
        n_max=K, n_sweep=(1, K), seeds=(seed,), size_of_data=1.0,
        correctness_criteria=("llm_judge",),
    )
    cache = GenerationCache(cache_root, config, seed)
    cache.write(items)
    _labels_path(cache).write_text(json.dumps(
        {"criteria": ["llm_judge"], "judge_model": "tau_reward", "labels": labels}, indent=2))
    return cache, K, name


def run_file(path, cache_root, device, seed):
    from hc_benchmark.stage_b_label import correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import summarize, sla_verdict

    cache, K, name = build_cache(path, cache_root, seed)
    n_sweep = sorted({1, max(1, K // 2), K})
    methods = build_bb_methods(K, device, include_verbalized=False)  # consistency only

    print(f"\n[{name}] {len(cache)} tasks, {K} trials/task, N-sweep {n_sweep} -- scoring ...")
    scores = score_stage_c(cache, methods, model="precomputed", tokenizer=None,
                           n_sweep=n_sweep, collect_timing=True)
    correctness = correctness_vector(cache, "llm_judge")
    fcorr, fscores = _filter_valid(correctness, scores)
    n_pos = sum(fcorr)
    print(f"[{name}] {len(fcorr)} tasks scorable, {n_pos} pass / {len(fcorr) - n_pos} fail")

    rows = {}
    for method in methods:
        mname = type(method).__name__
        ev = evaluate_method(fscores[mname], fcorr, truth_method=method, require_calibrated=False)
        for n, metrics in ev.items():
            tm = fscores[mname][n]["timing_ms"]
            row = {m: metrics.get(m) for m in
                   ("auroc", "auprc", "auarc", "prr", "ece", "ace", "mce", "brier")}
            if tm:
                summ = summarize(tm, warmup=0, label=mname)
                row["p50_ms"] = round(summ["p50_ms"], 3)
                row["fits_500ms"] = sla_verdict(summ)["verdicts"]["500ms"]
            rows[f"{mname}_N{n}"] = row
            if n == K:
                print(f"[{name}]   {mname:26s} N={n}: AUROC={row['auroc']:.3f} "
                      f"PRR={metrics.get('prr'):.3f}")
    return {"file": name, "n_items": len(fcorr), "n_positive": n_pos,
            "trials_per_task": K, "n_sweep": n_sweep, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Stage C->D on tau-bench agentic trials.")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/JasonLucas/data/tau_bench"))
    ap.add_argument("--files", nargs="+", default=None, help="specific json files (default: all)")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_tau"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_tau"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_root, exist_ok=True)

    files = args.files or sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    print(f"tau-bench Stage C->D: {len(files)} file(s), device={device}")
    out = os.path.join(args.results_root, "stage_cd_tau.json")
    cells = []
    for f in files:
        try:
            cells.append(run_file(f, args.cache_root, device, args.seed))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            cells.append({"file": os.path.basename(f), "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        # Write incrementally so a walltime kill keeps completed files (no per-run atomicity).
        json.dump({"cells": cells}, open(out, "w"), indent=2, default=str)
        print(f"  [saved {len(cells)}/{len(files)} files -> {out}]")
    print(f"\nDone. Wrote {out}")


if __name__ == "__main__":
    main()
