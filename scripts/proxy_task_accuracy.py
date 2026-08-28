#!/usr/bin/env python
"""Is the distilled proxy (student, qwen3-0.6b) as ACCURATE as the teacher (qwen3-8b) on the
TASKS themselves — not uncertainty — across every dataset (QA, MCQ, health)?

DisAAD distills the proxy from the teacher's outputs, but the proxy is only ever used for its
*logits* (uncertainty); its own task accuracy is never measured off the distillation set. This
runs the merged proxy as a GENERATOR (greedy, thinking off, primary answer only), judges it with
the same labeler the teacher used (MCQMatch for medqa/mmlu_med, local Qwen3-4B judge otherwise),
and compares proxy accuracy vs. the teacher's accuracy (read from the teacher's cached labels).

    python scripts/proxy_task_accuracy.py --seed 0 \
        --proxy-path ~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged
"""

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

QA = ["trivia_qa", "natural_qa", "pop_qa", "truthful_qa"]
HEALTH = ["medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"]
TEACHER_KEY = "qwen3-8b"


def teacher_accuracy(dataset, seed, qa_root, health_root):
    """Teacher (qwen3-8b) accuracy for a dataset = fraction correct in its cached labels."""
    from calibrate_eval import _labels_for
    root = health_root if dataset in HEALTH else qa_root
    pqs = glob.glob(os.path.join(root, f"stageA_{dataset}_{TEACHER_KEY}_*_seed{seed}.parquet"))
    if not pqs:
        return None, 0
    labels = _labels_for(pqs[0])
    if not labels:
        return None, 0
    vals = [v.get("correct_llm_judge") for v in labels.values()]
    vals = [v for v in vals if v in (0, 1)]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def main():
    ap = argparse.ArgumentParser(description="Proxy vs teacher TASK accuracy across datasets.")
    ap.add_argument("--proxy-path",
                    default=os.path.expanduser("~/JasonLucas/outputs/disaad/proxy_qwen3-0.6b_from_qwen3-8b/merged"))
    ap.add_argument("--generator-key", default="qwen3-0.6b-proxy")
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--judge-label", default="qwen3-4b-instruct-2507")
    ap.add_argument("--datasets", nargs="+", default=QA + HEALTH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--max-items", type=int, default=150)
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_proxy_acc"))
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results_proxy_acc"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from stage2_open import load_judge, _make_config, _mcq_fn, MCQ_DATASETS
    from hc_benchmark.stage_a_generate import generate_stage_a_local
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector

    os.makedirs(args.cache_root, exist_ok=True); os.makedirs(args.results_root, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"[proxy-acc] loading proxy {args.proxy_path} + judge {args.judge} on {device} ...")
    proxy = AutoModelForCausalLM.from_pretrained(args.proxy_path, torch_dtype=dt,
                                                 local_files_only=True).to(device).eval()
    proxy_tok = AutoTokenizer.from_pretrained(args.proxy_path, local_files_only=True)
    judge_fn, _jm, _jt = load_judge(args.judge, device, args.dtype)
    ct = {"enable_thinking": False}   # qwen3 proxy: answer only, no reasoning trace

    rows = []
    for ds in args.datasets:
        cfg = _make_config(ds, args.generator_key, n_max=1, size=args.size, seed=args.seed, n_sweep=(1,))
        cache = generate_stage_a_local(cfg, seed=args.seed, model=proxy, tokenizer=proxy_tok,
                                       cache_root=args.cache_root, chat_template_kwargs=ct,
                                       max_items=args.max_items)
        is_mcq = ds in MCQ_DATASETS
        label_fn = _mcq_fn() if is_mcq else judge_fn
        label_stage_b(cache, criteria=("llm_judge",), _judge_fn=label_fn,
                      judge_model=("mcq_match(letter)" if is_mcq else args.judge_label))
        corr = [c for c in correctness_vector(cache, "llm_judge") if c in (0, 1)]
        p_acc = sum(corr) / len(corr) if corr else None
        t_acc, t_n = teacher_accuracy(ds, args.seed, args.qa_cache_root, args.health_cache_root)
        delta = (p_acc - t_acc) if (p_acc is not None and t_acc is not None) else None
        rows.append({"dataset": ds, "task_type": "MCQ" if is_mcq else "QA",
                     "domain": "Health" if ds in HEALTH else "General",
                     "proxy_acc": p_acc, "teacher_acc": t_acc, "delta": delta,
                     "n_proxy": len(corr), "n_teacher": t_n})
        print(f"[{ds:12s}] proxy={p_acc:.3f}  teacher={t_acc:.3f}  Δ={delta:+.3f}"
              if (p_acc is not None and t_acc is not None) else f"[{ds}] incomplete")

    out = os.path.join(args.results_root, f"proxy_acc_seed{args.seed}.json")
    json.dump({"generator_key": args.generator_key, "teacher": TEACHER_KEY, "seed": args.seed,
               "rows": rows}, open(out, "w"), indent=2, default=str)
    print(f"\n[proxy-acc] wrote {out}")


if __name__ == "__main__":
    main()
