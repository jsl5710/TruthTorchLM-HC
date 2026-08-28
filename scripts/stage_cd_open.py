#!/usr/bin/env python
"""Stage A->D for OPEN targets with the pure black-box UQ method set (the real benchmark cell).

Unlike hc_benchmark/run.py (which uses the litellm API generation path), this drives the
local HuggingFace target, labels with the local judge, and scores the consistency-based
black-box methods -- producing the headline AUROC/PRR + auxiliary-compute latency per
method per N. First real research output on open models.

Pipeline per (dataset, seed):
  A  generate_stage_a_local: primary + n_max samples, thinking OFF (ANSWER_ONLY), cached
  B  label_stage_b with the local Qwen3-4B judge (trace-stripped)
  C  score_stage_c: LexicalSimilarity, DiscreteSemanticEntropy, EigV, NumSemanticSetUncertainty
     over the cached samples, N-sweep by truncation (no target re-call)
  D  evaluate_method: AUROC/AUPRC/PRR + auxiliary-compute p50/p95 + SLA verdict

The NLI backbone (microsoft/deberta-large-mnli) is loaded from a locally converted
SAFETENSORS copy -- the hub ships only a pickle .bin, which transformers 5 / torch 2.5
refuse to load. Convert once with scripts (see NLI_DIR) and it loads offline.

Never on the login node -- submit via scripts/stage_cd_open.slurm.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from smoke_open import _load_model, _warmup          # noqa: E402
from stage2_open import load_judge, _make_config, _mcq_fn, MCQ_DATASETS  # noqa: E402

NLI_DIR = os.path.expanduser("~/JasonLucas/hf_cache/converted/deberta-large-mnli")


def patch_thinking(tokenizer, enable: bool):
    """Make every apply_chat_template on this tokenizer default enable_thinking=<enable>.

    Method-agnostic way to run the verbalized family's elicitation with thinking off: they
    all call the target tokenizer's apply_chat_template with no enable_thinking, so this
    injects it (setdefault -- an explicit value from a caller, e.g. Stage A generation,
    still wins). No-op for non-thinking tokenizers (unknown template var).
    """
    orig = tokenizer.apply_chat_template

    def patched(*args, **kwargs):
        kwargs.setdefault("enable_thinking", enable)
        return orig(*args, **kwargs)

    tokenizer.apply_chat_template = patched
    return tokenizer


def build_bb_methods(n_max, device, include_verbalized=False, judge_model=None, judge_tok=None):
    """The pure black-box method set, with the injected safetensors NLI model.

    Default: 10 consistency/graph methods (LexicalSimilarity is similarity-only; the other
    nine share the one loaded NLI model). ``include_verbalized`` appends the verbalized
    family (VerbalizedConfidence, PTrue), which elicit a confidence answer from the target
    at score time -- their latency is now measured honestly (aux + that generation), but
    that elicitation currently runs with the target's default thinking mode, so keep them
    opt-in until the enable_thinking plumbing lands. All are asserted black-box.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from TruthTorchLM.truth_methods import (
        LexicalSimilarity, DiscreteSemanticEntropy, EigV, NumSemanticSetUncertainty,
        EccentricityConfidence, EccentricityUncertainty,
        MatrixDegreeConfidence, MatrixDegreeUncertainty,
        KernelLanguageEntropy, SumEigenUncertainty,
    )
    from TruthTorchLM.utils.access_level import is_black_box

    print(f"[nli] loading entailment model (safetensors) from {NLI_DIR} ...")
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_DIR).to(device)
    nli.eval()
    nli_tok = AutoTokenizer.from_pretrained(NLI_DIR)
    if device == "cuda":
        # Warm the NLI model so the FIRST timed clustering isn't inflated by cold-start
        # (cuDNN autotune, lazy allocation) -- protocol §5 measurement hygiene.
        import torch
        with torch.inference_mode():
            nli(**nli_tok("warm", "up", return_tensors="pt").to(device))
        torch.cuda.synchronize()
        print("[nli] warmup done")

    def _nli(cls):  # semantic methods share the one injected NLI model
        return cls(number_of_generations=n_max,
                   model_for_entailment=nli, tokenizer_for_entailment=nli_tok)

    methods = [
        LexicalSimilarity(number_of_generations=n_max),   # similarity-only, no NLI
        _nli(DiscreteSemanticEntropy),
        _nli(EigV),
        _nli(NumSemanticSetUncertainty),
        _nli(EccentricityConfidence),
        _nli(EccentricityUncertainty),
        _nli(MatrixDegreeConfidence),
        _nli(MatrixDegreeUncertainty),
        _nli(KernelLanguageEntropy),
        _nli(SumEigenUncertainty),
    ]
    if include_verbalized:
        # Only VerbalizedConfidence is pure black-box (parses the model's TEXT confidence).
        # PTrue is EXCLUDED: it reads the target's logprobs (P("true") token) -> grey-box, which
        # violates the text-only constraint. SelfDetection/CrossExamination excluded as impractical.
        from TruthTorchLM.truth_methods import VerbalizedConfidence
        methods += [VerbalizedConfidence()]
    bad = [type(m).__name__ for m in methods if not is_black_box(m)]
    if bad:
        raise ValueError(f"Non-black-box methods slipped in: {bad}")
    return methods


def _filter_valid(correctness, scores):
    """Drop items whose judge label is -1 (not-attempted/unparsed): AUROC needs binary 0/1.

    The same item order is shared by every method and N (aligned to cache.read()), so one
    keep-mask applies everywhere -- correctness, truth values, and per-item timing.
    """
    from hc_benchmark.stage_c_score import MethodScores

    list_fields = ("truth_values", "normalized_truth_values", "timing_ms", "aux_ms", "gen_ms")
    keep = [i for i, c in enumerate(correctness) if c in (0, 1)]
    fcorr = [correctness[i] for i in keep]
    fscores = {}
    for name, by_n in scores.items():
        fscores[name] = {}
        for n, ms in by_n.items():
            fms = MethodScores()
            for f in list_fields:
                vals = ms.get(f) or []
                fms[f] = [vals[i] for i in keep] if len(vals) == len(correctness) else vals
            fscores[name][n] = fms
    return fcorr, fscores


def run_cell(dataset, generator_key, target_model, target_tok, judge_fn, methods,
             size, n_max, n_sweep, seed, cache_root, ct_kwargs, max_items, judge_label):
    from hc_benchmark.stage_a_generate import generate_stage_a_local
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import summarize, sla_verdict

    config = _make_config(dataset, generator_key, n_max=n_max, size=size, seed=seed,
                          n_sweep=n_sweep)
    print(f"\n[A] {dataset}/{generator_key} seed{seed}: generate n_max={n_max} (thinking off) ...")
    cache = generate_stage_a_local(config, seed=seed, model=target_model, tokenizer=target_tok,
                                   cache_root=cache_root, chat_template_kwargs=ct_kwargs,
                                   max_items=max_items)
    # MCQ datasets (medqa, mmlu_med) are letter/option answers -> MCQMatch, NOT the LLM
    # judge (which fails on letter-vs-text). Everything else uses the judge.
    is_mcq = dataset in MCQ_DATASETS
    label_fn = _mcq_fn() if is_mcq else judge_fn
    print(f"[B] labelling with {'MCQMatch' if is_mcq else 'local judge'} ...")
    label_stage_b(cache, criteria=("llm_judge",), _judge_fn=label_fn,
                  judge_model=("mcq_match(letter)" if is_mcq else judge_label))
    correctness = correctness_vector(cache, "llm_judge")

    print(f"[C] scoring {len(methods)} black-box methods, N-sweep {n_sweep} ...")
    # Pass the LOADED target (not the key string): verbalized methods call it live at score
    # time; consistency methods ignore it and score from the cached samples.
    scores = score_stage_c(cache, methods, model=target_model, tokenizer=target_tok,
                           n_sweep=n_sweep, collect_timing=True)
    fcorr, fscores = _filter_valid(correctness, scores)
    n_items, n_pos = len(fcorr), sum(fcorr)
    print(f"[D] evaluating on {n_items} scorable items ({n_pos} correct / "
          f"{n_items - n_pos} incorrect; {len(correctness) - n_items} dropped as -1) ...")

    rows = {}
    for method in methods:
        name = type(method).__name__
        ev = evaluate_method(fscores[name], fcorr, truth_method=method, require_calibrated=False)
        for n, metrics in ev.items():
            tm = fscores[name][n]["timing_ms"]
            gm = fscores[name][n].get("gen_ms") or []
            # Save ALL metrics evaluate_method computed -- discrimination (auroc/auprc/auarc/
            # prr) AND calibration (ece/ace/mce/brier). NB: calibration metrics are on raw
            # (uncalibrated) scores here since require_calibrated=False.
            row = {m: metrics.get(m) for m in
                   ("auroc", "auprc", "auarc", "prr", "ece", "ace", "mce", "brier")}
            if tm:
                # p50/p95 are the TOTAL per-item method cost (aux compute + the method's own
                # score-time generation); gen_p50 isolates that generation (nonzero only for
                # verbalized methods that elicit a confidence answer).
                summ = summarize(tm, warmup=0, label=f"{name} N={n}")
                row["p50_ms"] = round(summ["p50_ms"], 3)
                row["p95_ms"] = round(summ["p95_ms"], 3)
                row["fits_500ms"] = sla_verdict(summ)["verdicts"]["500ms"]
                if gm and any(g > 0 for g in gm):
                    row["gen_p50_ms"] = round(summarize(gm, warmup=0, label=name)["p50_ms"], 3)
            rows[f"{name}_N{n}"] = row
            au = row["auroc"]
            gtxt = f" gen={row['gen_p50_ms']}ms" if "gen_p50_ms" in row else ""
            print(f"[D]   {name:26s} N={n:<2}: AUROC={au:.3f} "
                  f"PRR={metrics.get('prr'):.3f}"
                  + (f" | p50={row.get('p50_ms')}ms{gtxt} fits500={row.get('fits_500ms')}"
                     if tm else ""))
    return {"dataset": dataset, "n_items": n_items, "n_positive": n_pos,
            "n_dropped": len(correctness) - n_items, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Stage A->D black-box UQ benchmark (open target).")
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--generator-key", default="qwen3-1.7b")
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--judge-label", default="qwen3-4b-instruct-2507")
    ap.add_argument("--datasets", nargs="+", default=["trivia_qa", "simple_qa", "pop_qa"])
    ap.add_argument("--size", type=float, default=0.05)
    ap.add_argument("--max-items", type=int, default=80, help="cap items/dataset for the cell.")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--n-sweep", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--include-verbalized", action="store_true",
                    help="Also run VerbalizedConfidence + PTrue (live target call at score "
                         "time; latency counted honestly but elicitation uses default thinking).")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", help="target dtype: bf16|fp32|4bit (4bit fits a 70B on one A100)")
    ap.add_argument("--judge-dtype", default="bf16", help="judge dtype -- keep bf16 for label quality even when target is 4bit")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    ct_kwargs = {"enable_thinking": args.thinking == "on"}
    n_sweep = tuple(n for n in args.n_sweep if n <= args.n_max) or (1,)

    print(f"[load] target {args.target} (thinking={args.thinking}) ...")
    target_model, target_tok = _load_model(args.target, device, args.dtype)
    _warmup(target_model, target_tok, device)
    # With a multi-GPU target (dtype=4bit/bf16-auto reserves the LAST GPU), load the judge there
    # so it never competes with the target for cuda:0.
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    judge_device = f"cuda:{n_gpu - 1}" if (n_gpu > 1 and args.dtype in ("4bit", "bf16-auto")) else device
    print(f"[judge] loading ONCE: {args.judge} (dtype={args.judge_dtype}) on {judge_device} ...")
    judge_fn, judge_model, judge_tok = load_judge(args.judge, judge_device, args.judge_dtype)
    # Verbalized methods elicit via the target tokenizer; default its thinking mode to the
    # run's (off) so confidence calls match the ANSWER_ONLY base -- faster and cleaner.
    patch_thinking(target_tok, args.thinking == "on")
    methods = build_bb_methods(args.n_max, device, include_verbalized=args.include_verbalized,
                               judge_model=judge_model, judge_tok=judge_tok)
    print(f"[methods] {len(methods)}: {[type(m).__name__ for m in methods]}")

    t0 = time.perf_counter()
    cells = []
    for ds in args.datasets:
        try:
            cells.append(run_cell(ds, args.generator_key, target_model, target_tok, judge_fn,
                                  methods, args.size, args.n_max, n_sweep, args.seed,
                                  args.cache_root, ct_kwargs, args.max_items, args.judge_label))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            cells.append({"dataset": ds, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    elapsed = time.perf_counter() - t0

    bundle = {"target": args.target, "generator_key": args.generator_key, "judge": args.judge,
              "thinking": args.thinking, "n_max": args.n_max, "n_sweep": list(n_sweep),
              "seed": args.seed, "cells": cells, "wall_seconds": round(elapsed, 1)}
    out = Path(args.results_root) / f"stage_cd_{args.generator_key}_seed{args.seed}.json"
    out.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"\nStage A->D done in {elapsed:.1f}s -> {out}")


if __name__ == "__main__":
    main()
