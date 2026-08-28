#!/usr/bin/env python
"""Method-coverage probe: which black-box UQ methods actually run in this stack?

Round one shipped 4 methods (LexicalSimilarity, DiscreteSemanticEntropy, EigV,
NumSemanticSetUncertainty). This probes the REST of the plausibly-black-box M axis on a
small cached cell, each method isolated, and reports per method:
    constructed? | is_black_box? | scored? | AUROC | aux p50 ms   -- or the failure.

Two families are under test:
  * consistency / graph-spectral (cache + injected NLI): Eccentricity*, MatrixDegree*,
    KernelLanguageEntropy, SumEigenUncertainty -- expected to work like the shipped 4.
  * verbalized / prompt (a LIVE target call at score time): VerbalizedConfidence, PTrue,
    Confidence, SelfDetection, CrossExamination -- these need the loaded target; on a
    thinking model they may emit <think> traces (the known deferred wrinkle).

Adaptive construction: inspect each __init__ and pass only the kwargs it accepts
(number_of_generations / model_for_entailment / tokenizer_for_entailment / device), so one
probe handles heterogeneous constructors. Uses a tiny fresh cache (fast); it measures
viability, not precise AUROC. Never on the login node.
"""

import argparse
import inspect
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

from smoke_open import _load_model, _warmup            # noqa: E402
from stage2_open import load_judge, _make_config        # noqa: E402
from stage_cd_open import NLI_DIR, _filter_valid        # noqa: E402

# Candidates beyond the shipped 4. White-box (AttentionScore, Inside, SAPLMA, LARS, MARS,
# SAR*, DirectionalEntailmentGraph), external (GoogleSearchCheck, ContextCheck, MiniCheck),
# multi-model (MultiLLMCollab) and round-two (DisAAD, SPUQ) are intentionally omitted;
# is_black_box + try/except still guard anything misjudged.
CANDIDATES = [
    "EccentricityConfidence", "EccentricityUncertainty",
    "MatrixDegreeConfidence", "MatrixDegreeUncertainty",
    "KernelLanguageEntropy", "SumEigenUncertainty", "Entropy",
    "VerbalizedConfidence", "PTrue", "Confidence", "SelfDetection", "CrossExamination",
]
# Re-run the shipped 4 as a control that the harness itself is healthy.
CONTROL = ["LexicalSimilarity", "DiscreteSemanticEntropy"]


def adaptive_construct(cls, n, nli, nli_tok, device):
    params = inspect.signature(cls.__init__).parameters
    kw = {}
    if "number_of_generations" in params:
        kw["number_of_generations"] = n
    if "model_for_entailment" in params:
        kw["model_for_entailment"] = nli
    if "tokenizer_for_entailment" in params:
        kw["tokenizer_for_entailment"] = nli_tok
    if "entailment_model_device" in params and "model_for_entailment" not in kw:
        kw["entailment_model_device"] = device
    return cls(**kw), kw


def main():
    ap = argparse.ArgumentParser(description="Probe which black-box methods run.")
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--generator-key", default="qwen3-1.7b")
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="trivia_qa")
    ap.add_argument("--n", type=int, default=5, help="samples used per method (single N).")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--max-items", type=int, default=25)
    ap.add_argument("--size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_probe"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch
    import TruthTorchLM.truth_methods as TM
    from TruthTorchLM.utils.access_level import is_black_box
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from hc_benchmark.stage_a_generate import generate_stage_a_local
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import summarize

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.results_root).mkdir(parents=True, exist_ok=True)

    target_model, target_tok = _load_model(args.target, device, args.dtype)
    _warmup(target_model, target_tok, device)
    judge_fn, _a, _b = load_judge(args.judge, device, args.dtype)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_DIR).to(device)
    nli.eval()
    nli_tok = AutoTokenizer.from_pretrained(NLI_DIR)

    # Small cell: generate (thinking off) + label once; every method scores the same cache.
    config = _make_config(args.dataset, args.generator_key, n_max=args.n_max,
                          size=args.size, seed=args.seed, n_sweep=(args.n,))
    cache = generate_stage_a_local(config, seed=args.seed, model=target_model,
                                   tokenizer=target_tok, cache_root=args.cache_root,
                                   chat_template_kwargs={"enable_thinking": False},
                                   max_items=args.max_items)
    label_stage_b(cache, criteria=("llm_judge",), _judge_fn=judge_fn,
                  judge_model="qwen3-4b-instruct-2507")
    correctness = correctness_vector(cache, "llm_judge")
    valid = [c for c in correctness if c in (0, 1)]
    print(f"\n[probe] {args.dataset}: {len(valid)} scorable items "
          f"({sum(valid)} correct). Testing methods at N={args.n} ...\n")

    report = {}
    for name in CONTROL + CANDIDATES:
        rec = {"constructed": False, "black_box": None, "scored": False,
               "auroc": None, "aux_p50_ms": None, "error": None}
        try:
            cls = getattr(TM, name)
            method, kw = adaptive_construct(cls, args.n, nli, nli_tok, device)
            rec["constructed"] = True
            rec["kwargs"] = sorted(kw)
            rec["black_box"] = bool(is_black_box(method))
            if not rec["black_box"]:
                rec["error"] = "not black-box (needs logits/hidden state)"
            else:
                t0 = time.perf_counter()
                scores = score_stage_c(cache, [method], model=target_model,
                                       tokenizer=target_tok, n_sweep=(args.n,),
                                       collect_timing=True)
                fcorr, fscores = _filter_valid(correctness, scores)
                ev = evaluate_method(fscores[name], fcorr,
                                     truth_method=method, require_calibrated=False)
                rec["scored"] = True
                rec["auroc"] = round(float(ev[args.n].get("auroc")), 3)
                tm = fscores[name][args.n]["timing_ms"]
                if tm:
                    rec["aux_p50_ms"] = round(summarize(tm, warmup=0, label=name)["p50_ms"], 2)
                rec["wall_s"] = round(time.perf_counter() - t0, 1)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        report[name] = rec
        _print_row(name, rec)

    out = Path(args.results_root) / "method_coverage.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[probe] wrote {out}")


def _print_row(name, r):
    if r["error"] and not r["scored"]:
        tag = "not-bb" if r["black_box"] is False else ("construct-FAIL" if not r["constructed"] else "score-FAIL")
        print(f"  {name:28s} {tag:16s} {r['error'][:90]}")
    else:
        print(f"  {name:28s} OK   AUROC={r['auroc']}  aux_p50={r['aux_p50_ms']}ms  ({r.get('wall_s')}s)")


if __name__ == "__main__":
    main()
