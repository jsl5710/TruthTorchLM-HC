#!/usr/bin/env python
"""Stage-2 open-model launcher: local LLM-as-judge correctness labeling, no API.

This drives the pure black-box benchmark on OPEN targets with an OPEN judge. It exists
because run.py does not drive the HuggingFace generation path, and because the correctness
judge must be a locally-hosted model (no keys in Stage 2).

Design invariants (do not violate):
  * The JUDGE model is loaded ONCE per process and reused across every dataset and every
    item -- never reloaded per dataset or per batch (it is the dominant load cost).
  * Exact match is wrong for fuzzy free-form answers ("anti lamore" ~ "Ante' Lamore"), so
    those sets are labeled by the judge. MCQ sets (medqa, mmlu_med) keep MCQMatch -- a
    letter/option match is exact and cheaper there, and it doubles as the judge's gold.

Two-phase, gated:
  1. VALIDATE (default): run the judge on the MCQ sets, where MCQMatch already gives the
     known-correct label, and report judge-vs-gold agreement. Nothing is labeled "for
     real" in this phase. Inspect the agreement number first.
  2. LABEL (--proceed): only after validation, label the free-form sets with the judge.
     Aborts if agreement is below --agreement-threshold unless --force is passed.

Never run on the login node -- submit via scripts/stage2_open.slurm.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make imports work regardless of cwd: repo root (for hc_benchmark), src (for
# TruthTorchLM), and scripts/ (for smoke_open). Derive all from this file's location, since
# `python scripts/stage2_open.py` only puts scripts/ on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

# Reuse the smoke's vetted model-load + warmup helpers (same repo, same conventions).
from smoke_open import _load_model, _warmup  # noqa: E402


# --- dataset -> correctness criterion policy --------------------------------
# Fuzzy free-form QA: the judge earns its keep here (spelling/alias/paraphrase).
JUDGE_FREEFORM = [
    "trivia_qa", "natural_qa", "pop_qa", "simple_qa",
    "web_questions", "wikipedia_factual", "narrative_qa",
]
# Long-form health: judge (or BLEURT) -- also fuzzy, no exact match possible.
JUDGE_LONGFORM = ["kqa", "medlfqa", "bioasq"]
# MCQ: exact letter/option match is correct and cheap; used as judge-validation gold.
MCQ_DATASETS = ["medqa", "mmlu_med"]
# Free-form sets used by the construct-from-gold validation (in-format, known labels).
CONSTRUCT_DEFAULT = ["simple_qa", "trivia_qa", "natural_qa"]


def _make_config(dataset, generator_key, n_max, size, seed, n_sweep=(1,)):
    """A BenchmarkConfig for one (dataset, target) on the local HF backend.

    n_sweep is irrelevant to labeling (we only read primary_answer), so it is kept minimal
    and <= n_max to satisfy the config invariant.
    """
    from hc_benchmark.config import BenchmarkConfig

    n_sweep = tuple(n for n in n_sweep if n <= n_max) or (1,)
    return BenchmarkConfig(
        dataset=dataset,
        generator=generator_key,
        generator_backend="huggingface",
        n_max=n_max,
        n_sweep=n_sweep,
        seeds=(seed,),
        size_of_data=size,
        correctness_criteria=("llm_judge",),
    )


# --- judge (loaded ONCE, reused everywhere) ---------------------------------

import re as _re_mod

_THINK_RE = _re_mod.compile(r"<think>.*?</think>", _re_mod.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces, leaving the final answer (ANSWER_ONLY).

    Reasoning targets (Qwen3 thinking, DeepSeek-R1) emit a trace before their answer; the
    registry's reasoning_trace policy is ANSWER_ONLY, so the correctness judge must grade
    only the final answer, not the trace. No-op for non-thinking models. The Stage-A cache
    still holds the full output, so trace-generation latency stays recoverable for timing.
    """
    return _THINK_RE.sub("", text or "").strip()


def load_judge(judge_id, device, dtype, system_prompt=None, max_new_tokens=16):
    """Load the judge model+tokenizer once and return a reusable judge_fn.

    judge_fn(question, answer, ground_truths) -> int label (1/0/-1), and after each call
    exposes .last_verdict / .last_raw so label_stage_b can persist unparseable outputs.
    The answer has its reasoning trace stripped before judging (ANSWER_ONLY).
    """
    from TruthTorchLM.evaluators import ModelJudge

    model, tok = _load_model(judge_id, device, dtype)
    kwargs = {"max_new_tokens": max_new_tokens}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    judge = ModelJudge(model=model, tokenizer=tok, **kwargs)

    def judge_fn(question, answer, ground_truths):
        label = judge(question, _strip_think(answer), ground_truths, "")
        judge_fn.last_raw = judge.last_raw_output
        judge_fn.last_verdict = judge.last_verdict
        return label

    judge_fn.last_raw = None
    judge_fn.last_verdict = None
    judge_fn.judge = judge
    return judge_fn, model, tok


def _mcq_fn():
    """A judge_fn-shaped adapter around MCQMatch (no model call, exposes no raw)."""
    from TruthTorchLM.evaluators import MCQMatch

    mcq = MCQMatch()

    def fn(question, answer, ground_truths):
        return mcq(question, answer, ground_truths, "")

    fn.last_raw = None
    fn.last_verdict = None
    return fn


# --- phase 1: judge validation on MCQ (gold = MCQMatch) ---------------------

def validate_judge_on_mcq(judge_fn, gen_model, gen_tok, gen_key,
                          datasets, cache_root, size, n_max, seed,
                          chat_template_kwargs=None):
    """Generate MCQ answers, gold-label with MCQMatch, judge-label with the local judge,
    and report agreement. MCQMatch is the gold because letter match on MCQ is essentially
    ground truth; agreement is therefore a direct read on judge reliability.
    """
    from hc_benchmark.stage_a_generate import generate_stage_a_local
    from TruthTorchLM.evaluators import MCQMatch

    mcq = MCQMatch()
    report = {}
    for ds in datasets:
        config = _make_config(ds, gen_key, n_max=n_max, size=size, seed=seed)
        print(f"\n[validate:{ds}] generating answers with {gen_key} to judge against MCQMatch ...")
        cache = generate_stage_a_local(config, seed=seed, model=gen_model,
                                       tokenizer=gen_tok, cache_root=cache_root,
                                       chat_template_kwargs=chat_template_kwargs)
        items = cache.read()

        n_scored = agree = abstain = unparsed = 0
        gold_correct_judge_incorrect = gold_incorrect_judge_correct = 0
        examples = []
        for it in items:
            q, ans, gts = it["question"], it["primary_answer"], it["ground_truths"]
            gold = mcq(q, ans, gts, "")          # 1 / 0 / -1
            jl = judge_fn(q, ans, gts)           # 1 / 0 / -1
            if judge_fn.last_verdict == "unparsed":
                unparsed += 1
                if len(examples) < 5:
                    examples.append({"q": q[:80], "answer": ans[:80],
                                     "raw": (judge_fn.last_raw or "")[:80]})
            if gold == -1:
                continue  # not scorable by MCQMatch -> can't serve as gold
            n_scored += 1
            gold_bin = 1 if gold == 1 else 0
            if jl == -1:
                abstain += 1
                continue
            judge_bin = 1 if jl == 1 else 0
            if judge_bin == gold_bin:
                agree += 1
            elif gold_bin == 1:
                gold_correct_judge_incorrect += 1
            else:
                gold_incorrect_judge_correct += 1

        definite = n_scored - abstain
        rate = (agree / definite) if definite else float("nan")
        report[ds] = {
            "n_scored": n_scored,
            "judge_definite": definite,
            "judge_abstained": abstain,
            "judge_unparsed": unparsed,
            "agreement": rate,
            "gold_correct_judge_incorrect": gold_correct_judge_incorrect,
            "gold_incorrect_judge_correct": gold_incorrect_judge_correct,
            "unparsed_examples": examples,
        }
        _print_validation_row(ds, report[ds])
    return report


def _print_validation_row(ds, r):
    rate = r["agreement"]
    rate_s = f"{rate:.3f}" if rate == rate else "n/a"  # nan check
    print(f"[validate:{ds}] agreement={rate_s} on {r['judge_definite']} definite "
          f"(scored={r['n_scored']}, abstained={r['judge_abstained']}, "
          f"unparsed={r['judge_unparsed']}) | judge-wrong: "
          f"gold+/judge- = {r['gold_correct_judge_incorrect']}, "
          f"gold-/judge+ = {r['gold_incorrect_judge_correct']}")


def _overall_agreement(report):
    tot_agree = tot_def = 0
    for r in report.values():
        rate, d = r["agreement"], r["judge_definite"]
        if rate == rate and d:  # skip nan
            tot_agree += round(rate * d)
            tot_def += d
    return (tot_agree / tot_def) if tot_def else float("nan")


# --- phase 1 (alt): construct-from-gold validation (in-format, known labels) --

def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _perturb_answer(text: str) -> str:
    """A fuzzy-but-still-correct rewrite of a gold answer: the case exact match fails.

    Case-fold, strip accents, drop apostrophes/periods, collapse whitespace. Preserves
    meaning ("Ante' Lamore" -> "ante lamore", "Bill D'Elia" -> "bill delia") while breaking
    verbatim string equality -- so a CORRECT verdict here is the judge's real value-add.
    """
    import re as _re
    t = _strip_accents(text).lower()
    t = _re.sub(r"[.'’]", "", t)
    t = _re.sub(r"\s+", " ", t).strip()
    return t


def _norm(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", _strip_accents(str(s)).lower()).strip()


def _answer_type(s: str) -> str:
    """Coarse surface type of a gold answer, for same-type hard negatives."""
    import re as _re
    t = str(s).strip()
    if _re.fullmatch(r"(?:1|2)\d{3}", t):
        return "year"
    if _re.fullmatch(r"[\d][\d,]*(?:\.\d+)?", t):
        return "number"
    words = t.split()
    if 1 <= len(words) <= 5 and t[:1].isupper():
        return "entity"
    return "other"


def _numeric_hard_neg(s: str):
    """A nearby-but-wrong numeric answer (plausible model mistake). None if not numeric."""
    import re as _re
    t = str(s).strip()
    if _re.fullmatch(r"(?:1|2)\d{3}", t):          # year -> shift by 3 (stay 4-digit)
        y = int(t)
        return str(y + 3 if y + 3 <= 2099 else y - 3)
    if _re.fullmatch(r"[\d][\d,]*(?:\.\d+)?", t):  # count/measure -> off by one
        digits = t.replace(",", "")
        try:
            if "." in digits:
                return str(round(float(digits) + 1, 2))
            v = int(digits)
            return str(v + 1 if v != 0 else 1)
        except ValueError:
            return None
    return None


def _construct_cases(items, seed, max_items):
    """Build known-label judge test cases from a dataset's own ground truths.

    Per item: an exact positive (gold verbatim -> CORRECT), a variant positive (fuzzy gold
    -> CORRECT), and a negative (another item's gold -> INCORRECT, guarded against an
    accidental match). No target model is involved -- labels are known by construction.
    """
    import random as _random

    rng = _random.Random(seed)
    usable = [it for it in items
              if it.get("ground_truths") and str(it["ground_truths"][0]).strip()]
    rng.shuffle(usable)
    usable = usable[:max_items]
    n = len(usable)
    cases = []
    if n < 2:
        return cases
    perm = list(range(n))
    rng.shuffle(perm)

    # Pools of first-golds by surface type, for same-type hard negatives.
    by_type = {}
    for it in usable:
        g0 = str(it["ground_truths"][0])
        by_type.setdefault(_answer_type(g0), []).append(g0)

    def _hard_neg(gold, gold_norms):
        """A plausible-but-wrong answer: a nearby number, else a same-type distractor."""
        typ = _answer_type(gold)
        if typ in ("year", "number"):
            hn = _numeric_hard_neg(gold)
            if hn and _norm(hn) not in gold_norms:
                return hn
        pool = by_type.get(typ) or []
        for _ in range(6):
            if not pool:
                break
            cand = pool[rng.randrange(len(pool))]
            if _norm(cand) not in gold_norms:
                return cand
        return None

    for i, it in enumerate(usable):
        q = it["question"]
        gts = [str(g) for g in it["ground_truths"] if str(g).strip()]
        gold = gts[0]
        gold_norms = {_norm(g) for g in gts}
        cases.append({"question": q, "answer": gold, "gts": gts,
                      "expected": 1, "kind": "exact_pos"})
        variant = _perturb_answer(gold)
        if variant and variant != gold:
            cases.append({"question": q, "answer": variant, "gts": gts,
                          "expected": 1, "kind": "variant_pos"})
        # easy negative: an UNRELATED item's gold (sanity floor).
        j, tries = perm[i], 0
        while (j == i or _norm(usable[j]["ground_truths"][0]) in gold_norms) and tries < 6:
            j = (j + 1) % n
            tries += 1
        if j != i:
            cases.append({"question": q, "answer": str(usable[j]["ground_truths"][0]),
                          "gts": gts, "expected": 0, "kind": "neg"})
        # hard negative: a plausible-but-wrong answer of the SAME type as the gold.
        hard = _hard_neg(gold, gold_norms)
        if hard is not None:
            cases.append({"question": q, "answer": str(hard), "gts": gts,
                          "expected": 0, "kind": "hard_neg"})
    return cases


def validate_judge_construct(judge_fn, datasets, size, seed, max_items):
    """Run the judge on constructed known-label cases; report calibration per dataset.

    accuracy counts an abstention (-1) on a clear-cut constructed case as WRONG -- the
    judge is expected to give a definite verdict here. recall_exact / recall_variant /
    specificity break that down by case type.
    """
    from TruthTorchLM.utils.dataset_utils import get_dataset

    report = {}
    for ds in datasets:
        try:
            items = get_dataset(ds, size_of_data=size, seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[validate:{ds}] SKIP -- load failed: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        cases = _construct_cases(items, seed, max_items)
        if not cases:
            print(f"[validate:{ds}] SKIP -- no usable ground-truth items.")
            continue

        buckets = {"exact_pos": [0, 0], "variant_pos": [0, 0],
                   "neg": [0, 0], "hard_neg": [0, 0]}  # [ok, total]
        matched = abstain = unparsed = 0
        examples = []
        for c in cases:
            label = judge_fn(c["question"], c["answer"], c["gts"])
            if getattr(judge_fn, "last_verdict", None) == "unparsed":
                unparsed += 1
            if label == -1:
                abstain += 1
            hit = (label == c["expected"])
            matched += int(hit)
            b = buckets[c["kind"]]
            b[1] += 1
            b[0] += int(hit)
            if not hit and len(examples) < 8:
                examples.append({"kind": c["kind"], "answer": c["answer"][:60],
                                 "gold": c["gts"][0][:60], "label": label,
                                 "raw": (getattr(judge_fn, "last_raw", "") or "")[:60]})

        total = len(cases)

        def _rate(k):
            ok, tot = buckets[k]
            return (ok / tot) if tot else float("nan")

        accuracy = matched / total if total else float("nan")
        report[ds] = {
            "n_cases": total,
            "accuracy": accuracy,
            "recall_exact": _rate("exact_pos"),
            "recall_variant": _rate("variant_pos"),
            "specificity_easy": _rate("neg"),
            "specificity_hard": _rate("hard_neg"),
            "n_exact": buckets["exact_pos"][1],
            "n_variant": buckets["variant_pos"][1],
            "n_neg": buckets["neg"][1],
            "n_hard": buckets["hard_neg"][1],
            "abstain": abstain,
            "unparsed": unparsed,
            "error_examples": examples,
            # compat keys so _overall_agreement can weight construct results too:
            "agreement": accuracy,
            "judge_definite": total,
        }
        _print_construct_row(ds, report[ds])
    return report


def _print_construct_row(ds, r):
    def f(x):
        return f"{x:.3f}" if x == x else "n/a"  # nan check
    print(f"[validate:{ds}] accuracy={f(r['accuracy'])} on {r['n_cases']} cases | "
          f"recall_exact={f(r['recall_exact'])}({r['n_exact']}) "
          f"recall_variant={f(r['recall_variant'])}({r['n_variant']}) | "
          f"specificity_easy={f(r['specificity_easy'])}({r['n_neg']}) "
          f"specificity_hard={f(r['specificity_hard'])}({r['n_hard']}) | "
          f"abstain={r['abstain']} unparsed={r['unparsed']}")


# --- phase 2: label the free-form sets for real -----------------------------

def label_free_form(judge_fn, targets, datasets, cache_root, size, n_max, seed,
                    device, dtype, judge_label, chat_template_kwargs=None, max_items=None):
    """For each target: load once, Stage-A generate per dataset, Stage-B label with the
    right criterion (judge for free-form/long-form, MCQMatch for MCQ).

    Each (target, dataset) is wrapped so one dataset's failure is recorded and the run
    continues -- essential for a full-D-axis health check. The per-dataset status and, on
    success, the judged-correct distribution land in the returned summary.
    """
    import torch
    from hc_benchmark.stage_a_generate import generate_stage_a_local
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector

    mcq_fn = _mcq_fn()
    summary = {}
    for target_key, target_repo in targets:
        print(f"\n[label] loading target {target_key} ({target_repo}) ...")
        tmodel, ttok = _load_model(target_repo, device, dtype)
        _warmup(tmodel, ttok, device)
        summary[target_key] = {}
        for ds in datasets:
            is_mcq = ds in MCQ_DATASETS
            fn = mcq_fn if is_mcq else judge_fn
            provenance = "mcq_match(letter)" if is_mcq else judge_label
            print(f"[label] {target_key} x {ds}: Stage A -> Stage B ({provenance}) ...")
            try:
                config = _make_config(ds, target_key, n_max=n_max, size=size, seed=seed)
                cache = generate_stage_a_local(
                    config, seed=seed, model=tmodel, tokenizer=ttok, cache_root=cache_root,
                    chat_template_kwargs=chat_template_kwargs, max_items=max_items)
                label_stage_b(cache, criteria=("llm_judge",), _judge_fn=fn,
                              judge_model=provenance)
                vec = correctness_vector(cache, "llm_judge")
                dist = {"n": len(vec),
                        "correct": sum(v == 1 for v in vec),
                        "incorrect": sum(v == 0 for v in vec),
                        "not_attempted_or_unparsed": sum(v == -1 for v in vec)}
                summary[target_key][ds] = {"status": "ok", "criterion": provenance, **dist}
                print(f"[label]   OK {ds}: {dist}")
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                summary[target_key][ds] = {"status": "FAILED",
                                           "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                print(f"[label]   FAILED {ds}: {type(exc).__name__}: {str(exc)[:160]}")
        del tmodel
        if device == "cuda":
            torch.cuda.empty_cache()
    return summary


def _resolve_targets(keys):
    """Map registry generator keys -> (key, repo_id), keeping only HF-backed open models."""
    from hc_benchmark.generators import get_generator

    out = []
    for k in keys:
        spec = get_generator(k)
        if spec.backend != "huggingface":
            print(f"[warn] target '{k}' is backend={spec.backend}, not huggingface -- skipping.")
            continue
        out.append((k, spec.model_id))
    return out


def main():
    ap = argparse.ArgumentParser(description="Stage-2 open-model launcher (local judge).")
    ap.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HF repo id of the local judge (non-thinking model recommended).")
    ap.add_argument("--judge-label", default="qwen3-4b-instruct-2507",
                    help="Provenance string recorded into the labels file.")
    ap.add_argument("--targets", nargs="+", default=["qwen3-1.7b"],
                    help="Registry generator keys to label (huggingface backend).")
    ap.add_argument("--datasets", nargs="+", default=JUDGE_FREEFORM,
                    help="Free-form/long-form datasets to label in phase 2.")
    ap.add_argument("--validation-mode", choices=["construct", "mcq"], default="construct",
                    help="construct: known-label cases built from gold on free-form sets "
                         "(in-format, isolates judge quality; recommended). mcq: judge-vs-"
                         "MCQMatch agreement on MCQ (contaminated by letter-vs-text mismatch).")
    ap.add_argument("--validation-datasets", nargs="+", default=None,
                    help="Datasets for validation. Defaults to CONSTRUCT_DEFAULT (construct "
                         "mode) or MCQ_DATASETS (mcq mode).")
    ap.add_argument("--max-construct-items", type=int, default=60,
                    help="construct mode: cap items per dataset (each yields up to 3 cases).")
    ap.add_argument("--validation-target", default="Qwen/Qwen3-1.7B",
                    help="mcq mode only: HF repo id used to generate MCQ answers.")
    ap.add_argument("--validation-target-key", default="qwen3-1.7b",
                    help="mcq mode only: registry key (identity only) for the validation target.")
    ap.add_argument("--size", type=float, default=0.05,
                    help="size_of_data fraction sampled for validation.")
    ap.add_argument("--n-max", type=int, default=1,
                    help="Samples Stage A draws; labeling only reads primary_answer.")
    ap.add_argument("--max-label-items", type=int, default=None,
                    help="Phase 2: cap items generated/labeled per dataset (health check).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-max-new-tokens", type=int, default=16)
    ap.add_argument("--thinking", choices=["off", "on"], default="off",
                    help="Reasoning targets (Qwen3): 'off' passes enable_thinking=False so "
                         "they answer directly (fast, clean, ANSWER_ONLY); 'on' keeps the "
                         "thinking trace (needs a large max_tokens or answers get truncated).")
    ap.add_argument("--agreement-threshold", type=float, default=0.85,
                    help="Minimum judge-vs-gold agreement required to proceed to phase 2.")
    ap.add_argument("--proceed", action="store_true",
                    help="After validation, actually label the free-form sets.")
    ap.add_argument("--skip-validation", action="store_true",
                    help="DANGER: label without measuring judge agreement first.")
    ap.add_argument("--force", action="store_true",
                    help="Proceed to phase 2 even if agreement is below threshold.")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[warn] running on CPU -- fine for a dry check, never for a real run.")
    Path(args.results_root).mkdir(parents=True, exist_ok=True)

    if args.judge in {r for _, r in _resolve_targets(args.targets)}:
        print(f"[warn] judge '{args.judge}' is also a target -- a model judging itself is a "
              f"leakage risk. Consider a different judge.")

    print(f"[judge] loading ONCE: {args.judge} (reused across all datasets/items) ...")
    judge_fn, _jm, _jt = load_judge(args.judge, device, args.dtype,
                                    max_new_tokens=args.judge_max_new_tokens)

    # Reasoning targets answer directly when thinking is off (enable_thinking=False);
    # harmless no-op for non-thinking models (unknown template var).
    ct_kwargs = {"enable_thinking": args.thinking == "on"}
    print(f"[gen] target thinking = {args.thinking} (chat_template_kwargs={ct_kwargs})")

    out = {"judge": args.judge, "judge_label": args.judge_label, "device": device,
           "thinking": args.thinking}

    # -- phase 1: validation --------------------------------------------------
    if not args.skip_validation:
        mode = args.validation_mode
        if mode == "construct":
            datasets = args.validation_datasets or CONSTRUCT_DEFAULT
            print("\n=== PHASE 1: judge validation (construct-from-gold, free-form) ===")
            print(f"[validate] datasets={datasets} -- no target model; labels known by "
                  f"construction. Metric = accuracy vs constructed labels.")
            report = validate_judge_construct(
                judge_fn, datasets, args.size, args.seed, args.max_construct_items,
            )
            metric_name = "accuracy vs constructed labels"
        else:
            datasets = args.validation_datasets or MCQ_DATASETS
            print("\n=== PHASE 1: judge validation on MCQ (gold = MCQMatch) ===")
            print("[validate] NOTE: MCQ answers are letters and the judge grades text, so "
                  "this number is contaminated by format mismatch (see --validation-mode).")
            vt = _load_model(args.validation_target, device, args.dtype)
            _warmup(vt[0], vt[1], device)
            report = validate_judge_on_mcq(
                judge_fn, vt[0], vt[1], args.validation_target_key,
                datasets, args.cache_root, args.size, args.n_max, args.seed,
                chat_template_kwargs=ct_kwargs,
            )
            del vt
            if device == "cuda":
                torch.cuda.empty_cache()
            metric_name = "judge-vs-MCQMatch agreement"

        overall = _overall_agreement(report)
        overall_s = f"{overall:.3f}" if overall == overall else "n/a"
        out["validation"] = {"mode": mode, "metric": metric_name, "per_dataset": report,
                             "overall": overall, "threshold": args.agreement_threshold}
        (Path(args.results_root) / "stage2_judge_validation.json").write_text(
            json.dumps(out["validation"], indent=2, default=str))
        print(f"\n[validate] OVERALL {metric_name} = {overall_s} "
              f"(threshold {args.agreement_threshold}) -> "
              f"{Path(args.results_root) / 'stage2_judge_validation.json'}")

        if not args.proceed:
            print("\nValidation complete. Review the number above, then re-run with "
                  "--proceed to label the free-form sets for real.")
            return
        if overall == overall and overall < args.agreement_threshold and not args.force:
            print(f"\n[abort] {metric_name} {overall_s} < threshold {args.agreement_threshold}. "
                  f"Not labeling. Re-run with --force to override, or pick a stronger judge.")
            return
    elif not args.proceed:
        print("[note] --skip-validation set but not --proceed; nothing to do.")
        return

    # -- phase 2: real labeling ----------------------------------------------
    print("\n=== PHASE 2: labeling free-form sets with the local judge ===")
    targets = _resolve_targets(args.targets)
    t0 = time.perf_counter()
    summary = label_free_form(judge_fn, targets, args.datasets, args.cache_root,
                              args.size, args.n_max, args.seed, device, args.dtype,
                              args.judge_label, chat_template_kwargs=ct_kwargs,
                              max_items=args.max_label_items)
    elapsed = time.perf_counter() - t0
    out["labeling"] = {"summary": summary, "wall_seconds": round(elapsed, 2)}
    (Path(args.results_root) / "stage2_labeling.json").write_text(
        json.dumps(out["labeling"], indent=2, default=str))
    print(f"\nStage-2 labeling done in {elapsed:.1f}s -> "
          f"{Path(args.results_root) / 'stage2_labeling.json'}")


if __name__ == "__main__":
    main()
