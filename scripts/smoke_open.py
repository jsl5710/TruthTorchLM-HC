#!/usr/bin/env python
"""Stage-1 ZERO-API smoke: the Stage A->D pipeline on a tiny OPEN model, no API key.

The shipped scripts/smoke_run.py needs an OpenAI key (API target + LLM judge). This is
the no-API counterpart the staged plan's Stage 1 asks for: it drives an open HuggingFace
target through the same four stages, so it proves the G x D -> M -> V machinery on our own
hardware with zero external calls.

What it exercises, on a handful of items:
  * Stage A  -- generate_stage_a_local: primary answer + n_max samples from a local model,
                cached to Parquet (the shared-generation control);
  * Stage B  -- correctness labels via a DETERMINISTIC string-match stand-in judge
                (injected -- no API, no BLEURT download), stored under the llm_judge key;
  * Stage C  -- score with LexicalSimilarity ONLY: pure pairwise ROUGE-L over the cached
                samples, so no model is called and no NLI weights are downloaded;
  * Stage D  -- AUROC / PRR / ECE / MCE / Brier + per-method aux-compute ms and an SLA verdict.

PASS if every [D] row has numbers and both stages cached. AUROC ~0.5 on a handful of items
is expected -- this proves the pipeline, not a research result.

Run it on a GPU (or CPU) node, never the login node:
    python scripts/smoke_open.py --config hc_benchmark/configs/smoke_open.yaml \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --cache-root $WORK/outputs/cache --results-root $WORK/outputs/results
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Make imports work regardless of cwd: repo root (for hc_benchmark) and src (for
# TruthTorchLM). Running `python scripts/smoke_open.py` puts scripts/ on sys.path, NOT the
# repo root, so derive both from this file's location.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)


# --- deterministic, API-free correctness -----------------------------------

def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _match_judge(question, answer, ground_truths):
    """1 if any ground truth appears (normalized) in the answer, else 0.

    A stand-in for the LLM judge so Stage B needs no API. Crude on purpose -- the smoke
    validates wiring, not label quality.
    """
    ans = _normalize(answer)
    if not ans:
        return 0
    for gt in ground_truths or []:
        g = _normalize(str(gt))
        if g and g in ans:
            return 1
    return 0


# --- model loading ----------------------------------------------------------

def _load_model(repo_id, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {repo_id} -> device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(repo_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # left padding is what batch generation wants for decoder-only models
    tok.padding_side = "left"
    if dtype in ("4bit", "bf16-auto"):
        # Multi-GPU placement for a large target (multi-target RQ). Never call .to(device) on a
        # device_map-placed model. When >1 GPU is allocated, RESERVE the last GPU for the judge
        # (loaded separately at cuda:N-1) so the target does not crowd it off cuda:0.
        n = torch.cuda.device_count()
        reserve_judge = n > 1
        cap = lambda i: int(torch.cuda.get_device_properties(i).total_memory * 0.9 / (1024**3))
        usable = range(n - 1) if reserve_judge else range(n)
        max_mem = {i: f"{cap(i)}GiB" for i in usable}
        if reserve_judge:
            max_mem[n - 1] = "0GiB"                    # keep the last GPU empty for the judge
        kw = dict(device_map="auto", max_memory=max_mem)
        if dtype == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(repo_id, **kw)
        model.eval()
        return model, tok
    torch_dtype = torch.bfloat16 if (dtype == "bf16" and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tok


def _warmup(model, tok, device):
    """Convention: warm the GPU before any timing (protocol §5)."""
    import torch

    if device != "cuda":
        return
    inp = tok("warm up", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    print("[warmup] done")


# --- pipeline ---------------------------------------------------------------

def _run(config, model, tok, device, cache_root, results_root):
    from hc_benchmark.stage_a_generate import generate_stage_a_local
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.truth_methods import LexicalSimilarity
    from TruthTorchLM.utils.access_level import is_black_box
    from TruthTorchLM.instrumentation import sla_verdict, summarize

    method = LexicalSimilarity(number_of_generations=config.n_max)
    assert is_black_box(method), "smoke method must be pure black-box"

    seed = config.seeds[0]

    print(f"\n[A] generating (seed={seed}) ...")
    cache = generate_stage_a_local(config, seed=seed, model=model, tokenizer=tok,
                                   cache_root=cache_root)
    assert cache.exists(), "Stage A cache missing"
    print(f"[A] cached {len(cache)} items -> {cache.path.name}")

    print("[B] labelling with deterministic string-match stand-in judge (no API) ...")
    label_stage_b(cache, criteria=("llm_judge",), _judge_fn=_match_judge)
    correctness = correctness_vector(cache, "llm_judge")
    acc = sum(c == 1 for c in correctness)
    print(f"[B] labelled {len(correctness)} items (match-accuracy {acc}/{len(correctness)})")

    print("[C] scoring with LexicalSimilarity (no model call, no NLI) ...")
    scores = score_stage_c(cache, [method], model=config.generator,
                           n_sweep=config.n_sweep, collect_timing=True)

    name = type(method).__name__
    rows = {}
    ev = evaluate_method(scores[name], correctness, truth_method=method,
                         require_calibrated=False)
    for n, metrics in ev.items():
        summ = summarize(scores[name][n]["timing_ms"], warmup=0, label=f"{name} N={n}")
        verdict = sla_verdict(summ)
        rows[f"{name}_N{n}"] = {
            "auroc": metrics.get("auroc"), "prr": metrics.get("prr"),
            "ece": metrics.get("ece"), "mce": metrics.get("mce"),
            "brier": metrics.get("brier"),
            "aux_p50_ms": summ["p50_ms"], "aux_p95_ms": summ["p95_ms"],
            "fits_500ms": verdict["verdicts"]["500ms"],
        }
        print(f"[D] {name:20s} N={n}: AUROC={metrics.get('auroc'):.3f} "
              f"PRR={metrics.get('prr'):.3f} ECE={metrics.get('ece'):.3f} "
              f"Brier={metrics.get('brier'):.3f} | aux p50={summ['p50_ms']:.2f}ms "
              f"fits500ms={verdict['verdicts']['500ms']}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Stage-1 zero-API open-model smoke.")
    ap.add_argument("--config", default="hc_benchmark/configs/smoke_open.yaml")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF repo id of the tiny open target to load.")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache"))
    ap.add_argument("--results-root", default=os.path.expanduser("~/JasonLucas/outputs/results"))
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--dtype", default="bf16", help="bf16 (cuda) | fp32")
    args = ap.parse_args()

    from hc_benchmark.config import load_config
    import torch

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Smoke (open): model={args.model} device={device} "
          f"dataset={config.dataset} size={config.size_of_data} n_max={config.n_max}")
    if device == "cpu":
        print("[warn] running on CPU -- fine for this tiny smoke, but never for real runs.")

    model, tok = _load_model(args.model, device, args.dtype)
    _warmup(model, tok, device)

    t0 = time.perf_counter()
    rows = _run(config, model, tok, device, args.cache_root, args.results_root)
    elapsed = time.perf_counter() - t0

    results = {"generator": config.generator, "model": args.model, "device": device,
               "dataset": config.dataset, "rows": rows, "wall_seconds": round(elapsed, 2)}
    out_dir = Path(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "smoke_open.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSmoke (open) complete in {elapsed:.1f}s. Results -> {out}")
    print("PASS if every [D] row has numbers. AUROC~0.5 on a handful of items is expected.")


if __name__ == "__main__":
    main()
