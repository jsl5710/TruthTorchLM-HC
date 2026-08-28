#!/usr/bin/env python
"""RQ5 Stage 1 — build pure-black-box UNCERTAINTY LABELS for uncertainty-aware distillation.

For each of the DisAAD training prompts (the 376 tqa `sft_prompt`s), draw N teacher (qwen3-8b)
text samples and compute THREE black-box uncertainty oracles over them — the ones RQ3 ranked
highest on qwen3-8b:

    VerbalizedConfidence (0.685) · EccentricityUncertainty (0.672) · DiscreteSemanticEntropy (0.595)

Everything is TEXT-ONLY from the teacher (samples + a verbalized confidence call) — no logits,
no hidden states — so it stays in DisAAD's pure black-box setting. Emits an uncertainty label
u*(x) per prompt per oracle (raw + min-max normalized to [0,1], higher = more uncertain), aligned
to the existing sft_prompt/sft_text pairs, for the training stage to regress against.

    python scripts/rq5_build_labels.py --n 10
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

ORACLES = ["VerbalizedConfidence", "EccentricityUncertainty", "DiscreteSemanticEntropy"]
SYS = "You are a helpful assistant. Answer the question concisely."


class MiniCache:
    """Minimal cache so score_stage_c can consume in-memory items."""
    def __init__(self, items): self._items = items
    def read(self, n=None):
        return self._items if n is None else [{**it, "samples": list(it["samples"])[:n]} for it in self._items]
    def exists(self): return True
    def __len__(self): return len(self._items)


def _norm(vals):
    v = np.asarray([x if isinstance(x, (int, float)) and np.isfinite(x) else np.nan for x in vals], float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    out = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return [None if np.isnan(x) else float(x) for x in out]


def main():
    ap = argparse.ArgumentParser(description="RQ5 black-box uncertainty labels for distillation.")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--sft-data",
                    default=os.path.expanduser("~/JasonLucas/outputs/disaad/sft_qwen3-8b.raw_data.json"))
    ap.add_argument("--n", type=int, default=10, help="teacher samples per prompt for the oracles.")
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all (smoke with small).")
    ap.add_argument("--out", default=os.path.expanduser("~/JasonLucas/outputs/disaad/rq5_uncertainty_labels.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--tokenizer", default="",
                    help="tokenizer source (default: --teacher). Set a local stand-in (e.g. Qwen/Qwen3-0.6B) "
                         "for CLOSED teachers that have no local tokenizer -- the NLI oracles only cluster "
                         "the sample texts, so the specific tokenizer is a formality.")
    ap.add_argument("--use-existing-samples", action="store_true",
                    help="Compute the oracle over the sft_text samples already collected in Stage 4b "
                         "(no teacher reload, no re-sampling). Restricts oracles to the NLI-based ones "
                         "(Eccentricity, DiscreteSemanticEntropy) -- used for the multi-target RQ so the "
                         "70B teacher never has to be reloaded just to label.")
    ap.add_argument("--augment-vc", default="",
                    help="path to an existing mt label file; compute ONLY VerbalizedConfidence over its "
                         "stored prompts + primary answers (the teacher self-reports on each -- no "
                         "re-sampling of dispersion oracles), MERGE, and write to --out. Adds the VC "
                         "oracle to a file built with --use-existing-samples so 'Ours' can distill toward it.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="load the teacher 4-bit (nf4, device_map=auto) -- for large teachers (70B) that "
                         "don't fit bf16 on one GPU. VC only elicits a short confidence, so 4-bit is fine.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from stage_cd_open import build_bb_methods
    from hc_benchmark.stage_c_score import score_stage_c
    from TruthTorchLM.generation import sample_generations_hf_local

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    def _load_teacher():
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True)
            return AutoModelForCausalLM.from_pretrained(
                args.teacher, quantization_config=bnb, device_map="auto",
                local_files_only=True).eval()
        return AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=dt, local_files_only=True).to(device).eval()

    # --- focused mode: add ONLY the VerbalizedConfidence oracle to an existing label file ---
    if args.augment_vc:
        base = json.load(open(args.augment_vc))
        prompts, sft_text = base["prompts"], base["sft_text"]
        if args.max_prompts:
            prompts, sft_text = prompts[:args.max_prompts], sft_text[:args.max_prompts]
        print(f"[rq5-labels/vc] augment {args.augment_vc}: {len(prompts)} prompts, "
              f"teacher={args.teacher} 4bit={args.load_in_4bit}")
        teacher = _load_teacher()
        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.teacher, local_files_only=True)
        vc = [m for m in build_bb_methods(1, device, include_verbalized=True,
                                          judge_model=None, judge_tok=None)
              if type(m).__name__ == "VerbalizedConfidence"]
        print(f"[rq5-labels/vc] methods: {[type(m).__name__ for m in vc]}")

        def _pa(x):
            return x[0] if isinstance(x, list) and x else (x if isinstance(x, str) else "")
        items = [{"item_id": i, "question": p, "primary_answer": _pa(sft_text[i]),
                  "samples": [_pa(sft_text[i])], "ground_truths": [""]}
                 for i, p in enumerate(prompts)]
        scores = score_stage_c(MiniCache(items), vc, model=teacher, tokenizer=tok,
                               n_sweep=(1,), collect_timing=False)
        out = dict(base)
        out["oracles"] = dict(base.get("oracles", {}))
        tv = scores.get("VerbalizedConfidence", {}).get(1, {}).get("truth_values", [])
        u_raw = [(-x if isinstance(x, (int, float)) else None) for x in tv]
        out["oracles"]["VerbalizedConfidence"] = {"u_raw": u_raw, "u_norm": _norm(u_raw)}
        ok = [x for x in out["oracles"]["VerbalizedConfidence"]["u_norm"] if x is not None]
        print(f"  VerbalizedConfidence labels ok={len(ok)}/{len(prompts)}"
              + (f" mean_u={np.mean(ok):.3f}" if ok else " (NONE -- check elicitation!)"))
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print(f"[rq5-labels/vc] wrote {args.out} (oracles: {list(out['oracles'].keys())})")
        return

    d = json.load(open(args.sft_data))
    prompts, sft_text = d["sft_prompt"], d["sft_text"]
    if args.max_prompts:
        prompts, sft_text = prompts[:args.max_prompts], sft_text[:args.max_prompts]
    print(f"[rq5-labels] {len(prompts)} prompts; teacher={args.teacher}; N={args.n}; "
          f"existing_samples={args.use_existing_samples}")

    from tqdm import tqdm
    if args.use_existing_samples:
        # NLI-only oracles over the Stage-4b samples: no teacher WEIGHTS needed -- but the base
        # TruthMethod.__call__ still runs fix_tokenizer_chat(tokenizer, ...), so a valid tokenizer
        # is required (model stays None; forward_hf_local skips generation when samples are cached).
        teacher = None
        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.teacher, local_files_only=True)
        nli_oracles = ["EccentricityUncertainty", "DiscreteSemanticEntropy"]
        methods = [m for m in build_bb_methods(args.n, device, include_verbalized=False,
                                               judge_model=None, judge_tok=None)
                   if type(m).__name__ in nli_oracles]
        print(f"[rq5-labels] oracles (existing samples): {[type(m).__name__ for m in methods]}")
        items = []
        for i, p in enumerate(prompts):
            samples = list(sft_text[i]) if isinstance(sft_text[i], list) else [sft_text[i]]
            items.append({"item_id": i, "question": p, "primary_answer": samples[0] if samples else "",
                          "samples": samples[:args.n], "ground_truths": [""]})
    else:
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dt,
                                                       local_files_only=True).to(device).eval()
        tok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
        methods = [m for m in build_bb_methods(args.n, device, include_verbalized=True,
                                               judge_model=None, judge_tok=None)
                   if type(m).__name__ in ORACLES]
        print(f"[rq5-labels] oracles: {[type(m).__name__ for m in methods]}")
        items = []
        for i, p in enumerate(tqdm(prompts, desc="[rq5-labels] teacher sampling")):
            text = tok.apply_chat_template([{"role": "system", "content": SYS},
                                            {"role": "user", "content": p}],
                                           add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
            sd = sample_generations_hf_local(model=teacher, input_text=text, tokenizer=tok,
                                             number_of_generations=args.n, return_text=True,
                                             batch_generation=True, temperature=args.sample_temperature,
                                             top_p=0.95, max_new_tokens=args.max_new_tokens, do_sample=True)
            items.append({"item_id": i, "question": p, "primary_answer": sft_text[i],
                          "samples": (sd or {}).get("generated_texts", []), "ground_truths": [""]})

    scores = score_stage_c(MiniCache(items), methods, model=teacher, tokenizer=tok,
                           n_sweep=(args.n,), collect_timing=False)
    # uncertainty = -truth_value (truth_value: higher = more truthful = less uncertain)
    labels = {"teacher": args.teacher, "n": args.n, "prompts": prompts, "sft_text": sft_text,
              "oracles": {}}
    for m in methods:
        name = type(m).__name__
        tv = scores.get(name, {}).get(args.n, {}).get("truth_values", [])
        u_raw = [(-x if isinstance(x, (int, float)) else None) for x in tv]
        labels["oracles"][name] = {"u_raw": u_raw, "u_norm": _norm(u_raw)}
        ok = [x for x in labels["oracles"][name]["u_norm"] if x is not None]
        print(f"  {name:26s} labels ok={len(ok)}/{len(prompts)} mean_u={np.mean(ok):.3f}" if ok else f"  {name}: no labels")

    json.dump(labels, open(args.out, "w"), indent=1, default=str)
    print(f"\n[rq5-labels] wrote {args.out}")


if __name__ == "__main__":
    main()
