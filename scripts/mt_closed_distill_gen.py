#!/usr/bin/env python
"""Closed-teacher distillation-data generation (login node, gateway). For each of the FIXED
DisAAD-mixed prompts, draw the API teacher's samples and save in the exact sft_prompt/sft_text
format train_disaad.py / dald_train.py consume -- so a proxy can be distilled from a *closed*
frontier teacher (gpt-4o, claude-haiku). Reuses stage_cd_api's gateway call (GATEWAY_KEY env).

sft_text[i] = [greedy, sample1..sampleN-1]  (matches the open data_builder layout: [0] low-temp,
[1:] high-temp blackbox answers).

    source ~/JasonLucas/.gateway_key
    python scripts/mt_closed_distill_gen.py --model openai/gpt-4o --key jhu-gpt-4o
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gateway model string, e.g. openai/gpt-4o")
    ap.add_argument("--key", required=True, help="generator key, e.g. jhu-gpt-4o (names the output)")
    ap.add_argument("--prompts", default=os.path.expanduser("~/JasonLucas/data/distill/mixed_prompts"))
    ap.add_argument("--n", type=int, default=10, help="samples per prompt (incl. 1 greedy)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--out-root", default=os.path.expanduser("~/JasonLucas/outputs/disaad"))
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all; small for a smoke test.")
    args = ap.parse_args()

    from datasets import load_from_disk
    from stage_cd_api import _gateway_call

    if "GATEWAY_KEY" not in os.environ:
        sys.exit("GATEWAY_KEY not set (source ~/JasonLucas/.gateway_key)")

    ds = load_from_disk(args.prompts)
    prompts = [r["prompt"].strip() for r in ds if r.get("prompt", "").strip()]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"[closed-distill] {len(prompts)} prompts; teacher={args.model}; N={args.n}", flush=True)

    sft_prompt, sft_text = [], []
    from tqdm import tqdm
    for i, p in enumerate(tqdm(prompts, desc=f"[distill/{args.key}]")):
        greedy, _ = _gateway_call(args.model, p, 0.0, args.max_tokens)
        samples = [greedy]
        for _ in range(args.n - 1):
            s, _ = _gateway_call(args.model, p, args.sample_temperature, args.max_tokens)
            samples.append(s)
        # keep the prompt only if it has a usable greedy + enough non-empty samples
        good = [s for s in samples if s and len(s.split()) >= args.min_words]
        if len(good) >= max(2, args.n // 2):
            sft_prompt.append(p)
            sft_text.append((samples + [""] * args.n)[:args.n])   # pad to N, preserve order

    out = os.path.join(args.out_root, f"sft_{args.key}_mt")
    os.makedirs(args.out_root, exist_ok=True)
    json.dump({"sft_prompt": sft_prompt, "sft_text": sft_text},
              open(f"{out}.raw_data.json", "w"), indent=1)
    json.dump({"teacher": args.model, "key": args.key, "n": args.n, "n_prompts": len(sft_prompt)},
              open(f"{out}.args.json", "w"), indent=2)
    print(f"[closed-distill] wrote {out}.raw_data.json ({len(sft_prompt)} prompts x {args.n} samples)", flush=True)


if __name__ == "__main__":
    main()
