#!/usr/bin/env python
"""RQ5 — regenerate the VerbalizedConfidence oracle label (the degenerate one), fixed.

The shared VerbalizedConfidence method elicits the confidence WITHOUT thinking-off, so qwen3-8b
emits a <think> trace; its extract_confidence then concatenates every digit/dot -> multiple dots
-> float() fails -> 0.0 for every prompt (constant label). This re-elicits the SAME prompt with
enable_thinking=False and parses robustly (strip <think>, regex the first number), giving a varied
u*(x). Merges the result into rq5_uncertainty_labels.json, replacing only the VerbalizedConfidence
entry (Eccentricity / DiscreteSemanticEntropy untouched). Pure black-box (teacher text only).

    python scripts/rq5_verbalized_labels.py
"""

import argparse
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)


def parse_conf(text):
    """Robust: drop <think>…</think>, take the first number, map 0-100 -> [0,1]."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    v = float(m.group())
    return min(v, 100.0) / 100.0


def _norm(vals):
    v = np.asarray([x if isinstance(x, (int, float)) and np.isfinite(x) else np.nan for x in vals], float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    out = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return [None if np.isnan(x) else float(x) for x in out]


def main():
    ap = argparse.ArgumentParser(description="Fix + regenerate the VerbalizedConfidence RQ5 oracle.")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--labels", default=os.path.expanduser("~/JasonLucas/outputs/disaad/rq5_uncertainty_labels.json"))
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TruthTorchLM.templates import VC_SYSTEM_PROMPT, VC_USER_PROMPT

    d = json.load(open(args.labels))
    prompts, sft_text = d["prompts"], d["sft_text"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[rq5-verb] re-eliciting VerbalizedConfidence for {len(prompts)} prompts (thinking OFF)")

    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dt,
                                                 local_files_only=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)

    from tqdm import tqdm
    u_raw, n_ok, n_bad = [], 0, 0
    for p, ans in zip(tqdm(prompts, desc="[rq5-verb]"), sft_text):
        chat = [{"role": "system", "content": VC_SYSTEM_PROMPT},
                {"role": "user", "content": VC_USER_PROMPT.format(question=p, generated_text=ans)}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        conf = parse_conf(gen)
        if conf is None:
            n_bad += 1; u_raw.append(None)
        else:
            n_ok += 1; u_raw.append(-conf)          # uncertainty = -confidence (higher = more uncertain)

    u_norm = _norm(u_raw)
    distinct = len(set(round(x, 3) for x in u_norm if x is not None))
    print(f"[rq5-verb] parsed {n_ok} ok, {n_bad} unparsed | distinct normalized values: {distinct}")
    if distinct < 5:
        print("[rq5-verb] WARNING: still low-variance — inspect a few raw generations before trusting.")

    d["oracles"]["VerbalizedConfidence"] = {"u_raw": u_raw, "u_norm": u_norm}
    json.dump(d, open(args.labels, "w"), indent=1, default=str)
    print(f"[rq5-verb] merged fixed VerbalizedConfidence into {args.labels}")


if __name__ == "__main__":
    main()
