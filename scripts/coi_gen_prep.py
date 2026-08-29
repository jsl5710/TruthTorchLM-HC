#!/usr/bin/env python
"""Generate + gold-label a dataset for the CoI study on an open target (for datasets not already
cached, e.g. gsm8k / hotpot_qa on the 8B models). Phase 1: generate one answer per prompt with the
target (thinking OFF). Phase 2: label correctness with the local Qwen3-4B judge, GOLD-GROUNDED
(the judge sees the reference answer(s)). Writes a Stage-A GenerationCache + a {item_id:0/1} labels
JSON that scripts/coi_score.py reads via --labels-file.

    python scripts/coi_gen_prep.py --model meta-llama/Llama-3.1-8B-Instruct \
        --generator-key llama-3.1-8b --dataset gsm8k --size 150 --seed 1
"""
import argparse, gc, json, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)

GEN_SYS = "You are a helpful assistant. Answer the question correctly and concisely."
JUDGE_SYS = ("You are a strict grader. Given a question, the reference answer(s), and a candidate "
             "answer, decide if the candidate is correct (same final answer / factually equivalent). "
             "Reply with exactly one word: YES or NO.")


def _apply(tok, chat):
    try:
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator-key", required=True)
    ap.add_argument("--dataset", required=True, help="get_dataset key, e.g. gsm8k or hotpot_qa")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen-max-tokens", type=int, default=512)
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_coi_gen"))
    args = ap.parse_args()
    os.makedirs(args.cache_root, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TruthTorchLM.utils.dataset_utils import get_dataset
    from hc_benchmark.cache import GenerationCache
    from stage2_open import _make_config
    from tqdm import tqdm

    rows = get_dataset(args.dataset, size_of_data=args.size, seed=args.seed)
    prompts = [r["question"] for r in rows]
    golds = [r.get("ground_truths") or [] for r in rows]
    print(f"[gen-prep] {len(prompts)} {args.dataset} prompts", flush=True)

    def gen_batch(model, tok, texts, sys_prompt, max_new, enable_think=False):
        outs = []
        for t in tqdm(texts, desc=f"[gen-prep] generate", leave=False):
            p = _apply(tok, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": t}])
            ids = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=(tok.pad_token_id or tok.eos_token_id))
            outs.append(tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return outs

    # -- phase 1: generate answers (target) --
    print(f"[gen-prep] loading target {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               local_files_only=True, device_map="auto").eval()
    answers = gen_batch(tgt, tok, prompts, GEN_SYS, args.gen_max_tokens)
    del tgt; gc.collect(); torch.cuda.empty_cache()

    # -- phase 2: gold-grounded judge (Qwen3-4B) --
    print(f"[gen-prep] loading judge {args.judge_model} ...", flush=True)
    jtok = AutoTokenizer.from_pretrained(args.judge_model, local_files_only=True)
    if jtok.pad_token is None: jtok.pad_token = jtok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=torch.bfloat16,
                                                 local_files_only=True, device_map="auto").eval()

    def grade(q, gold, a):
        ref = "; ".join(str(g) for g in gold[:5]) if gold else "(none)"
        u = f"Question: {q}\nReference answer(s): {ref}\nCandidate answer: {a}\nIs the candidate correct?"
        p = _apply(jtok, [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": u}])
        ids = jtok(p, return_tensors="pt").to(judge.device)
        with torch.no_grad():
            o = judge.generate(**ids, max_new_tokens=8, do_sample=False,
                               pad_token_id=(jtok.pad_token_id or jtok.eos_token_id))
        txt = jtok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r'\b(yes|no)\b', txt, re.IGNORECASE)
        return 1 if (m and m.group(1).lower() == "yes") else 0

    labels_list = [grade(q, g, a) for q, g, a in tqdm(list(zip(prompts, golds, answers)), desc="[gen-prep] judge")]

    # -- write cache + labels --
    items, labels = [], {}
    for i, (q, g, a, c) in enumerate(zip(prompts, golds, answers, labels_list)):
        iid = f"{args.dataset[:3]}{i}"
        items.append({"item_id": iid, "question": q, "context": "", "ground_truths": list(g),
                      "primary_answer": a, "samples": [a], "stratum": args.dataset, "outcome_type": "free"})
        labels[iid] = int(c)
    cfg = _make_config(args.dataset, args.generator_key, n_max=1, size=1.0, seed=args.seed)
    cache = GenerationCache(args.cache_root, cfg, args.seed); cache.write(items)
    labfile = os.path.join(args.cache_root, f"labels_{args.dataset}_{args.generator_key}_seed{args.seed}.json")
    json.dump({str(k): v for k, v in labels.items()}, open(labfile, "w"), indent=1)
    npos = sum(labels.values())
    print(f"[gen-prep] wrote {cache.path}")
    print(f"[gen-prep] labels: correct={npos}/{len(labels)}; -> {labfile}", flush=True)


if __name__ == "__main__":
    main()
