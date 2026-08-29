#!/usr/bin/env python
"""LongFact prep for the CoI-Verbalized study: (1) generate long-form answers with a target model
(thinking OFF), (2) factuality-LABEL each answer with an offline SAFE substitute -- Llama-3.3-70B
(4-bit) decomposes the answer into atomic facts, rates each supported/unsupported from its own
knowledge, and returns a factuality fraction, binarized to correct/incorrect. Writes a Stage-A cache
(GenerationCache) + a {item_id: 0/1} labels JSON that scripts/coi_score.py reads via --labels-file.

Two phases (memory-safe): generate all (8B), free it, then load the 70B judge and label all.
    python scripts/coi_longfact_prep.py --model meta-llama/Llama-3.1-8B-Instruct \
        --generator-key llama-3.1-8b --size 150 --seed 1
"""
import argparse, gc, json, os, re, sys
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.dirname(_HERE)
sys.path.insert(0,os.path.join(_REPO,"src")); sys.path.insert(0,_REPO); sys.path.insert(0,_HERE)

GEN_SYS = "You are a knowledgeable assistant. Answer the question thoroughly and factually in a short paragraph."
JUDGE_SYS = ("You are a meticulous fact-checker. You are given a question and an answer. Decompose the "
             "answer into atomic factual claims and, using your own knowledge, rate each claim "
             "'supported' (true) or 'unsupported' (false/dubious/unverifiable). Output ONLY this JSON: "
             '{"claims":[{"claim":"...","verdict":"supported|unsupported"}], "factuality": <fraction supported 0-1>}')


def _apply(tok, chat):
    try:
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="target HF repo (generates the answers)")
    ap.add_argument("--generator-key", required=True, help="cache key, e.g. llama-3.1-8b")
    ap.add_argument("--judge-model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--branches", nargs="+", default=["longfact_objects","longfact_concepts"])
    ap.add_argument("--size", type=int, default=150, help="prompts per branch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen-max-tokens", type=int, default=320)
    ap.add_argument("--threshold", type=float, default=0.8, help="factuality>=thr -> correct")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_coi_longfact"))
    args=ap.parse_args()
    os.makedirs(args.cache_root, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from TruthTorchLM.long_form_generation.utils.dataset_utils import get_dataset
    from hc_benchmark.cache import GenerationCache
    from stage2_open import _make_config
    from tqdm import tqdm

    # -- prompts --
    prompts=[]
    for b in args.branches:
        for r in get_dataset(b, args.size, seed=args.seed):
            prompts.append(r["question"])
    print(f"[lf-prep] {len(prompts)} LongFact prompts ({args.branches})", flush=True)

    def gen_batch(model, tok, texts, max_new):
        outs=[]
        for t in tqdm(texts, desc="[lf-prep] generate", leave=False):
            p=_apply(tok, [{"role":"system","content":GEN_SYS},{"role":"user","content":t}])
            ids=tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o=model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=(tok.pad_token_id or tok.eos_token_id))
            outs.append(tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return outs

    # -- phase 1: generate answers (target 8B) --
    print(f"[lf-prep] loading target {args.model} ...", flush=True)
    tok=AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tgt=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True,
                                             device_map="auto").eval()
    answers=gen_batch(tgt, tok, prompts, args.gen_max_tokens)
    del tgt; gc.collect(); torch.cuda.empty_cache()
    print(f"[lf-prep] generated {len(answers)} answers; answer words median="
          f"{sorted(len(a.split()) for a in answers)[len(answers)//2]}", flush=True)

    # -- phase 2: factuality-judge (Llama-3.3-70B 4-bit) --
    print(f"[lf-prep] loading judge {args.judge_model} (4-bit) ...", flush=True)
    jtok=AutoTokenizer.from_pretrained(args.judge_model, local_files_only=True)
    if jtok.pad_token is None: jtok.pad_token=jtok.eos_token
    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    judge=AutoModelForCausalLM.from_pretrained(args.judge_model, quantization_config=bnb,
                                               device_map="auto", local_files_only=True).eval()
    def factuality(q, a):
        p=_apply(jtok, [{"role":"system","content":JUDGE_SYS},
                        {"role":"user","content":f"question: {q}\nanswer: {a}"}])
        ids=jtok(p, return_tensors="pt").to(judge.device)
        with torch.no_grad():
            o=judge.generate(**ids, max_new_tokens=1024, do_sample=False,
                             pad_token_id=(jtok.pad_token_id or jtok.eos_token_id))
        txt=jtok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m=re.search(r'"factuality"\s*:\s*([0-9]*\.?[0-9]+)', txt)
        if m: return float(m.group(1))
        vs=re.findall(r'"verdict"\s*:\s*"(supported|unsupported)"', txt)
        return (sum(v=="supported" for v in vs)/len(vs)) if vs else None
    facts=[factuality(q,a) for q,a in tqdm(list(zip(prompts,answers)), desc="[lf-prep] judge")]

    # -- write cache + labels --
    items=[]; labels={}
    for i,(q,a,f) in enumerate(zip(prompts,answers,facts)):
        iid=f"lf{i}"
        items.append({"item_id":iid,"question":q,"context":"","ground_truths":[],
                      "primary_answer":a,"samples":[a],"stratum":"longfact","outcome_type":"free"})
        labels[iid]=(int(f>=args.threshold) if isinstance(f,(int,float)) else -1)
    cfg=_make_config("longfact", args.generator_key, n_max=1, size=1.0, seed=args.seed)
    cache=GenerationCache(args.cache_root, cfg, args.seed); cache.write(items)
    labfile=os.path.join(args.cache_root, f"labels_longfact_{args.generator_key}_seed{args.seed}.json")
    json.dump({str(k):v for k,v in labels.items()}, open(labfile,"w"), indent=1)
    # keep the raw factuality fractions too (for threshold sweeps / analysis)
    json.dump({f"lf{i}":facts[i] for i in range(len(facts))},
              open(os.path.join(args.cache_root, f"factuality_{args.generator_key}_seed{args.seed}.json"),"w"), indent=1)
    good=[v for v in labels.values() if v in (0,1)]
    print(f"[lf-prep] wrote cache {cache.path}")
    print(f"[lf-prep] labels: correct={sum(good)}/{len(good)} (thr={args.threshold}); labels file -> {labfile}", flush=True)


if __name__=="__main__":
    main()
