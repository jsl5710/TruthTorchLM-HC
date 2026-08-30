#!/usr/bin/env python
"""Multi-judge validation of the correctness labels.

Every AUROC in these papers is computed against labels from ONE LLM judge. This script
re-labels the same cached (question, gold, answer) triples with THREE judges from different
model families, takes the majority vote, and reports:
  (1) pairwise agreement + Cohen's kappa between judges;
  (2) agreement of the original single judge with the 3-judge majority;
  (3) how many items flip, and---the part that matters---whether the paper's AUROC
      conclusions change when scored against majority-vote labels instead.

It also writes a random sample of items to `judge_spotcheck_sample.jsonl` for a human
(author) pass; we do NOT report a human number until that pass is actually done.

    python scripts/judge_ensemble.py --datasets trivia_qa medlfqa --generator llama-3.1-8b
"""
import argparse, glob, json, os, random, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)

JUDGES = [
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3-4b"),        # the judge used for the papers
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1-8b"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral-7b"),
]
SYS = ("You are a strict grader. Given a question, the reference answer(s), and a candidate "
       "answer, decide whether the candidate is correct (factually equivalent to a reference). "
       "Reply with exactly one word: YES or NO.")
HEALTH = ["medqa", "mmlu_med", "kqa", "medlfqa", "bioasq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="llama-3.1-8b")
    ap.add_argument("--datasets", nargs="+", default=["trivia_qa", "medlfqa"])
    ap.add_argument("--qa-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full"))
    ap.add_argument("--health-cache-root", default=os.path.expanduser("~/JasonLucas/outputs/cache_full_health"))
    ap.add_argument("--out-root", default=os.path.expanduser("~/JasonLucas/outputs/results_judge"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spotcheck-n", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    from tqdm import tqdm

    def apply(tok, chat):
        try:
            return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # ---- gather items + the ORIGINAL single-judge label ----
    cells = {}
    for ds in args.datasets:
        root = args.health_cache_root if ds in HEALTH else args.qa_cache_root
        pq = glob.glob(f"{root}/stageA_{ds}_{args.generator}_*_seed{args.seed}.parquet")
        lb = glob.glob(f"{root}/stageA_{ds}_{args.generator}_*_seed{args.seed}.labels.json")
        if not pq or not lb:
            print(f"[judge] {ds}: cache/labels missing -- skip"); continue
        df = pd.read_parquet(pq[0])
        labs = json.load(open(lb[0]))["labels"]
        items = []
        for i, r in df.iterrows():
            orig = labs.get(str(r["item_id"]), {}).get("correct_llm_judge")
            if orig not in (0, 1):
                continue
            items.append({"item_id": str(r["item_id"]), "question": str(r["question"]),
                          "gold": [str(g) for g in (r["ground_truths"] or [])][:5],
                          "answer": str(r["primary_answer"]), "orig": int(orig)})
        cells[ds] = items
        print(f"[judge] {ds}: {len(items)} items", flush=True)
    if not cells:
        sys.exit("no cells")

    # ---- run each judge ----
    votes = {ds: {it["item_id"]: [] for it in items} for ds, items in cells.items()}
    for repo, tag in JUDGES:
        print(f"[judge] loading {repo} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16,
                                                   local_files_only=True, device_map="auto").eval()
        for ds, items in cells.items():
            for it in tqdm(items, desc=f"[{tag}/{ds}]", leave=False):
                ref = "; ".join(it["gold"]) if it["gold"] else "(none)"
                u = (f"Question: {it['question']}\nReference answer(s): {ref}\n"
                     f"Candidate answer: {it['answer']}\nIs the candidate correct?")
                p = apply(tok, [{"role": "system", "content": SYS}, {"role": "user", "content": u}])
                ids = tok(p, return_tensors="pt").to(mdl.device)
                with torch.no_grad():
                    o = mdl.generate(**ids, max_new_tokens=6, do_sample=False,
                                     pad_token_id=(tok.pad_token_id or tok.eos_token_id))
                txt = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                m = re.search(r"\b(yes|no)\b", txt, re.IGNORECASE)
                votes[ds][it["item_id"]].append(1 if (m and m.group(1).lower() == "yes") else 0)
        del mdl; torch.cuda.empty_cache()

    # ---- agreement + majority vote ----
    out = {}
    for ds, items in cells.items():
        V = votes[ds]
        maj, orig = [], []
        for it in items:
            v = V[it["item_id"]]
            if len(v) != len(JUDGES):
                continue
            maj.append(1 if sum(v) >= 2 else 0); orig.append(it["orig"])
        pair = {}
        for a in range(len(JUDGES)):
            for b in range(a + 1, len(JUDGES)):
                va = [V[it["item_id"]][a] for it in items if len(V[it["item_id"]]) == len(JUDGES)]
                vb = [V[it["item_id"]][b] for it in items if len(V[it["item_id"]]) == len(JUDGES)]
                agree = sum(x == y for x, y in zip(va, vb)) / len(va)
                po = agree
                pe = (sum(va)/len(va))*(sum(vb)/len(vb)) + (1-sum(va)/len(va))*(1-sum(vb)/len(vb))
                kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
                pair[f"{JUDGES[a][1]}~{JUDGES[b][1]}"] = {"agreement": round(agree, 4),
                                                          "kappa": round(kappa, 4)}
        agree_orig = sum(m == o for m, o in zip(maj, orig)) / len(maj)
        out[ds] = {"n": len(maj), "pairwise": pair,
                   "orig_vs_majority_agreement": round(agree_orig, 4),
                   "n_flipped": int(sum(m != o for m, o in zip(maj, orig))),
                   "pos_rate_orig": round(sum(orig)/len(orig), 4),
                   "pos_rate_majority": round(sum(maj)/len(maj), 4),
                   "majority_labels": {it["item_id"]: m for it, m in zip(
                       [i for i in items if len(V[i["item_id"]]) == len(JUDGES)], maj)}}
        print(f"[judge] {ds}: orig-vs-majority agreement={agree_orig:.3f} "
              f"flipped={out[ds]['n_flipped']}/{len(maj)} pairwise={pair}", flush=True)

    dst = f"{args.out_root}/judge_ensemble_{args.generator}_seed{args.seed}.json"
    json.dump(out, open(dst, "w"), indent=2)
    # human spot-check sample (NOT scored here -- for the authors to review)
    rng = random.Random(0); sample = []
    for ds, items in cells.items():
        pick = rng.sample(items, min(args.spotcheck_n // max(1, len(cells)), len(items)))
        for it in pick:
            sample.append({"dataset": ds, **it, "judges": votes[ds][it["item_id"]]})
    sp = f"{args.out_root}/judge_spotcheck_sample.jsonl"
    with open(sp, "w") as f:
        for r in sample: f.write(json.dumps(r) + "\n")
    print(f"[judge] wrote {dst}\n[judge] spot-check sample ({len(sample)} items) -> {sp}")


if __name__ == "__main__":
    main()
