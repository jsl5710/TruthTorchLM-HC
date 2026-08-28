#!/usr/bin/env python
"""Build the FIXED DisAAD-mixed distillation prompt set for the multi-target RQ.

DisAAD recipe (paper 4.3): half in-domain eval prompts + half OOD conversational.
Here the in-domain half is sampled across the 7-dataset eval subset; the OOD half is
WildChat (pre-extracted). The SAME prompt set is then generated-from by every teacher
(Qwen3-32B, Llama-3.3-70B), so teacher size is the only variable that changes downstream.

Output: an arrow dataset at --out with columns {prompt, source}, exactly the format
DisAAD's data_builder.generate_finetune_dataset (what_to_do=local) consumes.

CPU-only (dataset loading) -> runs on the login node.

    PYTHONPATH=src python scripts/build_distill_prompts.py
"""
import argparse
import os
import sys

# in-domain half: one dataset per category (the 7-subset). Draw distillation prompts
# from the TRAIN split where it exists, else the test split (tail slice) so they are
# disjoint from the first-N eval items where possible. Overlap, if any, is method-neutral
# (every proxy trains on the same set; direct methods don't train) so the RQ comparison
# is unaffected.
IN_DOMAIN = ["trivia_qa", "bioasq", "medqa", "medlfqa", "gsm8k", "truthful_qa", "wikipedia_factual"]
HAS_TRAIN = {"trivia_qa", "gsm8k", "medqa"}          # datasets with a usable train split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=30, help="in-domain prompts per dataset")
    ap.add_argument("--wildchat", type=int, default=190, help="OOD conversational prompts")
    ap.add_argument("--eval-holdout", type=int, default=150, help="reserve first-N test items for eval (tail-slice distill)")
    ap.add_argument("--wildchat-file", default=os.path.expanduser("~/JasonLucas/data/distill/wildchat_ood_prompts.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/JasonLucas/data/distill/mixed_prompts"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import json
    import random
    from datasets import Dataset

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from TruthTorchLM.utils.dataset_utils import get_dataset

    rng = random.Random(args.seed)
    rows = []

    # ---- in-domain half ----
    for ds in IN_DOMAIN:
        split = "train" if ds in HAS_TRAIN else "test"
        data = get_dataset(ds, size_of_data=1.0, seed=args.seed, split=split)
        qs = [d["question"].strip() for d in data if d.get("question", "").strip()]
        if split == "train":
            pool = qs                                   # train split -> no eval overlap
        else:
            pool = qs[args.eval_holdout:] or qs         # tail slice, disjoint from first-N eval
        rng.shuffle(pool)
        take = pool[: args.per_dataset]
        rows += [{"prompt": p, "source": f"indomain:{ds}"} for p in take]
        print(f"[in-domain] {ds:18s} split={split:5s} pool={len(pool):5d} took={len(take)}")

    # ---- OOD conversational half (WildChat) ----
    wc = json.load(open(args.wildchat_file))
    rng.shuffle(wc)
    wc = wc[: args.wildchat]
    rows += [{"prompt": p, "source": "wildchat"} for p in wc]
    print(f"[ood]       wildchat took={len(wc)}")

    rng.shuffle(rows)
    out = Dataset.from_list(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.save_to_disk(args.out)
    n_in = sum(r["source"].startswith("indomain") for r in rows)
    print(f"\n[build] {len(rows)} prompts ({n_in} in-domain / {len(rows)-n_in} OOD) -> {args.out}")


if __name__ == "__main__":
    main()
