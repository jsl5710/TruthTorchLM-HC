#!/usr/bin/env python
"""Standalone LoRA-merge for a DALD/DisAAD proxy whose merged model was pruned to save home
quota. Loads base + the preserved adapter, merges, writes a full model to --out (put it on
scratch -- it is regenerable). stage_cd_disaad.py then scores from there.

    python scripts/mt_merge.py --base meta-llama/Llama-3.1-8B-Instruct \
        --proxy-dir ~/JasonLucas/outputs/disaad/proxy_mt_disaad_llama3.3-70b_llama3.1-8b \
        --out /weka/scratch/.../merged_disaad_llama3.3-70b_llama3.1-8b
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="student HF repo id")
    ap.add_argument("--proxy-dir", required=True, help="proxy_mt_<method>_<t>_<s> dir (holds the adapter)")
    ap.add_argument("--out", required=True, help="where to write the merged full model (use scratch)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hits = glob.glob(os.path.join(args.proxy_dir, "*", "logs", "saved_models", "best_model"))
    if not hits:
        raise SystemExit(f"no adapter (best_model) under {args.proxy_dir}")
    adapter = hits[0]
    print(f"[merge] base={args.base}\n[merge] adapter={adapter}\n[merge] out={args.out}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                local_files_only=True, trust_remote_code=True)
    PeftModel.from_pretrained(base, adapter).merge_and_unload().save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base, local_files_only=True).save_pretrained(args.out)
    print(f"[merge] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
