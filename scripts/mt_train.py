#!/usr/bin/env python
"""Multi-target RQ, Stage 6: train ONE proxy cell = (teacher, student, method).

Dispatches the three proxy objectives that differ only in training signal:
  dald   -> masked LoRA SFT on teacher outputs           (scripts/dald_train.py)
  disaad -> SFT + adversarial                            (train_disaad.py, uncertainty_mode=none)
  ours   -> SFT + adversarial + uncertainty-aware term   (train_disaad.py, uncertainty_mode=edl)

All use LoRA r32/alpha64 (DisAAD-match). Output dir: proxy_mt_<method>_<teacher>_<student>/,
with the adapter under <student>/logs/saved_models/best_model so rq5_score.py finds it.

    python scripts/mt_train.py --teacher qwen3-32b --student qwen3-0.6b \
        --student-repo Qwen/Qwen3-0.6B --method ours \
        --sft ~/JasonLucas/outputs/disaad/sft_qwen3-32b_mt.raw_data.json \
        --labels ~/JasonLucas/outputs/disaad/mt_labels_qwen3-32b.json
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DISAAD = os.path.join(_REPO, "third_party", "DisAAD", "scripts")
_CACHE = os.path.expanduser("~/JasonLucas/hf_cache")


def main():
    ap = argparse.ArgumentParser(description="Train one multi-target proxy cell.")
    ap.add_argument("--teacher", required=True, help="teacher/target key, e.g. qwen3-32b, llama3.3-70b")
    ap.add_argument("--student", required=True, help="student key (model.py registry), e.g. qwen3-0.6b")
    ap.add_argument("--student-repo", required=True, help="student HF repo id (for DALD's direct load)")
    ap.add_argument("--method", required=True, choices=["dald", "disaad", "ours"])
    ap.add_argument("--sft", required=True, help="teacher distillation data (<...>.raw_data.json)")
    ap.add_argument("--labels", default="", help="uncertainty labels json (required for method=ours)")
    ap.add_argument("--oracle", default="EccentricityUncertainty",
                    help="uncertainty oracle key inside the labels file (ours)")
    ap.add_argument("--lam", type=float, default=5.0, help="uncertainty loss weight (ours); RQ5 best ~5")
    ap.add_argument("--unc-mode", default="edl", choices=["edl", "head", "both"],
                    help="uncertainty representation for method=ours (RQ5 single-target winner was 'head')")
    ap.add_argument("--out-root", default=os.path.expanduser("~/JasonLucas/outputs/disaad"))
    ap.add_argument("--variant-tag", default="",
                    help="suffix distinguishing sweep configs, e.g. 'ecc_lam2' -> proxy_mt_ours_ecc_lam2_<t>_<s>")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=400, help="prompts in the sft set (train/val split 70/30)")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--merged-dir", default=None,
                    help="where dald/disaad merged full-models are written. Default: <scratch>/mt_merged "
                         "(derived from HF_HOME). Kept OFF the 94G home quota -- merges are regenerable.")
    args = ap.parse_args()

    if args.method == "ours" and not args.labels:
        sys.exit("[mt-train] method=ours requires --labels")

    if args.merged_dir is None:
        # scratch root = parent of the (symlinked) HF_HOME; matches scripts/mt_remerge_score.slurm
        _scratch = os.path.dirname(os.path.realpath(
            os.environ.get("HF_HOME", os.path.expanduser("~/JasonLucas/hf_cache"))))
        args.merged_dir = os.path.join(_scratch, "mt_merged")

    tag = f"_{args.variant_tag}" if args.variant_tag else ""
    out = os.path.join(args.out_root, f"proxy_mt_{args.method}{tag}_{args.teacher}_{args.student}")
    print(f"[mt-train] cell: teacher={args.teacher} student={args.student} method={args.method}\n"
          f"           sft={args.sft}\n           out={out}")

    if args.method == "dald":
        cmd = [sys.executable, os.path.join(_HERE, "dald_train.py"),
               "--sft-data", args.sft, "--student", args.student_repo,
               "--student-key", args.student, "--teacher", args.teacher,
               "--lora-r", str(args.lora_r), "--lora-alpha", str(args.lora_alpha),
               "--out", out]
    else:  # disaad or ours -> the adversarial trainer
        n = args.num_samples
        cmd = [sys.executable, os.path.join(_DISAAD, "train_disaad.py"),
               "--target_model_name", args.teacher,
               "--scoring_model_name", args.student,
               "--train_dataset_path", args.sft,
               "--output_path", out,
               "--train_num_samples", str(max(1, int(n * 0.7))),
               "--val_num_samples", str(max(1, int(n * 0.3))),
               "--epochs", "1", "--seed", "42",
               "--cache_dir", _CACHE, "--gpu_ids", args.gpu_ids,
               "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha)]
        if args.method == "ours":
            cmd += ["--uncertainty_mode", args.unc_mode,
                    "--uncertainty_labels", args.labels,
                    "--uncertainty_oracle", args.oracle,
                    "--uncertainty_lambda", str(args.lam)]
        # method=disaad -> uncertainty_mode defaults to 'none' (vanilla adversarial DisAAD)

    print("[mt-train] cmd:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # DALD/DisAAD are scored by stage_cd_disaad.py, which loads a MERGED full model (not a LoRA
    # adapter). Merge the trained adapter -> <out>/merged. ('ours' is scored by rq5_score.py, which
    # merges the adapter in-memory, so it needs no pre-merge.)
    if args.method in ("dald", "disaad"):
        import glob as _glob
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        hits = _glob.glob(os.path.join(out, "*", "logs", "saved_models", "best_model"))
        if not hits:
            sys.exit(f"[mt-train] trained but no adapter found under {out} to merge")
        adapter = hits[0]
        # merged full-model goes to SCRATCH (regenerable, keeps home quota clear); stage_cd_disaad.py
        # is pointed here by mt_score.slurm.
        os.makedirs(args.merged_dir, exist_ok=True)
        merged = os.path.join(args.merged_dir, f"{args.method}_{args.teacher}_{args.student}")
        print(f"[mt-train] merging {adapter} -> {merged}")
        base = AutoModelForCausalLM.from_pretrained(args.student_repo, torch_dtype=torch.bfloat16,
                                                    local_files_only=True, trust_remote_code=True)
        PeftModel.from_pretrained(base, adapter).merge_and_unload().save_pretrained(merged)
        AutoTokenizer.from_pretrained(args.student_repo, local_files_only=True).save_pretrained(merged)
        print(f"[mt-train] merged -> {merged}")

    print(f"[mt-train] done -> {out}")


if __name__ == "__main__":
    main()
