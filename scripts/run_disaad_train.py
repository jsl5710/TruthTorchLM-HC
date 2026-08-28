#!/usr/bin/env python
"""Drive DisAAD proxy distillation (teacher Qwen3-8B -> student Qwen3-0.6B).

Runs in the SEPARATE `disaad` venv (transformers 4.56 / torch 2.7+cu118). Two stages:
  data  : teacher generates responses for the source prompts -> SFT distillation set
  train : adversarially distil the small student proxy
  both  : data then train, then stamp the readiness manifest

Uses hc_benchmark/disaad_train.py's command builders (stdlib-only) but imports it by FILE
path so we don't pull the heavy hc_benchmark package into the disaad venv.

    python scripts/run_disaad_train.py --stage data --num-samples 50   # validate first
    python scripts/run_disaad_train.py --stage both --num-samples 800  # full
"""

import argparse
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_disaad_train():
    p = os.path.join(_REPO, "hc_benchmark", "disaad_train.py")
    spec = importlib.util.spec_from_file_location("disaad_train", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="DisAAD proxy distillation driver.")
    ap.add_argument("--stage", choices=["data", "train", "both"], default="both")
    ap.add_argument("--teacher", default="qwen3-8b")
    ap.add_argument("--student", default="qwen3-0.6b")
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--out-root", default=os.path.expanduser("~/JasonLucas/outputs/disaad"))
    args = ap.parse_args()

    dt = _load_disaad_train()
    cfg = dt.DisAADTrainingConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        teacher_is_api=False,
        dataset="tqa",
        ori_dataset_path=os.path.join(args.out_root, "ori_trivia"),
        sft_dataset_path=os.path.join(args.out_root, f"sft_{args.teacher}"),
        proxy_output_path=os.path.join(args.out_root, f"proxy_{args.student}_from_{args.teacher}"),
        cache_dir=os.path.expanduser("~/JasonLucas/hf_cache"),
        num_samples=args.num_samples,
        train_num_samples=max(1, int(args.num_samples * 0.7)),
        val_num_samples=max(1, int(args.num_samples * 0.3)),
        gpu_ids=args.gpu_ids,
    )
    print(f"[disaad] teacher={cfg.teacher_model} student={cfg.student_model} "
          f"num_samples={cfg.num_samples}\n  ori={cfg.ori_dataset_path}\n  sft={cfg.sft_dataset_path}"
          f"\n  proxy={cfg.proxy_output_path}")

    if args.stage in ("data", "both"):
        print("\n=== Stage 1-1: collect teacher distillation data ===")
        cmd = dt.build_data_builder_command(cfg)
        print("  cmd:", " ".join(cmd))
        import subprocess
        subprocess.run(cmd, check=True)
    if args.stage in ("train", "both"):
        print("\n=== Stage 1-2: distil the student proxy ===")
        cmd = dt.build_train_command(cfg)
        print("  cmd:", " ".join(cmd))
        import subprocess
        subprocess.run(cmd, check=True)
        dt.write_training_manifest(cfg)
        print(f"\n[disaad] wrote readiness manifest -> {cfg.proxy_output_path}")


if __name__ == "__main__":
    main()
