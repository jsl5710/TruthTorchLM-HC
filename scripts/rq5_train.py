#!/usr/bin/env python
"""RQ5 Stage 3 driver — train ONE uncertainty-aware proxy variant.

Reuses the EXISTING DisAAD SFT data (sft_qwen3-8b) and the RQ5 uncertainty labels — train-only,
no data regeneration. Adds the uncertainty term via `extra_train_args` (threaded into
train_disaad.py's new --uncertainty_* flags). One variant = (oracle, mode); a distinct output
path per variant so the six run concurrently without clobbering.

    python scripts/rq5_train.py --oracle EccentricityUncertainty --mode edl
"""

import argparse
import importlib.util
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_disaad_train():
    p = os.path.join(_REPO, "hc_benchmark", "disaad_train.py")
    spec = importlib.util.spec_from_file_location("disaad_train", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Train one RQ5 uncertainty-aware proxy variant.")
    ap.add_argument("--oracle", required=True,
                    choices=["VerbalizedConfidence", "EccentricityUncertainty", "DiscreteSemanticEntropy"])
    ap.add_argument("--mode", required=True, choices=["head", "edl", "both"])
    ap.add_argument("--uncertainty-lambda", type=float, default=1.0)
    ap.add_argument("--uncertainty-topk", type=int, default=10)
    ap.add_argument("--teacher", default="qwen3-8b")
    ap.add_argument("--student", default="qwen3-0.6b")
    ap.add_argument("--num-samples", type=int, default=400)
    ap.add_argument("--labels",
                    default=os.path.expanduser("~/JasonLucas/outputs/disaad/rq5_uncertainty_labels.json"))
    ap.add_argument("--out-root", default=os.path.expanduser("~/JasonLucas/outputs/disaad"))
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--train-seed", type=int, default=42,
                    help="training seed; !=42 gets a _s<seed> suffix (paper-grade error bars).")
    args = ap.parse_args()

    if not os.path.exists(args.labels):
        sys.exit(f"[rq5-train] uncertainty labels not found: {args.labels} (run rq5_build_labels.py first)")

    dt = _load_disaad_train()
    # lambda != 1.0 gets a suffix so a lambda-sweep doesn't clobber the base (lambda=1) variant
    lam = args.uncertainty_lambda
    tag = (f"rq5_{args.oracle}_{args.mode}" + ("" if lam == 1.0 else f"_lam{lam:g}")
           + ("" if args.train_seed == 42 else f"_s{args.train_seed}"))
    cfg = dt.DisAADTrainingConfig(
        seed=args.train_seed,
        teacher_model=args.teacher, student_model=args.student, teacher_is_api=False,
        dataset="tqa",
        ori_dataset_path=os.path.join(args.out_root, "ori_trivia"),
        sft_dataset_path=os.path.join(args.out_root, f"sft_{args.teacher}"),   # REUSE existing SFT data
        proxy_output_path=os.path.join(args.out_root, f"proxy_{tag}"),
        cache_dir=os.path.expanduser("~/JasonLucas/hf_cache"),
        num_samples=args.num_samples,
        train_num_samples=max(1, int(args.num_samples * 0.7)),
        val_num_samples=max(1, int(args.num_samples * 0.3)),
        gpu_ids=args.gpu_ids,
        extra_train_args=[
            "--uncertainty_mode", args.mode,
            "--uncertainty_labels", args.labels,
            "--uncertainty_oracle", args.oracle,
            "--uncertainty_lambda", str(args.uncertainty_lambda),
            "--uncertainty_topk", str(args.uncertainty_topk),
        ],
    )
    print(f"[rq5-train] variant={tag} lambda={args.uncertainty_lambda}\n  proxy -> {cfg.proxy_output_path}")
    cmd = dt.build_train_command(cfg)
    print("  cmd:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    dt.write_training_manifest(cfg)
    print(f"[rq5-train] done -> {cfg.proxy_output_path}")


if __name__ == "__main__":
    main()
