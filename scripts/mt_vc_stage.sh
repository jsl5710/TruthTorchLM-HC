#!/usr/bin/env bash
# VC-oracle test (Llama only): does distilling "Ours" toward VerbalizedConfidence (the strongest
# DIRECT method on Llama-70B, 0.707) beat the Eccentricity-oracle base on the best student cell?
# Single cell: llama3.3-70b -> llama3.2-3b, ours/edl/lam5, VC oracle. Three DEPENDENT jobs:
#   1) mt-vc-labels  : add VerbalizedConfidence oracle to the llama label file (4-bit 70B, ~364 calls)
#   2) mt-train      : train the VC-oracle Ours proxy (afterok:labels)
#   3) mt-score      : per-dataset AUROC (afterok:train), -> results_mt_score/ours_vc_lam5_...
# Compare to the existing base cell results_mt_score/ours_llama3.3-70b_llama3.2-3b (edl/ecc/lam5).
#   DRY=1 scripts/mt_vc_stage.sh      # preview the exact sbatch commands, submit nothing
#   scripts/mt_vc_stage.sh            # submit the chain
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"; mkdir -p "$LOG"

# Teacher/student parameterized (defaults = Llama arm). Override via env for the Qwen control arm:
#   TEACHER=qwen3-32b TEACHER_REPO=Qwen/Qwen3-32B STUDENT=qwen3-4b \
#     STUDENT_REPO=Qwen/Qwen3-4B-Instruct-2507 scripts/mt_vc_stage.sh
TEACHER="${TEACHER:-llama3.3-70b}"
TEACHER_REPO="${TEACHER_REPO:-meta-llama/Llama-3.3-70B-Instruct}"
STUDENT="${STUDENT:-llama3.2-3b}"
STUDENT_REPO="${STUDENT_REPO:-meta-llama/Llama-3.2-3B-Instruct}"
TAG="${TAG:-vc_lam5}"
VC_LABELS="$WORK/outputs/disaad/mt_labels_${TEACHER}_vc.json"
SFT="$WORK/outputs/disaad/sft_${TEACHER}_mt.raw_data.json"
common="--cpus-per-task=12"

SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }

# 1) VC labels
lb=$(TEACHER=$TEACHER TEACHER_REPO=$TEACHER_REPO \
     SUB --job-name=mt-vc-labels-$TEACHER --gpus=1 --time=04:00:00 \
        --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err scripts/mt_vc_labels.slurm)
echo "vc-labels job = $lb"

# 2) train the VC-oracle Ours proxy (depends on labels)
dep=""; [[ -n "$lb" ]] && dep="--dependency=afterok:$lb"
tr=$(TEACHER=$TEACHER STUDENT=$STUDENT STUDENT_REPO=$STUDENT_REPO METHOD=ours \
     ORACLE=VerbalizedConfidence LAM=5.0 MODE=edl TAG=$TAG LABELS="$VC_LABELS" SFT="$SFT" \
     SUB --job-name=mt-train-ours-$TAG-$TEACHER-$STUDENT --gpus=1 --time=20:00:00 $dep \
        --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err scripts/mt_train.slurm)
echo "train job = $tr"

# 3) score (depends on train)
sdep=""; [[ -n "$tr" ]] && sdep="--dependency=afterok:$tr"
sc=$(TEACHER=$TEACHER STUDENT=$STUDENT STUDENT_REPO=$STUDENT_REPO METHOD=ours \
     TAG=$TAG MODE=edl CACHE="$WORK/outputs/cache_mt" \
     SUB --job-name=mt-score-ours-$TAG-$TEACHER-$STUDENT --gpus=1 --time=06:00:00 $sdep \
        --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err scripts/mt_score.slurm)
echo "score job = $sc"
echo
echo "chain: labels($lb) -> train($tr) -> score($sc)"
echo "result will land in: $WORK/outputs/results_mt_score/ours_${TAG}_${TEACHER}_${STUDENT}/"
echo "compare against base: $WORK/outputs/results_mt_score/ours_${TEACHER}_${STUDENT}/ (edl/ecc/lam5)"
