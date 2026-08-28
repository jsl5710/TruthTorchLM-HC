#!/usr/bin/env bash
# Multi-target RQ orchestrator: submit the whole Stage 5-6 matrix with Slurm dependencies.
#   2 eval (direct methods on each target) + 2 label builds + 18 train (2 teachers x 3 students x
#   3 methods) + 18 score. Training waits on its teacher's datagen; 'ours' also waits on labels;
#   scoring waits on train + that teacher's eval.
#
# Usage: pass the RUNNING datagen job ids so dependents chain onto them.
#   DATAGEN_QWEN=49429 DATAGEN_LLAMA=49430 scripts/mt_submit_matrix.sh          # submit for real
#   DRY=1 DATAGEN_QWEN=49429 DATAGEN_LLAMA=49430 scripts/mt_submit_matrix.sh    # print only
set -uo pipefail           # NOT -e: one failed submission must not abort the whole matrix
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"
SUB() { if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
          local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
          [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }

# teacher -> (repo, datagen-jobid, eval GPUs); student -> repo
declare -A TREPO=( [qwen3-32b]=Qwen/Qwen3-32B [llama3.3-70b]=meta-llama/Llama-3.3-70B-Instruct )
declare -A EVAL_GPUS=( [qwen3-32b]=2 [llama3.3-70b]=3 )      # target + 1 reserved for the judge
# Stage 4b datagen is already COMPLETE (its sft_<teacher>_mt.raw_data.json exist on disk), so
# labels/train do NOT depend on it -- and an afterok on a purged completed job is in fact rejected
# ("Job dependency problem"). Only intra-matrix deps remain (score waits on train + eval).
declare -A SREPO=(
  [qwen3-0.6b]=Qwen/Qwen3-0.6B [qwen3-1.7b]=Qwen/Qwen3-1.7B [qwen3-4b]=Qwen/Qwen3-4B-Instruct-2507
  [llama3.2-1b]=meta-llama/Llama-3.2-1B-Instruct [llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct
  [llama3.1-8b]=meta-llama/Llama-3.1-8B-Instruct )
declare -A STUDENTS=( [qwen3-32b]="qwen3-0.6b qwen3-1.7b qwen3-4b"
                      [llama3.3-70b]="llama3.2-1b llama3.2-3b llama3.1-8b" )
METHODS="dald disaad ours"

common() { echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }

EVAL_ONLY="${EVAL_ONLY:-}"     # optional: comma list of teachers whose eval is already submitted -> skip
for T in qwen3-32b llama3.3-70b; do
  # --- Stage 5: direct methods on this target (no datagen dep) ---
  if [[ ",$EVAL_ONLY," == *",$T,"* ]]; then
    ev="${EVAL_ID:-}"; echo "eval[$T]=SKIPPED (reusing $ev)"
  else
    ev=$(TEACHER=$T TARGET="${TREPO[$T]}" SUB --job-name=mt-eval-$T --gpus="${EVAL_GPUS[$T]}" \
          --time=16:00:00 $(common) scripts/mt_eval.slurm)
    echo "eval[$T]=$ev"
  fi
  # --- Stage 4c: uncertainty labels (datagen already on disk, no dep) ---
  lb=$(TEACHER=$T TEACHER_REPO="${TREPO[$T]}" SUB --job-name=mt-labels-$T --gpus=1 \
        --time=03:00:00 $(common) scripts/mt_labels.slurm)
  echo "labels[$T]=$lb"
  # --- Stage 6: 3 students x 3 methods ---
  for S in ${STUDENTS[$T]}; do
    for M in $METHODS; do
      dep=""; [[ "$M" == "ours" && -n "$lb" ]] && dep="--dependency=afterok:$lb"
      tr=$(TEACHER=$T STUDENT=$S STUDENT_REPO="${SREPO[$S]}" METHOD=$M \
            SUB --job-name=mt-train-$M-$T-$S --gpus=1 --time=20:00:00 $dep $(common) scripts/mt_train.slurm)
      sdep=""; [[ -n "$tr" && -n "$ev" ]] && sdep="--dependency=afterok:$tr:$ev" || { [[ -n "$tr" ]] && sdep="--dependency=afterok:$tr"; }
      sc=$(TEACHER=$T STUDENT=$S STUDENT_REPO="${SREPO[$S]}" METHOD=$M \
            SUB --job-name=mt-score-$M-$T-$S --gpus=1 --time=06:00:00 $sdep $(common) scripts/mt_score.slurm)
      echo "  cell $M/$T/$S: train=$tr score=$sc"
    done
  done
done
echo "Submitted the multi-target matrix. After all mt-score jobs finish: python scripts/mt_report.py"
