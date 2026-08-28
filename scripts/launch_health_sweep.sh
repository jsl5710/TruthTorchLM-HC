#!/usr/bin/env bash
# Launch the OPEN-model HEALTH-dataset sweep: 6 generators x 5 health datasets x 3 seeds.
# Mirrors the completed general-QA sweep settings exactly (N-sweep 1/5/10, max-items 150,
# size 1.0, verbalized included -- open targets run locally so the live confidence call is
# free). MCQ sets (medqa, mmlu_med) auto-route to MCQMatch; the 3 free-form sets use the
# local Qwen3-4B judge. Separate output roots so results_full (general QA) is untouched.
#
#   bash scripts/launch_health_sweep.sh --dry-run   # print sbatch lines, submit nothing
#   bash scripts/launch_health_sweep.sh             # submit all 18 jobs
set -uo pipefail
cd "$(dirname "$0")/.."
WORK="$HOME/JasonLucas"

DRY=""; [[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

CACHE_ROOT="$WORK/outputs/cache_full_health"
RESULTS_ROOT="$WORK/outputs/results_full_health"
mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT" "$WORK/logs"

DATASETS="medqa mmlu_med kqa medlfqa bioasq"
COMMON="--datasets $DATASETS --size 1.0 --max-items 150 --n-max 10 --n-sweep 1 5 10 --include-verbalized --thinking off"
# ga129 has a flaky GPU + preempts jobs (whole session's failures traced to it) -> exclude
# by default. Override with EXCLUDE_NODES="" once admins confirm it's fixed.
EXCLUDE_NODES="${EXCLUDE_NODES:-ga129}"
EXCLUDE_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCLUDE_FLAG="--exclude=$EXCLUDE_NODES"

# key -> HF target id
declare -A TARGET=(
  [llama-3.1-8b]="meta-llama/Llama-3.1-8B-Instruct"
  [qwen3-8b]="Qwen/Qwen3-8B"
  [qwen3-1.7b]="Qwen/Qwen3-1.7B"
  [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [llama-3.2-3b]="meta-llama/Llama-3.2-3B-Instruct"
  [llama-3.2-1b]="meta-llama/Llama-3.2-1B-Instruct"
)

for key in "${!TARGET[@]}"; do
  for seed in 0 1 2; do
    jn="health-${key}-s${seed}"
    STAGE_CD_ARGS="--target ${TARGET[$key]} --generator-key ${key} ${COMMON} --seed ${seed}" \
    CACHE_ROOT="$CACHE_ROOT" RESULTS_ROOT="$RESULTS_ROOT" \
    scripts/submit.sh $DRY \
      --job-name="$jn" --gpus=1 --time=12:00:00 $EXCLUDE_FLAG \
      --output="$WORK/logs/%x-%j.out" --error="$WORK/logs/%x-%j.err" \
      scripts/stage_cd_open.slurm
  done
done
echo "[launch_health_sweep] ${DRY:+DRY-RUN }done: 6 gens x 3 seeds = 18 jobs -> $RESULTS_ROOT"
