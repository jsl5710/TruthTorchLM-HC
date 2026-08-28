#!/usr/bin/env bash
# Post-completion CALIBRATION pass over every finished regime: re-score cached generations,
# fit a cross-fitted isotonic normalizer per cell, and write honest ECE/ACE/MCE/Brier.
# One job per (generator, seed) since each cell fits its own normalizer.
#   open QA + health : N=10 (n_max was 10);  closed QA : N=5 (n_max was 5).
# ga129 excluded (flaky GPU / preemptions -- every failure this run traced to it).
#
#   bash scripts/launch_calibration.sh --dry-run   # print, submit nothing
#   bash scripts/launch_calibration.sh             # submit
set -uo pipefail
cd "$(dirname "$0")/.."
WORK="$HOME/JasonLucas"
DRY=""; [[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"
EXCLUDE="${EXCLUDE_NODES:-ga129}"; EXFLAG=""; [ -n "$EXCLUDE" ] && EXFLAG="--exclude=$EXCLUDE"

OPEN_GENS=(llama-3.1-8b llama-3.2-1b llama-3.2-3b mistral-7b qwen3-1.7b qwen3-8b)
CLOSED_GENS=(jhu-gpt-4o-mini jhu-claude-haiku-4.5)

submit () {  # gen seed cache_root results_root n
  local gen=$1 seed=$2 cache=$3 res=$4 n=$5
  CALIB_ARGS="--generator $gen --seed $seed --cache-root $cache --results-root $res --n-sweep $n" \
  scripts/submit.sh $DRY --job-name="calib-${gen}-s${seed}" --gpus=1 --cpus-per-task=12 --time=06:00:00 $EXFLAG \
    --output="$WORK/logs/%x-%j.out" --error="$WORK/logs/%x-%j.err" scripts/calibrate.slurm 2>&1 \
    | grep -oE "Submitted batch job [0-9]+|DRY.*"
}

echo "## OPEN QA (cache_full / results_full, N=10)"
for g in "${OPEN_GENS[@]}"; do for s in 0 1 2; do
  submit "$g" "$s" "$WORK/outputs/cache_full" "$WORK/outputs/results_full" 10; done; done

echo "## HEALTH (cache_full_health / results_full_health, N=10)"
for g in "${OPEN_GENS[@]}"; do for s in 0 1 2; do
  submit "$g" "$s" "$WORK/outputs/cache_full_health" "$WORK/outputs/results_full_health" 10; done; done

echo "## CLOSED QA (cache_closed_full / results_closed_full, N=5)"
for g in "${CLOSED_GENS[@]}"; do
  submit "$g" 0 "$WORK/outputs/cache_closed_full" "$WORK/outputs/results_closed_full" 5; done

echo "[launch_calibration] ${DRY:+DRY-RUN }done: 18 open + 18 health + 2 closed = 38 jobs"
