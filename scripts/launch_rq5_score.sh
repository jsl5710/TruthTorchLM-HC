#!/usr/bin/env bash
# Score all trained RQ5 variants (head+edl) that have a manifest. One GPU job each, concurrent.
#   bash scripts/launch_rq5_score.sh [--dry-run]
set -uo pipefail
cd "$(dirname "$0")/.."
WORK="$HOME/JasonLucas"; DIS="$WORK/outputs/disaad"
DRY=""; [[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"
EXCLUDE="${EXCLUDE_NODES:-ga129}"; EXFLAG=""; [ -n "$EXCLUDE" ] && EXFLAG="--exclude=$EXCLUDE"

n=0
for vd in "$DIS"/proxy_rq5_*; do
  [ -d "$vd" ] || continue
  # trained? needs the readiness manifest
  [ -f "$vd/disaad_ready.json" ] || { echo "  skip (not trained): $(basename "$vd")"; continue; }
  mode="${vd##*_}"                       # trailing head|edl
  case "$mode" in head|edl) ;; *) echo "  skip (bad mode): $(basename "$vd")"; continue;; esac
  jn="rq5s-$(basename "$vd" | sed 's/proxy_rq5_//; s/_/-/g')"
  RQ5_SCORE_ARGS="--variant-dir $vd --mode $mode" \
  scripts/submit.sh $DRY --job-name="$jn" --gpus=1 --cpus-per-task=12 --time=03:00:00 $EXFLAG \
    --output="$WORK/logs/%x-%j.out" --error="$WORK/logs/%x-%j.err" scripts/rq5_score.slurm 2>&1 \
    | grep -oE "Submitted batch job [0-9]+|DRY RUN.*"
  n=$((n+1))
done
echo "[launch_rq5_score] ${DRY:+DRY-RUN }submitted $n variant-scoring jobs -> results_rq5/"
