#!/usr/bin/env bash
# Launch all 6 RQ5 uncertainty-aware proxy variants: 3 oracles x 2 representations, one GPU each,
# concurrently. Requires the full labels (rq5_uncertainty_labels.json) to exist first.
#   bash scripts/launch_rq5_train.sh --dry-run   # print, submit nothing
#   bash scripts/launch_rq5_train.sh             # submit 6 jobs
set -uo pipefail
cd "$(dirname "$0")/.."
WORK="$HOME/JasonLucas"
DRY=""; [[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"
LABELS="$WORK/outputs/disaad/rq5_uncertainty_labels.json"
EXCLUDE="${EXCLUDE_NODES:-ga129}"; EXFLAG=""; [ -n "$EXCLUDE" ] && EXFLAG="--exclude=$EXCLUDE"

if [ ! -f "$LABELS" ] && [ -z "$DRY" ]; then
  echo "[launch_rq5] labels not found: $LABELS — run rq5_build_labels.py first."; exit 1
fi

ORACLES=(VerbalizedConfidence EccentricityUncertainty DiscreteSemanticEntropy)
SHORT=(verb ecc sem)   # short tags for job names
MODES=(head edl)

for i in "${!ORACLES[@]}"; do
  for mode in "${MODES[@]}"; do
    o="${ORACLES[$i]}"; s="${SHORT[$i]}"
    RQ5_TRAIN_ARGS="--oracle $o --mode $mode --uncertainty-lambda 1.0" \
    scripts/submit.sh $DRY --job-name="rq5-${s}-${mode}" --gpus=1 --cpus-per-task=12 --time=20:00:00 $EXFLAG \
      --output="$WORK/logs/%x-%j.out" --error="$WORK/logs/%x-%j.err" scripts/rq5_train.slurm 2>&1 \
      | grep -oE "Submitted batch job [0-9]+|DRY RUN.*"
  done
done
echo "[launch_rq5] ${DRY:+DRY-RUN }done: 3 oracles x 2 modes = 6 variants -> proxy_rq5_<oracle>_<mode>"
