#!/usr/bin/env bash
# Method-coverage probe: which black-box UQ methods run in this stack? Small cached cell.
# Submit via: scripts/submit.sh --job-name=method-probe --gpus=1 --time=01:00:00 \
#   --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/method_coverage.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=method-probe
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
PROBE_ARGS="${PROBE_ARGS:-}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?}"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python scripts/method_coverage.py $PROBE_ARGS &
wait $!
echo "[$(date -Is)] method-probe done rc=$?"
