#!/usr/bin/env bash
# Phase 2 of the closed run: GPU scoring of gateway-generated caches (Stage B/C/D).
# Stage A is already cached (generated on the login node), so NO gateway access is needed
# here. Submit:
#   CLOSED_ARGS="--model openai/gpt-4o-mini --generator-key jhu-gpt-4o-mini --datasets trivia_qa --max-items 40 --n-max 5 --n-sweep 1 3 5" \
#   scripts/submit.sh --job-name=closed-jhu-gpt-4o-mini --gpus=1 --cpus-per-task=12 \
#     --time=01:00:00 --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/stage_cd_api.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=closed-run
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
CLOSED_ARGS="${CLOSED_ARGS:-}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?}"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu"
export NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# NB: no --generate-only -> Stage A cache-hits (must be pre-generated), then B/C/D on GPU.
# cache/results roots come from env so a run can point at its own dirs (e.g. the full run).
# shellcheck disable=SC2086
python scripts/stage_cd_api.py $CLOSED_ARGS \
    --cache-root "${CACHE_ROOT:-$WORK/outputs/cache_closed}" \
    --results-root "${RESULTS_ROOT:-$WORK/outputs/results_closed}" &
wait $!
echo "[$(date -Is)] closed-run done rc=$?"
