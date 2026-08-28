#!/usr/bin/env bash
# Post-completion calibration pass for one generator: re-score cached gens at N (default 10),
# cross-fit an isotonic normalizer, compute meaningful ECE/MCE/Brier. Run AFTER the main runs.
#   CALIB_ARGS="--generator qwen3-8b --seed 0 --cache-root $WORK/outputs/cache_full --results-root $WORK/outputs/results_full" \
#   scripts/submit.sh --job-name=calib-qwen3-8b-s0 --gpus=1 --cpus-per-task=12 --time=02:00:00 \
#     --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/calibrate.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=calib
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
CALIB_ARGS="${CALIB_ARGS:-}"

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

# shellcheck disable=SC2086
python scripts/calibrate_eval.py $CALIB_ARGS &
wait $!
echo "[$(date -Is)] calib done rc=$?"
