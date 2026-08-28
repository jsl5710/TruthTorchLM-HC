#!/usr/bin/env bash
# RQ5 Stage 1 — black-box uncertainty labels (teacher qwen3-8b samples + 3 UQ oracles). ttlm venv.
#   RQ5_LABEL_ARGS="--n 10" scripts/submit.sh --job-name=rq5-labels --gpus=1 --cpus-per-task=12 \
#     --time=03:00:00 --exclude=ga129 \
#     --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/rq5_build_labels.slurm
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=rq5-labels
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
RQ5_LABEL_ARGS="${RQ5_LABEL_ARGS:-}"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu" NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
python scripts/rq5_build_labels.py $RQ5_LABEL_ARGS &
wait $!
echo "[$(date -Is)] rq5_build_labels done rc=$?"
