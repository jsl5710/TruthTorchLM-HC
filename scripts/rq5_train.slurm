#!/usr/bin/env bash
# RQ5 Stage 3 — train ONE uncertainty-aware proxy variant. Uses the SEPARATE `disaad` venv
# (train_disaad.py lives there). Reuses the banked sft_qwen3-8b data + rq5 labels (train-only).
# Walltime matches DisAAD training (~11.7h/epoch) -> 20h. One job per (oracle, mode).
#   RQ5_TRAIN_ARGS="--oracle EccentricityUncertainty --mode edl" scripts/submit.sh \
#     --job-name=rq5-ecc-edl --gpus=1 --cpus-per-task=12 --time=20:00:00 --exclude=ga129 \
#     --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/rq5_train.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=rq5-train
#SBATCH --time=20:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
RQ5_TRAIN_ARGS="${RQ5_TRAIN_ARGS:-}"
echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?}"
module load python/3.11.9
source "$WORK/envs/disaad/bin/activate"          # DisAAD-compatible venv
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-$WORK/nltk_data}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu"
export NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
python scripts/rq5_train.py $RQ5_TRAIN_ARGS &
wait $!
echo "[$(date -Is)] rq5_train done rc=$?"
