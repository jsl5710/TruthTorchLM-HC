#!/usr/bin/env bash
# Calibration for the verbalized methods (VerbalizedConfidence, PTrue) — needs the LIVE target.
# ttlm venv, GPU. One job per seed for one generator.
#   CV_ARGS="--generator qwen3-8b --target Qwen/Qwen3-8B --seed 0" scripts/submit.sh \
#     --job-name=calibverb-s0 --gpus=1 --cpus-per-task=12 --time=04:00:00 --exclude=ga129 \
#     --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/calibrate_verbalized.slurm
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=calibverb
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
CV_ARGS="${CV_ARGS:-}"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu" NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name --format=csv,noheader || true
# shellcheck disable=SC2086
python scripts/calibrate_verbalized.py $CV_ARGS &
wait $!
echo "[$(date -Is)] calibrate_verbalized done rc=$?"
