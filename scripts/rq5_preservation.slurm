#!/usr/bin/env bash
# RQ5 Stage 4 — score one uncertainty-aware proxy variant (head/edl). ttlm venv, GPU.
#   RQ5_PRES_ARGS="--variant-dir .../proxy_rq5_EccentricityUncertainty_edl --mode edl" \
#   scripts/submit.sh --job-name=rq5s-ecc-edl --gpus=1 --cpus-per-task=12 --time=03:00:00 \
#     --exclude=ga129 --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/rq5_score.slurm
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=rq5-pres
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
RQ5_PRES_ARGS="${RQ5_PRES_ARGS:-}"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu" NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name --format=csv,noheader || true
# shellcheck disable=SC2086
python scripts/rq5_preservation.py $RQ5_PRES_ARGS &
wait $!
echo "[$(date -Is)] rq5_score done rc=$?"
