#!/usr/bin/env bash
# Stage C->D for the DisAAD proxy method: score cached qwen3-8b generations with the trained
# proxy (evidential UQ) -> AUROC + latency on the M-axis. Runs in the `ttlm` venv (the
# scoring/eval stack), NOT the `disaad` training venv. Reads existing Stage-A caches, no
# regeneration. Submit via the wrapper (space-containing reservation name):
#   scripts/submit.sh --job-name=disaad-score --gpus=1 --cpus-per-task=12 --time=03:00:00 \
#     --exclude=ga129 --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err \
#     scripts/stage_cd_disaad.slurm
# Override args via DISAAD_SCORE_ARGS.

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=disaad-score
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
DISAAD_SCORE_ARGS="${DISAAD_SCORE_ARGS:-}"

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
python scripts/stage_cd_disaad.py $DISAAD_SCORE_ARGS &
wait $!
echo "[$(date -Is)] disaad-score done rc=$?"
