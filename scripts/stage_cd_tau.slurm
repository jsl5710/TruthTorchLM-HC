#!/usr/bin/env bash
# tau-bench agentic-trials Stage C->D (dataset #17). Small/fast; needs a GPU for the NLI.
# Submit: scripts/submit.sh --job-name=tau-trials --gpus=1 --cpus-per-task=12 --time=01:00:00 \
#   --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/stage_cd_tau.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=tau-trials
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"

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

python scripts/stage_cd_tau.py \
    --data-dir "$WORK/data/tau_bench" \
    --cache-root "$WORK/outputs/cache_tau" \
    --results-root "$WORK/outputs/results_tau" &
wait $!
echo "[$(date -Is)] tau-trials done rc=$?"
