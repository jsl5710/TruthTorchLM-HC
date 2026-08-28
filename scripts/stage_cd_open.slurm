#!/usr/bin/env bash
# Stage A->D black-box UQ benchmark on an open target (the first real research cell).
# Submit via the wrapper (REQUIRED for the space-containing reservation name):
#   scripts/submit.sh --job-name=stage-cd --gpus=1 --time=03:00:00 \
#       --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/stage_cd_open.slurm
#
# Override args via STAGE_CD_ARGS, e.g.
#   STAGE_CD_ARGS="--datasets trivia_qa --max-items 150" scripts/submit.sh ...

# ---- Fixed flags (reservation comes from submit.sh; space-containing name) --
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost

# ---- This job ---------------------------------------------------------------
#SBATCH --job-name=stage-cd
#SBATCH --time=03:00:00
#SBATCH --gpus=1
# 8 GPUs + 96 cores per node -> 12 cores/GPU-job. Without this, jobs request 0 CPUs and
# each spawns threads across all 96 cores; 8 co-located jobs then oversubscribe the CPUs
# ~8x and CPU-bound methods (ROUGE) crawl (observed 32s/item). cgroup-bind each job.
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
CACHE_ROOT="${CACHE_ROOT:-$WORK/outputs/cache}"
RESULTS_ROOT="${RESULTS_ROOT:-$WORK/outputs/results}"
STAGE_CD_ARGS="${STAGE_CD_ARGS:-}"

_on_signal() {
  echo "[$(date -Is)] caught USR1 -- requeuing ${SLURM_JOB_ID:-?}" >&2
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol requeue "$SLURM_JOB_ID" || true; fi
  exit 0
}
trap _on_signal USR1

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0}"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
# Bound CPU threads to the allocated cores so co-located jobs don't oversubscribe (the
# 32s/item cause). Matches --cpus-per-task; torch/numpy/BLAS/tokenizers all honor these.
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu"
export NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
# Target, judge, NLI (converted safetensors), and datasets are all prefetched to scratch.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# shellcheck disable=SC2086
python scripts/stage_cd_open.py \
    --cache-root "$CACHE_ROOT" --results-root "$RESULTS_ROOT" \
    $STAGE_CD_ARGS &
wait $!
echo "[$(date -Is)] stage-cd done rc=$?"
