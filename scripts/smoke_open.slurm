#!/usr/bin/env bash
# Stage-1 zero-API open-model smoke as a Slurm batch job.
#
# Submit via the wrapper (REQUIRED for the reservation -- its name "JSALT 2026" contains
# a space, which an #SBATCH line cannot represent: sbatch would split it into
# `--reservation=JSALT` plus a stray `2026`. submit.sh passes it as a single quoted argv
# and auto-drops it once the reservation expires):
#   scripts/submit.sh --job-name=smoke-open --gpus=1 --time=00:30:00 \
#       --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/smoke_open.slurm
#
# Running `sbatch scripts/smoke_open.slurm` directly still works, but WITHOUT the
# reservation -- it queues against the general a100 pool.
#
# A batch job runs on the compute node under the scheduler -- it is DETACHED from your
# SSH/VS Code session, so disconnecting or going offline does not affect it. --requeue
# lets it survive preemption (it resumes at shard level via the Stage-A cache).

# ---- Fixed flags (never change) --------------------------------------------
# The reservation is NOT set here -- its space-containing name must come from submit.sh
# (see above). partition/account/comment are space-free and safe as #SBATCH directives.
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost

# ---- This job ---------------------------------------------------------------
#SBATCH --job-name=smoke-open
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
CACHE_ROOT="${CACHE_ROOT:-$WORK/outputs/cache}"
RESULTS_ROOT="${RESULTS_ROOT:-$WORK/outputs/results}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

_on_signal() {
  echo "[$(date -Is)] caught USR1 -- requeuing ${SLURM_JOB_ID:-?} for shard-level resume" >&2
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol requeue "$SLURM_JOB_ID" || true; fi
  exit 0
}
trap _on_signal USR1

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0}"

module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
# Model + dataset are pre-fetched to scratch on the login node, so run cache-only:
# no network on the compute node, which avoids hangs on air-gapped nodes.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT"
cd "$REPO"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"

python scripts/smoke_open.py --config hc_benchmark/configs/smoke_open.yaml \
    --model "$MODEL" --cache-root "$CACHE_ROOT" --results-root "$RESULTS_ROOT" &
wait $!
echo "[$(date -Is)] smoke-open done rc=$?"
