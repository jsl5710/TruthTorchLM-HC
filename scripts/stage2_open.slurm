#!/usr/bin/env bash
# Stage-2 open-model judge as a Slurm batch job. DRAFT / STAGED -- do not submit until the
# judge-validation number has been reviewed.
#
# Default run is PHASE 1 ONLY (construct-from-gold judge validation): it loads Qwen3-4B
# once and grades known-label cases built from gold answers on free-form sets (exact +
# fuzzy-variant positives, mismatched negatives), reporting recall/specificity/accuracy. No
# target model, nothing labeled for real. To use the older MCQ proxy instead, set
# STAGE2_ARGS="--validation-mode mcq". To then label the free-form sets, add --proceed to
# STAGE2_ARGS (and review the number first).
#
# Submit via the wrapper (REQUIRED for the space-containing reservation name):
#   scripts/submit.sh --job-name=stage2-validate --gpus=1 --time=01:00:00 \
#       --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/stage2_open.slurm
#
# Running `sbatch scripts/stage2_open.slurm` directly works too, but WITHOUT the
# reservation -- it queues against the general a100 pool.

# ---- Fixed flags (never change) --------------------------------------------
# Reservation is NOT set here -- its name "JSALT 2026" contains a space, which an #SBATCH
# line cannot represent; submit.sh supplies it as a single quoted argv and auto-drops it
# once it expires. partition/account/comment are space-free and safe as directives.
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost

# ---- This job ---------------------------------------------------------------
#SBATCH --job-name=stage2-validate
#SBATCH --time=01:00:00
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
# Phase-1-only by default. To label for real, append --proceed (and optionally
# --targets ... --datasets ...): STAGE2_ARGS="--proceed" sbatch ... (via submit.sh)
STAGE2_ARGS="${STAGE2_ARGS:-}"

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
# Judge (Qwen3-4B-Instruct-2507), validation target (Qwen3-1.7B), and MCQ datasets are
# pre-fetched to scratch on the login node, so run cache-only (no network on the node).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT"
cd "$REPO"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"

# shellcheck disable=SC2086  # STAGE2_ARGS is intentionally word-split
python scripts/stage2_open.py \
    --judge Qwen/Qwen3-4B-Instruct-2507 \
    --cache-root "$CACHE_ROOT" --results-root "$RESULTS_ROOT" \
    $STAGE2_ARGS &
wait $!
echo "[$(date -Is)] stage2 done rc=$?"
