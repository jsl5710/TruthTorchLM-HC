#!/usr/bin/env bash
# ============================================================================
# Parameterized Rockfish job template for the TruthTorchLM-HC pipeline.
#
# The four FIXED flags below never change for this account. The varying flags
# have safe defaults and are meant to be overridden on the command line, e.g.:
#
#   scripts/submit.sh --job-name=stageA-open --gpus=1 --time=08:00:00 \
#       --export=ALL,CONFIG=hc_benchmark/configs/open_llama.yaml \
#       scripts/job_template.slurm
#
# or directly (the #SBATCH lines make it self-contained):
#
#   sbatch --job-name=stageA-open --gpus=1 --export=ALL,CONFIG=... scripts/job_template.slurm
#
# Parameterize by exporting these before submit (or via --export):
#   CONFIG        hc_benchmark YAML to run     (required unless CMD is set)
#   CACHE_ROOT    Stage-A cache dir            (default $WORK/outputs/cache)
#   RESULTS_ROOT  results bundle dir           (default $WORK/outputs/results)
#   CMD           arbitrary command to run instead of `python -m hc_benchmark.run`
#
# RESERVATION NOTE: after "JSALT 2026" expires, comment out the #SBATCH
# --reservation line below (one edit) OR submit via scripts/submit.sh, which
# drops the reservation automatically once it is no longer ACTIVE.
# ============================================================================

# ---- Fixed flags (never change) --------------------------------------------
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --reservation=JSALT 2026

# ---- Varying flags (override on the CLI; %x=job-name %j=job-id) -------------
#SBATCH --job-name=ttlm-hc
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ---- Resumption: allow requeue and get a signal 120s before the walltime ---
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

: "${WORK:=$HOME/JasonLucas}"
CACHE_ROOT="${CACHE_ROOT:-$WORK/outputs/cache}"
RESULTS_ROOT="${RESULTS_ROOT:-$WORK/outputs/results}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"

# On preemption / near-timeout, Slurm sends USR1 (see --signal above). Requeue so
# the job restarts; finished (config,seed) shards are skipped by cache.exists(),
# so it resumes at shard granularity. (Per-item checkpointing is a later change.)
_on_signal() {
  echo "[$(date -Is)] caught USR1 -- requeuing job ${SLURM_JOB_ID:-?} for shard-level resume" >&2
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol requeue "$SLURM_JOB_ID" || true; fi
  exit 0
}
trap _on_signal USR1

echo "[$(date -Is)] job=${SLURM_JOB_NAME:-?} id=${SLURM_JOB_ID:-?} node=$(hostname) restarts=${SLURM_RESTART_COUNT:-0}"

# ---- Environment -----------------------------------------------------------
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"       # weights+datasets -> scratch symlink
mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT"
cd "$REPO"

# Show the GPU we actually got (latency runs care about this).
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"

# ---- Run -------------------------------------------------------------------
# Run in the background and `wait` so the USR1 trap can fire promptly (a
# foreground child would otherwise delay signal handling until it returns).
if [[ -n "${CMD:-}" ]]; then
  eval "$CMD" &
else
  : "${CONFIG:?set CONFIG=hc_benchmark/configs/<x>.yaml (or set CMD=...)}"
  python -m hc_benchmark.run --config "$CONFIG" \
      --cache-root "$CACHE_ROOT" --results-root "$RESULTS_ROOT" &
fi
wait $!
echo "[$(date -Is)] job complete rc=$?"
