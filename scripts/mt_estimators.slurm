#!/usr/bin/env bash
# Generalized RQ2: run ALL SIX estimators (EDL-AU/EU, MSP, Entropy, Energy, LogTokU) on ONE trained
# proxy's logits, over the cache_mt eval data. ttlm venv, small GPU job (~15 min/proxy).
#   PROXY_DIR=~/JasonLucas/outputs/disaad/proxy_mt_disaad_qwen3-32b_qwen3-0.6b \
#   BASE=Qwen/Qwen3-0.6B TEACHER=qwen3-32b sbatch ... scripts/mt_estimators.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=mt-estimators
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
BASE="${BASE:?}"; TEACHER="${TEACHER:?}"
PROXY_DIR="${PROXY_DIR:-}"; NO_ADAPTER="${NO_ADAPTER:-0}"; LOAD_4BIT="${LOAD_4BIT:-0}"; TAG="${TAG:-}"
echo "[$(date -Is)] node=$(hostname) estimators base=$BASE no_adapter=$NO_ADAPTER 4bit=$LOAD_4BIT"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"
CACHE="${CACHE:-$WORK/outputs/cache_mt}"          # cache_closed for closed teachers
extra=()
[[ "$NO_ADAPTER" == "1" ]] && extra+=(--no-adapter) || extra+=(--variant-dir "$PROXY_DIR")
[[ "$LOAD_4BIT" == "1" ]] && extra+=(--load-in-4bit)
[[ -n "$TAG" ]] && extra+=(--tag "$TAG")
# shellcheck disable=SC2086
python scripts/mt_estimators.py --base "$BASE" --teacher-key "$TEACHER" --seeds 0 \
    --cache-roots "$CACHE" "${extra[@]}" &
wait $!
echo "[$(date -Is)] mt-estimators done rc=$?"
