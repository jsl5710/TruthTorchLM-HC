#!/usr/bin/env bash
# Closed-model arm, PHASE 2 (GPU node, OFFLINE): cache-hit Stage A (populated by mt_closed_gen.sh) +
# local Qwen3-4B judge + consistency direct methods + AUROC/latency. No gateway call, no key needed.
# One job per model.
#   GENKEY=jhu-gpt-4o MODEL=openai/gpt-4o sbatch --job-name=mt-closed-score-jhu-gpt-4o ... scripts/mt_closed_score.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh.
#SBATCH --job-name=mt-closed-score
# ~10 consistency methods x N-sweep{1,3,5,10} x 7 datasets x 150 items is ~7h+ of NLI -> 6h TIMED OUT.
#SBATCH --time=16:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
GENKEY="${GENKEY:?set GENKEY=jhu-gpt-4o ...}"
MODEL="${MODEL:?set MODEL=openai/gpt-4o ...}"
DATASETS="${DATASETS:-trivia_qa bioasq medqa medlfqa gsm8k truthful_qa wikipedia_factual}"
SEED="${SEED:-0}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} score closed=$GENKEY"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Phase 2: NO --generate-only -> cache-hit Stage A (must already exist), then judge + methods + eval.
# shellcheck disable=SC2086
python scripts/stage_cd_api.py --model "$MODEL" --generator-key "$GENKEY" \
    --datasets $DATASETS --size 1.0 --max-items 150 --n-max 10 --n-sweep 1 3 5 10 \
    --seed "$SEED" --judge Qwen/Qwen3-4B-Instruct-2507 &
wait $!
echo "[$(date -Is)] mt-closed-score done $GENKEY rc=$?"
