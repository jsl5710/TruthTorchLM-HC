#!/usr/bin/env bash
# Multi-target RQ, Stage 4c: build uncertainty labels for a teacher (used by method=ours).
# NLI-based oracle over the Stage-4b samples already collected -- no teacher reload, no re-sampling.
# One small-GPU job per teacher. Run AFTER mt_datagen for that teacher completes.
#   TEACHER=qwen3-32b TEACHER_REPO=Qwen/Qwen3-32B sbatch ... scripts/mt_labels.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh.
#SBATCH --job-name=mt-labels
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
TEACHER="${TEACHER:?set TEACHER=qwen3-32b or llama3.3-70b}"
TEACHER_REPO="${TEACHER_REPO:?set TEACHER_REPO=<hf id>}"
SFT="${SFT:-$WORK/outputs/disaad/sft_${TEACHER}_mt.raw_data.json}"
OUT="${OUT:-$WORK/outputs/disaad/mt_labels_${TEACHER}.json}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} teacher=$TEACHER"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"

python scripts/rq5_build_labels.py \
    --teacher "$TEACHER_REPO" --sft-data "$SFT" --out "$OUT" \
    --n 10 --use-existing-samples &
wait $!
echo "[$(date -Is)] mt-labels done teacher=$TEACHER -> $OUT rc=$?"
