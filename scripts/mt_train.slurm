#!/usr/bin/env bash
# Multi-target RQ, Stage 6 (train): train ONE proxy cell = (TEACHER, STUDENT, METHOD).
# DisAAD-compatible venv (transformers 4.56). Trains + merges the adapter (dald/disaad).
#   TEACHER=qwen3-32b STUDENT=qwen3-0.6b STUDENT_REPO=Qwen/Qwen3-0.6B METHOD=ours \
#     sbatch --job-name=mt-train-ours-qwen3-32b-qwen3-0.6b ... scripts/mt_train.slurm
# For METHOD=ours also set LABELS (defaults to mt_labels_<teacher>.json).

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh.
#SBATCH --job-name=mt-train
#SBATCH --time=20:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
TEACHER="${TEACHER:?set TEACHER}"; STUDENT="${STUDENT:?set STUDENT}"
STUDENT_REPO="${STUDENT_REPO:?set STUDENT_REPO}"; METHOD="${METHOD:?set METHOD=dald|disaad|ours}"
SFT="${SFT:-$WORK/outputs/disaad/sft_${TEACHER}_mt.raw_data.json}"
LABELS="${LABELS:-$WORK/outputs/disaad/mt_labels_${TEACHER}.json}"
ORACLE="${ORACLE:-EccentricityUncertainty}"; LAM="${LAM:-5.0}"; TAG="${TAG:-}"; MODE="${MODE:-edl}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} cell=$METHOD/$TEACHER/$STUDENT"
module load python/3.11.9
source "$WORK/envs/disaad/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-$WORK/nltk_data}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

extra=()
[[ "$METHOD" == "ours" ]] && extra=(--labels "$LABELS" --oracle "$ORACLE" --lam "$LAM" --unc-mode "$MODE")
[[ -n "$TAG" ]] && extra+=(--variant-tag "$TAG")
python scripts/mt_train.py \
    --teacher "$TEACHER" --student "$STUDENT" --student-repo "$STUDENT_REPO" \
    --method "$METHOD" --sft "$SFT" \
    --lora-r 32 --lora-alpha 64 --gpu-ids 0 "${extra[@]}" &
wait $!
echo "[$(date -Is)] mt-train done cell=$METHOD/$TEACHER/$STUDENT rc=$?"
