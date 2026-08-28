#!/usr/bin/env bash
# VC-oracle test (Llama only), step 1: add the VerbalizedConfidence oracle to the existing
# llama3.3-70b label file. The teacher (4-bit) self-reports a confidence on each distill prompt's
# primary answer -- NO re-sampling of the dispersion oracles, so it's ~364 short greedy calls.
# Output: mt_labels_<TEACHER>_vc.json (Ecc + DSE preserved, VerbalizedConfidence added).
#   TEACHER=llama3.3-70b TEACHER_REPO=meta-llama/Llama-3.3-70B-Instruct sbatch ... scripts/mt_vc_labels.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh.
#SBATCH --job-name=mt-vc-labels
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
TEACHER="${TEACHER:?set TEACHER}"; TEACHER_REPO="${TEACHER_REPO:?set TEACHER_REPO}"
BASE_LABELS="${BASE_LABELS:-$WORK/outputs/disaad/mt_labels_${TEACHER}.json}"
OUT="${OUT:-$WORK/outputs/disaad/mt_labels_${TEACHER}_vc.json}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} VC-labels teacher=$TEACHER"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/rq5_build_labels.py \
    --teacher "$TEACHER_REPO" --load-in-4bit \
    --augment-vc "$BASE_LABELS" --out "$OUT" &
wait $!
echo "[$(date -Is)] VC-labels done rc=$? -> $OUT"
