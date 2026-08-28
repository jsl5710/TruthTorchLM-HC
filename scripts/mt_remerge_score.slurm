#!/usr/bin/env bash
# One-off: re-merge a DALD/DisAAD proxy (adapter -> full model on SCRATCH, regenerable) then score it.
# Used to recover a cell whose merged model was pruned to free home quota.
#   TEACHER=llama3.3-70b STUDENT=llama3.1-8b STUDENT_REPO=meta-llama/Llama-3.1-8B-Instruct \
#     METHOD=disaad sbatch ... scripts/mt_remerge_score.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=mt-remerge-score
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
TEACHER="${TEACHER:?}"; STUDENT="${STUDENT:?}"; STUDENT_REPO="${STUDENT_REPO:?}"; METHOD="${METHOD:?}"
SCRATCH="/weka/scratch/jhu/jsalt2026-lgarci27/hcoach/users/jsalt2026-ext-jlucas32/mt_merged"
PROXY_DIR="$WORK/outputs/disaad/proxy_mt_${METHOD}_${TEACHER}_${STUDENT}"
MERGED="$SCRATCH/${METHOD}_${TEACHER}_${STUDENT}"
CACHE="$WORK/outputs/cache_mt"
RESULTS="$WORK/outputs/results_mt_score/${METHOD}_${TEACHER}_${STUDENT}"
DATASETS="trivia_qa bioasq medqa medlfqa gsm8k truthful_qa wikipedia_factual"

echo "[$(date -Is)] node=$(hostname) re-merge+score $METHOD/$TEACHER/$STUDENT"
module load python/3.11.9
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"; mkdir -p "$SCRATCH" "$RESULTS"

# 1) merge (disaad venv)
source "$WORK/envs/disaad/bin/activate"
python scripts/mt_merge.py --base "$STUDENT_REPO" --proxy-dir "$PROXY_DIR" --out "$MERGED"
deactivate 2>/dev/null || true

# 2) score (ttlm venv)
source "$WORK/envs/ttlm/bin/activate"
# shellcheck disable=SC2086
python scripts/stage_cd_disaad.py --proxy-path "$MERGED" --generator-key "$TEACHER" \
    --datasets $DATASETS --modes au eu --seeds 0 \
    --qa-cache-root "$CACHE" --health-cache-root "$CACHE" \
    --results-root "$RESULTS" --allow-unverified
echo "[$(date -Is)] re-merge+score done rc=$?"
