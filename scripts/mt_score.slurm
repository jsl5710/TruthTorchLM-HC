#!/usr/bin/env bash
# Multi-target RQ, Stage 6 (score): score ONE trained proxy cell over the target's cached eval
# generations (from mt_eval). ttlm venv. Dispatches by method:
#   dald/disaad -> stage_cd_disaad.py (logit LogTokU au+eu) on <out>/merged
#   ours        -> rq5_score.py --mode edl (EDL evidence), merges adapter in-memory
#   TEACHER=qwen3-32b STUDENT=qwen3-0.6b STUDENT_REPO=Qwen/Qwen3-0.6B METHOD=ours sbatch ... scripts/mt_score.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh.
#SBATCH --job-name=mt-score
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
TEACHER="${TEACHER:?set TEACHER}"; STUDENT="${STUDENT:?set STUDENT}"
STUDENT_REPO="${STUDENT_REPO:?set STUDENT_REPO}"; METHOD="${METHOD:?set METHOD}"
TAG="${TAG:-}"; _t="${TAG:+_$TAG}"                 # optional sweep-variant suffix (e.g. ecc_lam2)
OUT="${OUT:-$WORK/outputs/disaad/proxy_mt_${METHOD}${_t}_${TEACHER}_${STUDENT}}"
CACHE="${CACHE:-$WORK/outputs/cache_mt}"
# per-CELL results dir: stage_cd_disaad.py names its file only by generator_key, so same-teacher
# cells (different students/methods/variants) would otherwise overwrite each other.
RESULTS="${RESULTS:-$WORK/outputs/results_mt_score/${METHOD}${_t}_${TEACHER}_${STUDENT}}"
SEEDS="${SEEDS:-0}"
MODE="${MODE:-edl}"                                 # ours read-out: edl | head | both
DATASETS="${DATASETS:-trivia_qa bioasq medqa medlfqa gsm8k truthful_qa wikipedia_factual}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} score cell=$METHOD/$TEACHER/$STUDENT"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
mkdir -p "$RESULTS"
# dald/disaad merged full-model lives on SCRATCH (mt_train.py writes it there to spare home quota).
_scratch="$(dirname "$(readlink -f "${HF_HOME}")")"
MERGED_PATH="${MERGED_PATH:-$_scratch/mt_merged/${METHOD}_${TEACHER}_${STUDENT}}"

# shellcheck disable=SC2086
if [[ "$METHOD" == "ours" ]]; then
    python scripts/rq5_score.py \
        --variant-dir "$OUT" --mode "$MODE" --base "$STUDENT_REPO" \
        --teacher-key "$TEACHER" --seeds $SEEDS \
        --qa-cache-root "$CACHE" --health-cache-root "$CACHE" \
        --results-root "$RESULTS" &
else
    python scripts/stage_cd_disaad.py \
        --proxy-path "$MERGED_PATH" --generator-key "$TEACHER" \
        --datasets $DATASETS --modes au eu --seeds $SEEDS \
        --qa-cache-root "$CACHE" --health-cache-root "$CACHE" \
        --results-root "$RESULTS" --allow-unverified &
fi
wait $!
echo "[$(date -Is)] mt-score done cell=$METHOD/$TEACHER/$STUDENT rc=$?"
