#!/usr/bin/env bash
# 5-signal stacker -- per-item DUMP for ONE best-student cell.
# Re-merges the (purged) DALD/DisAAD proxies to scratch, then dumps per-item uncertainties for all
# five read-outs over the target's cached eval generations, aligned to (dataset, item_id, correct):
#   ours_u (EDL-AU, rq5_score) | dald_au/dald_eu, disaad_au/disaad_eu (LogTokU, stage_cd_disaad).
# The score-level stacker (scripts/mt_stack_fit.py) then fits leave-one-dataset-out over these.
#   TEACHER=qwen3-32b STUDENT=qwen3-4b STUDENT_REPO=Qwen/Qwen3-4B-Instruct-2507 sbatch ... scripts/mt_stack_dump.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh.
#SBATCH --job-name=mt-stack-dump
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
TEACHER="${TEACHER:?set TEACHER}"; STUDENT="${STUDENT:?set STUDENT}"; STUDENT_REPO="${STUDENT_REPO:?set STUDENT_REPO}"
SEED="${SEED:-0}"
SCRATCH="/weka/scratch/jhu/jsalt2026-lgarci27/hcoach/users/jsalt2026-ext-jlucas32/mt_merged"
CACHE="${CACHE:-$WORK/outputs/cache_mt}"
DUMP="$WORK/outputs/mt_stack/${TEACHER}_${STUDENT}"
TMPRES="$DUMP/_scoretmp"
DATASETS="trivia_qa bioasq medqa medlfqa gsm8k truthful_qa wikipedia_factual"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} stack-dump cell=$TEACHER/$STUDENT"
module load python/3.11.9
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"; mkdir -p "$SCRATCH" "$DUMP" "$TMPRES"

# 1) re-merge DALD + DisAAD adapters -> full models on scratch (disaad venv), if missing.
source "$WORK/envs/disaad/bin/activate"
for M in dald disaad; do
  MERGED="$SCRATCH/${M}_${TEACHER}_${STUDENT}"
  if [[ -f "$MERGED/config.json" ]]; then echo "[merge] $M already present -> $MERGED"; else
    python scripts/mt_merge.py --base "$STUDENT_REPO" \
      --proxy-dir "$WORK/outputs/disaad/proxy_mt_${M}_${TEACHER}_${STUDENT}" --out "$MERGED"
  fi
done
deactivate 2>/dev/null || true

# 2) dump all five per-item read-outs (ttlm venv).
source "$WORK/envs/ttlm/bin/activate"
# shellcheck disable=SC2086
echo "[dump] Ours (EDL-AU) via rq5_score ..."
python scripts/rq5_score.py \
    --variant-dir "$WORK/outputs/disaad/proxy_mt_ours_${TEACHER}_${STUDENT}" \
    --mode edl --base "$STUDENT_REPO" --teacher-key "$TEACHER" --seeds "$SEED" \
    --qa-cache-root "$CACHE" --health-cache-root "$CACHE" --results-root "$TMPRES" \
    --dump-items "$DUMP/items_ours_seedSEED.json"

for M in dald disaad; do
  echo "[dump] $M (LogTokU au/eu) via stage_cd_disaad ..."
  # shellcheck disable=SC2086
  python scripts/stage_cd_disaad.py \
      --proxy-path "$SCRATCH/${M}_${TEACHER}_${STUDENT}" --generator-key "$TEACHER" \
      --datasets $DATASETS --modes au eu --seeds "$SEED" \
      --qa-cache-root "$CACHE" --health-cache-root "$CACHE" \
      --results-root "$TMPRES" --allow-unverified \
      --dump-tag "$M" --dump-items "$DUMP/items_${M}_seedSEED.json"
done
echo "[$(date -Is)] stack-dump done cell=$TEACHER/$STUDENT rc=$? -> $DUMP"
