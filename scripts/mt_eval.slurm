#!/usr/bin/env bash
# Multi-target RQ, Stage 5: eval-data generation + DIRECT black-box methods on a teacher/target,
# over the 7-dataset subset (full 150 items). Writes stage_cd_<key>_seed<seed>.json used by the
# multi-target report as the "direct methods on this target" reference. 4-bit target on 1-2 A100s
# (consistent with the 4-bit precision used to collect its distillation samples).
#
#   TEACHER=qwen3-32b    TARGET=Qwen/Qwen3-32B                 sbatch ... scripts/mt_eval.slurm
#   TEACHER=llama3.3-70b TARGET=meta-llama/Llama-3.3-70B-Instruct sbatch ... scripts/mt_eval.slurm
# (submit via scripts/submit.sh; give the 70B --gpus=2 for headroom)

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# --reservation injected by scripts/submit.sh (do NOT hardcode; space breaks #SBATCH).
#SBATCH --job-name=mt-eval
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
TEACHER="${TEACHER:?set TEACHER=qwen3-32b or llama3.3-70b}"
TARGET="${TARGET:?set TARGET=<hf repo id>}"
SEED="${SEED:-0}"
DATASETS="${DATASETS:-trivia_qa bioasq medqa medlfqa gsm8k truthful_qa wikipedia_factual}"
RESULTS="${RESULTS:-$WORK/outputs/results_mt}"
CACHE="${CACHE:-$WORK/outputs/cache_mt}"    # stageA parquet cache the proxy scorers read back

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} target=$TARGET seed=$SEED"
module load python/3.11.9
source "$WORK/envs/ttlm/bin/activate"            # library venv (transformers 5.x) + bitsandbytes
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
mkdir -p "$RESULTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python scripts/stage_cd_open.py \
    --target "$TARGET" --generator-key "$TEACHER" \
    --judge Qwen/Qwen3-4B-Instruct-2507 --judge-label qwen3-4b-instruct-2507 \
    --datasets $DATASETS \
    --size 1.0 --max-items 150 --n-max 10 --n-sweep 1 3 5 10 \
    --seed "$SEED" --thinking off --include-verbalized \
    --dtype bf16-auto --judge-dtype bf16 \
    --cache-root "$CACHE" --results-root "$RESULTS" &
wait $!
echo "[$(date -Is)] mt-eval done target=$TARGET seed=$SEED rc=$?"
