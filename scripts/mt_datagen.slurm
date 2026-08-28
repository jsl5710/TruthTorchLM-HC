#!/usr/bin/env bash
# Multi-target RQ, Stage 4b: collect a teacher's distillation samples (10 per prompt) over the
# FIXED DisAAD-mixed prompt set (scripts/build_distill_prompts.py -> ~/JasonLucas/data/distill/mixed_prompts).
# One job per teacher. bf16 teacher (faithful + fast): Qwen3-32B -> 1 A100, Llama-3.3-70B -> 2 A100.
# Resumable: data_builder checkpoints to <sft>.partial.json every SAVE_EVERY prompts.
#
#   TEACHER=qwen3-32b    GPUIDS=0   sbatch --gpus=1 ... scripts/mt_datagen.slurm
#   TEACHER=llama3.3-70b GPUIDS=0,1 sbatch --gpus=2 ... scripts/mt_datagen.slurm
# (submit via scripts/submit.sh so the reservation flag is handled automatically)

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
# NOTE: --reservation is injected by scripts/submit.sh (it quotes "JSALT 2026" and auto-drops
# the flag once the reservation expires). Do NOT hardcode it here -- the space breaks #SBATCH.
#SBATCH --job-name=mt-datagen
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
TEACHER="${TEACHER:?set TEACHER=qwen3-32b or llama3.3-70b}"
GPUIDS="${GPUIDS:-0}"                       # "0" for 32B (1 GPU); "0,1" for 70B (2 GPUs)
ORI="${ORI:-$WORK/data/distill/mixed_prompts}"
NUM_SAMPLES="${NUM_SAMPLES:-400}"          # prompts to generate from (fixed set has 400)
GEN_BATCH="${GEN_BATCH:-5}"                # high-temp samples per batched generate call
OUT="${OUT:-$WORK/outputs/disaad/sft_${TEACHER}_mt}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?} teacher=$TEACHER gpuids=$GPUIDS"
module load python/3.11.9
source "$WORK/envs/disaad/bin/activate"          # DisAAD-compatible venv (transformers 4.56)
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-$WORK/nltk_data}"
export SAVE_EVERY="${SAVE_EVERY:-10}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# bf16 teacher (no --use_quantization), device_map=auto spreads a 70B over the allocated GPUs.
python third_party/DisAAD/scripts/data_builder.py \
    --what_to_do local \
    --base_model_name "$TEACHER" \
    --ori_dataset_path "$ORI" \
    --sft_dataset_path "$OUT" \
    --num_samples "$NUM_SAMPLES" \
    --cache_dir "$WORK/hf_cache" \
    --gen_batch "$GEN_BATCH" \
    --do_temperature --temperature 0.7 --do_top_p --top_p 0.9 \
    --seed 42 --gpuids "$GPUIDS" &
wait $!
echo "[$(date -Is)] mt-datagen done teacher=$TEACHER -> ${OUT}.raw_data.json rc=$?"
