#!/usr/bin/env bash
# DisAAD proxy distillation (teacher Qwen3-8B -> student Qwen3-0.6B). Uses the SEPARATE
# `disaad` venv (transformers 4.56 / torch 2.7+cu118). GPU job.
#   DISAAD_ARGS="--stage data --num-samples 50" scripts/submit.sh --job-name=disaad-data \
#     --gpus=1 --cpus-per-task=12 --time=04:00:00 \
#     --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err scripts/disaad_train.slurm

#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=disaad-train
# train stage = 1 epoch x 1260 steps x ~33s/it ~= 11.7h, checkpoint ONLY at epoch end (no
# mid-epoch save, no resume). 6h is far too short -> job dies at ~50% with nothing saved.
# Default to 20h; override with --time on the CLI. (Data stage is faster but shares this.)
#SBATCH --time=20:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
DISAAD_ARGS="${DISAAD_ARGS:---stage both --num-samples 800}"

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-?}"
module load python/3.11.9
source "$WORK/envs/disaad/bin/activate"          # <-- the DisAAD-compatible venv
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# train_disaad.py's word_tokenize needs NLTK punkt/punkt_tab, prefetched here (not a
# default NLTK search path) — export so the offline compute node finds it.
export NLTK_DATA="${NLTK_DATA:-$WORK/nltk_data}"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu" OPENBLAS_NUM_THREADS="$_ncpu"
export NUMEXPR_NUM_THREADS="$_ncpu" TOKENIZERS_PARALLELISM=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch,transformers;print('disaad env: torch',torch.__version__,'transformers',transformers.__version__)"

# shellcheck disable=SC2086
python scripts/run_disaad_train.py $DISAAD_ARGS &
wait $!
echo "[$(date -Is)] disaad-train done rc=$?"
