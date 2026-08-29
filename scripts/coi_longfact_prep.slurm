#!/usr/bin/env bash
# LongFact prep: generate answers (target 8B) + factuality-label (Llama-3.3-70B 4-bit).
#   MODEL=meta-llama/Llama-3.1-8B-Instruct GENKEY=llama-3.1-8b SIZE=150 sbatch ... scripts/coi_longfact_prep.slurm
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=coi-lfprep
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue
set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"; REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
module load python/3.11.9; source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"
python scripts/coi_longfact_prep.py --model "${MODEL:?}" --generator-key "${GENKEY:?}" \
    --size "${SIZE:-150}" --seed "${SEED:-1}" --threshold "${THR:-1.0}" ${BRANCHES:+--branches $BRANCHES} &
wait $!; echo "[lf-prep] done rc=$?"
