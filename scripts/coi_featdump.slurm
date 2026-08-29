#!/usr/bin/env bash
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=coi-featdump
#SBATCH --time=00:40:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"; REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
module load python/3.11.9; source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"; python scripts/coi_featdump.py & wait $!; echo "[featdump] done rc=$?"
