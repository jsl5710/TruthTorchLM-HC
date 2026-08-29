#!/usr/bin/env bash
# CoI-Verbalized smoke: 8 items x {n=1,2,3} on llama-3.1-8b / trivia_qa (cached) -- validates the
# full pipeline (live chain generation, JSON parse, phi, metrics, latency) end to end on GPU.
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=coi-smoke
#SBATCH --time=01:00:00
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
python scripts/coi_score.py --generator "${GEN:-llama-3.1-8b}" --model "${MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
    --datasets ${DATASETS:-trivia_qa} --chain-counts ${NS:-1 2 3} --seeds "${SEED:-1}" \
    --max-items "${MAXN:-8}" --results-root "$WORK/outputs/results_coi_smoke" &
wait $!
echo "[coi-smoke] done rc=$?"
