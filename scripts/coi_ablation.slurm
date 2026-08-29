#!/usr/bin/env bash
# CoI-Verbalized full Section-6 ablation: n=1..5 x {trivia_qa, medlfqa} for ONE generator (judge=self).
#   GEN=llama-3.1-8b MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch ... scripts/coi_ablation.slurm
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=coi-abl
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=ga129
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue
set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"; REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
GEN="${GEN:?set GEN}"; MODEL="${MODEL:?set MODEL}"
module load python/3.11.9; source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"
# optional overrides: QA_CACHE (e.g. LongFact cache), LABELS (precomputed correctness -> skips judge)
EXTRA=()
[ -n "${QA_CACHE:-}" ]     && EXTRA+=(--qa-cache-root "$QA_CACHE")
[ -n "${HEALTH_CACHE:-}" ] && EXTRA+=(--health-cache-root "$HEALTH_CACHE")
[ -n "${LABELS:-}" ]       && EXTRA+=(--labels-file "$LABELS")
[ "${LOAD4BIT:-0}" = 1 ]   && EXTRA+=(--load-4bit)
python scripts/coi_score.py --generator "$GEN" --model "$MODEL" \
    --datasets ${DATASETS:-trivia_qa medlfqa} --chain-counts ${NS:-1 2 3 4 5} --seeds "${SEED:-1}" \
    --results-root "${RESULTS:-$WORK/outputs/results_coi}" "${EXTRA[@]}" &
wait $!
echo "[coi-abl] done GEN=$GEN rc=$?"
