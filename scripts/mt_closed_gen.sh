#!/usr/bin/env bash
# Closed-model arm, PHASE 1 (login node — needs network): generate + cache JHU-gateway outputs for
# the 7-dataset subset, for each chosen model. No GPU. Sources the gitignored key file so
# GATEWAY_KEY never lands in the repo or the transcript.
#
#   # one-time, in YOUR terminal (not committed):
#   printf "export GATEWAY_KEY='...'\n" > ~/JasonLucas/.gateway_key && chmod 600 ~/JasonLucas/.gateway_key
#   # then:
#   MODELS="jhu-gpt-4o jhu-claude-haiku-4.5 jhu-gemini-2.5-pro" scripts/mt_closed_gen.sh
set -euo pipefail
: "${WORK:=$HOME/JasonLucas}"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"
KEYFILE="${KEYFILE:-$WORK/.gateway_key}"
DATASETS="${DATASETS:-trivia_qa bioasq medqa medlfqa gsm8k truthful_qa wikipedia_factual}"
MAXITEMS="${MAXITEMS:-150}"; NMAX="${NMAX:-10}"; SEED="${SEED:-0}"
MODELS="${MODELS:-jhu-gpt-4o jhu-claude-haiku-4.5 jhu-gemini-2.5-pro}"

# generator-key -> exact gateway --model string
declare -A MODEL_ID=(
  [jhu-gpt-4o]=openai/gpt-4o
  [jhu-gpt-4o-mini]=openai/gpt-4o-mini
  [jhu-claude-opus-4.8]=anthropic/claude-opus-4.8
  [jhu-claude-sonnet-5]=anthropic/claude-sonnet-5-20260630
  [jhu-claude-haiku-4.5]=anthropic/claude-haiku-4.5
  [jhu-gemini-2.5-pro]=google-ai-studio/gemini-2.5-pro
  [jhu-gemini-3.1-flash-lite]=google-ai-studio/gemini-3.1-flash-lite
)

[ -f "$KEYFILE" ] || { echo "ERROR: $KEYFILE missing. In your terminal:"; \
  echo "  printf \"export GATEWAY_KEY='...'\\n\" > $KEYFILE && chmod 600 $KEYFILE"; exit 1; }
# shellcheck disable=SC1090
source "$KEYFILE"
: "${GATEWAY_KEY:?GATEWAY_KEY not set after sourcing $KEYFILE}"
echo "[closed-gen] GATEWAY_KEY loaded (len=${#GATEWAY_KEY}); models: $MODELS"

cd "$REPO"
module load python/3.11.9 2>/dev/null || true
source "$WORK/envs/ttlm/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" HF_HOME="$WORK/hf_cache" HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

for gk in $MODELS; do
  mid="${MODEL_ID[$gk]:-}"
  [ -n "$mid" ] || { echo "  !! unknown model key '$gk' -- skipping"; continue; }
  echo "=== generating $gk ($mid) over 7 datasets ==="
  # shellcheck disable=SC2086
  python scripts/stage_cd_api.py --model "$mid" --generator-key "$gk" \
      --datasets $DATASETS --size 1.0 --max-items "$MAXITEMS" \
      --n-max "$NMAX" --n-sweep 1 3 5 10 --seed "$SEED" \
      --generate-only
done
echo "[closed-gen] done. caches in $WORK/outputs/cache_closed. Now submit scripts/mt_closed_score.slurm per model."
