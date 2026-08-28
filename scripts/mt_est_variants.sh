#!/usr/bin/env bash
# Run the 9-estimator sweep (incl. Perplexity) on EVERY Ours variant of the two best-student cells,
# so we can rank variants under the Perplexity read-out. Auto-skips variants that already have
# Perplexity on file, and variants with no adapter. -> results_mt_estimators/est_ours_*_seed0.json
# Then: python scripts/mt_variant_ppl.py
#   DRY=1 scripts/mt_est_variants.sh   # preview (how many jobs)
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"; EST="$WORK/outputs/results_mt_estimators"; mkdir -p "$LOG"
SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }
c(){ echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }

declare -A REPO=( [qwen3-32b_qwen3-4b]=Qwen/Qwen3-4B-Instruct-2507
                  [llama3.3-70b_llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct )
declare -A TEACH=( [qwen3-32b_qwen3-4b]=qwen3-32b [llama3.3-70b_llama3.2-3b]=llama3.3-70b )
n=0; skip=0
for cell in qwen3-32b_qwen3-4b llama3.3-70b_llama3.2-3b; do
  for d in $(ls -d "$WORK"/outputs/disaad/proxy_mt_ours*_${cell} 2>/dev/null); do
    tag=$(basename "$d"); tag=${tag#proxy_mt_}
    # skip if no adapter
    find "$d" -name adapter_model.safetensors -path '*best_model*' 2>/dev/null | head -1 | grep -q . || { echo "skip $tag (no adapter)"; ((skip++)); continue; }
    # skip if est file already has Perplexity
    if [[ -f "$EST/est_${tag}_seed0.json" ]] && grep -q '"Perplexity"' "$EST/est_${tag}_seed0.json"; then
      echo "skip $tag (Perplexity already present)"; ((skip++)); continue; fi
    id=$(PROXY_DIR="$d" BASE="${REPO[$cell]}" TEACHER="${TEACH[$cell]}" \
         SUB --job-name=mt-est-${tag} --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
    echo "run  $tag -> $id"; ((n++))
  done
done
echo "submitted=$n skipped=$skip"
echo "when done: python scripts/mt_variant_ppl.py"
