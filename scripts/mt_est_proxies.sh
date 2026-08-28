#!/usr/bin/env bash
# Fair read-out comparison: read the DALD and DisAAD proxies with the SAME 9 estimators as Ours
# (incl. Perplexity), so "is Ours a better PROXY?" isn't confounded by Ours getting the good read-out
# and DALD/DisAAD their weak native LogTokU-au. 4 small jobs on the two best-student cells.
#   -> results_mt_estimators/est_{dald,disaad}_<teacher>_<student>_seed0.json
# Then: python scripts/mt_proxy_fair.py
#   DRY=1 scripts/mt_est_proxies.sh   # preview
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"; mkdir -p "$LOG"
SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }
c(){ echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }

# (method, teacher, student, student-repo)
run_cell(){ local M=$1 T=$2 S=$3 REPO=$4
  local id
  id=$(PROXY_DIR="$WORK/outputs/disaad/proxy_mt_${M}_${T}_${S}" BASE="$REPO" TEACHER="$T" \
       SUB --job-name=mt-est-${M}-${T}-${S} --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
  echo "  ${M}/${T}/${S}: $id"
}
echo "=== staging DALD/DisAAD estimator jobs (9 read-outs incl. Perplexity) ==="
for M in dald disaad; do
  run_cell "$M" qwen3-32b   qwen3-4b   Qwen/Qwen3-4B-Instruct-2507
  run_cell "$M" llama3.3-70b llama3.2-3b meta-llama/Llama-3.2-3B-Instruct
done
echo "when done: python scripts/mt_proxy_fair.py"
