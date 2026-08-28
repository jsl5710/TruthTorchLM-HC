#!/usr/bin/env bash
# Complete the proxy read-out grid: run the 9-read-out sweep (incl Perplexity) + full metric suite
# (AUPRC, calibrated ECE/MCE/Brier) on the cells that only have native/partial scores:
#   (A) CLOSED proxies: dald/disaad/ours x 6 students x {gpt-4o, claude-haiku}  (cache_closed)  = 36
#   (B) OPEN non-best students: dald/disaad/ours x {q0.6b,q1.7b}|{l1b,l8b}      (cache_mt)       = 12
# All offline GPU, ~3 min/cell. Writes results_mt_estimators/est_<method>_<teacher>_<student>_seed0.json.
#   DRY=1 scripts/mt_est_complete.sh     # preview + counts
#   CLOSED_ONLY=1 / OPEN_ONLY=1          # run just one batch
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"; mkdir -p "$LOG"
SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }
c(){ echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }
declare -A REPO=( [qwen3-0.6b]=Qwen/Qwen3-0.6B [qwen3-1.7b]=Qwen/Qwen3-1.7B [qwen3-4b]=Qwen/Qwen3-4B-Instruct-2507
  [llama3.2-1b]=meta-llama/Llama-3.2-1B-Instruct [llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct [llama3.1-8b]=meta-llama/Llama-3.1-8B-Instruct )
QSTUD="qwen3-0.6b qwen3-1.7b qwen3-4b"; LSTUD="llama3.2-1b llama3.2-3b llama3.1-8b"
n=0
runcell(){ local M=$1 T=$2 S=$3 CACHE=$4
  local d="$WORK/outputs/disaad/proxy_mt_${M}_${T}_${S}"
  find "$d" -name adapter_model.safetensors -path '*best_model*' 2>/dev/null | head -1 | grep -q . || { echo "skip $M/$T/$S (no adapter)"; return; }
  local id; id=$(PROXY_DIR="$d" BASE="${REPO[$S]}" TEACHER="$T" CACHE="$CACHE" \
        SUB --job-name=mt-estC-${M}-${T}-${S} --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
  echo "run  $M/$T/$S -> $id"; ((n++)) || true
}
# (A) CLOSED proxies
if [[ "${OPEN_ONLY:-0}" != "1" ]]; then
  echo "### (A) closed proxies (cache_closed) ###"
  for T in jhu-gpt-4o jhu-claude-haiku-4.5; do for M in dald disaad ours; do
    for S in $QSTUD $LSTUD; do runcell "$M" "$T" "$S" "$WORK/outputs/cache_closed"; done
  done; done
fi
# (B) OPEN non-best students
if [[ "${CLOSED_ONLY:-0}" != "1" ]]; then
  echo "### (B) open non-best students (cache_mt) ###"
  for M in dald disaad ours; do
    for S in qwen3-0.6b qwen3-1.7b; do runcell "$M" qwen3-32b "$S" "$WORK/outputs/cache_mt"; done
    for S in llama3.2-1b llama3.1-8b; do runcell "$M" llama3.3-70b "$S" "$WORK/outputs/cache_mt"; done
  done
fi
echo "submitted=$n"
echo "when done: python scripts/mt_proxy_fair.py ; python scripts/mt_est_metrics_report.py"
