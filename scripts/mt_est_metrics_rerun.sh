#!/usr/bin/env bash
# STAGE-3 of the multi-metric add: re-run the 9-read-out sweep (now emitting AUPRC + cross-fit
# calibrated ECE/MCE/Brier alongside AUROC) on the deployable proxy cells, so the read-out comparison
# (Perplexity vs EDL vs ...) can be reported on ranking AND calibration. Existing est_* files carry
# AUROC only; this overwrites them with the full `metrics` field (the scalar `rows` AUROC is preserved,
# so mt_proxy_fair.py / mt_variant_ppl.py / mt_est_bb_vs_gb.py keep working unchanged).
# 6 core cells (Ours-base, DALD, DisAAD x 2 teachers); add --all to also redo every Ours variant.
#   DRY=1 scripts/mt_est_metrics_rerun.sh        # preview
#   scripts/mt_est_metrics_rerun.sh              # submit the 6 core cells
#   ALL=1 scripts/mt_est_metrics_rerun.sh        # also redo all Ours variants (24 cells)
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"; ALL="${ALL:-0}"; mkdir -p "$LOG"
SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }
c(){ echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }

declare -A REPO=( [qwen3-32b_qwen3-4b]=Qwen/Qwen3-4B-Instruct-2507
                  [llama3.3-70b_llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct )
declare -A TEACH=( [qwen3-32b_qwen3-4b]=qwen3-32b [llama3.3-70b_llama3.2-3b]=llama3.3-70b )
n=0
for cell in qwen3-32b_qwen3-4b llama3.3-70b_llama3.2-3b; do
  # core deployable proxies
  for M in ours dald disaad; do
    d="$WORK/outputs/disaad/proxy_mt_${M}_${cell}"
    [[ -d "$d" ]] || { echo "skip $M/$cell (no proxy dir)"; continue; }
    id=$(PROXY_DIR="$d" BASE="${REPO[$cell]}" TEACHER="${TEACH[$cell]}" \
         SUB --job-name=mt-estM-${M}-${cell} --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
    echo "run  ${M}/${cell} -> $id"; ((n++))
  done
  # optional: all Ours variants
  if [[ "$ALL" == "1" ]]; then
    for d in $(ls -d "$WORK"/outputs/disaad/proxy_mt_ours*_${cell} 2>/dev/null); do
      tag=$(basename "$d"); tag=${tag#proxy_mt_}
      find "$d" -name adapter_model.safetensors -path '*best_model*' 2>/dev/null | head -1 | grep -q . || continue
      id=$(PROXY_DIR="$d" BASE="${REPO[$cell]}" TEACHER="${TEACH[$cell]}" \
           SUB --job-name=mt-estM-${tag} --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
      echo "run  variant ${tag} -> $id"; ((n++))
    done
  fi
done
echo "submitted=$n  (ALL=$ALL)"
echo "when done: python scripts/mt_est_metrics_report.py"
