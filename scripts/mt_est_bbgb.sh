#!/usr/bin/env bash
# Fast logit read-outs, BLACK-BOX (our proxy logits) vs GREY-BOX (target's own logits) ceiling.
# 9 read-outs each: EDL-AU/EU, MSP, Entropy, Energy, LogTokU + the 3 new likelihood-based ones
# (Perplexity, MaxNLL, LogitMargin -- these use which token was emitted, not just distribution shape).
# 4 jobs on the two best-student cells:
#   black-box: Ours-base proxy (q4b / l3b)      -> est_ours_<t>_<s>_seed0.json (updated w/ 3 new)
#   grey-box : the TARGET itself, 4-bit         -> est_greybox_<teacher>_seed0.json
# PTrue (grey-box) is already in results_mt. Compare with scripts/mt_est_bb_vs_gb.py.
#   DRY=1 scripts/mt_est_bbgb.sh      # preview
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"; mkdir -p "$LOG"
SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }
c(){ echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }

# --- black-box: Ours-base proxy on the two best students ---
bb1=$(PROXY_DIR="$WORK/outputs/disaad/proxy_mt_ours_qwen3-32b_qwen3-4b" BASE=Qwen/Qwen3-4B-Instruct-2507 \
      TEACHER=qwen3-32b SUB --job-name=mt-est-bb-qwen3-4b --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
bb2=$(PROXY_DIR="$WORK/outputs/disaad/proxy_mt_ours_llama3.3-70b_llama3.2-3b" BASE=meta-llama/Llama-3.2-3B-Instruct \
      TEACHER=llama3.3-70b SUB --job-name=mt-est-bb-llama3.2-3b --gpus=1 --time=03:00:00 $(c) scripts/mt_estimators.slurm)
echo "black-box (proxy): qwen=$bb1 llama=$bb2"

# --- grey-box: the TARGET's own logits (4-bit) ---
gb1=$(NO_ADAPTER=1 LOAD_4BIT=1 BASE=Qwen/Qwen3-32B TEACHER=qwen3-32b TAG=greybox_qwen3-32b \
      SUB --job-name=mt-est-gb-qwen3-32b --gpus=1 --time=04:00:00 $(c) scripts/mt_estimators.slurm)
gb2=$(NO_ADAPTER=1 LOAD_4BIT=1 BASE=meta-llama/Llama-3.3-70B-Instruct TEACHER=llama3.3-70b TAG=greybox_llama3.3-70b \
      SUB --job-name=mt-est-gb-llama3.3-70b --gpus=1 --time=06:00:00 $(c) scripts/mt_estimators.slurm)
echo "grey-box (target): qwen=$gb1 llama=$gb2"
echo
echo "jobs: bb[$bb1 $bb2] gb[$gb1 $gb2]"
echo "when done: python scripts/mt_est_bb_vs_gb.py"
