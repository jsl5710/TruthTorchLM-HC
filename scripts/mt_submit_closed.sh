#!/usr/bin/env bash
# Closed-teacher proxy matrix: 2 API teachers (gpt-4o, claude-haiku) x 6 open students x 3 methods
# (DALD/DisAAD/Ours) = 36 proxies, distilled from the teacher's gateway-generated TEXT (train never
# loads the teacher). Mirrors mt_submit_matrix.sh but: SFT = sft_<key>_mt (closed distill data),
# labels use a stand-in tokenizer, scoring reads cache_closed. Run AFTER mt_closed_distill_gen.py
# has written both sft_<key>_mt.raw_data.json files.
#   DRY=1 scripts/mt_submit_closed.sh    # preview
set -uo pipefail
: "${WORK:=$HOME/JasonLucas}"; export WORK
cd "$(dirname "$0")/.."
LOG="$WORK/logs"; DRY="${DRY:-0}"
TOK="Qwen/Qwen3-0.6B"                       # stand-in tokenizer for closed-teacher labels
CACHE="$WORK/outputs/cache_closed"
SUB(){ if [[ "$DRY" == "1" ]]; then echo "DRY sbatch $*" >&2; echo "000$RANDOM"; else
         local id; id=$(scripts/submit.sh "$@" 2>/dev/null | grep -oP 'Submitted batch job \K[0-9]+' || true)
         [[ -z "$id" ]] && echo "SUBMIT-FAILED" >&2 && echo "" || echo "$id"; fi; }

declare -A TEACHERS=( [jhu-gpt-4o]=openai/gpt-4o [jhu-claude-haiku-4.5]=anthropic/claude-haiku-4.5 )
declare -A SREPO=(
  [qwen3-0.6b]=Qwen/Qwen3-0.6B [qwen3-1.7b]=Qwen/Qwen3-1.7B [qwen3-4b]=Qwen/Qwen3-4B-Instruct-2507
  [llama3.2-1b]=meta-llama/Llama-3.2-1B-Instruct [llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct
  [llama3.1-8b]=meta-llama/Llama-3.1-8B-Instruct )
STUDENTS="qwen3-0.6b qwen3-1.7b qwen3-4b llama3.2-1b llama3.2-3b llama3.1-8b"
METHODS="dald disaad ours"
common(){ echo "--cpus-per-task=12 --output=$LOG/%x-%j.out --error=$LOG/%x-%j.err"; }

for T in "${!TEACHERS[@]}"; do
  SFT="$WORK/outputs/disaad/sft_${T}_mt.raw_data.json"
  if [[ ! -f "$SFT" ]]; then echo "!! $SFT missing (run mt_closed_distill_gen for $T first) -- skipping $T"; continue; fi
  # Stage 4c labels (stand-in tokenizer; NLI over the API teacher's samples)
  lb=$(TEACHER=$T TEACHER_REPO=$TOK SFT="$SFT" SUB --job-name=mt-labels-$T --gpus=1 --time=03:00:00 $(common) scripts/mt_labels.slurm)
  echo "labels[$T]=$lb"
  for S in $STUDENTS; do
    for M in $METHODS; do
      dep=""; [[ "$M" == "ours" && -n "$lb" ]] && dep="--dependency=afterok:$lb"
      tr=$(TEACHER=$T STUDENT=$S STUDENT_REPO="${SREPO[$S]}" METHOD=$M SFT="$SFT" LABELS="$WORK/outputs/disaad/mt_labels_${T}.json" \
            SUB --job-name=mt-train-$M-$T-$S --gpus=1 --time=20:00:00 $dep $(common) scripts/mt_train.slurm)
      sdep=""; [[ -n "$tr" ]] && sdep="--dependency=afterok:$tr"
      sc=$(TEACHER=$T STUDENT=$S STUDENT_REPO="${SREPO[$S]}" METHOD=$M CACHE="$CACHE" \
            SUB --job-name=mt-score-$M-$T-$S --gpus=1 --time=06:00:00 $sdep $(common) scripts/mt_score.slurm)
      echo "  cell $M/$T/$S: train=$tr score=$sc"
    done
  done
done
echo "Closed matrix submitted. Scoring reads cache_closed + the closed eval correctness labels (from the direct-method scoring)."
