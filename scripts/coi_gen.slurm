#!/usr/bin/env bash
# Generate+label gsm8k/hotpot on an 8B target (not previously cached), then CoI-score rungs 1/3/6/7.
#   GEN=llama-3.1-8b MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch ... scripts/coi_gen.slurm
#SBATCH --partition=a100
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --job-name=coi-gen
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
DSETS="${DATASETS:-gsm8k hotpot_qa}"; SEED="${SEED:-1}"
CGEN="$WORK/outputs/cache_coi_gen"; RES="$WORK/outputs/results_coi_gen"
module load python/3.11.9; source "$WORK/envs/ttlm/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
cd "$REPO"
# phase 1: generate + gold-label each dataset (item_ids are dataset-prefixed -> collision-free)
for ds in $DSETS; do
  echo "[coi-gen] ===== prep $ds ($GEN) ====="
  python scripts/coi_gen_prep.py --model "$MODEL" --generator-key "$GEN" --dataset "$ds" \
      --size 150 --seed "$SEED" --cache-root "$CGEN"
done
# merge the per-dataset labels into one (keys are unique across datasets)
MERGED="$CGEN/labels_merged_${GEN}_seed${SEED}.json"
python - "$CGEN" "$GEN" "$SEED" "$MERGED" $DSETS <<'PY'
import json, sys, os
cgen, gen, seed, merged = sys.argv[1:5]; dsets = sys.argv[5:]
m = {}
for ds in dsets:
    f = os.path.join(cgen, f"labels_{ds}_{gen}_seed{seed}.json")
    m.update(json.load(open(f)))
json.dump(m, open(merged, "w"), indent=1); print(f"[merge] {len(m)} labels -> {merged}")
PY
# phase 2: score all datasets in ONE call (avoids clobbering the single results file)
echo "[coi-gen] ===== score $DSETS ($GEN) ====="
python scripts/coi_score.py --generator "$GEN" --model "$MODEL" --datasets $DSETS \
    --chain-counts 1 3 6 7 --seeds "$SEED" --qa-cache-root "$CGEN" \
    --labels-file "$MERGED" --results-root "$RES"
echo "[coi-gen] done GEN=$GEN"
