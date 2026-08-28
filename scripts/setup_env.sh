#!/usr/bin/env bash
# Build the TruthTorchLM-HC Python env on Rockfish. LOGIN-NODE SAFE (install only,
# no GPU, no timed work). Idempotent-ish: re-running reuses the venv unless --fresh.
#
#   scripts/setup_env.sh            # create/update $WORK/envs/ttlm
#   scripts/setup_env.sh --fresh    # delete and rebuild from scratch
#
# Choices (locked with the user):
#   * venv on python/3.11.9 (repo requires >=3.10; no gcc/conda dance needed)
#   * torch pinned to a bundled-CUDA wheel -> inference needs no `cuda` module
#   * pip cache on scratch, venv in $WORK/envs (home). Model weights never touch home.
set -euo pipefail

: "${WORK:=$HOME/JasonLucas}"
ENV_DIR="$WORK/envs/ttlm"
REPO="${REPO:-$WORK/code/TruthTorchLM-HC}"

# torch wheel: cu121 is compatible with A100 SXM + any recent driver. If the GPU node
# reports CUDA unavailable, re-run with TORCH_INDEX pointing at the matching wheel.
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if [[ "${1:-}" == "--fresh" && -d "$ENV_DIR" ]]; then
  echo "--fresh: removing $ENV_DIR"; rm -rf "$ENV_DIR"
fi

# Keep pip's (large, regenerable) cache off the 94G home quota -> scratch, via hf_cache.
SCRATCH_ROOT="$(dirname "$(readlink -f "$WORK/hf_cache")")"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}"
mkdir -p "$PIP_CACHE_DIR"
echo "pip cache -> $PIP_CACHE_DIR"

module load python/3.11.9
if [[ ! -d "$ENV_DIR" ]]; then
  echo "creating venv at $ENV_DIR"
  python -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python --version

python -m pip install --upgrade pip wheel setuptools

# 1) torch FIRST from the CUDA wheel index, so the later unpinned `torch` in
#    requirements.txt is already satisfied and pip won't pull a different build.
echo ">>> installing $TORCH_SPEC from $TORCH_INDEX"
python -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

# 2) the rest of the deps, then the package itself (editable).
echo ">>> installing repo requirements + editable package"
python -m pip install -r "$REPO/requirements.txt"
python -m pip install -e "$REPO"

# 3) verify imports (CPU-only checks here; CUDA is confirmed inside a GPU job).
echo ">>> import check"
python - <<'PY'
import importlib
mods = ["torch","transformers","litellm","datasets","pyarrow","pandas",
        "sentence_transformers","sklearn","numpy","yaml"]
for m in mods:
    importlib.import_module(m)
    print(f"  ok  {m}")
import torch
print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}  "
      f"cuda_available_here={torch.cuda.is_available()} (False on login node is expected)")
# harness imports
import sys; sys.path.insert(0, "src")
from hc_benchmark.config import load_config          # noqa
from hc_benchmark.generators import GENERATORS       # noqa
print(f"  ok  hc_benchmark ({len(GENERATORS)} generators in registry)")
PY

echo
echo "DONE. Activate with:  module load python/3.11.9 && source $ENV_DIR/bin/activate"
