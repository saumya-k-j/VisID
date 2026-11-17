#!/usr/bin/env bash
# One command: FAISS vs MongoDB Atlas Vector Search on the real Beauty catalog, on this machine.
#   bash scripts/atlas_bench.sh
# Needs: Docker (Docker Desktop or Colima) and the trained two-tower at
# results/beauty2014/twotower_text/seed0/model.pt. Missing processed data is rebuilt with the
# repo's own `prepare` (same split_seed; it stops if the dataset stats differ from the published ones).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1   # macOS: torch + faiss each ship OpenMP
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # ...and two runtimes segfault in parallel regions; FAISS is benchmarked on 1 thread anyway

test -f results/beauty2014/twotower_text/seed0/model.pt || {
  echo "missing results/beauty2014/twotower_text/seed0/model.pt (copy it from the Colab/Drive run)"; exit 1; }
$PY -m pip install -q "pymongo>=4.7" pyyaml faiss-cpu numpy matplotlib pytest

if [ ! -f data/processed/beauty2014/split.json ]; then
  echo "== rebuilding data/processed/beauty2014 (downloads the 2014 Beauty files)"
  $PY -m visid.cli prepare --config configs/beauty2014.yaml
fi
if [ ! -f data/embeddings/beauty2014/text.npy ]; then
  echo "== rebuilding text embeddings (the two-tower's content buffer itself is loaded from model.pt)"
  $PY -m pip install -q "sentence-transformers>=2.7" "transformers>=4.40"
  $PY -m visid.cli embed --config configs/beauty2014.yaml --variant text
fi

if ! docker info >/dev/null 2>&1; then
  if command -v colima >/dev/null; then colima start --cpu 4 --memory 6; else
    echo "Docker is not running: start Docker Desktop and rerun"; exit 1; fi
fi
echo "== starting Atlas Local (first run downloads the image)"
docker compose up -d --wait atlas
export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017/?directConnection=true}"

echo "== spec tests against Atlas"
$PY -m pytest -q tests/test_atlas.py
echo "== benchmark"
$PY -m visid.cli bench --config configs/atlas_local.yaml --seed 0
