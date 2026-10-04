#!/usr/bin/env bash
# Full study, resumable: every finished stage leaves an output file, and the stage is skipped
# when that file exists. Safe to re-run after a Colab disconnect.
#   bash scripts/run_resumable.sh [config] [seeds...]
set -euo pipefail
C=${1:-configs/beauty2014.yaml}; shift || true
SEEDS=${*:-0 1 2}
V="python -m visid.cli"
NAME=$(python -c "from visid.utils import load_config; print(load_config('$C')['data']['name'])")
RES=$(python -c "from visid.utils import load_config; print(load_config('$C').get('results_dir','results'))")/$NAME
step() { local done_file=$1; shift; if [ -e "$done_file" ]; then echo "skip (done): $*"; else echo ">>> $*"; "$@"; fi; }

step data/processed/$NAME/split.json        $V prepare --config $C
step data/processed/$NAME/image_check.json  $V check-images --config $C
step data/images/$NAME/_download_report.json $V download-images --config $C
for v in text image fused; do step data/embeddings/$NAME/$v.npy $V embed --config $C --variant $v; done
for s in $SEEDS; do
  for v in text image fused; do step data/ids/$NAME/$v/seed$s/ids.npy $V build-ids --config $C --variant $v --seed $s; done
  step $RES/sasrec/seed$s/metrics.json      $V train-sasrec --config $C --seed $s
  step $RES/sasrec_bce/seed$s/metrics.json  $V train-sasrec --config $C --seed $s --set sasrec.loss=bce sasrec.epochs=400 sasrec.patience=10
  step $RES/twotower_text/seed$s/metrics.json $V train-twotower --config $C --seed $s
  for v in text image fused; do step $RES/tiger_$v/seed$s/metrics.json $V train-tiger --config $C --variant $v --seed $s; done
done
step $RES/bench/seed0/bench.json $V bench --config $C --seed 0
$V analyze --config $C --seed 0
$V table --config $C
