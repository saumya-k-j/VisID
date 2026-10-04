#!/usr/bin/env bash
# CPU sandbox run on the real Beauty 5-core interactions with attribute-bag IDs.
set -euo pipefail
C=configs/beauty_s3rec_attr.yaml
V="python -m visid.cli"
SEED=${SEED:-0}
$V train-sasrec --config $C --seed $SEED "$@"
$V train-twotower --config $C --seed $SEED "$@"
$V train-tiger --config $C --variant attr --seed $SEED "$@"
$V table --config $C
