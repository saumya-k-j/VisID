#!/usr/bin/env bash
# End-to-end smoke run on the synthetic catalog (minutes on CPU). Checks plumbing, not results.
set -euo pipefail
C=configs/smoke.yaml
V="python -m visid.cli"
$V prepare --config $C
$V embed --config $C --variant fused
for v in text image fused; do $V build-ids --config $C --variant $v --seed 0; done
$V train-sasrec --config $C --seed 0
$V train-twotower --config $C --seed 0
for v in text image fused; do $V train-tiger --config $C --variant $v --seed 0; done
$V bench --config $C --seed 0 --variants text,image,fused
$V table --config $C
