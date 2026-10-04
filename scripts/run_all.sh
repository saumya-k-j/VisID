#!/usr/bin/env bash
# Full study on a GPU machine: bash scripts/run_all.sh [config]
# Produces results/<data>/table.md, results/<data>/bench/seed0/recall_vs_latency.png and
# results/<data>/analysis/seed0.md. TIGER training dominates the run time (9 runs: 3 variants
# x 3 seeds); each run's train_log.jsonl records seconds per epoch so you can budget.
set -euo pipefail
C=${1:-configs/beauty2014.yaml}
make all C=$C
