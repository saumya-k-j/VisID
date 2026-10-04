# One command per table. C selects the dataset config; SEEDS the seeds for the main table.
C     ?= configs/beauty2014.yaml
SEEDS ?= 0 1 2
V      = python -m visid.cli

.PHONY: test smoke data images embed ids baselines tiger bench analyze table results all

test:            ## spec tests (splits, metrics, trie decoding, RQ-VAE round trip, models)
	python -m pytest -q

smoke:           ## whole pipeline on a synthetic catalog, ~8 min on CPU
	bash scripts/smoke.sh

data:            ## Day 1: sequences + split (fails loudly if stats != published), image URL check
	$(V) prepare --config $(C)
	$(V) check-images --config $(C)

images:
	$(V) download-images --config $(C)

embed:
	$(V) embed --config $(C) --variant text
	$(V) embed --config $(C) --variant image
	$(V) embed --config $(C) --variant fused

ids:
	for s in $(SEEDS); do for v in text image fused; do $(V) build-ids --config $(C) --variant $$v --seed $$s; done; done

baselines:
	for s in $(SEEDS); do \
	  $(V) train-sasrec --config $(C) --seed $$s; \
	  $(V) train-sasrec --config $(C) --seed $$s --set sasrec.loss=bce sasrec.epochs=400 sasrec.patience=10; \
	  $(V) train-twotower --config $(C) --seed $$s; done

tiger:
	for s in $(SEEDS); do for v in text image fused; do $(V) train-tiger --config $(C) --variant $$v --seed $$s; done; done

bench:           ## recall vs p50/p95 latency, batch 1 and 64
	$(V) bench --config $(C) --seed 0

analyze:         ## why-diagnostics: prefix agreement, kNN co-occurrence precision, collisions
	$(V) analyze --config $(C) --seed 0

table:           ## aggregate every metrics.json -> results/<data>/table.md (mean ± std)
	$(V) table --config $(C)

results: ids baselines tiger table   ## main results table from embeddings onward

all: data images embed results bench analyze
