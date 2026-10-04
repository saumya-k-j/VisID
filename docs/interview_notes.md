# Interview notes

Grounded in this repo's code and the 3-seed results in `results/beauty2014/`.

## Why do semantic IDs help cold start?
An atomic-ID model (SASRec) gives every item its own embedding row, learned only from that
item's interactions. A cold item's row is never trained, so it is effectively never retrieved.
On real Beauty interactions this shows up as SASRec's cold Recall@10 (see the sandbox table in
the README).

A semantic ID is a short code sequence derived from content. A cold item shares its first
codes with similar warm items, and the generator learned to predict those shared prefixes from
warm data. Probability mass therefore reaches cold items through tokens that *were* trained.
Two limits worth stating:
- The last code (and TIGER's collision token) for a cold item may never have been a target, so
  plain beam search rarely completes it. TIGER's ε reserves a share of the top-K for unseen
  items whose first 3 codes match a generated beam (`models/train.py::mix_cold`).
- The benefit is bounded by how well content similarity predicts behaviour. `make analyze`
  measures this directly: prefix agreement of consecutive purchases (vs chance) and content
  kNN co-occurrence precision, per modality.

## What is codebook collapse and how did you detect it?
Collapse is when a few codes absorb most inputs at some RQ level. IDs become coarse, many
items collide, the disambiguation token gets long, and the generator has to memorize instead of
generalize. Detection: every `log_every` epochs the RQ-VAE logs per-level utilization (share of
codes used) and perplexity (exp entropy; 256 = uniform, 1 = total collapse), the unique-prefix
fraction, and raises `collapse_warning_l{k}` below 20% utilization.

It happened for real: on the attribute IDs, level-0 utilization dropped to 17% at epoch 50.
Mitigations in the code: k-means initialization of every level, and re-seeding codes unused
for `reset_dead_every` epochs at random current residuals. Utilization recovered to 95% by epoch
100 and 100% by 300. A separate effect looks similar but is not collapse: when the *input*
cannot distinguish items (3,120 distinct attribute bags for 12,101 items), every level can be
fully used and you still get 87% prefix collisions.

## Why do trie constraints matter?
The ID space is 256 × 256 × 256 × K, about 10^7–10^9 strings, against roughly 12K real items.
Unconstrained beam search can spend top-K slots on IDs that map to nothing. The prefix trie masks
invalid next tokens at each of the 4 steps, so every finished beam is a real item and the whole
beam budget is spent on candidates. Implementation: each prefix is a mixed-radix integer; one
`searchsorted` per step resolves a whole batch of beams (`ids/trie.py`). Cost is one gather per
step. The stretch metric "invalid-ID rate without constraints" is in each TIGER `metrics.json`.
Measured: only 0.01–0.03% of unconstrained top-10 IDs were invalid and recall was identical, because
with ~12K items and 3.8–7.3% collisions the model rarely strays. Say this honestly: the trie is
a correctness guarantee that becomes a recall gain only when IDs are sparser or the model weaker.

## When would you choose ANN over generative retrieval in production?
Choose ANN (two-tower + FAISS) when:
- Latency and throughput dominate. Generative retrieval runs L = 4 sequential decoder steps with
  beam B per request. ANN is one user-tower forward plus an index lookup. Measured: at batch 1,
  TIGER beam 20 p95 = 27.5 ms vs HNSW efSearch 64 p95 = 2.3 ms (≈12×); at batch 64, 33.8 ms vs
  7.3 ms (4.6×) for 0.67× the recall.
- The catalog churns quickly and you need new items retrievable immediately. ANN needs only an
  item vector; generative retrieval needs a stable ID for the item, and the generator has a weak
  prior for codes it never saw as targets.
- You need hard filters (in stock, region): mature in ANN libraries, custom work in a trie.
Choose generative when cold-start and long-tail recall matter, memory for a huge item embedding
table is a problem (the generator stores codebooks, not one vector per item), or retrieval should
share a model with downstream ranking/generation (PinRec-style).
