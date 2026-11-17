RESUME FACTS (measured in this run; nothing estimated)
catalog size: 12101 items | embedding dim: 64 | device: cpu | test users: 4096 | Atlas server 9.0.4, index build 4.1s
batch  1, efSearch = numCandidates =  16: FAISS HNSW Recall@10 vs Flat 0.762, p95 0.58 ms | Atlas Recall@10 vs Flat 0.929, p95 9.25 ms
batch  1, efSearch = numCandidates =  32: FAISS HNSW Recall@10 vs Flat 0.868, p95 0.65 ms | Atlas Recall@10 vs Flat 0.974, p95 6.96 ms
batch  1, efSearch = numCandidates =  64: FAISS HNSW Recall@10 vs Flat 0.937, p95 0.59 ms | Atlas Recall@10 vs Flat 0.993, p95 6.69 ms
batch  1, efSearch = numCandidates = 128: FAISS HNSW Recall@10 vs Flat 0.971, p95 1.77 ms | Atlas Recall@10 vs Flat 0.998, p95 7.19 ms
batch  1, efSearch = numCandidates = 256: FAISS HNSW Recall@10 vs Flat 0.990, p95 1.26 ms | Atlas Recall@10 vs Flat 1.000, p95 8.04 ms
batch 64, efSearch = numCandidates =  16: FAISS HNSW Recall@10 vs Flat 0.769, p95 31.47 ms | Atlas Recall@10 vs Flat 0.931, p95 393.66 ms
batch 64, efSearch = numCandidates =  32: FAISS HNSW Recall@10 vs Flat 0.868, p95 18.19 ms | Atlas Recall@10 vs Flat 0.975, p95 404.96 ms
batch 64, efSearch = numCandidates =  64: FAISS HNSW Recall@10 vs Flat 0.932, p95 18.53 ms | Atlas Recall@10 vs Flat 0.993, p95 410.49 ms
batch 64, efSearch = numCandidates = 128: FAISS HNSW Recall@10 vs Flat 0.972, p95 22.59 ms | Atlas Recall@10 vs Flat 0.998, p95 412.14 ms
batch 64, efSearch = numCandidates = 256: FAISS HNSW Recall@10 vs Flat 0.990, p95 30.98 ms | Atlas Recall@10 vs Flat 0.999, p95 421.23 ms
filtered (category pre-filter, median selectivity 0.270), two-tower+atlas at 64: Recall@10 vs exact filtered 1.000, in-category 1.000, p95 7.53 ms
filtered (category pre-filter, median selectivity 0.270), two-tower+faiss at 64: Recall@10 vs exact filtered 0.896, in-category 1.000, p95 0.80 ms
filtered (category pre-filter, median selectivity 0.270), two-tower+atlas at 256: Recall@10 vs exact filtered 1.000, in-category 1.000, p95 7.06 ms
filtered (category pre-filter, median selectivity 0.270), two-tower+faiss at 256: Recall@10 vs exact filtered 0.979, in-category 1.000, p95 1.01 ms
