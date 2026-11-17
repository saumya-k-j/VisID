Catalog: 12101 items, embedding dim 64, device cpu, 4096 test users, seed 0.

## Batch size 1

| system | index | knob | ANN R@10 (vs Flat) | R@10 | warm R@10 | cold R@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| two-tower+faiss | flat | – | 1.0000 | 0.0740 | 0.0774 | 0.0000 | 0.68 | 0.71 |
| two-tower+faiss | hnsw | 16 | 0.7618 | 0.0720 | 0.0753 | 0.0000 | 0.54 | 0.58 |
| two-tower+faiss | hnsw | 32 | 0.8684 | 0.0740 | 0.0774 | 0.0000 | 0.49 | 0.65 |
| two-tower+faiss | hnsw | 64 | 0.9370 | 0.0740 | 0.0774 | 0.0000 | 0.57 | 0.59 |
| two-tower+faiss | hnsw | 128 | 0.9714 | 0.0740 | 0.0774 | 0.0000 | 0.74 | 1.77 |
| two-tower+faiss | hnsw | 256 | 0.9902 | 0.0740 | 0.0774 | 0.0000 | 0.82 | 1.26 |
| two-tower+faiss | ivf | 1 | 0.4236 | 0.0640 | 0.0669 | 0.0000 | 0.56 | 0.96 |
| two-tower+faiss | ivf | 4 | 0.7300 | 0.0800 | 0.0837 | 0.0000 | 0.54 | 0.80 |
| two-tower+faiss | ivf | 16 | 0.9292 | 0.0740 | 0.0774 | 0.0000 | 0.54 | 0.59 |
| two-tower+faiss | ivf | 64 | 0.9956 | 0.0740 | 0.0774 | 0.0000 | 0.55 | 0.57 |
| two-tower+atlas | hnsw | 16 | 0.9292 | 0.0720 | 0.0753 | 0.0000 | 5.76 | 9.25 |
| two-tower+atlas | hnsw | 32 | 0.9742 | 0.0740 | 0.0774 | 0.0000 | 5.19 | 6.96 |
| two-tower+atlas | hnsw | 64 | 0.9928 | 0.0740 | 0.0774 | 0.0000 | 5.16 | 6.69 |
| two-tower+atlas | hnsw | 128 | 0.9978 | 0.0740 | 0.0774 | 0.0000 | 5.13 | 7.19 |
| two-tower+atlas | hnsw | 256 | 0.9996 | 0.0740 | 0.0774 | 0.0000 | 5.51 | 8.04 |
| two-tower+atlas | enn | – | 1.0000 | 0.0740 | 0.0774 | 0.0000 | 5.81 | 7.89 |

## Batch size 64

| system | index | knob | ANN R@10 (vs Flat) | R@10 | warm R@10 | cold R@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| two-tower+faiss | flat | – | 1.0000 | 0.0835 | 0.0869 | 0.0155 | 19.42 | 34.95 |
| two-tower+faiss | hnsw | 16 | 0.7690 | 0.0818 | 0.0853 | 0.0103 | 16.81 | 31.47 |
| two-tower+faiss | hnsw | 32 | 0.8676 | 0.0837 | 0.0874 | 0.0103 | 14.88 | 18.19 |
| two-tower+faiss | hnsw | 64 | 0.9318 | 0.0835 | 0.0869 | 0.0155 | 16.90 | 18.53 |
| two-tower+faiss | hnsw | 128 | 0.9718 | 0.0837 | 0.0871 | 0.0155 | 20.87 | 22.59 |
| two-tower+faiss | hnsw | 256 | 0.9896 | 0.0835 | 0.0869 | 0.0155 | 29.49 | 30.98 |
| two-tower+faiss | ivf | 1 | 0.4186 | 0.0720 | 0.0748 | 0.0155 | 13.66 | 15.42 |
| two-tower+faiss | ivf | 4 | 0.7323 | 0.0796 | 0.0830 | 0.0103 | 14.70 | 83.66 |
| two-tower+faiss | ivf | 16 | 0.9270 | 0.0835 | 0.0869 | 0.0155 | 15.71 | 33.89 |
| two-tower+faiss | ivf | 64 | 0.9960 | 0.0835 | 0.0869 | 0.0155 | 19.55 | 26.04 |
| two-tower+atlas | hnsw | 16 | 0.9315 | 0.0828 | 0.0864 | 0.0103 | 329.63 | 393.66 |
| two-tower+atlas | hnsw | 32 | 0.9749 | 0.0840 | 0.0876 | 0.0103 | 371.15 | 404.96 |
| two-tower+atlas | hnsw | 64 | 0.9927 | 0.0833 | 0.0869 | 0.0103 | 381.26 | 410.49 |
| two-tower+atlas | hnsw | 128 | 0.9976 | 0.0835 | 0.0869 | 0.0155 | 378.30 | 412.14 |
| two-tower+atlas | hnsw | 256 | 0.9991 | 0.0835 | 0.0869 | 0.0155 | 369.72 | 421.23 |
| two-tower+atlas | enn | – | 1.0000 | 0.0835 | 0.0869 | 0.0155 | 383.43 | 446.83 |

## Filtered search (category of the user's last item, batch 1)

Ground truth: exact inner-product top-10 within the category. Atlas pre-filters with `$vectorSearch.filter`; FAISS uses HNSW with an IDSelector at the same efSearch.

| system | knob | ANN R@10 (filtered) | in-category rate | median selectivity | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| two-tower+atlas | 64 | 0.9998 | 1.0000 | 0.270 | 5.50 | 7.53 |
| two-tower+faiss | 64 | 0.8956 | 1.0000 | 0.270 | 0.74 | 0.80 |
| two-tower+atlas | 256 | 1.0000 | 1.0000 | 0.270 | 5.15 | 7.06 |
| two-tower+faiss | 256 | 0.9786 | 1.0000 | 0.270 | 0.91 | 1.01 |
