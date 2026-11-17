"""MongoDB Atlas Vector Search as a third retrieval backend (next to FAISS HNSW / IVF).

Pure tests (index definition, query pipeline, documents, overlap metric) always run.
Server tests need a running Atlas deployment, e.g. Atlas Local in Docker:
    docker compose up -d atlas
    export MONGODB_URI="mongodb://localhost:27017/?directConnection=true"
They are skipped cleanly when MONGODB_URI is unset.
"""
import os
import uuid

import numpy as np
import pytest

from visid.bench.atlas import (atlas_similarity, category_of, create_vector_index, index_definition,
                               item_documents, load_collection, search_topk, vector_search_pipeline)
from visid.eval.metrics import overlap_at_k

faiss = pytest.importorskip("faiss")


# ----------------------------------------------------------------------------- pure
def test_overlap_at_k_hand_computed():
    topk = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    truth = np.array([[4, 3, 9, 9], [5, 6, 7, 8]])
    # row 0 shares {3, 4} of 4 -> 0.5; row 1 shares all 4 -> 1.0; mean 0.75
    assert overlap_at_k(topk, truth, k=4) == pytest.approx(0.75)
    # padding (-1) never counts as a hit
    assert overlap_at_k(np.array([[-1, -1]]), np.array([[-1, -1]]), k=2) == 0.0


def test_similarity_matches_faiss_metric():
    assert atlas_similarity(faiss.METRIC_INNER_PRODUCT) == "dotProduct"
    assert atlas_similarity(faiss.METRIC_L2) == "euclidean"


def test_index_definition_has_vector_and_filter_fields():
    d = index_definition(dim=64, similarity="dotProduct", path="embedding",
                         filter_paths=("category", "is_cold"), hnsw_options={"maxEdges": 32, "numEdgeCandidates": 200})
    vec = [f for f in d["fields"] if f["type"] == "vector"]
    assert vec == [{"type": "vector", "path": "embedding", "numDimensions": 64, "similarity": "dotProduct",
                    "hnswOptions": {"maxEdges": 32, "numEdgeCandidates": 200}}]
    assert sorted(f["path"] for f in d["fields"] if f["type"] == "filter") == ["category", "is_cold"]
    # no hnswOptions key at all when not configured (server defaults)
    assert "hnswOptions" not in index_definition(8, "dotProduct")["fields"][0]


def test_vector_search_pipeline():
    p = vector_search_pipeline([0.1, 0.2], k=10, num_candidates=64, index="ix", path="embedding",
                               filter={"category": "Makeup"})
    vs = p[0]["$vectorSearch"]
    assert vs == {"index": "ix", "path": "embedding", "queryVector": [0.1, 0.2], "numCandidates": 64,
                  "limit": 10, "filter": {"category": "Makeup"}}
    assert p[1] == {"$project": {"_id": 1}}
    exact = vector_search_pipeline([0.1], k=5, num_candidates=None, index="ix", path="embedding", exact=True)
    assert exact[0]["$vectorSearch"]["exact"] is True and "numCandidates" not in exact[0]["$vectorSearch"]
    with pytest.raises(ValueError):  # ANN needs numCandidates >= limit
        vector_search_pipeline([0.1], k=10, num_candidates=5, index="ix", path="embedding")


def test_item_documents_skip_padding_row_and_mark_cold():
    vecs = np.arange(12, dtype=np.float32).reshape(3, 4)  # rows are items 1..3 (padding row already dropped)
    docs = item_documents(vecs, ["a", "b", "a"], cold_items={2})
    assert [d["_id"] for d in docs] == [1, 2, 3]
    assert [d["is_cold"] for d in docs] == [False, True, False]
    assert docs[1]["embedding"] == [4.0, 5.0, 6.0, 7.0] and docs[2]["category"] == "a"


def test_category_of_level_with_fallback():
    assert category_of({"categories": ["Beauty", "Makeup", "Eyes"]}, level=1) == "Makeup"
    assert category_of({"categories": ["c3"]}, level=1) == "c3"          # shallower list: deepest level
    assert category_of({"categories": []}, level=1) == "unknown"


# ----------------------------------------------------------------------------- server (Atlas)
N_ITEMS, DIM, N_CATS, K = 1500, 32, 5, 10


@pytest.fixture(scope="module")
def atlas():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set: start Atlas Local (docker compose up -d atlas) to run server tests")
    pymongo = pytest.importorskip("pymongo")
    rng = np.random.default_rng(0)
    # Unnormalized vectors with very different norms, like two-tower item embeddings, so the test
    # fails if the server ranks by cosine instead of the raw inner product FAISS uses.
    vecs = (rng.normal(size=(N_ITEMS, DIM)) * rng.uniform(0.2, 3.0, size=(N_ITEMS, 1))).astype(np.float32)
    cats = [f"cat{i % N_CATS}" for i in range(N_ITEMS)]
    cold = set(range(1, N_ITEMS + 1, 7))
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[f"visid_test_{uuid.uuid4().hex[:8]}"]
    coll = db["items"]
    load_collection(coll, item_documents(vecs, cats, cold))
    create_vector_index(coll, index_definition(DIM, "dotProduct", filter_paths=("category", "is_cold")),
                        name="item_embedding", timeout_s=600)
    yield {"coll": coll, "vecs": vecs, "cats": np.array(cats), "cold": cold}
    client.drop_database(db.name)
    client.close()


def _flat_topk(vecs, queries, k, mask=None):
    """FAISS Flat inner-product ground truth (item ids are 1-based)."""
    if mask is None:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index.search(queries, k)[1] + 1
    ids = np.where(mask)[0]
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs[ids])
    return ids[index.search(queries, k)[1]] + 1


def test_index_round_trip(atlas):
    coll, vecs = atlas["coll"], atlas["vecs"]
    assert coll.count_documents({}) == N_ITEMS
    doc = coll.find_one({"_id": 42})
    np.testing.assert_allclose(doc["embedding"], vecs[41], rtol=1e-6)
    assert doc["category"] == atlas["cats"][41] and doc["is_cold"] == (42 in atlas["cold"])
    # every document is in the vector index: exact search over the whole catalog returns all of them
    got = search_topk(coll, vecs[0], k=N_ITEMS, num_candidates=None, index="item_embedding", exact=True)
    assert sorted(got.tolist()) == list(range(1, N_ITEMS + 1))


def test_top10_at_max_num_candidates_matches_faiss_flat(atlas):
    vecs = atlas["vecs"]
    q = np.random.default_rng(1).normal(size=(50, DIM)).astype(np.float32)
    truth = _flat_topk(vecs, q, K)
    exact = np.stack([search_topk(atlas["coll"], v, K, None, "item_embedding", exact=True) for v in q])
    ann = np.stack([search_topk(atlas["coll"], v, K, 256, "item_embedding") for v in q])
    assert overlap_at_k(exact, truth, K) >= 0.99   # same metric as FAISS (float ties aside)
    assert overlap_at_k(ann, truth, K) >= 0.95     # HNSW at the top of the sweep


def test_filter_returns_only_matching_items(atlas):
    coll, vecs, cats = atlas["coll"], atlas["vecs"], atlas["cats"]
    q = np.random.default_rng(2).normal(size=(20, DIM)).astype(np.float32)
    for i, v in enumerate(q):
        c = f"cat{i % N_CATS}"
        got = search_topk(coll, v, K, 256, "item_embedding", filter={"category": c})
        assert len(got) == K and all(cats[g - 1] == c for g in got)
        warm = search_topk(coll, v, K, 256, "item_embedding", filter={"is_cold": False})
        assert not set(warm.tolist()) & atlas["cold"]
    # and the filtered ANN result is the right one, not just a matching one
    c = "cat0"
    truth = _flat_topk(vecs, q, K, mask=cats == c)
    ann = np.stack([search_topk(coll, v, K, 256, "item_embedding", filter={"category": c}) for v in q])
    assert overlap_at_k(ann, truth, K) >= 0.95
