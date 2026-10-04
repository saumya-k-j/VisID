"""Behavioural tests for the models: shapes, causality, and the ability to memorize a tiny
dataset end to end (training -> constrained generation -> correct item)."""
import numpy as np
import torch

from visid.ids.trie import SemanticIDTrie
from visid.models.sasrec import SeqRecModel, IDItemTower
from visid.models.tiger import TIGER, TigerTokenizer


def test_sasrec_is_causal():
    torch.manual_seed(0)
    m = SeqRecModel(IDItemTower(20, 16), d_model=16, n_layers=2, n_heads=2, max_len=6, dropout=0.0).eval()
    seq = torch.tensor([[0, 3, 4, 5, 6, 7]])
    h1 = m.encode_sequence(seq)
    seq2 = seq.clone()
    seq2[0, -1] = 9  # changing the last item must not change earlier positions
    h2 = m.encode_sequence(seq2)
    assert torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-5)
    assert not torch.allclose(h1[:, -1], h2[:, -1])


def test_sasrec_scores_shape():
    m = SeqRecModel(IDItemTower(20, 16), d_model=16, n_layers=1, n_heads=2, max_len=5)
    s = m.score(torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))
    assert s.shape == (2, 21)


def _tiny_semantic_ids():
    ids = np.full((7, 4), -1, dtype=np.int64)
    ids[1:] = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 2, 0], [1, 1, 1, 0], [2, 0, 1, 0], [2, 0, 1, 1]]
    return ids, [3, 3, 3, 2]


def test_tiger_memorizes_tiny_data():
    torch.manual_seed(0)
    ids, sizes = _tiny_semantic_ids()
    tok = TigerTokenizer(ids, sizes, n_user_buckets=4, max_hist=3)
    model = TIGER(tok.vocab_size, sizes, tok.level_offsets, d_model=32, n_heads=2,
                  n_enc_layers=1, n_dec_layers=1, d_ff=64, dropout=0.0, max_src_len=tok.max_src_len)
    data = [(1, [1, 2], 3), (2, [4, 5], 6), (3, [6, 1], 2), (0, [3], 4)]
    src = tok.encode_histories([u for u, _, _ in data], [h for _, h, _ in data])
    tgt = tok.target_tokens([t for _, _, t in data])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = model.loss(src, tgt)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    model.eval()
    trie = SemanticIDTrie(ids, sizes)
    items, _ = model.generate(src, trie, beam_size=4)
    assert items[:, 0].tolist() == [t for _, _, t in data]


def test_cold_items_get_no_training_signal():
    """Spec: cold items are held out of training *entirely*, including as softmax negatives.
    After a training step, no gradient may reach a cold item's embedding row."""
    torch.manual_seed(0)
    cold = {7, 9}
    m = SeqRecModel(IDItemTower(12, 8), d_model=8, n_layers=1, n_heads=2, max_len=4, dropout=0.0)
    inp = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])
    tgt = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 8]])
    for kind in ("ce", "bce"):
        m.zero_grad()
        m.loss(inp, tgt, kind=kind, exclude_items=cold).backward()
        g = m.items.emb.weight.grad
        for c in cold:
            assert g[c].abs().sum() == 0, f"{kind}: cold item {c} received gradient"
        assert g[8].abs().sum() > 0


def test_mix_cold_respects_quota_and_order():
    from visid.models.train import mix_cold
    items = np.array([[1, 2, 3, 4]])
    scores = np.array([[-1.0, -2.0, -3.0, -4.0]])
    tokens = np.array([[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
    cold = {10, 11, 12}
    cmap = {(0, 0, 1): [10], (1, 0, 0): [11, 12]}
    out = mix_cold(items, scores, tokens, cold, cmap, eps=0.5, k=4)
    # quota = 2 cold slots: 10 (inherits -2.0) and 11 (inherits -4.0); 2 warm slots: 1, 2
    assert sorted(out[0].tolist()) == [1, 2, 10, 11]
    assert out[0].tolist() == [1, 2, 10, 11] or out[0].tolist() == [1, 10, 2, 11]
    assert set(out[0].tolist()) & cold == {10, 11}
    out0 = mix_cold(items, scores, tokens, cold, cmap, eps=0.0, k=4)
    assert out0[0].tolist() == [1, 2, 3, 4]


def test_tokenizer_alignment():
    ids, sizes = _tiny_semantic_ids()
    for align in ("left", "right"):
        tok = TigerTokenizer(ids, sizes, n_user_buckets=4, max_hist=3, align=align)
        src = tok.encode_histories([1, 1], [[1, 2], [3]])
        assert src.shape == (2, 1 + 3 * 4)
        assert (src[:, 0] >= tok.user_offset).all()                       # user token first
        last = torch.tensor(ids[2]) + torch.tensor(tok.level_offsets)     # tokens of item 2
        if align == "left":
            assert torch.equal(src[0, 5:9], last) and (src[0, 9:] == 0).all()
        else:
            assert torch.equal(src[0, -4:], last) and (src[0, 1:5] == 0).all()
            assert torch.equal(src[1, -4:], torch.tensor(ids[3]) + torch.tensor(tok.level_offsets))
