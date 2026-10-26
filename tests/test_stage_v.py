"""
test_stage_v.py — Stage V tests: minibatch_csr.py (--minibatch_sampler
vectorized / dgl), the numpy/CSR twin of rrea_minibatch.py's python loops.

The critical invariant: for the SAME seeds, the vectorized backend must
produce exactly the same sampled layers (fanout=None), nested active sets,
self-block sparse matrix, attention edge multiset, relation-profile
features and therefore the same RREA.forward_blocks output as the python
backend. The vectorized backend exists only to make per-batch block
construction faster; it must not change the sampled structure.

DGL-free (the 'dgl' backend is exercised only when DGL happens to be
importable; otherwise its fallback-to-vectorized path is what gets
tested). No GPU, no torch.distributed. <15 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minibatch_csr import (
    build_rrea_attention_block_csr,
    build_rrea_self_block_csr,
    compute_rel_profile_features_csr,
    nested_active_sets_np,
    resolve_sampler_backend,
    sample_multi_hop_csr,
)
from minibatch_sampler import sample_multi_hop
from models.gcn import build_entity_features
from models.rrea import RREA
from rrea_minibatch import (
    RREAMinibatchIndex,
    build_rrea_attention_block,
    build_rrea_self_block,
    compute_rel_profile_features,
    nested_active_sets,
    prepare_rrea_batch,
)

_REL_NUM = 5
_N = 60


def _random_triples(seed=0, n=_N, m=220, rel_num=_REL_NUM):
    """Random multi-relational graph with deliberate duplicate (h,t) pairs
    (multi-relation edges -> 1/multiplicity values, arc-count degrees) and
    a few isolated entities (ids >= every triple id) to exercise the
    'no arcs' default paths."""
    rng = np.random.default_rng(seed)
    active_n = n - 5  # ids n-5..n-1 never appear in any triple
    h = rng.integers(0, active_n, m)
    t = rng.integers(0, active_n, m)
    r = rng.integers(0, rel_num, m)
    triples = [(int(a), int(b), int(c)) for a, b, c in zip(h, r, t, strict=True) if a != b]
    triples += [(0, 1, 1), (0, 2, 1), (1, 3, 0)]  # forced multi-relation pair
    return triples


def _both_indexes(triples):
    py = RREAMinibatchIndex.build(triples, _REL_NUM, backend="python")
    vec = RREAMinibatchIndex.build(triples, _REL_NUM, backend="vectorized", n_entities=_N)
    assert py.csr is None and vec.csr is not None and vec.backend == "vectorized"
    return py, vec


def _dense(block):
    return block.adj_t.to_dense()


# ---------------------------------------------------------------------------


def test_sample_multi_hop_csr_matches_python_when_unrestricted():
    triples = _random_triples()
    py, vec = _both_indexes(triples)
    for seeds in (np.array([3]), np.array([0, 7, 12]), np.array([_N - 1]), np.array([5, _N - 2])):
        a = sample_multi_hop(seeds, py.plain_neighbor_index(), num_layers=3, fanout=None)
        b = sample_multi_hop_csr(seeds, vec.csr, num_layers=3, fanout=None)
        assert len(a) == len(b), (seeds, [len(x) for x in a], [len(x) for x in b])
        for la, lb in zip(a, b, strict=True):
            assert np.array_equal(la, lb), (seeds, la, lb)
    print("  test_sample_multi_hop_csr_matches_python_when_unrestricted PASSED")


def test_sample_multi_hop_csr_fanout_caps_and_keeps_layers_disjoint():
    triples = _random_triples()
    _py, vec = _both_indexes(triples)
    seeds = np.array([0, 7, 12])
    rng = np.random.default_rng(1)
    layers = sample_multi_hop_csr(seeds, vec.csr, num_layers=2, fanout=2, rng=rng)
    inner_first = list(reversed(layers))
    assert np.array_equal(inner_first[0], np.unique(seeds))
    all_ids = np.concatenate(layers)
    assert len(np.unique(all_ids)) == len(all_ids), "an entity appeared in two layers"
    for prev, nxt in zip(inner_first, inner_first[1:], strict=False):
        assert len(nxt) <= 2 * len(prev), (len(prev), len(nxt))
    again = sample_multi_hop_csr(seeds, vec.csr, num_layers=2, fanout=2, rng=np.random.default_rng(1))
    assert all(np.array_equal(x, y) for x, y in zip(layers, again, strict=True)), "seeded rng not reproducible"
    print("  test_sample_multi_hop_csr_fanout_caps_and_keeps_layers_disjoint PASSED")


def test_nested_active_sets_np_matches_python():
    layers = [np.array([3, 9]), np.array([0, 2]), np.array([1, 7])]
    a = nested_active_sets(layers)
    b = nested_active_sets_np(layers)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True)), (a, b)
    print("  test_nested_active_sets_np_matches_python PASSED")


def test_self_block_csr_matches_python():
    triples = _random_triples()
    py, vec = _both_indexes(triples)
    device = torch.device("cpu")
    for seeds in (np.array([3]), np.array([0, 7, 12, _N - 1])):
        layers = sample_multi_hop(seeds, py.plain_neighbor_index(), num_layers=3, fanout=None)
        active = nested_active_sets(layers)
        a = build_rrea_self_block(active[0], active[1], py.relation_index, py.self_degrees, device)
        b = build_rrea_self_block_csr(active[0], active[1], vec.csr, device)
        assert torch.allclose(_dense(a), _dense(b), atol=1e-6), (seeds, _dense(a), _dense(b))
    print("  test_self_block_csr_matches_python PASSED")


def test_attention_block_csr_matches_python_edge_multiset():
    triples = _random_triples()
    py, vec = _both_indexes(triples)
    seeds = np.array([0, 7, 12])
    layers = sample_multi_hop(seeds, py.plain_neighbor_index(), num_layers=3, fanout=None)
    active = nested_active_sets(layers)
    for k in range(2):
        a = build_rrea_attention_block(active[k + 1], active[k + 2], py.relation_index)
        b = build_rrea_attention_block_csr(active[k + 1], active[k + 2], vec.csr)
        # both paths end up as float32 tensors in forward_block_hop; compare there
        va = np.asarray(a[3], dtype=np.float32).round(5).tolist()
        vb = np.asarray(b[3], dtype=np.float32).round(5).tolist()
        ea = sorted(zip(a[0], a[1], a[2], va, strict=True))
        eb = sorted(zip(b[0].tolist(), b[1].tolist(), b[2].tolist(), vb, strict=True))
        assert ea == eb, (k, ea[:5], eb[:5])
    print("  test_attention_block_csr_matches_python_edge_multiset PASSED")


def test_rel_profile_features_csr_matches_python():
    triples = _random_triples()
    py, vec = _both_indexes(triples)
    torch.manual_seed(0)
    rel_emb = torch.randn(2 * _REL_NUM, 6)
    ids = np.array([0, 1, 7, 12, _N - 1])
    a = compute_rel_profile_features(ids, py.rel_out_index, py.rel_in_index, py.self_degrees, _REL_NUM, rel_emb)
    b = compute_rel_profile_features_csr(ids, vec.csr, rel_emb)
    assert torch.allclose(a, b, atol=1e-5), (a, b)
    print("  test_rel_profile_features_csr_matches_python PASSED")


def test_forward_blocks_identical_across_backends():
    """End-to-end: same seeds, fanout=None, python vs vectorized index ->
    prepare_rrea_batch -> RREA.forward_blocks must agree numerically."""
    triples = _random_triples()
    py, vec = _both_indexes(triples)
    torch.manual_seed(0)
    model = RREA(rel_num=_REL_NUM, output_dim=4, num_layers=2, dropout=0.0, attn_heads=1)
    model.eval()
    device = torch.device("cpu")
    seeds = np.array([0, 7, 12])
    ba = prepare_rrea_batch(seeds, py, num_layers=2, fanout=None, device=device)
    bb = prepare_rrea_batch(seeds, vec, num_layers=2, fanout=None, device=device)
    assert np.array_equal(ba.outer_ids, bb.outer_ids)
    assert all(np.array_equal(x, y) for x, y in zip(ba.active, bb.active, strict=True))
    h = build_entity_features(ba.outer_ids, 4, device)
    oa = model.forward_blocks(ba, py, h)
    ob = model.forward_blocks(bb, vec, h)
    assert torch.allclose(oa, ob, atol=1e-5), (oa, ob)
    print("  test_forward_blocks_identical_across_backends PASSED")


def test_isolated_seed_entity_has_no_arcs_and_degree_one():
    triples = _random_triples()
    _py, vec = _both_indexes(triples)
    iso = np.array([_N - 1])
    layers = sample_multi_hop_csr(iso, vec.csr, num_layers=3, fanout=None)
    ref = sample_multi_hop(iso, _py.plain_neighbor_index(), num_layers=3, fanout=None)
    # python semantics: one empty frontier is appended before the early
    # stop -> [empty, seeds]; the CSR twin must match that exactly.
    assert len(layers) == len(ref) == 2, (layers, ref)
    assert len(layers[0]) == 0 and np.array_equal(layers[1], iso), layers
    assert vec.csr.degrees(iso).tolist() == [1.0]
    e = build_rrea_attention_block_csr(iso, iso, vec.csr)
    assert len(e[0]) == 0
    blk = build_rrea_self_block_csr(iso, iso, vec.csr, torch.device("cpu"))
    assert torch.allclose(blk.adj_t.to_dense(), torch.ones(1, 1))  # self-loop only, deg 1
    print("  test_isolated_seed_entity_has_no_arcs_and_degree_one PASSED")


def test_resolve_backend_dgl_falls_back_or_samples():
    from minibatch_csr import import_dgl
    try:
        import_dgl()
        have_dgl = True
    except Exception:
        have_dgl = False
    resolved = resolve_sampler_backend("dgl")
    if not have_dgl:
        assert resolved == "vectorized", resolved
        idx = RREAMinibatchIndex.build(_random_triples(), _REL_NUM, backend="dgl", n_entities=_N)
        assert idx.backend == "vectorized"
    else:
        from minibatch_csr import sample_multi_hop_dgl
        idx = RREAMinibatchIndex.build(_random_triples(), _REL_NUM, backend="dgl", n_entities=_N)
        seeds = np.array([0, 7, 12])
        layers = sample_multi_hop_dgl(seeds, idx.csr, num_layers=2, fanout=None)
        ref = sample_multi_hop_csr(seeds, idx.csr, num_layers=2, fanout=None)
        assert all(np.array_equal(x, y) for x, y in zip(layers, ref, strict=True))
    try:
        resolve_sampler_backend("nope")
        raise AssertionError("unknown backend must raise")
    except ValueError:
        pass
    print("  test_resolve_backend_dgl_falls_back_or_samples PASSED")


if __name__ == "__main__":
    print("Stage V tests:")
    test_sample_multi_hop_csr_matches_python_when_unrestricted()
    test_sample_multi_hop_csr_fanout_caps_and_keeps_layers_disjoint()
    test_nested_active_sets_np_matches_python()
    test_self_block_csr_matches_python()
    test_attention_block_csr_matches_python_edge_multiset()
    test_rel_profile_features_csr_matches_python()
    test_forward_blocks_identical_across_backends()
    test_isolated_seed_entity_has_no_arcs_and_degree_one()
    test_resolve_backend_dgl_falls_back_or_samples()
    print("All Stage V tests PASSED.")
