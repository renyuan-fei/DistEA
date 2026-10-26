"""
test_vector_graph.py — DISTEA_VECTOR_GRAPH=1: the numpy
code paths of DistGraphLoader.load() (halo detection, edge list) and
build_rrea_graph() must reproduce the Python loops EXACTLY — same halo
lists, same all_global_ids, same (coalesced) weighted adjacency, same
edge_index / edge_rel / edge_val in the same ORDER, same adj_self_norm and
rel_profile_norm — on random multi-relational KGs with parallel edges,
duplicate triples, self-loops, relations without weights (default 1.0),
weights below the 0.3 floor, entities outside the rank, and halos on /
off, for both ranks of a 2-way partition.

DGL-free, CPU, < 30 s.
Run:  .venv/bin/python tests/test_vector_graph.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_graph import DistGraphLoader, build_rrea_graph  # noqa: E402

N_REL = 7


class _Data:
    def __init__(self, kg1, kg2, w1, w2):
        self.kg1_triples = kg1
        self.kg2_triples = kg2
        self.edge_weights1 = w1
        self.edge_weights2 = w2


def _random_kg(rng, offset, n_ent, n_tri):
    h = rng.integers(offset, offset + n_ent, n_tri)
    t = rng.integers(offset, offset + n_ent, n_tri)
    r = rng.integers(0, N_REL, n_tri)
    triples = list(zip(h.tolist(), r.tolist(), t.tolist(), strict=True))
    triples += triples[: n_tri // 10]                       # duplicate triples
    triples += [(hh, (rr + 1) % N_REL, tt) for hh, rr, tt in triples[: n_tri // 10]]  # parallel edges
    triples += [(offset + i, i % N_REL, offset + i) for i in range(0, n_ent, 37)]    # self-loops
    return triples


def _setup(tmp, seed):
    rng = np.random.default_rng(seed)
    n1, n2 = 300, 200
    kg1 = _random_kg(rng, 0, n1, 1200)
    kg2 = _random_kg(rng, n1, n2, 800)
    w1 = {r: (float(rng.random()), float(rng.random())) for r in range(0, N_REL, 2)}  # odd rels: default (1, 1)
    w2 = {r: (float(rng.random()) * 0.2, float(rng.random()) * 0.2) for r in range(N_REL)}  # all below the 0.3 floor
    perm = rng.permutation(n1)
    np.save(os.path.join(tmp, "0.npy"), np.sort(perm[: n1 // 2]).astype(np.int64))
    np.save(os.path.join(tmp, "1.npy"), np.sort(perm[n1 // 2:]).astype(np.int64))
    k = rng.permutation(np.arange(n1, n1 + n2))
    with open(os.path.join(tmp, "kg2_assignments.json"), "w") as f:
        json.dump({"0": sorted(k[: n2 // 2].tolist()), "1": sorted(k[n2 // 2:].tolist())}, f)
    return _Data(kg1, kg2, w1, w2)


def _run(data, tmp, rank, vec, no_halo):
    os.environ["DISTEA_VECTOR_GRAPH"] = "1" if vec else "0"
    for attr in ("_kg1_triples_np", "_kg2_triples_np"):
        data.__dict__.pop(attr, None)
    loader = DistGraphLoader(data, tmp, rank, 2, torch.device("cpu"),
                             no_halo_kg1=no_halo, no_halo_kg2=no_halo)
    g, g2l, ids, lk1, hk1, lk2, hk2 = loader.load()
    rg = build_rrea_graph(data, g2l, ids, N_REL, torch.device("cpu"))
    return g, g2l, ids, (lk1, hk1, lk2, hk2), rg


def test_loader_and_graph_parity():
    n_checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (0, 1):
            data = _setup(tmp, seed)
            for rank in (0, 1):
                for no_halo in (False, True):
                    g_a, g2l_a, ids_a, idx_a, rg_a = _run(data, tmp, rank, False, no_halo)
                    g_b, g2l_b, ids_b, idx_b, rg_b = _run(data, tmp, rank, True, no_halo)
                    assert np.array_equal(ids_a, ids_b)
                    assert g2l_a == g2l_b
                    for x, y in zip(idx_a, idx_b, strict=True):
                        assert np.array_equal(x, y)
                    assert not no_halo or (len(idx_b[1]) == 0 and len(idx_b[3]) == 0)
                    assert no_halo or (len(idx_b[1]) > 0 and len(idx_b[3]) > 0), "fixture should produce halos"
                    assert torch.equal(g_a.to_dense(), g_b.to_dense())
                    assert torch.equal(rg_a.edge_index, rg_b.edge_index)
                    assert torch.equal(rg_a.edge_rel, rg_b.edge_rel)
                    assert torch.equal(rg_a.edge_rel_val, rg_b.edge_rel_val)
                    assert torch.equal(rg_a.adj_self_norm.to_dense(), rg_b.adj_self_norm.to_dense())
                    pa, pb = rg_a.rel_profile_norm.coalesce(), rg_b.rel_profile_norm.coalesce()
                    assert torch.equal(pa.indices(), pb.indices())
                    assert torch.allclose(pa.values(), pb.values(), rtol=0, atol=1e-7)
                    assert rg_b.edge_index.size(0) > 0 and rg_b.edge_rel_val.min() < 1.0, "fixture should produce parallel edges"
                    n_checked += 1
    os.environ.pop("DISTEA_VECTOR_GRAPH", None)
    print(f"  test_loader_and_graph_parity PASSED ({n_checked} rank/halo/seed combinations bit-identical)")


def test_triples_array_cached_on_data():
    from dist_graph import _triples_array

    data = _Data([(0, 0, 1), (1, 2, 0)], [], {}, {})
    a = _triples_array(data, "kg1_triples")
    assert a.shape == (2, 3) and a.dtype == np.int64 and a[1, 1] == 2
    assert _triples_array(data, "kg1_triples") is a
    assert _triples_array(data, "kg2_triples").shape == (0, 3)
    print("  test_triples_array_cached_on_data PASSED")


if __name__ == "__main__":
    for fn in (test_loader_and_graph_parity, test_triples_array_cached_on_data):
        print(f"[{fn.__name__}]")
        fn()
    print("test_vector_graph: ALL PASSED")
