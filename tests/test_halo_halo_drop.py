"""
test_halo_halo_drop.py — --drop_halo_halo_edges: build_rrea_graph(...,
local_mask=owned rows) must equal build_rrea_graph on the same data with
the halo-halo triples (both ends halo rows) removed beforehand -- same
edge_index / edge_rel / edge_val in the same order, same adj_self_norm and
rel_profile_norm -- on both the vectorised and the Python path, and no
kept edge may join two halo rows. local_mask=None is the unchanged graph.

DGL-free, CPU, < 30 s.
Run:  .venv/bin/python tests/test_halo_halo_drop.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_graph import DistGraphLoader, build_rrea_graph  # noqa: E402
from tests.test_vector_graph import N_REL, _Data, _setup  # noqa: E402

_FIELDS = ("edge_index", "edge_rel", "edge_rel_val", "adj_self_norm", "rel_profile_norm")


def _dense(x):
    return x.coalesce().to_dense() if x.is_sparse else x


def _same(a, b):
    for k in _FIELDS:
        if hasattr(a, k):
            assert torch.equal(_dense(getattr(a, k)), _dense(getattr(b, k))), k


def test_mask_equals_prefiltered_triples():
    n = 0
    for seed in (0, 1):
        with tempfile.TemporaryDirectory() as tmp:
            data = _setup(tmp, seed)
            for rank in (0, 1):
                for vec in ("0", "1"):
                    os.environ["DISTEA_VECTOR_GRAPH"] = vec
                    try:
                        _, g2l, ids, i1, _h1, i2, _h2 = DistGraphLoader(
                            data, tmp, rank, 2, torch.device("cpu")).load()
                        own = np.zeros(len(ids), dtype=bool)
                        own[i1] = True
                        own[i2] = True
                        owned_ids = set(np.asarray(ids)[own].tolist())
                        dropped = _Data(
                            [x for x in data.kg1_triples if x[0] in owned_ids or x[2] in owned_ids],
                            [x for x in data.kg2_triples if x[0] in owned_ids or x[2] in owned_ids],
                            data.edge_weights1, data.edge_weights2)
                        got = build_rrea_graph(data, g2l, ids, N_REL, torch.device("cpu"), local_mask=own)
                        ref = build_rrea_graph(dropped, g2l, ids, N_REL, torch.device("cpu"))
                        _same(got, ref)
                        full = build_rrea_graph(data, g2l, ids, N_REL, torch.device("cpu"))
                        e = got.edge_index.numpy()
                        assert np.all(own[e[:, 0]] | own[e[:, 1]]), "an edge joins two halo rows"
                        assert got.edge_index.size(0) < full.edge_index.size(0), "nothing was dropped"
                        _same(build_rrea_graph(data, g2l, ids, N_REL, torch.device("cpu"), local_mask=None), full)
                        n += 1
                    finally:
                        os.environ.pop("DISTEA_VECTOR_GRAPH", None)
    print(f"  test_mask_equals_prefiltered_triples PASSED ({n} seed/rank/path combinations)")


if __name__ == "__main__":
    for fn in (test_mask_equals_prefiltered_triples,):
        print(f"[{fn.__name__}]")
        fn()
    print("test_halo_halo_drop: ALL PASSED")
