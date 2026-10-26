"""--ablate_full_graph: every rank holds the whole graph, ownership unchanged."""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from dist_graph import DistGraphLoader


def _fixture(tmp):
    # KG1 ids 0-5 (ranks: {0,1,2} / {3,4,5}), KG2 ids 10-15 (ranks: {10,11,12} / {13,14,15})
    np.save(os.path.join(tmp, "0.npy"), np.array([0, 1, 2]))
    np.save(os.path.join(tmp, "1.npy"), np.array([3, 4, 5]))
    with open(os.path.join(tmp, "kg2_assignments.json"), "w") as f:
        json.dump({"0": [10, 11, 12], "1": [13, 14, 15]}, f)
    kg1 = [(0, 0, 1), (1, 0, 2), (2, 0, 3), (4, 0, 5)]          # 2-3 crosses ranks; 4-5 is rank-1 only
    kg2 = [(10, 0, 11), (12, 0, 13), (14, 0, 15)]                # 12-13 crosses; 14-15 rank-1 only
    return types.SimpleNamespace(kg1_triples=kg1, kg2_triples=kg2, rel_num=1,
                                 edge_weights1={}, edge_weights2={})


def _load(tmp, data, rank, full_graph):
    ldr = DistGraphLoader(data, tmp, rank, 2, torch.device("cpu"), full_graph=full_graph)
    return ldr.load()


def main():
    for vec in ("0", "1"):
        os.environ["DISTEA_VECTOR_GRAPH"] = vec
        with tempfile.TemporaryDirectory() as tmp:
            data = _fixture(tmp)
            g, g2l, ids, lk1, hk1, lk2, hk2 = _load(tmp, data, 0, True)
            assert ids.tolist() == [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], ids
            assert lk1.tolist() == [0, 1, 2] and lk2.tolist() == [6, 7, 8], "ownership = partition"
            assert hk1.tolist() == [3, 4, 5] and hk2.tolist() == [9, 10, 11]
            # every edge present (both directions are added by the loader): 7 triples
            n_edges = g.number_of_edges() if hasattr(g, "number_of_edges") else g._nnz()
            assert n_edges >= 14, n_edges
            # baseline (1-hop halo) for comparison keeps rank 0 away from 4-5 / 14-15
            _, _, ids0, _, hk1_0, _, hk2_0 = _load(tmp, data, 0, False)
            assert set(ids0.tolist()) == {0, 1, 2, 3, 10, 11, 12, 13}, ids0
            # pair sharding by owned KG1 rows
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from framework import DistEAFramework
            fw = types.SimpleNamespace(train_links=[(0, 10), (3, 13), (5, 15)])
            allp = DistEAFramework._map_links_to_local(fw, g2l)
            own = DistEAFramework._map_links_to_local(fw, g2l, owned_kg1_n=len(lk1))
            assert len(allp) == 3 and own == [(0, 6)], (allp, own)
    print("test_full_graph_ablation: OK")


if __name__ == "__main__":
    main()
