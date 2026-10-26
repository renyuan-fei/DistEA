"""
test_stage_u.py — Stage U tests: sample_halo_rrea_graph
(BNS-GCN-style per-epoch halo node sampling, dist_graph.py, --halo_sample_p):
p=1 returns the graph unchanged, dropped halo nodes lose all their edges, the
adjacency is renormalised, and each call draws a fresh sample.

DGL-free, no GPU, no torch.distributed. <10 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_graph import sample_halo_rrea_graph
from models.rrea import RREAGraph


def _toy_graph(N=30, E=100, R=4, seed=0):
    torch.manual_seed(seed)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, dst], dim=1)
    edge_rel = torch.randint(0, 2 * R, (E,))
    edge_rel_val = torch.ones(E)
    eye = torch.arange(N)
    a_src = torch.cat([src, eye])
    a_dst = torch.cat([dst, eye])
    a_hat = torch.sparse_coo_tensor(
        torch.stack([a_src, a_dst]), torch.ones(E + N), (N, N)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    dis = deg.pow(-0.5)
    idx = a_hat.indices()
    vals = a_hat.values() * dis[idx[0]] * dis[idx[1]]
    adj = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    rel_prof = torch.sparse_coo_tensor(
        torch.stack([src, edge_rel]), torch.full((E,), 0.2), (N, 2 * R)).coalesce()
    return RREAGraph(adj, rel_prof, edge_index, edge_rel, edge_rel_val, R)


def test_p1_returns_same_object():
    g = _toy_graph()
    halo1 = np.array([20, 21, 22])
    halo2 = np.array([25, 26])
    assert sample_halo_rrea_graph(g, halo1, halo2, 1.0, 1.0) is g
    assert sample_halo_rrea_graph(g, np.array([]), np.array([]), 0.5, 0.5) is g
    print("  test_p1_returns_same_object PASSED")


def test_dropped_halo_nodes_have_no_edges():
    g = _toy_graph()
    halo1 = np.arange(18, 24)
    halo2 = np.arange(24, 30)
    torch.manual_seed(1)
    gs = sample_halo_rrea_graph(g, halo1, halo2, 0.5, 0.5)
    assert gs is not g

    halo_all = set(halo1.tolist()) | set(halo2.tolist())
    kept_ends = set(gs.edge_index.flatten().tolist())
    dropped = halo_all - kept_ends
    # keep budget: ceil(0.5*6)=3 kept per side -> exactly 3 dropped per
    # side at most CAN still appear via zero-degree coincidence, so check
    # the other direction: every edge endpoint that IS a halo node must be
    # one of the kept ones (count <= ceil per side).
    kept_halo1 = kept_ends & set(halo1.tolist())
    kept_halo2 = kept_ends & set(halo2.tolist())
    assert len(kept_halo1) <= int(np.ceil(0.5 * len(halo1)))
    assert len(kept_halo2) <= int(np.ceil(0.5 * len(halo2)))
    assert len(dropped) >= 1  # something was actually dropped

    # Non-halo edges all survive
    src, dst = g.edge_index[:, 0], g.edge_index[:, 1]
    is_halo = torch.zeros(30, dtype=torch.bool)
    is_halo[torch.as_tensor(sorted(halo_all))] = True
    n_nonhalo = int((~(is_halo[src] | is_halo[dst])).sum())
    assert gs.edge_index.size(0) >= n_nonhalo
    # edge_rel / edge_rel_val filtered consistently
    assert gs.edge_rel.size(0) == gs.edge_index.size(0)
    assert gs.edge_rel_val.size(0) == gs.edge_index.size(0)
    print("  test_dropped_halo_nodes_have_no_edges PASSED")


def test_shapes_and_renormalisation():
    g = _toy_graph()
    N = 30
    halo1 = np.arange(18, 24)
    torch.manual_seed(2)
    gs = sample_halo_rrea_graph(g, halo1, np.array([]), 0.2, 1.0)
    # N never changes (dropped nodes keep a self-loop row)
    assert gs.adj_self_norm.size() == (N, N)
    assert gs.rel_profile_norm is g.rel_profile_norm
    # Renormalisation: rows sum to <= ~1 and diagonal present for all nodes
    dense = gs.adj_self_norm.to_dense()
    assert torch.isfinite(dense).all()
    assert (dense.diagonal() > 0).all()  # every node keeps its self-loop
    # A fully isolated (dropped) node's row is exactly its self-loop = 1.0
    kept_ends = set(gs.edge_index.flatten().tolist())
    dropped = [i for i in halo1.tolist() if i not in kept_ends]
    for i in dropped:
        row = dense[i]
        assert abs(row[i].item() - 1.0) < 1e-5
        assert abs(row.sum().item() - 1.0) < 1e-5
    print("  test_shapes_and_renormalisation PASSED")


def test_fresh_draw_each_call():
    g = _toy_graph()
    halo1 = np.arange(10, 30)
    torch.manual_seed(3)
    a = sample_halo_rrea_graph(g, halo1, np.array([]), 0.5, 1.0)
    b = sample_halo_rrea_graph(g, halo1, np.array([]), 0.5, 1.0)
    # Overwhelmingly likely different edge sets across draws
    assert a.edge_index.size(0) != b.edge_index.size(0) or not torch.equal(
        a.edge_index, b.edge_index)
    print("  test_fresh_draw_each_call PASSED")


if __name__ == "__main__":
    print("Stage U tests:")
    test_p1_returns_same_object()
    test_dropped_halo_nodes_have_no_edges()
    test_shapes_and_renormalisation()
    test_fresh_draw_each_call()
    print("All Stage U tests PASSED.")
