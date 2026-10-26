"""
test_gas_metis_csr.py — --gas_cluster_method metis_csr
(gas.py _metis_csr_assign): METIS through the vendored package's C-level
CSR entry point, with hubs of degree > hub_degree removed first.

1. On a ring + hub graph the assignment is valid (every node, K non-empty,
   balanced parts), the hub is assigned, and the non-hub edge-cut is lower
   than BFS-chunk's (on a ring BFS order alternates sides, so its chunks
   are not arcs).
2. build_gas_clusters(method="metis_csr") end to end: the multi-cluster
   forward with exact history matches RREA.forward() (test_stage_w's gate).
3. If the METIS call raises, _assign_clusters degrades to the BFS-chunk
   assignment with a warning, exactly like the networkx path.

Tests 1-2 need the vendored Linux extension (nxmetis.metis.part_graph);
they print SKIPPED where it is not importable (e.g. macOS), test 3 always
runs.

Run:  .venv/bin/python tests/test_gas_metis_csr.py
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import (  # noqa: E402
    HUB,
    N_HUB,
    _exact_history,
    _graph_from_edges,
    _hub_graph,
)
from test_stage_w import _model_and_h  # noqa: E402

import gas  # noqa: E402
from gas import _assign_clusters, build_gas_clusters, gas_forward_batch  # noqa: E402

CPU = torch.device("cpu")


def _metis_available() -> bool:
    """A METIS backend that actually RUNS: pymetis (release build, preferred
    by _metis_csr_assign) or the vendored networkx-metis Linux extension.
    Probed with a 2-node partition rather than a bare import -- a pymetis
    install without its compiled module imports fine and only fails inside
    part_graph (seen on macOS), which would make _metis_csr_assign fall
    back to BFS-chunk and the cut comparison below meaningless."""
    try:
        import pymetis
        pymetis.part_graph(2, xadj=[0, 1, 2], adjncy=[1, 0])
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from nxmetis import metis as _nxm
        _nxm.part_graph([0, 1, 2], [1, 0], 2)
        return True
    except Exception:  # noqa: BLE001
        return False


def _edge_cut(edges: np.ndarray, assign: np.ndarray, exclude: np.ndarray) -> int:
    s, d = edges[:, 0], edges[:, 1]
    keep = (s != d) & ~exclude[s] & ~exclude[d]
    return int((assign[s[keep]] != assign[d[keep]]).sum())


def test_metis_csr_assignment_is_valid_and_cuts_less_than_bfs():
    if not _metis_available():
        print("  test_metis_csr_assignment_is_valid_and_cuts_less_than_bfs SKIPPED (nxmetis extension not importable)")
        return
    g = _hub_graph()
    edges = g.edge_index.numpy()
    K = 4
    a_metis = _assign_clusters(N_HUB, edges, K, "metis_csr", hub_degree=10)
    a_bfs = _assign_clusters(N_HUB, edges, K, "bfs")
    assert a_metis.shape == (N_HUB,) and a_metis.min() >= 0 and a_metis.max() < K
    sizes = np.bincount(a_metis, minlength=K)
    assert (sizes > 0).all(), sizes
    assert sizes.max() <= 1.5 * (N_HUB / K), sizes  # METIS balance constraint
    # the hub (degree ~N) is excluded from METIS and assigned afterwards
    deg = np.bincount(np.concatenate([edges[:, 0], edges[:, 1]]), minlength=N_HUB)
    hubs = deg > 10
    assert hubs[HUB]
    cut_metis = _edge_cut(edges, a_metis, hubs)
    cut_bfs = _edge_cut(edges, a_bfs, hubs)
    assert cut_metis < cut_bfs, (cut_metis, cut_bfs)
    print(f"  test_metis_csr_assignment_is_valid_and_cuts_less_than_bfs PASSED "
          f"(non-hub edge-cut metis_csr {cut_metis} vs bfs {cut_bfs}, parts {sizes.tolist()})")


def test_metis_csr_clusters_forward_parity():
    if not _metis_available():
        print("  test_metis_csr_clusters_forward_parity SKIPPED (nxmetis extension not importable)")
        return
    g = _hub_graph()
    model, h = _model_and_h(g, N=N_HUB)
    with torch.no_grad():
        full = model(g, h)
    hist = _exact_history(g, model, h)
    cl = build_gas_clusters(g, 4, method="metis_csr", hub_degree=10)
    assert sum(c.batch.size(0) for c in cl) == N_HUB
    outs = torch.zeros_like(full)
    with torch.no_grad():
        for c in cl:
            outs[c.batch] = gas_forward_batch(model, c, h, hist, CPU)
    assert torch.allclose(full, outs, atol=1e-4), (full - outs).abs().max().item()
    print("  test_metis_csr_clusters_forward_parity PASSED")


def test_rcm_assignment_valid_and_cuts_less_than_random():
    """rcm needs only scipy, so it runs everywhere: valid assignment, hub
    excluded from the ordering but still assigned, equal-size chunks, and a
    much lower non-hub edge-cut than random chunks. (Not compared with
    BFS-chunk: on this toy graph BFS starts AT the hub, so its order is the
    ring's index order and its chunks are perfect arcs -- an accident of
    the toy, not a property of real graphs.)"""
    g = _hub_graph()
    edges = g.edge_index.numpy()
    K = 4
    a_rcm = _assign_clusters(N_HUB, edges, K, "rcm", hub_degree=10)
    assert a_rcm.shape == (N_HUB,) and a_rcm.min() >= 0 and a_rcm.max() < K
    sizes = np.bincount(a_rcm, minlength=K)
    assert sizes.tolist() == [15, 15, 15, 15], sizes
    deg = np.bincount(np.concatenate([edges[:, 0], edges[:, 1]]), minlength=N_HUB)
    hubs = deg > 10
    cut_rcm = _edge_cut(edges, a_rcm, hubs)
    rng = np.random.default_rng(0)
    cut_rand = int(np.mean([
        _edge_cut(edges, rng.permutation(N_HUB) // 15, hubs) for _ in range(20)
    ]))
    # 40 random chords on 60 nodes make this toy small-world, so RCM's
    # bandwidth order is only mildly better than random here; the strong
    # check is the pure ring below. RCM's BFS from a pseudo-peripheral
    # node walks a cycle in BOTH directions at once, so each contiguous
    # chunk is at most two arcs: cut <= 2K (measured 6 for K=4), while a
    # wrongly-applied permutation vector scatters the chunks (cut ~N/2).
    assert cut_rcm < cut_rand, (cut_rcm, cut_rand)
    ring = _graph_from_edges(torch.arange(N_HUB), (torch.arange(N_HUB) + 1) % N_HUB, N_HUB, 5)
    a_ring = _assign_clusters(N_HUB, ring.edge_index.numpy(), K, "rcm", hub_degree=0)
    cut_ring = _edge_cut(ring.edge_index.numpy(), a_ring, np.zeros(N_HUB, dtype=bool))
    assert cut_ring <= 2 * K, cut_ring
    # end to end: clusters build and the forward with exact history matches
    model, h = _model_and_h(g, N=N_HUB)
    with torch.no_grad():
        full = model(g, h)
    hist = _exact_history(g, model, h)
    cl = build_gas_clusters(g, K, method="rcm", hub_degree=10)
    outs = torch.zeros_like(full)
    with torch.no_grad():
        for c in cl:
            outs[c.batch] = gas_forward_batch(model, c, h, hist, CPU)
    assert torch.allclose(full, outs, atol=1e-4)
    print(f"  test_rcm_assignment_valid_and_cuts_less_than_random PASSED "
          f"(non-hub edge-cut rcm {cut_rcm} vs random-chunk mean {cut_rand}; pure ring cut {cut_ring})")


def test_metis_csr_failure_falls_back_to_bfs():
    g = _hub_graph()
    edges = g.edge_index.numpy()
    orig = gas._metis_csr_assign

    def _boom(*a, **k):
        raise RuntimeError("simulated METIS failure")

    gas._metis_csr_assign = _boom
    try:
        a = _assign_clusters(N_HUB, edges, 4, "metis_csr", hub_degree=10)
    finally:
        gas._metis_csr_assign = orig
    assert np.array_equal(a, _assign_clusters(N_HUB, edges, 4, "bfs"))
    print("  test_metis_csr_failure_falls_back_to_bfs PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for fn in (
        test_metis_csr_assignment_is_valid_and_cuts_less_than_bfs,
        test_metis_csr_clusters_forward_parity,
        test_rcm_assignment_valid_and_cuts_less_than_random,
        test_metis_csr_failure_falls_back_to_bfs,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_metis_csr: ALL PASSED")
