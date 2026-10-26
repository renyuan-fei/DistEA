"""
test_gas_bfs_chunk_scale.py — gas.py::_assign_clusters's BFS-chunk fallback
must stay vectorized (scipy CSR + compiled BFS), not regress to a
pure-Python per-edge/per-node loop, which does not finish in reasonable
time on DBP1M's ~640K-node/3.4M-edge per-rank graphs. Large graphs skip
METIS and always take this path, so its performance matters, not just its
correctness.

Run:  uv run python tests/test_gas_bfs_chunk_scale.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gas import _assign_clusters, _bfs_chunk_order  # noqa: E402


def test_bfs_chunk_order_is_a_valid_permutation():
    rng = np.random.default_rng(0)
    N, E = 5_000, 20_000
    edges = np.stack([rng.integers(0, N, E), rng.integers(0, N, E)], axis=1)
    order = _bfs_chunk_order(N, edges)
    assert sorted(order.tolist()) == list(range(N))
    print("  test_bfs_chunk_order_is_a_valid_permutation PASSED")


def test_assign_clusters_bfs_partitions_every_node_exactly_once():
    rng = np.random.default_rng(1)
    N, E, K = 3_000, 12_000, 6
    edges = np.stack([rng.integers(0, N, E), rng.integers(0, N, E)], axis=1)
    assign = _assign_clusters(N, edges, K, "bfs")
    assert assign.shape == (N,)
    assert set(np.unique(assign).tolist()) <= set(range(K))
    counts = np.bincount(assign, minlength=K)
    assert counts.sum() == N
    print("  test_assign_clusters_bfs_partitions_every_node_exactly_once PASSED")


def test_disconnected_graph_no_edges():
    # len(edges) == 0 takes the special-cased identity-order path.
    assign = _assign_clusters(50, np.empty((0, 2), dtype=np.int64), 5, "bfs")
    assert assign.shape == (50,)
    assert np.array_equal(np.sort(np.unique(assign)), np.arange(5))
    print("  test_disconnected_graph_no_edges PASSED")


def test_scale_representative_of_dbp1m_rank_shard():
    # A DBP1M/N=4 rank holds ~644K local+halo nodes and 3.38M edges. Use a
    # smaller but same-order-of-magnitude case
    # so the test stays fast while still exercising the vectorized path,
    # not a toy graph that would pass even with an accidental Python-loop
    # regression.
    rng = np.random.default_rng(2)
    N, E = 200_000, 1_000_000
    edges = np.stack([rng.integers(0, N, E), rng.integers(0, N, E)], axis=1)
    t0 = time.time()
    order = _bfs_chunk_order(N, edges)
    elapsed = time.time() - t0
    assert sorted(order.tolist()) == list(range(N))
    # A pure-Python loop takes far longer on a ~3x larger real graph; a
    # generous 30s bound here
    # only catches a regression back to that complexity class, not normal
    # machine-to-machine variance.
    assert elapsed < 30, f"BFS-chunk order took {elapsed:.1f}s at N={N} E={E} (expected < 30s)"
    print(f"  test_scale_representative_of_dbp1m_rank_shard PASSED ({elapsed:.2f}s)")


def test_size_guard_skips_nxmetis_entirely():
    # Above the guard, method="metis" must never touch nxmetis: the networkx
    # build + METIS on a 644K-node graph takes too long, so even a fast
    # fallback is useless if the attempt runs first. Inject a fake nxmetis whose
    # partition() raises if called.
    import types

    fake = types.ModuleType("nxmetis")

    def _boom(*a, **k):
        raise AssertionError("nxmetis.partition must not be called above the size guard")

    fake.partition = _boom
    saved = sys.modules.get("nxmetis")
    sys.modules["nxmetis"] = fake
    os.environ["DISTEA_GAS_METIS_MAX_NODES"] = "10"
    try:
        rng = np.random.default_rng(3)
        N, E, K = 500, 2_000, 4
        edges = np.stack([rng.integers(0, N, E), rng.integers(0, N, E)], axis=1)
        assign = _assign_clusters(N, edges, K, "metis")
        assert assign.shape == (N,) and np.bincount(assign, minlength=K).sum() == N
    finally:
        del os.environ["DISTEA_GAS_METIS_MAX_NODES"]
        if saved is None:
            del sys.modules["nxmetis"]
        else:
            sys.modules["nxmetis"] = saved
    print("  test_size_guard_skips_nxmetis_entirely PASSED")


if __name__ == "__main__":
    test_bfs_chunk_order_is_a_valid_permutation()
    test_assign_clusters_bfs_partitions_every_node_exactly_once()
    test_disconnected_graph_no_edges()
    test_scale_representative_of_dbp1m_rank_shard()
    test_size_guard_skips_nxmetis_entirely()
    print("all gas BFS-chunk scale tests passed")
