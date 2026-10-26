"""
test_gas_metis_fallback.py — gas.py::_assign_clusters must fall back to the
BFS-chunk heuristic on ANY nxmetis.partition failure, not just ImportError.

Background: the vendored `modules/networkx-metis`'s own exception-conversion
decorator (`nxmetis/__init__.py::_convert_exceptions`) has a Python 2->3 bug
-- `except catch_types as e: exc = e` binds `exc` to the exception INSTANCE,
then `exc[1]` (assuming `exc` is sys.exc_info()'s 3-tuple) crashes with
"'MetisError' object is not subscriptable" whenever nxmetis.partition's own
METIS call raises MetisError/ValueError/TypeError directly -- masking
whatever real METIS failure triggered it. It occurs on large local+halo
graphs (DBP1M at N=16). If `_assign_clusters` caught only ImportError, any
such failure would crash the whole run instead of degrading to the
BFS-chunk fallback.

Cannot be tested with the real nxmetis (not in the DGL-free test
environment's dependencies, and the exact METIS-internal trigger condition
at DBP1M scale isn't reproducible from a small toy graph) -- this test
injects a fake `nxmetis` module into sys.modules whose `partition()` raises
the exact class of exception this bug produces, and checks
`_assign_clusters` recovers instead of propagating it.

DGL-free, no GPU, no torch.distributed. <5 sec.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gas import _assign_clusters, build_gas_clusters  # noqa: E402
from models.rrea import RREAGraph  # noqa: E402


def _install_fake_nxmetis(exc: Exception):
    """Makes `import nxmetis` succeed and `nxmetis.partition(...)` raise
    `exc`, mimicking the real library reaching its own broken
    exception-conversion path. Returns a restore function."""
    fake = types.ModuleType("nxmetis")

    def _partition(G, n_clusters, recursive=True):
        raise exc

    fake.partition = _partition
    had_prior = "nxmetis" in sys.modules
    prior = sys.modules.get("nxmetis")
    sys.modules["nxmetis"] = fake

    def _restore():
        if had_prior:
            sys.modules["nxmetis"] = prior
        else:
            del sys.modules["nxmetis"]

    return _restore


def _toy_edges(N=60, E=260, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    keep = src != dst
    return np.stack([src[keep], dst[keep]], axis=1)


def test_assign_clusters_falls_back_on_typeerror_from_broken_reraise():
    """The exact failure mode: nxmetis.partition raises TypeError (the
    library's own botched reraise), _assign_clusters must not propagate
    it and must still return a valid, complete assignment."""
    N, K = 60, 8
    edges = _toy_edges(N)
    restore = _install_fake_nxmetis(
        TypeError("'MetisError' object is not subscriptable")
    )
    try:
        assign = _assign_clusters(N, edges, K, "metis")
    finally:
        restore()
    assert assign.shape == (N,)
    assert assign.min() >= 0 and assign.max() < K
    # every node assigned (BFS fallback covers all N, no node left at a
    # sentinel/garbage value)
    assert np.unique(assign).size >= 1
    print("  test_assign_clusters_falls_back_on_typeerror_from_broken_reraise PASSED")


def test_assign_clusters_falls_back_on_genuine_metis_error():
    """A real MetisError (if the reraise bug is ever fixed upstream and
    METIS's own failure surfaces cleanly) must ALSO fall back, not just
    the TypeError proxy -- the except clause catches Exception broadly."""
    N, K = 60, 8
    edges = _toy_edges(N, seed=1)

    class FakeMetisError(Exception):
        pass

    restore = _install_fake_nxmetis(FakeMetisError("METIS_ERROR: bad input"))
    try:
        assign = _assign_clusters(N, edges, K, "metis")
    finally:
        restore()
    assert assign.shape == (N,)
    assert assign.min() >= 0 and assign.max() < K
    print("  test_assign_clusters_falls_back_on_genuine_metis_error PASSED")


def _toy_graph(N=50, E=220, R=4, seed=2):
    """Same construction as test_stage_w.py's _toy_graph (kept local so
    this file has no cross-test-file dependency)."""
    torch.manual_seed(seed)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    keep = src != dst
    src, dst = src[keep], dst[keep]
    E = src.size(0)
    edge_index = torch.stack([src, dst], dim=1)
    edge_rel = torch.randint(0, 2 * R, (E,))
    edge_rel_val = torch.ones(E)
    eye = torch.arange(N)
    a_hat = torch.sparse_coo_tensor(
        torch.stack([torch.cat([src, eye]), torch.cat([dst, eye])]),
        torch.ones(E + N), (N, N)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    dis = deg.pow(-0.5)
    idx = a_hat.indices()
    vals = a_hat.values() * dis[idx[0]] * dis[idx[1]]
    adj = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    rel_prof = torch.sparse_coo_tensor(
        torch.stack([src, edge_rel]), torch.full((E,), 0.2), (N, 2 * R)).coalesce()
    return RREAGraph(adj, rel_prof, edge_index, edge_rel, edge_rel_val, R)


def test_build_gas_clusters_end_to_end_survives_metis_failure():
    """One level up: build_gas_clusters (what framework.py actually calls)
    must produce usable clusters covering every node, not crash, when the
    metis path fails underneath it."""
    N, K = 50, 6
    g = _toy_graph(N=N)
    restore = _install_fake_nxmetis(TypeError("'MetisError' object is not subscriptable"))
    try:
        clusters = build_gas_clusters(g, K)
    finally:
        restore()
    covered = torch.cat([c.batch for c in clusters]).sort().values
    assert torch.equal(covered, torch.arange(N)), "every node must land in exactly one batch"
    print("  test_build_gas_clusters_end_to_end_survives_metis_failure PASSED")


if __name__ == "__main__":
    print("GAS METIS-fallback tests:")
    test_assign_clusters_falls_back_on_typeerror_from_broken_reraise()
    test_assign_clusters_falls_back_on_genuine_metis_error()
    test_build_gas_clusters_end_to_end_survives_metis_failure()
    print("All GAS METIS-fallback tests PASSED.")
