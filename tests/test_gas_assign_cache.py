"""
test_gas_assign_cache.py — DISTEA_GAS_FAST_BUILD=1 caches the METIS
cluster assignment of a rank's local graph (gas._assign_clusters_cached):
off -> computed every call; on -> the first call computes and writes, the
second returns the identical array without recomputing; a different edge
set, K or hub threshold is a different key; a corrupt or out-of-range file
is recomputed; only metis_csr is cached, and a METIS failure's BFS
fallback is never written. Also: with the flag, _incidence_pair built on
a device equals the CPU-built matrix (distinct (row, col) pairs).

DGL-free, CPU, < 10 s.
Run:  .venv/bin/python tests/test_gas_assign_cache.py
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gas  # noqa: E402

CALLS = []


def _stub(N, edges, n_clusters, method, hub_degree=0):
    CALLS.append((N, n_clusters, method, hub_degree))
    return (np.arange(N, dtype=np.int64) * 7 + int(edges.sum())) % n_clusters


def _metis_stub(N, edges, n_clusters, hub_degree=0):
    return _stub(N, edges, n_clusters, "metis_csr", hub_degree)


def _metis_fail(N, edges, n_clusters, hub_degree=0):
    CALLS.append("metis_fail")
    raise RuntimeError("simulated METIS failure")


def _run(edges, N=500, K=8, method="metis_csr", hub=1000):
    return gas._assign_clusters_cached(N, edges, K, method, hub)


def test_cache_hits_and_keys():
    real, real_metis = gas._assign_clusters, gas._metis_csr_assign
    gas._assign_clusters, gas._metis_csr_assign = _stub, _metis_stub
    rng = np.random.default_rng(0)
    e1 = rng.integers(0, 500, (3000, 2))
    e2 = e1.copy()
    e2[0, 0] = (e2[0, 0] + 1) % 500
    try:
        with tempfile.TemporaryDirectory() as cdir:
            os.environ["DISTEA_DATA_CACHE_DIR"] = cdir
            CALLS.clear()
            a0 = _run(e1)
            a0b = _run(e1)
            assert len(CALLS) == 2 and not glob.glob(os.path.join(cdir, "*.npy")), "cache off: no cache"
            os.environ["DISTEA_GAS_FAST_BUILD"] = "1"
            CALLS.clear()
            a1 = _run(e1)
            assert len(CALLS) == 1 and len(glob.glob(os.path.join(cdir, "gas_assign_*.npy"))) == 1
            a2 = _run(e1)
            assert len(CALLS) == 1, "second call must hit the cache"
            assert a2.dtype == np.int64 and np.array_equal(a2, a1) and np.array_equal(a1, a0)
            assert np.array_equal(a0, a0b)
            for kw in ({"K": 4}, {"hub": 10}):
                _run(e1, **kw)
            _run(e2)
            assert len(CALLS) == 4, "K / hub / edges are part of the key"
            for m in ("bfs", "bfs", "metis", "metis"):
                _run(e1, method=m)
            assert len(CALLS) == 8, "only metis_csr is cached"
            gas._metis_csr_assign = _metis_fail
            n_files = len(glob.glob(os.path.join(cdir, "gas_assign_*.npy")))
            CALLS.clear()
            fb = _run(e1, K=5)
            assert CALLS[0] == "metis_fail" and CALLS[1][2] == "bfs", CALLS
            assert len(glob.glob(os.path.join(cdir, "gas_assign_*.npy"))) == n_files, "a fallback is never cached"
            assert np.array_equal(fb, _stub(500, e1, 5, "bfs", 1000))
            gas._metis_csr_assign = _metis_stub
            CALLS.clear()
            path = glob.glob(os.path.join(cdir, "gas_assign_*.npy"))
            for p in path:
                with open(p, "wb") as f:
                    f.write(b"garbage")
            a3 = _run(e1)
            assert len(CALLS) == 1 and np.array_equal(a3, a1), "a corrupt file is recomputed"
            np.save(os.path.join(cdir, os.path.basename(path[0])), np.full(500, 99, dtype=np.int16))
            CALLS.clear()
            for p in glob.glob(os.path.join(cdir, "gas_assign_*.npy")):
                np.save(p, np.full(500, 99, dtype=np.int16))
            a4 = _run(e1)
            assert len(CALLS) == 1 and np.array_equal(a4, a1), "an out-of-range file is recomputed"
    finally:
        gas._assign_clusters, gas._metis_csr_assign = real, real_metis
        os.environ.pop("DISTEA_GAS_FAST_BUILD", None)
        os.environ.pop("DISTEA_DATA_CACHE_DIR", None)
    print("  test_cache_hits_and_keys PASSED")


def test_incidence_pair_device_equal():
    rng = np.random.default_rng(1)
    n_e, n_rows = 4000, 900
    rows = torch.arange(n_e)
    cols = torch.from_numpy(rng.integers(0, n_rows, n_e))
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ref, ref_t = gas._incidence_pair(rows, cols, (n_e, n_rows))
    os.environ["DISTEA_GAS_FAST_BUILD"] = "1"
    for d in devs:
        m, m_t = gas._incidence_pair(rows, cols, (n_e, n_rows), device=d)
        assert torch.equal(m.to_dense().cpu(), ref.to_dense()) and torch.equal(m_t.to_dense().cpu(), ref_t.to_dense()), d
        sel = torch.from_numpy(np.sort(rng.choice(n_e, 1500, replace=False)))
        b = torch.from_numpy(rng.integers(0, 300, 1500))
        s0, s0t = gas._incidence_pair(b, sel, (300, n_e))
        s1, s1t = gas._incidence_pair(b, sel, (300, n_e), device=d)
        assert torch.equal(s1.to_dense().cpu(), s0.to_dense()) and torch.equal(s1t.to_dense().cpu(), s0t.to_dense()), d
    os.environ.pop("DISTEA_GAS_FAST_BUILD", None)
    print(f"  test_incidence_pair_device_equal PASSED ({', '.join(devs)})")


if __name__ == "__main__":
    for fn in (test_cache_hits_and_keys, test_incidence_pair_device_equal):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_assign_cache: ALL PASSED")
