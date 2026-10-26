"""
test_gas_row_slice_index.py — gas._RowSliceIndex: O(slice)
row slicing of a coalesced COO matrix via precomputed row ranges, replacing
the per-call np.isin over every stored entry in build_gas_clusters.

1. Slices equal the reference (mask-based) implementation on random sparse
   matrices for sorted row subsets, including empty rows, the empty subset
   and the full row set; results are coalesced.
2. build_gas_clusters output is unchanged (dense equality of adj_rows /
   relprof_rows against the reference slicer) with and without csr_spmm.

Run:  .venv/bin/python tests/test_gas_row_slice_index.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import _hub_graph  # noqa: E402

from gas import _RowSliceIndex, build_gas_clusters  # noqa: E402


def _reference(mat: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    idx, vals = mat.indices(), mat.values()
    row_np = idx[0].cpu().numpy()
    kept = np.nonzero(np.isin(row_np, rows))[0]
    new_row = np.searchsorted(rows, row_np[kept])
    new_idx = torch.stack([torch.from_numpy(new_row.astype(np.int64)),
                           idx[1][torch.from_numpy(kept)].cpu()])
    return torch.sparse_coo_tensor(new_idx, vals[torch.from_numpy(kept)].cpu(),
                                   (len(rows), mat.size(1))).coalesce()


def test_matches_reference():
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    for n_rows, n_cols, density in ((30, 17, 0.2), (200, 50, 0.05), (64, 64, 0.0)):
        dense = torch.rand(n_rows, n_cols, generator=gen)
        dense[torch.rand(n_rows, n_cols, generator=gen) > density] = 0.0
        if density == 0.0:
            dense[3, 5] = 1.0  # a single entry, every other row empty
        mat = dense.to_sparse().coalesce()
        index = _RowSliceIndex(mat)
        subsets = [
            np.sort(rng.choice(n_rows, size=max(1, n_rows // 3), replace=False)),
            np.arange(n_rows), np.array([0]), np.array([n_rows - 1]),
            np.array([], dtype=np.int64),
        ]
        for rows in subsets:
            got, ref = index.slice(rows), _reference(mat, rows)
            assert got.is_coalesced()
            assert tuple(got.shape) == tuple(ref.shape)
            assert torch.equal(got.to_dense(), ref.to_dense()), (n_rows, rows[:5])
            assert torch.equal(got.indices(), ref.indices())
    print("  test_matches_reference PASSED")


def test_build_unchanged():
    g = _hub_graph()
    for csr in (False, True):
        cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True, csr_spmm=csr)
        for c in cl:
            ref_adj = _reference(g.adj_self_norm, c.needed.numpy())
            ref_rel = _reference(g.rel_profile_norm, c.needed.numpy())
            assert torch.allclose(c.adj_rows.to_dense(), ref_adj.to_dense())
            assert torch.allclose(c.relprof_rows.to_dense(), ref_rel.to_dense())
            if csr:
                assert torch.allclose(c.adj_rows_t.to_dense(), ref_adj.to_dense().t())
    print("  test_build_unchanged PASSED")


if __name__ == "__main__":
    for fn in (test_matches_reference, test_build_unchanged):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_row_slice_index: ALL PASSED")
