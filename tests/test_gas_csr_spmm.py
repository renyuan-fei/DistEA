"""
test_gas_csr_spmm.py — --gas_csr_spmm (gas.py _SpMMFixedT,
_csr_pair, build_gas_clusters(csr_spmm=True), _layer0_mm).

The layer-0 products adj_rows @ h_init / relprof_rows @ rel_emb keep their
values; only the backward changes from torch.sparse.mm's per-call
transpose + coalesce to one spmm with a CSR transpose built once.

1. _SpMMFixedT == torch.sparse.mm forward and backward on random sparse
   matrices (values and the dense operand's gradient).
2. build_gas_clusters(csr_spmm=True): CSR slices + CSR transposes of the
   right shapes, same batches/edges as the COO build; GASCluster.to_ keeps
   them.
3. gas_forward_batch parity: csr clusters vs coo clusters with identical
   histories give the same outputs and the same h_init / rel_emb
   gradients.

Run:  .venv/bin/python tests/test_gas_csr_spmm.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import N_HUB, _hub_graph  # noqa: E402
from test_stage_w import _model_and_h  # noqa: E402

from gas import (  # noqa: E402
    GASHistory,
    _csr_pair,
    _SpMMFixedT,
    build_gas_clusters,
    gas_forward_batch,
)

CPU = torch.device("cpu")


def test_fixed_transpose_matches_sparse_mm():
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for rows, cols, dim in ((7, 11, 4), (50, 30, 8), (3, 40, 2)):
        dense = torch.rand(rows, cols, generator=gen)
        dense[torch.rand(rows, cols, generator=gen) > 0.3] = 0.0
        coo = dense.to_sparse().coalesce()
        csr, csr_t = _csr_pair(coo)
        assert csr.layout == torch.sparse_csr and csr_t.layout == torch.sparse_csr
        assert tuple(csr_t.shape) == (cols, rows)
        x1 = torch.randn(cols, dim, generator=gen).requires_grad_(True)
        x2 = x1.detach().clone().requires_grad_(True)
        y1 = torch.sparse.mm(coo, x1)
        y2 = _SpMMFixedT.apply(x2, csr, csr_t)
        g = torch.randn(rows, dim, generator=gen)
        y1.backward(g)
        y2.backward(g)
        assert torch.allclose(y1, y2, atol=1e-6)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
        worst = max(worst, (y1 - y2).abs().max().item(), (x1.grad - x2.grad).abs().max().item())
    print(f"  test_fixed_transpose_matches_sparse_mm PASSED (max |diff| {worst:.2e})")


def test_build_csr_clusters():
    g = _hub_graph()
    a = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    b = build_gas_clusters(g, 4, method="bfs", softmax_history=True, csr_spmm=True)
    for ca, cb in zip(a, b, strict=True):
        assert torch.equal(ca.batch, cb.batch) and torch.equal(ca.e_gid, cb.e_gid)
        assert ca.adj_rows_t is None and ca.adj_rows.layout == torch.sparse_coo
        assert cb.adj_rows.layout == torch.sparse_csr and cb.adj_rows_t.layout == torch.sparse_csr
        assert tuple(cb.adj_rows_t.shape) == (ca.adj_rows.size(1), ca.adj_rows.size(0))
        assert tuple(cb.relprof_rows_t.shape) == (ca.relprof_rows.size(1), ca.relprof_rows.size(0))
        assert torch.allclose(ca.adj_rows.to_dense(), cb.adj_rows.to_dense())
        assert torch.allclose(ca.adj_rows.to_dense().t(), cb.adj_rows_t.to_dense())
        cb.to_("cpu")
        assert cb.adj_rows_t.layout == torch.sparse_csr
    print("  test_build_csr_clusters PASSED")


def test_forward_backward_parity_with_coo():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    outs = []
    for csr in (False, True):
        hist = GASHistory(N_HUB, model.output_dim, model.encoder.depth,
                          edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads)
        warm = build_gas_clusters(g, 1)
        with torch.no_grad():
            gas_forward_batch(model, warm[0], h0, hist, CPU)
        cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True, csr_spmm=csr)
        h = torch.nn.Parameter(h0.clone())
        model.zero_grad(set_to_none=True)
        out = gas_forward_batch(model, cl[2], h, hist, CPU)
        out.sum().backward()
        outs.append((out.detach().clone(), h.grad.clone(), model.rel_emb.grad.clone()))
    (oa, ha, ra), (ob, hb, rb) = outs
    assert torch.allclose(oa, ob, atol=1e-6), (oa - ob).abs().max()
    assert torch.allclose(ha, hb, atol=1e-6), (ha - hb).abs().max()
    assert torch.allclose(ra, rb, atol=1e-6), (ra - rb).abs().max()
    assert ha.abs().sum() > 0 and ra.abs().sum() > 0
    print("  test_forward_backward_parity_with_coo PASSED")


if __name__ == "__main__":
    for fn in (test_fixed_transpose_matches_sparse_mm, test_build_csr_clusters,
               test_forward_backward_parity_with_coo):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_csr_spmm: ALL PASSED")
