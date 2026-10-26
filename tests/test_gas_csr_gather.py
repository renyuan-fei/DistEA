"""
test_gas_csr_gather.py — --gas_csr_gather (gas._incidence_pair,
build_gas_clusters(csr_gather=True), _gas_attention_head(csr_gather=...)).

The per-edge dst/src gathers and the message scatter of an attention head
become spmm's with per-cluster 0/1 incidence matrices; values and gradients
are unchanged, only the backward kernels differ (segmented reduction
instead of a sort).

1. _incidence_pair: M @ X == X[idx] and M.t() @ G == scatter-add of G by
   idx, on random inputs.
2. Clusters carry incidence matrices of the right shapes; to_ keeps them.
3. gas_forward_batch parity: csr_gather clusters vs plain clusters with
   identical histories -> same outputs, same h_init / rel_emb gradients,
   same history pushes (softmax-history mode, hub graph).
4. gas_train_epoch end to end (gloo, world_size 1): finite losses.

Run:  .venv/bin/python tests/test_gas_csr_gather.py   (test 4 binds a loopback
port -- run outside a sandbox that forbids it)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import N_HUB, _hub_graph  # noqa: E402
from test_stage_w import _model_and_h  # noqa: E402

from gas import (  # noqa: E402
    GASHistory,
    _incidence_pair,
    _SpMMFixedT,
    build_gas_clusters,
    gas_forward_batch,
    gas_train_epoch,
)

CPU = torch.device("cpu")


def test_incidence_pair_gather_and_scatter():
    gen = torch.Generator().manual_seed(0)
    n_e, n_rows, dim = 50, 12, 6
    idx = torch.randint(0, n_rows, (n_e,), generator=gen)
    m, m_t = _incidence_pair(torch.arange(n_e), idx, (n_e, n_rows))
    assert m.layout == torch.sparse_csr and tuple(m_t.shape) == (n_rows, n_e)
    x1 = torch.randn(n_rows, dim, generator=gen).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = x1[idx]
    y2 = _SpMMFixedT.apply(x2, m, m_t)
    assert torch.allclose(y1, y2)
    g = torch.randn(n_e, dim, generator=gen)
    y1.backward(g)
    y2.backward(g)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    # scatter: S (n_rows x n_e) @ msg == index_add of msg into rows
    s, s_t = _incidence_pair(idx, torch.arange(n_e), (n_rows, n_e))
    msg = torch.randn(n_e, dim, generator=gen)
    ref = torch.zeros(n_rows, dim).index_add_(0, idx, msg)
    assert torch.allclose(_SpMMFixedT.apply(msg, s, s_t), ref, atol=1e-5)
    print("  test_incidence_pair_gather_and_scatter PASSED")


def test_cluster_matrices():
    g = _hub_graph()
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True, csr_gather=True)
    for c in cl:
        n_e, n_nd, n_b = c.e_rel.numel(), c.needed.numel(), c.batch.numel()
        assert tuple(c.g_dst.shape) == (n_e, n_nd) and tuple(c.g_dst_t.shape) == (n_nd, n_e)
        assert tuple(c.g_src.shape) == (n_e, n_nd) and tuple(c.s_mat.shape) == (n_b, n_e)
        assert tuple(c.s_mat_t.shape) == (n_e, n_b)
        d = c.g_dst.to_dense()
        assert torch.equal(d.argmax(dim=1), c.e_dst_pos) and torch.equal(d.sum(dim=1), torch.ones(n_e))
        sm = c.s_mat.to_dense()
        assert torch.equal(sm.sum(dim=0), c.e_src_batch_mask.float())
        assert torch.equal(sm.argmax(dim=0)[c.e_src_batch_mask], c.e_src_batch_pos)
        c.to_("cpu")
        assert c.g_dst_t.layout == torch.sparse_csr
    plain = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    assert all(p.g_dst is None for p in plain)
    print("  test_cluster_matrices PASSED")


def _warm_hist(g, model, h0):
    hist = GASHistory(N_HUB, model.output_dim, model.encoder.depth,
                      edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h0, hist, CPU)
    return hist


def test_forward_backward_parity():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    outs = []
    for csr in (False, True):
        hist = _warm_hist(g, model, h0)
        cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True, csr_gather=csr)
        h = torch.nn.Parameter(h0.clone())
        model.zero_grad(set_to_none=True)
        out = gas_forward_batch(model, cl[1], h, hist, CPU)
        out.sum().backward()
        outs.append((out.detach().clone(), h.grad.clone(), model.rel_emb.grad.clone(),
                     {k: v.clone() for k, v in hist.tables.items()},
                     {k: v.clone() for k, v in hist.edge_exp.items()}))
    (oa, ha, ra, ta, ea), (ob, hb, rb, tb, eb) = outs
    assert torch.allclose(oa, ob, atol=1e-5), (oa - ob).abs().max()
    assert torch.allclose(ha, hb, atol=1e-5), (ha - hb).abs().max()
    assert torch.allclose(ra, rb, atol=1e-5), (ra - rb).abs().max()
    for k in ta:
        assert torch.allclose(ta[k], tb[k], atol=1e-5), k
    for k in ea:
        assert torch.allclose(ea[k], eb[k], rtol=1e-6, atol=1e-9), k
    assert ha.abs().sum() > 0 and ra.abs().sum() > 0
    print("  test_forward_backward_parity PASSED")


def test_train_epoch_end_to_end():
    import torch.distributed as dist

    from dist_trainer import DistTrainer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29621")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt = torch.optim.Adam(list(model.parameters()) + [h, lw], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1, neg_k=1, margin=6.0, loss_weight=lw)
    hist = GASHistory(N_HUB, model.output_dim, model.encoder.depth,
                      edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True, csr_gather=True)
    final = torch.zeros(N_HUB, model.output_width)
    pairs = torch.tensor([[1, 31], [2, 32], [5, 35], [7, 40], [9, 44]])
    losses = [gas_train_epoch(trainer, cl, hist, final, h, pairs, 30, CPU) for _ in range(3)]
    assert all(np.isfinite(v) for v in losses), losses
    print(f"  test_train_epoch_end_to_end PASSED (losses {[round(v, 4) for v in losses]})")


if __name__ == "__main__":
    for fn in (test_incidence_pair_gather_and_scatter, test_cluster_matrices,
               test_forward_backward_parity, test_train_epoch_end_to_end):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_csr_gather: ALL PASSED")
