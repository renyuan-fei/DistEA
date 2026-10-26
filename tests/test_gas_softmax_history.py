"""
test_gas_softmax_history.py — --gas_softmax_history
(gas.py GASHistory edge_exp/dst_sum tables, build_gas_clusters(
softmax_history=True), _gas_attention_head's stale-denominator path).

With the denominator history a cluster processes ONLY its batch-src
edges; dst d's softmax denominator is the fresh sum over those edges plus
(stale per-dst total of d minus the stale values of the same edges).

1. Cluster build: each cluster's edges are exactly the edges whose src is
   in its batch (mask all True, no weights), every local edge lands in
   exactly one cluster, needed = batch U dsts, positions consistent.
2. Parity: a single-cluster closure pass is exact AND fills the edge
   tables (every in-edge of every dst is processed, so stale terms
   cancel); after it, the multi-cluster history forward reproduces
   RREA.forward() to float tolerance, twice in a row, and the per-dst
   totals equal the scatter-sum of the edge tables (refresh is a no-op).
3. Gradients flow into the tables; DISTEA_GAS_CHECKPOINT=1 reproduces the
   un-checkpointed outputs, gradients and history tables (the tables are
   written outside the checkpointed region).
4. gas_train_epoch end to end (gloo, world_size 1): finite loss, dst
   totals rebuilt with negligible drift.

Run:  .venv/bin/python tests/test_gas_softmax_history.py   (test 4 binds a
loopback port -- run outside a sandbox that forbids it)
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

from gas import GASHistory, build_gas_clusters, gas_forward_batch, gas_train_epoch  # noqa: E402

CPU = torch.device("cpu")


def _smh_history(g, model) -> GASHistory:
    return GASHistory(
        g.adj_self_norm.size(0), model.output_dim, model.encoder.depth,
        edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads,
    )


def _check_dst_sums(hist: GASHistory) -> None:
    for key, ee in hist.edge_exp.items():
        ref = torch.zeros_like(hist.dst_sum[key]).index_add_(0, hist.edge_dst, ee)
        assert torch.allclose(ref, hist.dst_sum[key], rtol=1e-9, atol=1e-12), key


def test_clusters_are_batch_src_edges_only():
    g = _hub_graph()
    edges = g.edge_index.numpy()
    E = edges.shape[0]
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    seen = np.zeros(E, dtype=np.int64)
    for c in cl:
        gid = c.e_gid.numpy()
        seen[gid] += 1
        b = c.batch.numpy()
        assert np.isin(edges[gid, 0], b).all()
        assert gid.shape[0] == int(np.isin(edges[:, 0], b).sum())
        assert bool(c.e_src_batch_mask.all()) and c.e_w is None
        assert np.array_equal(c.needed.numpy(), np.unique(np.concatenate([b, edges[gid, 1]])))
        assert np.array_equal(c.needed[c.e_dst_pos].numpy(), edges[gid, 1])
        assert np.array_equal(c.needed[c.e_src_pos].numpy(), edges[gid, 0])
        assert np.array_equal(c.batch[c.e_src_batch_pos].numpy(), edges[gid, 0])
    assert (seen == 1).all()
    total = sum(c.e_rel.size(0) for c in cl)
    print(f"  test_clusters_are_batch_src_edges_only PASSED (edges processed per epoch = {total} = 1.00x E)")


def test_parity_after_exact_warmup():
    g = _hub_graph()
    model, h = _model_and_h(g, N=N_HUB)
    with torch.no_grad():
        full = model(g, h)
    hist = _smh_history(g, model)
    warm = build_gas_clusters(g, 1)  # closure mode, one cluster: exact, fills both histories
    with torch.no_grad():
        out_warm = gas_forward_batch(model, warm[0], h, hist, CPU)
    assert torch.allclose(full[warm[0].batch], out_warm, atol=1e-5)
    _check_dst_sums(hist)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    for rep in range(2):
        outs = torch.zeros_like(full)
        with torch.no_grad():
            for c in cl:
                outs[c.batch] = gas_forward_batch(model, c, h, hist, CPU)
        assert torch.allclose(full, outs, atol=1e-4), (rep, (full - outs).abs().max().item())
        _check_dst_sums(hist)
        assert hist.refresh_dst_sums() < 1e-9
    print("  test_parity_after_exact_warmup PASSED")


def test_gradients_and_checkpoint_parity():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    res = {}
    for ck in ("0", "1"):
        os.environ["DISTEA_GAS_CHECKPOINT"] = ck
        hist = _smh_history(g, model)
        warm = build_gas_clusters(g, 1)
        with torch.no_grad():
            gas_forward_batch(model, warm[0], h0, hist, CPU)
        cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
        model.train()
        model.zero_grad(set_to_none=True)
        h = torch.nn.Parameter(h0.clone())
        out = gas_forward_batch(model, cl[1], h, hist, CPU)
        out.sum().backward()
        assert h.grad is not None and h.grad.abs().sum() > 0
        assert model.rel_emb.grad is not None and model.rel_emb.grad.abs().sum() > 0
        assert all(not t.requires_grad for t in hist.edge_exp.values())
        res[ck] = (
            out.detach().clone(), h.grad.clone(), model.rel_emb.grad.clone(),
            {k: v.clone() for k, v in hist.edge_exp.items()},
            {k: v.clone() for k, v in hist.dst_sum.items()},
        )
    os.environ.pop("DISTEA_GAS_CHECKPOINT", None)
    a, b = res["0"], res["1"]
    for x, y in zip(a[:3], b[:3], strict=True):
        assert torch.allclose(x, y, atol=1e-6), (x - y).abs().max().item()
    for k in a[3]:
        assert torch.equal(a[3][k], b[3][k]) and torch.equal(a[4][k], b[4][k]), k
    print("  test_gradients_and_checkpoint_parity PASSED")


def test_train_epoch_end_to_end():
    import torch.distributed as dist

    from dist_trainer import DistTrainer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt = torch.optim.Adam(list(model.parameters()) + [h, lw], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1, neg_k=1, margin=6.0, loss_weight=lw)
    hist = _smh_history(g, model)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    final = torch.zeros(N_HUB, model.output_width)
    pairs = torch.tensor([[1, 31], [2, 32], [5, 35], [7, 40], [9, 44]])
    losses = [gas_train_epoch(trainer, cl, hist, final, h, pairs, 30, CPU) for _ in range(3)]
    assert all(np.isfinite(v) for v in losses), losses
    _check_dst_sums(hist)
    assert hist.refresh_dst_sums() < 1e-9
    print(f"  test_train_epoch_end_to_end PASSED (losses {[round(v, 4) for v in losses]})")


if __name__ == "__main__":
    for fn in (
        test_clusters_are_batch_src_edges_only,
        test_parity_after_exact_warmup,
        test_gradients_and_checkpoint_parity,
        test_train_epoch_end_to_end,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_softmax_history: ALL PASSED")
