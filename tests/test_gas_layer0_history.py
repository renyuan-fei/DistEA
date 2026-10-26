"""
test_gas_layer0_history.py — --gas_layer0_history
(gas.py GASHistory(layer0=True) + warm_layer0, build_gas_clusters(
layer0_history=True), gas_forward_batch's batch-only layer-0 spmm).

With the layer-0 history a cluster's adj_rows / relprof_rows cover its
BATCH rows only; the other needed rows' layer-0 features come from the
history tables (stale <= 1 epoch), the fresh batch rows are pushed back.

1. Cluster build: layer-0 matrices have len(batch) rows (not len(needed)),
   l0_batch_only is set; everything else identical to the softmax-history
   build (same batches, edges, positions).
2. Parity at the fixed point: with the layer-0 tables warmed by the exact
   whole-graph rows (warm_layer0) and the other tables warmed identically,
   every cluster's forward equals the exact-layer-0 forward (float
   tolerance), twice in a row.
3. Gradients: h_init.grad is non-zero and supported only on the batch's
   own in-neighbourhood (rows read by the batch-only spmm) -- the stale
   neighbour rows carry no gradient, by construction.
4. A cluster built with layer0_history against a history without layer-0
   tables is rejected.
5. gas_train_epoch end to end (gloo, world_size 1): finite losses.

Run:  .venv/bin/python tests/test_gas_layer0_history.py   (test 5 binds a
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


def _history(g, model, layer0: bool) -> GASHistory:
    return GASHistory(
        g.adj_self_norm.size(0), model.output_dim, model.encoder.depth,
        edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads,
        layer0=layer0,
    )


def _warm(g, model, h0, hist: GASHistory) -> None:
    """Exact warm-up: one single-cluster closure pass fills the layer>=1 and
    edge tables; warm_layer0 fills the layer-0 tables when present."""
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h0, hist, CPU)
    hist.warm_layer0(g, h0, model.rel_emb, CPU)


def test_clusters_slice_batch_rows_only():
    g = _hub_graph()
    a = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    b = build_gas_clusters(g, 4, method="bfs", softmax_history=True, layer0_history=True)
    assert len(a) == len(b) == 4
    for ca, cb in zip(a, b, strict=True):
        assert torch.equal(ca.batch, cb.batch) and torch.equal(ca.needed, cb.needed)
        assert torch.equal(ca.e_gid, cb.e_gid) and torch.equal(ca.e_dst_pos, cb.e_dst_pos)
        assert ca.adj_rows.size(0) == ca.needed.numel() and not ca.l0_batch_only
        assert cb.adj_rows.size(0) == cb.batch.numel() and cb.l0_batch_only
        assert cb.relprof_rows.size(0) == cb.batch.numel()
        assert cb.needed.numel() > cb.batch.numel()  # the slice really shrank
        # The batch rows of the exact slice are the batch-only slice.
        pos = cb.batch_pos_in_needed
        assert torch.allclose(ca.adj_rows.to_dense()[pos], cb.adj_rows.to_dense())
        assert torch.allclose(ca.relprof_rows.to_dense()[pos], cb.relprof_rows.to_dense())
    print("  test_clusters_slice_batch_rows_only PASSED")


def test_parity_at_fixed_point():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.eval()
    ha, hb = _history(g, model, layer0=False), _history(g, model, layer0=True)
    _warm(g, model, h0, ha)
    _warm(g, model, h0, hb)
    assert ("ent", 0) in hb.tables and ("ent", 0) not in ha.tables
    a = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    b = build_gas_clusters(g, 4, method="bfs", softmax_history=True, layer0_history=True)
    worst = 0.0
    with torch.no_grad():
        for _ in range(2):
            for ca, cb in zip(a, b, strict=True):
                oa = gas_forward_batch(model, ca, h0, ha, CPU)
                ob = gas_forward_batch(model, cb, h0, hb, CPU)
                worst = max(worst, (oa - ob).abs().max().item())
                assert torch.allclose(oa, ob, atol=1e-5), worst
    # The layer-0 tables still hold the exact rows after the pushes.
    f_ent = torch.sparse.mm(g.adj_self_norm, h0)
    assert torch.allclose(hb.tables[("ent", 0)], f_ent, atol=1e-6)
    print(f"  test_parity_at_fixed_point PASSED (max |diff| {worst:.2e})")


def test_gradients_only_through_batch_in_edges():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    hb = _history(g, model, layer0=True)
    _warm(g, model, h0, hb)
    b = build_gas_clusters(g, 4, method="bfs", softmax_history=True, layer0_history=True)
    c = b[1]
    h = torch.nn.Parameter(h0.clone())
    model.zero_grad(set_to_none=True)
    out = gas_forward_batch(model, c, h, hb, CPU)
    out.sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert model.rel_emb.grad is not None and model.rel_emb.grad.abs().sum() > 0
    got = set(torch.nonzero(h.grad.abs().sum(1) > 0).flatten().tolist())
    idx = g.adj_self_norm.coalesce().indices()
    in_batch = torch.isin(idx[0], c.batch)
    allowed = set(idx[1][in_batch].tolist()) | set(c.batch.tolist())
    assert got <= allowed, sorted(got - allowed)[:10]
    # The exact path reaches further (in-neighbours of every needed row).
    ha = _history(g, model, layer0=False)
    _warm(g, model, h0, ha)
    a = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    h2 = torch.nn.Parameter(h0.clone())
    model.zero_grad(set_to_none=True)
    gas_forward_batch(model, a[1], h2, ha, CPU).sum().backward()
    got_exact = set(torch.nonzero(h2.grad.abs().sum(1) > 0).flatten().tolist())
    assert got <= got_exact and len(got_exact) > len(got), (len(got), len(got_exact))
    print(f"  test_gradients_only_through_batch_in_edges PASSED "
          f"(grad rows {len(got)} vs exact path {len(got_exact)})")


def test_layer0_cluster_needs_layer0_history():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    ha = _history(g, model, layer0=False)
    b = build_gas_clusters(g, 4, method="bfs", softmax_history=True, layer0_history=True)
    try:
        with torch.no_grad():
            gas_forward_batch(model, b[0], h0, ha, CPU)
    except ValueError as exc:
        assert "layer0" in str(exc)
        print("  test_layer0_cluster_needs_layer0_history PASSED")
        return
    raise AssertionError("expected ValueError")


def test_train_epoch_end_to_end():
    import torch.distributed as dist

    from dist_trainer import DistTrainer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29619")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt = torch.optim.Adam(list(model.parameters()) + [h, lw], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1, neg_k=1, margin=6.0, loss_weight=lw)
    hist = _history(g, model, layer0=True)
    hist.warm_layer0(g, h, model.rel_emb, CPU)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True, layer0_history=True)
    final = torch.zeros(N_HUB, model.output_width)
    pairs = torch.tensor([[1, 31], [2, 32], [5, 35], [7, 40], [9, 44]])
    losses = [gas_train_epoch(trainer, cl, hist, final, h, pairs, 30, CPU) for _ in range(3)]
    assert all(np.isfinite(v) for v in losses), losses
    # Layer-0 tables track the moving h: every row was refreshed this epoch.
    f_ent = torch.sparse.mm(g.adj_self_norm, h.detach())
    stale = (hist.tables[("ent", 0)] - f_ent).abs().max().item()
    assert stale < 1.0, stale  # bounded by one epoch of Adam steps at lr 1e-2
    print(f"  test_train_epoch_end_to_end PASSED (losses {[round(v, 4) for v in losses]}, "
          f"layer-0 staleness {stale:.2e})")


if __name__ == "__main__":
    for fn in (
        test_clusters_slice_batch_rows_only,
        test_parity_at_fixed_point,
        test_gradients_only_through_batch_in_edges,
        test_layer0_cluster_needs_layer0_history,
        test_train_epoch_end_to_end,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_layer0_history: ALL PASSED")
