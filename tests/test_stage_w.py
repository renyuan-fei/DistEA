"""
test_stage_w.py — Stage W tests: GNNAutoScale-style (GAS)
historical-embedding forward and training epoch for RREA (gas.py).

The load-bearing check is test_multicluster_matches_full_with_exact_history:
a single-cluster pass fills the history tables with the exact per-layer
features, after which a multi-cluster forward must reproduce
RREA.forward() to float tolerance — proving the closure edge sets, the
positional index precomputation, and the history read/write boundaries
are all correct. Any staleness in a real run then comes ONLY from the
GAS trade (out-of-batch neighbors' history lag), not from wiring bugs.

DGL-free, no GPU, no torch.distributed. <15 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gas import GASHistory, build_gas_clusters, gas_forward_batch
from models.rrea import RREA, RREAGraph

CPU = torch.device("cpu")


def _toy_graph(N=40, E=180, R=5, seed=0):
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


def _model_and_h(g, N=40, dim=16, seed=0):
    torch.manual_seed(seed)
    model = RREA(rel_num=g.rel_num, output_dim=dim, num_layers=2,
                 dropout=0.0, attn_heads=2)
    model.eval()
    h = torch.randn(N, dim) * 0.3
    return model, h


def test_single_cluster_matches_full_forward():
    g = _toy_graph()
    model, h = _model_and_h(g)
    with torch.no_grad():
        full = model(g, h)
    clusters = build_gas_clusters(g, 1)
    assert len(clusters) == 1
    hist = GASHistory(40, model.output_dim, model.encoder.depth)
    with torch.no_grad():
        out = gas_forward_batch(model, clusters[0], h, hist, CPU)
    batch = clusters[0].batch
    assert torch.allclose(full[batch], out, atol=1e-5), \
        (full[batch] - out).abs().max().item()
    print("  test_single_cluster_matches_full_forward PASSED")


def test_multicluster_matches_full_with_exact_history():
    g = _toy_graph()
    model, h = _model_and_h(g)
    with torch.no_grad():
        full = model(g, h)

    # Warm the history with exact per-layer features (single-cluster pass)
    hist = GASHistory(40, model.output_dim, model.encoder.depth)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h, hist, CPU)

    for method in ("metis", "bfs"):
        clusters = build_gas_clusters(g, 4, method=method)
        assert sum(len(c.batch) for c in clusters) == 40
        outs = torch.zeros_like(full)
        with torch.no_grad():
            for c in clusters:
                outs[c.batch] = gas_forward_batch(model, c, h, hist, CPU)
        assert torch.allclose(full, outs, atol=1e-4), \
            (method, (full - outs).abs().max().item())
    print("  test_multicluster_matches_full_with_exact_history PASSED")


def test_gradients_flow_to_tables():
    g = _toy_graph()
    model, h0 = _model_and_h(g)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    hist = GASHistory(40, model.output_dim, model.encoder.depth)
    clusters = build_gas_clusters(g, 3)
    out = gas_forward_batch(model, clusters[0], h, hist, CPU)
    out.sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert model.rel_emb.grad is not None and model.rel_emb.grad.abs().sum() > 0
    # history rows are detached: no grad path through pulled features
    assert all(not t.requires_grad for t in hist.tables.values())
    print("  test_gradients_flow_to_tables PASSED")


def test_closure_and_memory_shrink():
    # A sparse ring-with-shortcuts graph: locality is real (unlike the
    # dense random toy graph, where 2-hop closure legitimately covers
    # everything -- avg degree ~13 on 60 nodes). Real KGs sit at avg
    # degree ~4 with METIS locality, which this approximates.
    N, R = 60, 5
    src = torch.arange(N)
    dst = (src + 1) % N
    extra_src = torch.arange(0, N, 6)
    extra_dst = (extra_src + 3) % N
    s = torch.cat([src, extra_src])
    d = torch.cat([dst, extra_dst])
    E = s.size(0)
    edge_index = torch.stack([s, d], dim=1)
    edge_rel = torch.randint(0, 2 * R, (E,))
    eye = torch.arange(N)
    a_hat = torch.sparse_coo_tensor(
        torch.stack([torch.cat([s, eye]), torch.cat([d, eye])]),
        torch.ones(E + N), (N, N)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    dis = deg.pow(-0.5)
    idx = a_hat.indices()
    vals = a_hat.values() * dis[idx[0]] * dis[idx[1]]
    adj = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    rel_prof = torch.sparse_coo_tensor(
        torch.stack([s, edge_rel]), torch.full((E,), 0.2), (N, 2 * R)).coalesce()
    g = RREAGraph(adj, rel_prof, edge_index, edge_rel, torch.ones(E), R)

    clusters = build_gas_clusters(g, 6)
    E_full = g.edge_index.size(0)
    for c in clusters:
        assert c.e_rel.size(0) <= E_full
        assert c.needed.size(0) <= N
        # batch is a subset of needed at the recorded positions
        assert torch.equal(c.needed[c.batch_pos_in_needed], c.batch)
    # On a sparse local graph every cluster's working set must shrink —
    # this is where GAS saves memory.
    assert all(c.e_rel.size(0) < E_full for c in clusters)
    assert all(c.needed.size(0) < N for c in clusters)
    print("  test_closure_and_memory_shrink PASSED")


def test_history_updates_on_push():
    g = _toy_graph()
    model, h = _model_and_h(g)
    hist = GASHistory(40, model.output_dim, model.encoder.depth)
    clusters = build_gas_clusters(g, 4)
    with torch.no_grad():
        gas_forward_batch(model, clusters[0], h, hist, CPU)
    b = clusters[0].batch
    others = np.setdiff1d(np.arange(40), b.numpy())
    for layer in (1, 2):
        t = hist.tables[("ent", layer)]
        assert t[b].abs().sum() > 0          # batch rows written
        assert t[others].abs().sum() == 0    # untouched rows still zero
    print("  test_history_updates_on_push PASSED")




def test_gas_train_epoch_end_to_end():
    """3 GAS epochs through the real DistTrainer config: loss finite and
    decreasing, final-history table populated, single-process gloo group
    for the allreduce path."""
    import torch.distributed as dist

    from dist_trainer import DistTrainer
    from gas import gas_train_epoch

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29614")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    g = _toy_graph()
    model, h0 = _model_and_h(g)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt = torch.optim.Adam(list(model.parameters()) + [h, lw], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1,
                          neg_k=1, margin=6.0, loss_weight=lw)

    clusters = build_gas_clusters(g, 4)
    hist = GASHistory(40, model.output_dim, model.encoder.depth)
    final = torch.zeros(40, model.output_width)
    pairs = torch.tensor([[0, 20], [1, 21], [2, 22], [5, 25], [7, 30]])

    losses = [gas_train_epoch(trainer, clusters, hist, final, h, pairs, 20, CPU)
              for _ in range(3)]
    assert all(np.isfinite(v) for v in losses), losses
    assert losses[-1] < losses[0], losses
    assert final.abs().sum() > 0 and torch.isfinite(final).all()

    # nhsm variant runs too
    trainer2 = DistTrainer(model, opt, rank=0, world_size=1,
                           loss_fn="nhsm", loss_weight=lw)
    v = gas_train_epoch(trainer2, clusters, hist, final, h, pairs, 20, CPU)
    assert np.isfinite(v)
    dist.destroy_process_group()
    print("  test_gas_train_epoch_end_to_end PASSED")


if __name__ == "__main__":
    print("Stage W tests:")
    test_single_cluster_matches_full_forward()
    test_multicluster_matches_full_with_exact_history()
    test_gradients_flow_to_tables()
    test_closure_and_memory_shrink()
    test_history_updates_on_push()
    test_gas_train_epoch_end_to_end()
    print("All Stage W tests PASSED.")
