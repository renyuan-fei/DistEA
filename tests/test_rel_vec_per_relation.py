"""
test_rel_vec_per_relation.py — per-relation normalisation of
the relation vectors (gas._gas_attention_head(e_val_pos=True),
rrea._head_messages): normalize(v * r) == normalize(r) for scalar v > 0, so
with all-positive edge weights the (E, dim) multiply + norm + divide per
head call collapses to one (n_rel, dim) normalisation plus a gather.

1. Algebra: for random positive v the two formulations agree to fp32
   rounding, forward and gradient w.r.t. the relation table; for a
   non-positive v they differ (so the guard matters).
2. GAS head: outputs and gradients (h_init / rel_emb) with e_val_pos=True
   equal the per-edge path on the hub graph (whose e_val are all 1/count).
3. Clusters record e_val_pos; a cluster with a zero weight does not.
4. RREA.forward (full graph) is unchanged against the explicit per-edge
   formula on the hub graph.

Run:  .venv/bin/python tests/test_rel_vec_per_relation.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import N_HUB, _hub_graph  # noqa: E402
from test_stage_w import _model_and_h  # noqa: E402

from gas import GASHistory, _gas_attention_head, build_gas_clusters, gas_forward_batch  # noqa: E402

CPU = torch.device("cpu")


def test_algebra():
    gen = torch.Generator().manual_seed(0)
    rel = torch.randn(7, 5, generator=gen).requires_grad_(True)
    rel2 = rel.detach().clone().requires_grad_(True)
    idx = torch.randint(0, 7, (40,), generator=gen)
    v = torch.rand(40, generator=gen) * 3 + 0.01
    a = F.normalize(rel[idx] * v.unsqueeze(1), p=2, dim=1)
    b = F.embedding(idx, F.normalize(rel2, p=2, dim=1))
    assert torch.allclose(a, b, atol=1e-6)
    g = torch.randn(40, 5, generator=gen)
    a.backward(g)
    b.backward(g)
    assert torch.allclose(rel.grad, rel2.grad, atol=1e-5), (rel.grad - rel2.grad).abs().max()
    v_bad = v.clone()
    v_bad[3] = 0.0
    c = F.normalize(rel.detach()[idx] * v_bad.unsqueeze(1), p=2, dim=1)
    assert not torch.allclose(c, b.detach(), atol=1e-6)
    print("  test_algebra PASSED")


def test_gas_head_parity():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    assert all(c.e_val_pos for c in cl)
    c = cl[1]
    hist = GASHistory(N_HUB, model.output_dim, model.encoder.depth,
                      edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h0, hist, CPU)
    enc = model.encoder
    outs = []
    for pos in (False, True):
        h = torch.nn.Parameter(h0.clone())
        rel = torch.nn.Parameter(model.rel_emb.detach().clone())
        prev = torch.sparse.mm(c.adj_rows, h)
        key = ("ent", 0, 0)
        e_stale, dst_sum = hist.pull_denominators(key, c.e_gid, c.needed, CPU)
        out, _, _ = _gas_attention_head(
            prev, rel, enc._kernel(0, 0), c.e_rel, c.e_val, c.e_dst_pos, c.e_src_pos,
            c.e_src_batch_mask, c.e_src_batch_pos, c.needed.numel(), c.batch.numel(),
            enc.dim, None, e_stale, dst_sum, e_val_pos=pos)
        out.sum().backward()
        outs.append((out.detach().clone(), h.grad.clone(), rel.grad.clone()))
    (oa, ha, ra), (ob, hb, rb) = outs
    assert torch.allclose(oa, ob, atol=1e-6), (oa - ob).abs().max()
    assert torch.allclose(ha, hb, atol=1e-6), (ha - hb).abs().max()
    assert torch.allclose(ra, rb, atol=1e-5), (ra - rb).abs().max()
    assert ra.abs().sum() > 0
    print("  test_gas_head_parity PASSED")


def test_cluster_flag_guards_zero_weight():
    g = _hub_graph()
    g.edge_rel_val = g.edge_rel_val.clone()
    g.edge_rel_val[0] = 0.0
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    src0 = int(g.edge_index[0, 0])
    flags = {bool(torch.isin(torch.tensor(src0), c.batch)): c.e_val_pos for c in cl}
    assert flags[True] is False, "the cluster holding the zero-weight edge must keep the per-edge path"
    assert any(c.e_val_pos for c in cl if not bool(torch.isin(torch.tensor(src0), c.batch)))
    print("  test_cluster_flag_guards_zero_weight PASSED")


def test_full_forward_unchanged():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.eval()
    with torch.no_grad():
        out = model(g, h0)
        # explicit per-edge formula for layer 0 of the ent channel, all heads
        from models.rrea import _sparse_softmax_by_dst
        enc = model.encoder
        rel_vec = F.normalize(model.rel_emb[g.edge_rel] * g.edge_rel_val.unsqueeze(1), p=2, dim=1)
        ent = torch.sparse.mm(g.adj_self_norm, h0)
        src, dst = g.edge_index[:, 0], g.edge_index[:, 1]
        neighs, selfs = ent[dst], ent[src]
        bias = (neighs * rel_vec).sum(dim=1, keepdim=True) * rel_vec
        nr = neighs - 2 * bias
        att_in = torch.cat([selfs, nr, rel_vec], dim=1)
        heads = []
        for head in range(enc.attn_heads):
            logits = (att_in @ enc._kernel(0, head)).squeeze(-1)
            att = _sparse_softmax_by_dst(logits, dst, N_HUB)
            heads.append(torch.zeros(N_HUB, enc.dim).index_add_(0, src, nr * att.unsqueeze(1)))
        ref_layer1 = torch.relu(torch.stack(heads, dim=0).mean(dim=0))
    assert torch.allclose(out[:, enc.dim:2 * enc.dim], ref_layer1, atol=1e-5), \
        (out[:, enc.dim:2 * enc.dim] - ref_layer1).abs().max()
    print("  test_full_forward_unchanged PASSED")


if __name__ == "__main__":
    for fn in (test_algebra, test_gas_head_parity, test_cluster_flag_guards_zero_weight,
               test_full_forward_unchanged):
        print(f"[{fn.__name__}]")
        fn()
    print("test_rel_vec_per_relation: ALL PASSED")
