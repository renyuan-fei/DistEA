"""
test_attn_softmax_axis.py — --attn_softmax {dst,src}
and --rrea_input_relu.

1. Default (dst) is bit-identical to a model built without the arguments.
2. src grouping: the attention mass summed per SELF node equals 1 for every
   node with out-edges (the original layer.py semantics); under dst grouping
   it does not.
3. GAS parity under src grouping: after an exact warm-up, gas_forward_batch
   on softmax-history clusters reproduces RREA.forward's rows for every
   cluster (the batch-src edge set holds each batch node's complete group).
4. input_relu: the depth-0 slice of each channel is the relu of the
   pre-aggregated features; off = signed features.
5. Edge-chunked forward with src grouping equals the unchunked one.

Run:  .venv/bin/python tests/test_attn_softmax_axis.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_softmax_history import N_HUB, _hub_graph, _model_and_h, _smh_history  # noqa: E402

import models.rrea as rrea_mod  # noqa: E402
from gas import build_gas_clusters, gas_forward_batch  # noqa: E402
from models.rrea import RREA  # noqa: E402

CPU = torch.device("cpu")


def _clone(model, **kw):
    m = RREA(rel_num=model.rel_num, output_dim=model.output_dim, num_layers=model.encoder.depth,
             attn_heads=model.encoder.attn_heads, **kw)
    m.load_state_dict(model.state_dict())
    m.eval()
    return m


def test_default_identical():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.eval()
    m_dst = _clone(model, attn_softmax="dst")
    with torch.no_grad():
        assert torch.equal(model(g, h0), m_dst(g, h0))
    print("  [ok] attn_softmax=dst is bit-identical to the default")


def test_src_mass_sums_to_one():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    seen = {}
    real = rrea_mod._sparse_softmax_by_dst

    def rec(logits, key, N):
        att = real(logits, key, N)
        seen.setdefault("calls", []).append((key.clone(), att.clone(), N))
        return att

    rrea_mod._sparse_softmax_by_dst = rec
    try:
        for axis in ("src", "dst"):
            seen.clear()
            m = _clone(model, attn_softmax=axis)
            with torch.no_grad():
                m(g, h0)
            key, att, N = seen["calls"][0]
            src = g.edge_index[:, 0]
            mass = torch.zeros(N).index_add_(0, src, att)       # mass landing on each self node
            has_out = torch.zeros(N, dtype=torch.bool)
            has_out[src] = True
            if axis == "src":
                assert torch.allclose(mass[has_out], torch.ones(int(has_out.sum())), atol=1e-5)
            else:
                assert not torch.allclose(mass[has_out], torch.ones(int(has_out.sum())), atol=1e-2)
    finally:
        rrea_mod._sparse_softmax_by_dst = real
    print("  [ok] src grouping: unit attention mass per self node; dst grouping does not")


def test_gas_parity_src():
    g = _hub_graph()
    base, h0 = _model_and_h(g, N=N_HUB)
    model = _clone(base, attn_softmax="src")
    hist = _smh_history(g, model)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h0, hist, CPU)
        full = model(g, h0)
        for c in build_gas_clusters(g, 4, method="bfs", softmax_history=True):
            out = gas_forward_batch(model, c, h0, hist, CPU)
            assert torch.allclose(out, full[c.batch], atol=1e-5), (out - full[c.batch]).abs().max()
    print("  [ok] GAS batches reproduce the full forward under src grouping")


def test_input_relu():
    g = _hub_graph()
    base, h0 = _model_and_h(g, N=N_HUB)
    m_on = _clone(base, input_relu=True)
    m_off = _clone(base, input_relu=False)
    d = base.output_dim
    with torch.no_grad():
        ent_feature = torch.sparse.mm(g.adj_self_norm, h0)
        on = m_on(g, h0)
        off = m_off(g, h0)
    assert torch.allclose(on[:, :d], torch.relu(ent_feature), atol=1e-6)
    assert torch.allclose(off[:, :d], ent_feature, atol=1e-6)
    assert bool((on[:, :d] >= 0).all()) and bool((off[:, :d] < 0).any())
    # GAS parity with input_relu on
    hist = _smh_history(g, m_on)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(m_on, warm[0], h0, hist, CPU)
        for c in build_gas_clusters(g, 4, method="bfs", softmax_history=True):
            out = gas_forward_batch(m_on, c, h0, hist, CPU)
            assert torch.allclose(out, on[c.batch], atol=1e-5)
    print("  [ok] input_relu: depth-0 slice = relu(features); GAS parity holds")


def test_chunked_src():
    g = _hub_graph()
    base, h0 = _model_and_h(g, N=N_HUB)
    m = _clone(base, attn_softmax="src")
    m_chunk = _clone(base, attn_softmax="src", edge_chunk=7)
    with torch.no_grad():
        assert torch.allclose(m(g, h0), m_chunk(g, h0), atol=1e-6)
    print("  [ok] edge-chunked forward matches under src grouping")


if __name__ == "__main__":
    test_default_identical()
    test_src_mass_sums_to_one()
    test_gas_parity_src()
    test_input_relu()
    test_chunked_src()
    print("test_attn_softmax_axis: ALL PASSED")
