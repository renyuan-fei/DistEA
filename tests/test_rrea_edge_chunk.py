"""
test_rrea_edge_chunk.py — --rrea_edge_chunk
(models/rrea.py NRGraphAttention.edge_chunk, _dst_chunk_slices).

The full-graph RREA forward processed in dst-grouped edge chunks must be
EXACT: chunk boundaries never split a softmax group, every edge is used
exactly once, and outputs AND gradients match the single-pass forward.
Checked on the ring + hub toy graph with a chunk (7) far smaller than the
hub's in-degree (59), so one chunk is the hub group alone, and with a
chunk larger than E (no chunking) as the identity case.

Run:  .venv/bin/python tests/test_rrea_edge_chunk.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import N_HUB, _hub_graph  # noqa: E402

from models.rrea import RREA, _dst_chunk_slices  # noqa: E402


def test_chunk_slices_cover_each_edge_once_and_keep_groups_whole():
    g = _hub_graph()
    dst = g.edge_index[:, 1]
    E = dst.numel()
    for chunk in (1, 7, 50, E - 1):
        slices = _dst_chunk_slices(dst, chunk)
        seen = torch.zeros(E, dtype=torch.long)
        groups_of = {}
        for k, idx in enumerate(slices):
            assert idx.numel() > 0
            seen[idx] += 1
            for d in dst[idx].unique().tolist():
                assert groups_of.setdefault(d, k) == k, (chunk, d)  # a dst lives in one chunk only
            n_groups = dst[idx].unique().numel()
            # a chunk is either within the size budget or a single whole group
            assert idx.numel() <= chunk or n_groups == 1, (chunk, idx.numel(), n_groups)
        assert torch.equal(seen, torch.ones(E, dtype=torch.long)), chunk
    print("  test_chunk_slices_cover_each_edge_once_and_keep_groups_whole PASSED")


def _model(g, edge_chunk, seed=0):
    torch.manual_seed(seed)
    m = RREA(rel_num=g.rel_num, output_dim=16, num_layers=2, dropout=0.0,
             attn_heads=2, edge_chunk=edge_chunk)
    return m


def test_chunked_forward_and_backward_match_single_pass():
    g = _hub_graph()
    torch.manual_seed(1)
    h0 = torch.randn(N_HUB, 16) * 0.3
    ref = _model(g, 0)
    outs = {}
    for chunk in (0, 7, 10_000):
        m = _model(g, chunk)
        m.load_state_dict(ref.state_dict())
        m.train()
        h = torch.nn.Parameter(h0.clone())
        out = m(g, h)
        (out * torch.arange(1, out.size(1) + 1)).sum().backward()
        outs[chunk] = (out.detach(), h.grad.clone(), m.rel_emb.grad.clone(),
                       [k.grad.clone() for k in m.encoder.attn_kernels])
    # Gradients are sums over all edges whose summation ORDER changes with
    # chunking (index_add_ per chunk); large contributions cancel in some
    # elements, so an elementwise rtol is meaningless there. Compare the max
    # abs difference against the tensor's max magnitude instead: measured
    # ~1e-7 to 4e-7 (fp32 rounding), e.g. rel_emb.grad 2.1e-4 vs 8.5e2.
    def _close_scaled(x, y, tol=1e-5):
        return float((x - y).abs().max()) <= tol * max(1.0, float(y.abs().max()))

    for chunk in (7, 10_000):
        a, b = outs[0], outs[chunk]
        assert torch.allclose(a[0], b[0], atol=1e-5), (chunk, (a[0] - b[0]).abs().max().item())
        assert _close_scaled(a[1], b[1]), (chunk, "h.grad", (a[1] - b[1]).abs().max().item())
        assert _close_scaled(a[2], b[2]), (chunk, "rel_emb.grad", (a[2] - b[2]).abs().max().item())
        for ka, kb in zip(a[3], b[3], strict=True):
            assert _close_scaled(ka, kb), (chunk, "kernel.grad")
    m = _model(g, 10_000)
    assert m.encoder.edge_chunk == 10_000 and ref.encoder.edge_chunk == 0
    print("  test_chunked_forward_and_backward_match_single_pass PASSED")


if __name__ == "__main__":
    for fn in (
        test_chunk_slices_cover_each_edge_once_and_keep_groups_whole,
        test_chunked_forward_and_backward_match_single_pass,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_rrea_edge_chunk: ALL PASSED")
