"""
test_nhsm_chunked.py — nhsm_loss's row-chunked path (torch.utils.checkpoint
over blocks of pairs) must be numerically identical to the dense form, in
both the loss value and the gradient w.r.t. the embedding table. The dense
form is the one verified against ClusterEA's reference (tests/test_stage_t.py);
this test pins the chunked form to it, including a chunk size that does not
divide the pair count evenly and the auto (chunk_rows=None) selection.

Run:  uv run python tests/test_nhsm_chunked.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gcn import nhsm_loss  # noqa: E402


def _case(seed: int, n: int = 400, d: int = 16, p: int = 90):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(n, d, generator=g)
    left = torch.randperm(n // 2, generator=g)[:p]
    right = n // 2 + torch.randperm(n - n // 2, generator=g)[:p]
    return emb, torch.stack([left, right], dim=1)


def _loss_and_grad(emb, pairs, **kw):
    e = emb.clone().requires_grad_(True)
    loss = nhsm_loss(e, pairs, gamma=1.0, **kw)
    loss.backward()
    return loss.detach(), e.grad.detach()


def test_chunked_matches_dense():
    for seed in (0, 1, 2):
        emb, pairs = _case(seed)
        dense_l, dense_g = _loss_and_grad(emb, pairs, chunk_rows=10_000)  # P <= chunk -> dense
        for chunk in (7, 30, 89, 90):  # uneven, several, one-short, exact
            loss, g = _loss_and_grad(emb, pairs, chunk_rows=chunk)
            assert torch.allclose(loss, dense_l, rtol=1e-6, atol=1e-6), (seed, chunk, loss, dense_l)
            assert torch.allclose(g, dense_g, rtol=1e-5, atol=1e-6), (seed, chunk)
    print("chunked == dense: loss and grad match for chunk_rows in {7, 30, 89, 90}")


def test_auto_chunk_small_graph_is_dense():
    emb, pairs = _case(3)
    auto_l, auto_g = _loss_and_grad(emb, pairs)                    # chunk_rows=None
    dense_l, dense_g = _loss_and_grad(emb, pairs, chunk_rows=10_000)
    assert torch.equal(auto_l, dense_l) and torch.equal(auto_g, dense_g)
    print("auto chunk_rows on a small graph takes the dense path bit-for-bit")


def test_auto_chunk_bound():
    # ~1 GB fp32 per (rows, N) temporary: N = 200_000 -> 1342 rows per block.
    rows = max(256, (1 << 28) // 200_000)
    assert 1000 <= rows <= 1400, rows
    print(f"auto chunk_rows at N=200K = {rows} rows (~{rows * 200_000 * 4 / 2**30:.2f} GB per temporary)")


if __name__ == "__main__":
    test_chunked_matches_dense()
    test_auto_chunk_small_graph_is_dense()
    test_auto_chunk_bound()
    print("all nhsm chunked tests passed")
