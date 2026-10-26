"""
test_gas_compact_loss.py — gas._compact_loss_rows: the GAS
epoch evaluates InfoNCE + margin on a compact table of exactly the rows the
loss reads (fresh batch rows with grad, the rest from final_hist) instead of
an (N, width) index_copy view whose every indexing op made autograd allocate
and zero an N-sized gradient (2.6 GB x 12 per cluster at DBP1M).

1. Loss values and gradients (w.r.t. the batch embeddings and the loss
   weight) are identical between the full view and the compact table, for
   combined (InfoNCE + margin) and plain margin, with and without batch
   rows among the negatives / pairs.
2. KG1 row count keeps alignment_loss's boundary guard meaningful.
3. GASCluster.to_ moves every tensor field and leaves the bool alone.

Run:  .venv/bin/python tests/test_gas_compact_loss.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import _hub_graph  # noqa: E402

from gas import GASCluster, _compact_loss_rows, build_gas_clusters  # noqa: E402
from models.gcn import alignment_loss, combined_align_loss  # noqa: E402


def _case(seed: int, n: int = 60, width: int = 12, kg1: int = 30, n_batch: int = 15, p: int = 9):
    gen = torch.Generator().manual_seed(seed)
    final_hist = torch.randn(n, width, generator=gen)
    batch = torch.sort(torch.randperm(n, generator=gen)[:n_batch]).values
    emb_batch = torch.randn(n_batch, width, generator=gen)
    in_batch = torch.zeros(n, dtype=torch.bool)
    in_batch[batch] = True
    kg1_idx = torch.randint(0, kg1, (p,), generator=gen)
    kg2_idx = torch.randint(kg1, n, (p,), generator=gen)
    pairs = torch.stack([kg1_idx, kg2_idx], dim=1)
    neg1 = torch.randint(0, n, (p, 2), generator=gen)
    neg2 = torch.randint(0, n, (p, 2), generator=gen)
    return final_hist, batch, emb_batch, in_batch, pairs, neg1, neg2, kg1


def _both(seed: int, combined: bool):
    final_hist, batch, emb_batch0, in_batch, pairs, neg1, neg2, kg1 = _case(seed)
    outs = []
    for compact in (False, True):
        emb_batch = emb_batch0.clone().requires_grad_(True)
        w = torch.nn.Parameter(torch.tensor([0.3, 0.7]))
        if compact:
            rows, pairs_c, neg1_c, neg2_c, kg1_c = _compact_loss_rows(
                final_hist, emb_batch, batch, in_batch, pairs, neg1, neg2, kg1)
            assert rows.size(0) <= pairs.numel() + neg1.numel() + neg2.numel()
            args = (rows, pairs_c, kg1_c)
            kw = {"neg_kg1": neg1_c, "neg_kg2": neg2_c}
        else:
            emb_full = final_hist.detach().index_copy(0, batch, emb_batch)
            args = (emb_full, pairs, kg1)
            kw = {"neg_kg1": neg1, "neg_kg2": neg2}
        if combined:
            loss = combined_align_loss(*args, w, neg_k=2, margin=6.0, infonce_temp=0.1, **kw)
        else:
            loss = alignment_loss(*args, neg_k=2, margin=6.0, **kw)
        loss.backward()
        outs.append((loss.detach().clone(), emb_batch.grad.clone(),
                     None if w.grad is None else w.grad.clone()))
    return outs


def test_values_and_gradients_match():
    worst = 0.0
    for seed in range(6):
        for combined in (True, False):
            (la, ga, wa), (lb, gb, wb) = _both(seed, combined)
            assert torch.allclose(la, lb, atol=1e-6, rtol=1e-6), (seed, combined, la, lb)
            assert torch.allclose(ga, gb, atol=1e-6), (seed, combined, (ga - gb).abs().max())
            assert ga.abs().sum() > 0  # some batch row really takes part
            if combined:
                assert wa is not None and torch.allclose(wa, wb, atol=1e-7)
            worst = max(worst, (ga - gb).abs().max().item(), (la - lb).abs().item())
    print(f"  test_values_and_gradients_match PASSED (max |diff| {worst:.2e})")


def test_kg1_count_and_no_batch_rows():
    final_hist, batch, emb_batch, in_batch, pairs, neg1, neg2, kg1 = _case(11)
    # No batch node among the touched rows -> rows come purely from final_hist.
    in_none = torch.zeros_like(in_batch)
    rows, pairs_c, neg1_c, neg2_c, kg1_c = _compact_loss_rows(
        final_hist, emb_batch, batch, in_none, pairs, neg1, neg2, kg1)
    ids = torch.unique(torch.cat([pairs.reshape(-1), neg1.reshape(-1), neg2.reshape(-1)]))
    assert torch.equal(rows, final_hist[ids])
    assert kg1_c == int((ids < kg1).sum()) and 0 < kg1_c < rows.size(0)
    assert torch.equal(ids[pairs_c], pairs) and torch.equal(ids[neg1_c], neg1)
    assert torch.equal(ids[neg2_c], neg2)
    print("  test_kg1_count_and_no_batch_rows PASSED")


def test_cluster_to_device_moves_tensors():
    g = _hub_graph()
    cl = build_gas_clusters(g, 3, method="bfs", softmax_history=True, layer0_history=True)
    c: GASCluster = cl[0]
    before = {k: getattr(c, k) for k in c.__slots__}
    c.to_("cpu")  # the only device available here; must be a no-op in values
    for k, v in before.items():
        now = getattr(c, k)
        if torch.is_tensor(v):
            assert torch.is_tensor(now) and now.device.type == "cpu"
            if v.layout != torch.strided:  # COO or CSR
                assert torch.equal(v.to_dense(), now.to_dense())
            else:
                assert torch.equal(v, now)
        else:
            assert now == v
    assert c.l0_batch_only is True
    print("  test_cluster_to_device_moves_tensors PASSED")


if __name__ == "__main__":
    for fn in (
        test_values_and_gradients_match,
        test_kg1_count_and_no_batch_rows,
        test_cluster_to_device_moves_tensors,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_compact_loss: ALL PASSED")
