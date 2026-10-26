"""
test_gas_hard_neg.py — hard negatives inside the GAS
training epoch (gas.gas_train_epoch(hard_neg=..., hard_neg_k=...)).

1. Off-path parity: hard_neg=None and (pools, k=0) consume the same RNG and
   give the same loss / table / parameters as the plain call -- the default
   recipe is untouched when the switch is off.
2. On-path: with pools and k=2 the epoch runs, the loss is finite, and the
   margin loss really saw (neg_k + k) negatives per pair (checked through a
   recording wrapper around alignment_loss).
3. Pool/pair length mismatch raises.

Run (gloo; binds a loopback port):  .venv/bin/python tests/test_gas_hard_neg.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_softmax_history import N_HUB, _hub_graph, _model_and_h, _smh_history  # noqa: E402

import gas as gas_mod  # noqa: E402
from gas import build_gas_clusters, gas_forward_batch, gas_train_epoch  # noqa: E402
from hard_neg import mine_hard_negatives  # noqa: E402

CPU = torch.device("cpu")
PAIRS = torch.tensor([[1, 31], [2, 32], [5, 35], [7, 40], [9, 44]])


def _setup(seed=0):
    from dist_trainer import DistTrainer

    torch.manual_seed(seed)
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt = torch.optim.Adam(list(model.parameters()) + [h, lw], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1, neg_k=1, margin=6.0, loss_weight=lw)
    hist = _smh_history(g, model)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h, hist, CPU)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    final = torch.zeros(N_HUB, model.output_width)
    return trainer, cl, hist, final, h, model


def _run(hard_neg, k, seed=0):
    trainer, cl, hist, final, h, model = _setup(seed)
    torch.manual_seed(123)
    loss = gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU, hard_neg=hard_neg, hard_neg_k=k)
    return loss, final.clone(), h.detach().clone(), model.rel_emb.detach().clone()


def test_off_path_parity():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29623")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    base = _run(None, 0)
    trainer, cl, hist, final, h, model = _setup(0)
    pools = mine_hard_negatives(torch.randn(N_HUB, model.output_width), PAIRS, 30, pool=4, dtype=torch.float32)
    off = _run(pools, 0)
    for a, b in zip(base[1:], off[1:], strict=True):
        assert torch.equal(a, b)
    assert abs(base[0] - off[0]) < 1e-12
    print(f"  [ok] k=0 with pools == plain epoch (loss {base[0]:.5f})")


def test_on_path_uses_extra_negatives():
    trainer, cl, hist, final, h, model = _setup(0)
    # one plain epoch so final_hist is non-zero, then mine from it
    gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU)
    pools = mine_hard_negatives(final, PAIRS, 30, pool=3, dtype=torch.float32)
    assert pools is not None and len(pools) == PAIRS.size(0)
    seen: list[tuple[int, int]] = []
    orig = gas_mod._align_loss if hasattr(gas_mod, "_align_loss") else None
    import models.gcn as gcn

    real_align = gcn.alignment_loss

    def recording(emb, pairs, kg1_size, neg_k=10, margin=3.0, pair_weights=None, neg_kg1=None, neg_kg2=None):
        seen.append((int(neg_kg1.size(1)), int(neg_kg2.size(1))))
        return real_align(emb, pairs, kg1_size, neg_k=neg_k, margin=margin,
                          pair_weights=pair_weights, neg_kg1=neg_kg1, neg_kg2=neg_kg2)

    gcn.alignment_loss = recording
    try:
        loss = gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU, hard_neg=pools, hard_neg_k=2)
    finally:
        gcn.alignment_loss = real_align
    assert np.isfinite(loss), loss
    assert seen, "margin loss never called"
    assert all(s == (1 + 2, 1 + 2) for s in seen), seen
    # pools stay inside the KG ranges the loss corrupts
    assert bool((pools.neg_kg1 < 30).all()) and bool((pools.neg_kg2 >= 30).all())
    print(f"  [ok] k=2: margin loss saw {seen[0]} negatives per pair in {len(seen)} cluster steps (loss {loss:.4f})")
    _ = orig


def test_length_mismatch_raises():
    trainer, cl, hist, final, h, model = _setup(0)
    pools = mine_hard_negatives(torch.randn(N_HUB, model.output_width), PAIRS[:3], 30, pool=2, dtype=torch.float32)
    try:
        gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU, hard_neg=pools, hard_neg_k=1)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched pools must raise")
    print("  [ok] pool/pair length mismatch raises")


if __name__ == "__main__":
    test_off_path_parity()
    test_on_path_uses_extra_negatives()
    test_length_mismatch_raises()
    print("test_gas_hard_neg: ALL PASSED")
