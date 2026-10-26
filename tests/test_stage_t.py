"""
test_stage_t.py — Stage T tests: nhsm_loss (Dual-AMN Normalized
Hard Sample Mining, ported from ClusterEA/src/prev_models/duala/loss.py).

The critical check is test_nhsm_matches_literal_onehot_reference: our port
zeroes the positive pair's own columns via index_put instead of the
reference's materialized (P, N) one-hot multiply (a pure memory
optimization). The two must be numerically equivalent; this is checked
against a literal transcription of the reference formula.

DGL-free, no GPU, no torch.distributed. <10 sec.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gcn import build_entity_features, nhsm_loss


def _literal_reference_nhsm(emb, pairs, gamma):
    """Line-for-line transcription of ClusterEA's align_loss (duala/loss.py),
    one-hot masking and all -- the ground truth our port must match."""
    node_size = emb.size(0)
    left = pairs[:, 0]
    right = pairs[:, 1]
    l_emb = emb[left]
    r_emb = emb[right]

    def squared_dist(a, b):
        ra = torch.sum(torch.square(a), dim=1).reshape(-1, 1)
        rb = torch.sum(torch.square(b), dim=1).reshape(1, -1)
        return ra + rb - 2 * torch.matmul(a, torch.transpose(b, 0, 1))

    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
    l_neg_dis = squared_dist(l_emb, emb)
    r_neg_dis = squared_dist(r_emb, emb)

    mask = 1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)

    l_loss = (pos_dis - l_neg_dis + gamma) * mask
    r_loss = (pos_dis - r_neg_dis + gamma) * mask

    with torch.no_grad():
        r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
        r_std = torch.std(r_loss, dim=-1, keepdim=True)
        l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
        l_std = torch.std(l_loss, dim=-1, keepdim=True)
    r_loss = (r_loss - r_mean) / r_std
    l_loss = (l_loss - l_mean) / l_std

    lamb, tau = 30, 10
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)


def test_nhsm_matches_literal_onehot_reference():
    torch.manual_seed(0)
    emb = build_entity_features(np.arange(12), 8, torch.device("cpu")).clone()
    pairs = torch.tensor([[0, 6], [1, 7], [2, 9]])  # kg1 rows 0-5, kg2 rows 6-11

    ours = nhsm_loss(emb, pairs, gamma=1.0)
    ref = _literal_reference_nhsm(emb, pairs, gamma=1.0)
    assert torch.allclose(ours, ref, atol=1e-5), (ours.item(), ref.item())
    print("  test_nhsm_matches_literal_onehot_reference PASSED")


def test_nhsm_optimizable_and_pulls_pairs_together():
    """The per-row z-score makes NHSM's absolute level nearly invariant to
    the embedding table (any table scores ~logsumexp of N standardized
    values), so a level comparison between tables is meaningless -- the
    training signal lives in the gradients. Sanity-check that signal:
    SGD on the loss must reduce it AND shrink pair distances relative to
    the mean distance to everything else."""
    torch.manual_seed(0)
    emb = torch.nn.Parameter(torch.randn(20, 8))
    pairs = torch.tensor([[0, 10], [1, 11], [2, 12], [3, 13]])
    opt = torch.optim.SGD([emb], lr=0.05)

    def pair_gap(e):
        d_pos = (e[pairs[:, 0]] - e[pairs[:, 1]]).pow(2).sum(-1).mean()
        d_all = torch.cdist(e[pairs[:, 0]], e).pow(2).mean()
        return (d_pos / d_all).item()

    loss0 = nhsm_loss(emb, pairs, gamma=1.0).item()
    gap0 = pair_gap(emb.detach())
    for _ in range(50):
        opt.zero_grad()
        loss = nhsm_loss(emb, pairs, gamma=1.0)
        loss.backward()
        opt.step()
    loss1 = nhsm_loss(emb, pairs, gamma=1.0).item()
    gap1 = pair_gap(emb.detach())

    assert loss1 < loss0, (loss0, loss1)
    assert gap1 < gap0, (gap0, gap1)
    print("  test_nhsm_optimizable_and_pulls_pairs_together PASSED")


def test_nhsm_backprop_flows_to_embeddings():
    torch.manual_seed(0)
    emb = torch.randn(10, 8, requires_grad=True)
    pairs = torch.tensor([[0, 5], [1, 6]])
    loss = nhsm_loss(emb, pairs, gamma=1.0)
    loss.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert emb.grad.abs().sum() > 0
    print("  test_nhsm_backprop_flows_to_embeddings PASSED")


if __name__ == "__main__":
    print("Stage T tests:")
    test_nhsm_matches_literal_onehot_reference()
    test_nhsm_optimizable_and_pulls_pairs_together()
    test_nhsm_backprop_flows_to_embeddings()
    print("All Stage T tests PASSED.")
