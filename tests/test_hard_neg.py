"""
test_hard_neg.py — hard_neg.mine_hard_negatives /
sample_hard_negatives / merge_negatives.

1. side="other": neg_kg1 pools lie in [0, kg1_size), neg_kg2 pools in
   [kg1_size, N); a pair's own positive never appears in its pool.
2. side="any": pools may span both KGs but never contain the pair's e1 or e2.
3. Chunked mining (chunk=3) == brute-force dense top-k (same ids, same order).
4. Pools really are the nearest rows: a planted near-duplicate of the anchor
   on the candidate side is the pool's first entry.
5. sample_hard_negatives draws only from the row's own pool; merge_negatives
   with no pool returns the uniform tensor itself (bit-identical recipe when
   mining is off), with a pool it appends k columns.
6. pool <= 0 / empty pairs -> None; bad side raises.

Run:  .venv/bin/python tests/test_hard_neg.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hard_neg import merge_negatives, mine_hard_negatives, sample_hard_negatives  # noqa: E402


def _table(seed=0, n=40, kg1=20, d=8, p=12):
    gen = torch.Generator().manual_seed(seed)
    emb = torch.randn(n, d, generator=gen)
    e1 = torch.randperm(kg1, generator=gen)[:p]
    e2 = torch.randperm(n - kg1, generator=gen)[:p] + kg1
    pairs = torch.stack([e1, e2], dim=1)
    return emb, pairs, kg1


def _brute(emb, pairs, kg1, m, side):
    normed = torch.nn.functional.normalize(emb, dim=1)
    sim = normed @ normed.t()
    out1, out2 = [], []
    for e1, e2 in pairs.tolist():
        s1 = sim[e2].clone()
        s2 = sim[e1].clone()
        if side == "other":
            s1[kg1:] = -1e9
            s2[:kg1] = -1e9
            s1[e1] = -1e9
            s2[e2] = -1e9
        else:
            s1[e1] = -1e9
            s1[e2] = -1e9
            s2[e2] = -1e9
            s2[e1] = -1e9
        out1.append(torch.topk(s1, m).indices)
        out2.append(torch.topk(s2, m).indices)
    return torch.stack(out1), torch.stack(out2)


def test_side_other_ranges_and_no_positive():
    emb, pairs, kg1 = _table()
    pools = mine_hard_negatives(emb, pairs, kg1, pool=5, side="other", dtype=torch.float32)
    assert pools.neg_kg1.shape == (12, 5) and pools.neg_kg2.shape == (12, 5)
    assert bool((pools.neg_kg1 < kg1).all()) and bool((pools.neg_kg2 >= kg1).all())
    for i, (e1, e2) in enumerate(pairs.tolist()):
        assert e1 not in pools.neg_kg1[i].tolist()
        assert e2 not in pools.neg_kg2[i].tolist()
    print("  [ok] side=other: KG ranges respected, positive excluded")


def test_side_any_excludes_pair():
    emb, pairs, kg1 = _table(seed=1)
    pools = mine_hard_negatives(emb, pairs, kg1, pool=6, side="any", dtype=torch.float32)
    spans_both = bool((pools.neg_kg1 < kg1).any()) and bool((pools.neg_kg1 >= kg1).any())
    assert spans_both, "side=any should be allowed to pick from both KGs"
    for i, (e1, e2) in enumerate(pairs.tolist()):
        row1 = pools.neg_kg1[i].tolist()
        row2 = pools.neg_kg2[i].tolist()
        assert e1 not in row1 and e2 not in row1
        assert e1 not in row2 and e2 not in row2
    print("  [ok] side=any: never the positive nor the anchor")


def test_chunked_equals_dense():
    for side in ("other", "any"):
        emb, pairs, kg1 = _table(seed=2)
        pools = mine_hard_negatives(emb, pairs, kg1, pool=4, side=side, chunk=3, dtype=torch.float32)
        b1, b2 = _brute(emb, pairs, kg1, 4, side)
        assert torch.equal(pools.neg_kg1, b1), (side, pools.neg_kg1, b1)
        assert torch.equal(pools.neg_kg2, b2), (side, pools.neg_kg2, b2)
    print("  [ok] chunked top-k == dense brute force (both sides)")


def test_planted_neighbour_is_first():
    emb, pairs, kg1 = _table(seed=3)
    e1, e2 = pairs[0].tolist()
    victim1 = (e1 + 1) % kg1                # another KG1 row -> near-copy of e2
    victim2 = kg1 + (e2 - kg1 + 1) % (emb.size(0) - kg1)  # another KG2 row -> near-copy of e1
    emb[victim1] = emb[e2] + 1e-3
    emb[victim2] = emb[e1] + 1e-3
    pools = mine_hard_negatives(emb, pairs, kg1, pool=3, dtype=torch.float32)
    assert pools.neg_kg1[0, 0].item() == victim1
    assert pools.neg_kg2[0, 0].item() == victim2
    print("  [ok] nearest wrong entity is the first pool entry")


def test_sampling_and_merge():
    emb, pairs, kg1 = _table(seed=4)
    pools = mine_hard_negatives(emb, pairs, kg1, pool=5, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    draw = sample_hard_negatives(pools.neg_kg1, 3, gen)
    assert draw.shape == (12, 3)
    for i in range(12):
        assert set(draw[i].tolist()) <= set(pools.neg_kg1[i].tolist())
    uni = torch.randint(0, emb.size(0), (12, 1))
    same = merge_negatives(uni, None, 2)
    assert same is uni
    assert merge_negatives(uni, pools.neg_kg1, 0) is uni
    merged = merge_negatives(uni, pools.neg_kg1, 2, gen)
    assert merged.shape == (12, 3) and torch.equal(merged[:, :1], uni)
    sub = torch.tensor([True, False, True] * 4)
    merged_sub = merge_negatives(uni[sub], pools.neg_kg1[sub], 1, gen)
    assert merged_sub.shape == (8, 2)
    print("  [ok] sampling stays inside the pool; merge is identity when off")


def test_edge_cases():
    emb, pairs, kg1 = _table(seed=5)
    assert mine_hard_negatives(emb, pairs, kg1, pool=0) is None
    assert mine_hard_negatives(emb, pairs[:0], kg1, pool=5) is None
    try:
        mine_hard_negatives(emb, pairs, kg1, pool=5, side="same")
    except ValueError:
        pass
    else:
        raise AssertionError("bad side must raise")
    # pool larger than the candidate side still returns (P, pool) valid ids
    big = mine_hard_negatives(emb, pairs, kg1, pool=25, dtype=torch.float32)
    assert big.neg_kg1.shape == (12, 25) and bool((big.neg_kg1 < kg1).all())
    for i, (e1, _) in enumerate(pairs.tolist()):
        assert e1 not in big.neg_kg1[i].tolist()
    print("  [ok] edge cases: pool<=0 / no pairs -> None, bad side raises, oversize pool padded")


if __name__ == "__main__":
    test_side_other_ranges_and_no_positive()
    test_side_any_excludes_pair()
    test_chunked_equals_dense()
    test_planted_neighbour_is_first()
    test_sampling_and_merge()
    test_edge_cases()
    print("test_hard_neg: all passed")
