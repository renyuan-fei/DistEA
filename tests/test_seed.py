"""
test_seed.py — seeding.seed_everything: the `--seed`
contract.

1. Same seed + same rank -> identical torch.randint / torch.randperm /
   numpy / random streams (the negative sampler and GAS cluster order).
2. Same seed, different ranks -> different streams (seed + rank).
3. seed=None is a no-op and returns None.
4. Re-seeding does not change ID-seeded entity features
   (models.gcn.build_entity_features) -- halo rows must still agree across
   ranks regardless of --seed.

Run:  .venv/bin/python tests/test_seed.py
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gcn import build_entity_features  # noqa: E402
from seeding import seed_everything  # noqa: E402


def _draw():
    return (
        torch.randint(0, 1000, (8,)).tolist(),
        torch.randperm(16).tolist(),
        np.random.randint(0, 1000, size=4).tolist(),
        [random.random() for _ in range(3)],
    )


def test_reproducible_same_rank():
    assert seed_everything(123, rank=0) == 123
    a = _draw()
    assert seed_everything(123, rank=0) == 123
    b = _draw()
    assert a == b
    print("  [ok] same seed + rank reproduces randint / randperm / numpy / random")


def test_ranks_differ():
    seed_everything(7, rank=0)
    a = _draw()
    seed_everything(7, rank=1)
    b = _draw()
    assert a[0] != b[0] or a[1] != b[1]
    assert seed_everything(7, rank=3) == 10
    print("  [ok] seed + rank gives each rank its own stream")


def test_none_is_noop():
    torch.manual_seed(99)
    before = torch.randint(0, 1000, (4,)).tolist()
    torch.manual_seed(99)
    assert seed_everything(None, rank=5) is None
    after = torch.randint(0, 1000, (4,)).tolist()
    assert before == after
    print("  [ok] seed=None leaves the RNG state alone")


def test_entity_features_unaffected():
    ids = np.array([3, 17, 1000, 123456], dtype=np.int64)
    seed_everything(1, rank=0)
    f1 = build_entity_features(ids, 16, torch.device("cpu"))
    seed_everything(2, rank=4)
    f2 = build_entity_features(ids, 16, torch.device("cpu"))
    assert torch.equal(f1, f2)
    print("  [ok] ID-seeded entity features ignore --seed")


if __name__ == "__main__":
    test_reproducible_same_rank()
    test_ranks_differ()
    test_none_is_noop()
    test_entity_features_unaffected()
    print("test_seed: all passed")
