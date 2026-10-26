"""
test_stage_r.py — Stage R tests: minibatch_eval.py's world_size<=1 pieces.

Functions that issue a real torch.distributed collective (all_reduce,
all_gather_object) need a live multi-rank process group and can't run here.
_synchronized_batch_count and _gather_emb_dicts both have an explicit
world_size<=1 fast path that skips the collective entirely -- that path is
what's covered here. compute_minibatch_embeddings / minibatch_selective_eval /
minibatch_generate_pseudo_seeds call gather_minibatch_features, which calls
dist.all_gather_object regardless of world_size, so they need an actual
torchrun run and are not covered here.

DGL-free, no GPU, no torch.distributed required. <5 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minibatch_eval import _gather_emb_dicts, _synchronized_batch_count


def test_synchronized_batch_count_world_size_1_matches_local_count():
    device = torch.device("cpu")
    assert _synchronized_batch_count(n_local=10, batch_size=4, world_size=1, device=device) == 3
    assert _synchronized_batch_count(n_local=0, batch_size=4, world_size=1, device=device) == 0
    assert _synchronized_batch_count(n_local=4, batch_size=4, world_size=1, device=device) == 1
    print("  test_synchronized_batch_count_world_size_1_matches_local_count PASSED")


def test_gather_emb_dicts_world_size_1_is_identity():
    local = {1: np.array([1.0, 2.0]), 2: np.array([3.0, 4.0])}
    merged = _gather_emb_dicts(local, rank=0, world_size=1)
    assert merged is local
    print("  test_gather_emb_dicts_world_size_1_is_identity PASSED")


if __name__ == "__main__":
    print("Stage R tests:")
    test_synchronized_batch_count_world_size_1_matches_local_count()
    test_gather_emb_dicts_world_size_1_is_identity()
    print("All Stage R tests PASSED.")
