"""
seeding.py — `--seed`: one call that seeds Python's
`random`, NumPy's legacy global RNG and torch (CPU + every CUDA device)
with `seed + rank`.

What it controls in DistEA: the negative sampler (`torch.randint` in
gas.py / gcn.py / dist_trainer.py), the GAS cluster visiting order
(`torch.randperm` in gas.py) and any dropout. What it deliberately does NOT
touch: the entity table `h_init` (build_entity_features seeds each row by
its global ID), RREA's `rel_emb`/attention kernels (local torch.Generator
seeded from a fixed kernel_seed) and the METIS/BFS clustering
(`closure_seed`) -- those are deterministic by construction and must stay
identical across ranks/runs for halo rows to agree.

`seed + rank` (not `seed`) so that ranks draw DIFFERENT negatives; the
all_reduce'd model parameters are unaffected by which negatives each rank
saw. Seven unseeded DBP1M runs of one recipe spread Hit@1 0.1126-0.1210
(sigma ~0.3pp), so repeated-seed averaging needs this to be reproducible.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int | None, rank: int = 0) -> int | None:
    """Seed random / numpy / torch with `seed + rank`; return the value used.

    `seed=None` is a no-op (returns None) so callers can pass the CLI value
    straight through.
    """
    if seed is None:
        return None
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s % (2**32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s
