"""
test_joint_metis_partition.py — scripts/make_joint_metis_partition.py's
joint_metis_assign on a planted graph: KG1 and KG2 are the same 4-community
graph (KG2 ids offset), 40% of the entity pairs are train links. Checks:
every train pair lands on one rank; the parts' vertex weights (1 + KG1 + KG2
triple degree) are balanced; most held-out pairs are on one rank; KG2
entities follow their community instead of piling onto a few ranks; the
result is deterministic; n_parts=1 puts everything on rank 0.

DGL-free, CPU, needs pymetis, < 5 s.
Run:  .venv/bin/python tests/test_joint_metis_partition.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_joint_metis_partition import joint_metis_assign  # noqa: E402


def _planted(n_comm=4, size=60, p_in=0.15, p_out=0.004, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n_comm * size
    comm = np.repeat(np.arange(n_comm), size)
    same = comm[:, None] == comm[None, :]
    prob = np.where(same, p_in, p_out)
    iu = np.triu_indices(n1, 1)
    hit = rng.random(iu[0].shape[0]) < prob[iu]
    h, t = iu[0][hit], iu[1][hit]
    rel = rng.integers(0, 5, h.shape[0])
    kg1_tri = np.stack([h, rel, t], 1)
    kg2_tri = np.stack([h + n1, rel, t + n1], 1)
    pairs = np.stack([np.arange(n1), np.arange(n1) + n1], 1)
    perm = rng.permutation(n1)
    train = pairs[perm[: int(0.4 * n1)]]
    held = pairs[perm[int(0.4 * n1):]]
    return np.arange(n1), np.arange(n1) + n1, kg1_tri, kg2_tri, train, held, comm


def test_joint_assign():
    kg1, kg2, t1, t2, train, held, comm = _planted()
    n_parts = 4
    r1, r2, cut = joint_metis_assign(kg1, kg2, t1, t2, train, n_parts)
    assert (r1[kg1] >= 0).all() and (r2[kg2] >= 0).all() and r1[kg1].max() < n_parts and r2[kg2].max() < n_parts
    assert (r2[kg1] == -1).all() and (r1[kg2] == -1).all()
    assert (r1[train[:, 0]] == r2[train[:, 1]]).all(), "a train pair was split"
    tri = np.concatenate([t1, t2])
    deg = np.bincount(np.concatenate([tri[:, 0], tri[:, 2]]), minlength=len(kg1) + len(kg2))
    w = np.zeros(n_parts)
    np.add.at(w, r1[kg1], deg[kg1] + 1)
    free = np.setdiff1d(kg2, train[:, 1])
    np.add.at(w, r2[kg2], deg[kg2])
    np.add.at(w, r2[free], 1)
    assert w.max() / w.mean() < 1.15, f"vertex weights unbalanced: {w}"
    cross = float(np.mean(r1[held[:, 0]] != r2[held[:, 1]]))
    assert cross < 0.10, f"held-out pairs split too often: {cross:.3f}"
    load2 = np.bincount(r2[kg2], minlength=n_parts)
    assert load2.max() / load2.mean() < 1.3, f"KG2 piled onto few ranks: {load2}"
    purity = np.mean([np.bincount(r1[kg1][comm == c]).max() / (comm == c).sum() for c in range(4)])
    assert purity > 0.9, f"communities split: {purity:.3f}"
    r1b, r2b, cutb = joint_metis_assign(kg1, kg2, t1, t2, train, n_parts)
    assert np.array_equal(r1, r1b) and np.array_equal(r2, r2b) and cut == cutb, "not deterministic"
    s1, s2, c1 = joint_metis_assign(kg1, kg2, t1, t2, train, 1)
    assert (s1[kg1] == 0).all() and (s2[kg2] == 0).all() and c1 == 0
    print(f"  test_joint_assign PASSED (cut {cut}, weight max/mean {w.max() / w.mean():.3f}, held-out cross {cross:.3f})")


if __name__ == "__main__":
    print("[test_joint_assign]")
    test_joint_assign()
    print("test_joint_metis_partition: ALL PASSED")
