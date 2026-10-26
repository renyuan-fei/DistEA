"""
test_pseudo_seed_policy.py -- --pseudo_seed_policy unaligned.

1. _merge_pseudo_seeds: the legacy call (test-link set passed) rejects a
   candidate equal to a test pair; the unaligned call (empty set) accepts it;
   entity-level freezing still holds.
2. generate_pseudo_seeds: with candidate_kg1/kg2 the mutual-NN search runs
   among the candidate rows only (train-linked entities never reappear), and
   the vectorised mutual check returns the same pairs as before.
3. _pseudo_seed_candidates: accumulate keeps earlier pseudo-seeds in the
   base and out of the pool; replace re-mines on the original train links and
   the merge drops the previous round's (possibly wrong) pairs.
4. top_frac: only the most confident mutual pairs are kept, the same pairs
   on the dense and the chunked path (CSLS on); _chunked_mutual_nn's row-best
   scores equal the dense CSLS row maxima.

Run (gloo; binds a loopback port):  .venv/bin/python tests/test_pseudo_seed_policy.py
"""
from __future__ import annotations

import os
import socket
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _Identity:
    def eval(self):
        return self

    def __call__(self, g, h):
        return h


def test_merge():
    from framework import _merge_pseudo_seeds

    train = [(0, 10)]
    test = {(1, 11), (2, 12)}
    cands = [(1, 11), (3, 13), (0, 14), (2, 12)]
    legacy, n_leg, _ = _merge_pseudo_seeds(train, cands, test, set())
    assert (1, 11) not in legacy and (2, 12) not in legacy and (3, 13) in legacy, legacy
    assert (0, 14) not in legacy, "entity 0 already used"
    new, n_new, correct = _merge_pseudo_seeds(train, cands, set(), test)
    assert (1, 11) in new and (2, 12) in new and (3, 13) in new and (0, 14) not in new, new
    assert n_new == 3 and correct == 2, (n_new, correct)
    print("  [ok] merge: legacy drops correct test pairs, unaligned keeps them")


def test_candidates():
    from eval import generate_pseudo_seeds

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        g = torch.Generator().manual_seed(0)
        base = torch.randn(5, 8, generator=g)
        h = torch.cat([base, base + 0.01 * torch.randn(5, 8, generator=g)])
        ids = np.arange(10)
        k1, k2 = np.arange(5), np.arange(5, 10)
        full = generate_pseudo_seeds(_Identity(), None, h, ids, k1, k2, 0, 1, torch.device("cpu"),
                                     metric="cosine", use_csls=False)
        assert full == [(i, i + 5) for i in range(5)], full
        part = generate_pseudo_seeds(_Identity(), None, h, ids, k1, k2, 0, 1, torch.device("cpu"),
                                     metric="cosine", use_csls=False,
                                     candidate_kg1=np.array([1, 2, 3, 4]), candidate_kg2=np.array([6, 7, 8, 9]),
                                     sim_on_device=True)
        assert part == [(i, i + 5) for i in range(1, 5)], part
        print("  [ok] candidate restriction + vectorised mutual check")
    finally:
        dist.destroy_process_group()


def test_refresh():
    from framework import _merge_pseudo_seeds, _pseudo_seed_candidates

    orig = [(0, 10)]
    valid = [(1, 11)]
    test = [(2, 12), (3, 13), (4, 14)]
    cur = orig + [(2, 13)]  # an earlier, wrong pseudo-seed
    kg1, kg2 = range(5), range(10, 15)
    base, c1, c2 = _pseudo_seed_candidates("testside", "accumulate", cur, orig, valid, test, kg1, kg2)
    assert base == cur and c1.tolist() == [3, 4] and c2.tolist() == [12, 14], (base, c1, c2)
    base, c1, c2 = _pseudo_seed_candidates("testside", "replace", cur, orig, valid, test, kg1, kg2)
    assert base == orig and c1.tolist() == [2, 3, 4] and c2.tolist() == [12, 13, 14], (base, c1, c2)
    new, n, correct = _merge_pseudo_seeds(base, [(2, 12), (3, 13)], set(), set(test))
    assert new == orig + [(2, 12), (3, 13)] and n == 2 and correct == 2, new
    _, c1, c2 = _pseudo_seed_candidates("unaligned", "replace", cur, orig, valid, [], kg1, kg2)
    assert c1.tolist() == [2, 3, 4] and c2.tolist() == [12, 13, 14], (c1, c2)
    print("  [ok] refresh: accumulate freezes earlier pairs, replace re-mines and drops them")


def test_top_frac():
    from eval import _apply_csls, _chunked_mutual_nn, _compute_sim, _csls_thr, generate_pseudo_seeds

    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=0, world_size=1)
    cpu = torch.device("cpu")
    try:
        # 4 one-hot KG1 rows; KG2 row i = e_i + eps_i * e_(4+i): all mutual,
        # cosine 1/sqrt(1+eps^2) decreasing with eps
        eps = [0.05, 0.1, 0.8, 1.0]
        k1 = torch.eye(4, 8)
        k2 = torch.stack([torch.eye(8)[i] + eps[i] * torch.eye(8)[4 + i] for i in range(4)])
        h = torch.cat([k1, k2])
        ids = np.arange(8)
        a1, a2 = np.arange(4), np.arange(4, 8)
        allp = generate_pseudo_seeds(_Identity(), None, h, ids, a1, a2, 0, 1, cpu,
                                     metric="cosine", use_csls=False)
        assert allp == [(i, i + 4) for i in range(4)], allp
        half = generate_pseudo_seeds(_Identity(), None, h, ids, a1, a2, 0, 1, cpu,
                                     metric="cosine", use_csls=False, top_frac=0.5)
        assert half == [(0, 4), (1, 5)], half

        g = torch.Generator().manual_seed(1)
        base = torch.randn(60, 16, generator=g)
        e1 = base + 0.3 * torch.randn(60, 16, generator=g)
        e2 = base + 0.3 * torch.randn(60, 16, generator=g)
        sim = _compute_sim("cosine", e1, e2, cpu)
        sim = _apply_csls(sim, _csls_thr(sim, 5, cpu), _csls_thr(sim.t(), 5, cpu), cpu)
        b2, b1, rb = _chunked_mutual_nn("cosine", e1, e2, cpu, use_csls=True, csls_k=5,
                                        chunk=7, return_row_best=True)
        assert torch.equal(b2, sim.argmax(dim=1)) and torch.equal(b1, sim.argmax(dim=0))
        assert torch.allclose(rb, sim.max(dim=1).values, atol=1e-5), (rb - sim.max(dim=1).values).abs().max()

        h2 = torch.cat([e1, e2])
        ids2 = np.arange(120)
        dense = generate_pseudo_seeds(_Identity(), None, h2, ids2, np.arange(60), np.arange(60, 120), 0, 1, cpu,
                                      metric="cosine", use_csls=True, csls_k=5, top_frac=0.5)
        os.environ["DISTEA_EVAL_DENSE_LIMIT_GB"] = "1e-9"
        try:
            chunked = generate_pseudo_seeds(_Identity(), None, h2, ids2, np.arange(60), np.arange(60, 120), 0, 1,
                                            cpu, metric="cosine", use_csls=True, csls_k=5, top_frac=0.5)
        finally:
            del os.environ["DISTEA_EVAL_DENSE_LIMIT_GB"]
        full = generate_pseudo_seeds(_Identity(), None, h2, ids2, np.arange(60), np.arange(60, 120), 0, 1, cpu,
                                     metric="cosine", use_csls=True, csls_k=5)
        assert dense == chunked and set(dense) <= set(full), (dense, chunked)
        assert len(dense) == int(np.ceil(0.5 * len(full))), (len(dense), len(full))
        print(f"  [ok] top_frac: {len(dense)} of {len(full)} mutual pairs kept, dense == chunked")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_merge()
    test_candidates()
    test_refresh()
    test_top_frac()
    print("test_pseudo_seed_policy: ALL PASSED")
