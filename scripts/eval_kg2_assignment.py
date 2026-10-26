#!/usr/bin/env python
"""
eval_kg2_assignment.py -- offline quality check of KG2-to-rank assignments
(no training).

Reads:  the dataset/fold (OpenEAData) and a partition directory
        ({r}.npy KG1 partitions + kg2_assignments.json).
Writes: nothing; prints one line per assignment variant.

Multi-node DistEA loses almost all of its accuracy on pairs whose KG1 and KG2
entities land on different ranks (e.g. EN_DE_100K at N=2: cross-rank Hit@1
0.017 vs same-rank 0.551), so the share of such pairs is a direct target for
the KG2 assignment. For one KG1 partition ({r}.npy) this reports, per
variant: the cross-rank share of VALID pairs (selection target) and TEST
pairs (report only), KG2 load per rank (max / mean), and the share of
KG2-KG2 edges cut by the assignment (drives halo_kg2 size, i.e. speed).

Variants, all seeded by the fold's TRAIN links only (KG2 partner of a train
link stays on its KG1 partner's rank):
  stored        the partition's kg2_assignments.json (BFS-layered plurality vote)
  lp<k>         stored + k synchronous rounds of label propagation: each
                non-seed KG2 entity moves to the plurality rank of ALL its KG2
                neighbours (ties keep the current rank)
  lp<k>_sw<w>   the same with seed neighbours' votes weighted w
  kg1vote       each non-seed KG2 entity votes with its seed neighbours: seed
                s2 (partner s1) contributes the rank distribution of s1 and
                s1's KG1 neighbours (where s2's neighbourhood counterpart most
                likely lives); entities without seed neighbours keep stored
  kg1vote_lp<k> kg1vote followed by k label-propagation rounds

Usage:
    python scripts/eval_kg2_assignment.py \
        --data_dir data/OpenEA_dataset_v2.0/EN_DE_100K_V1 \
        --fold_dir data/OpenEA_dataset_v2.0/EN_DE_100K_V1/721_5fold/1 \
        --partition_dir outputs/partitions/EN_DE_100K_V1/721_5fold_1/dist_True/4_metis_kg2locality
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataload import OpenEAData  # noqa: E402


def _adj(triples, n):
    t = np.asarray(list(triples), dtype=np.int64).reshape(-1, 3)
    r = np.concatenate([t[:, 0], t[:, 2]])
    c = np.concatenate([t[:, 2], t[:, 0]])
    a = sp.csr_matrix((np.ones(r.size, dtype=np.float32), (r, c)), shape=(n, n))
    a.data[:] = 1.0  # binary: multiplicity collapsed
    return a


def _onehot(ranks, idx, w, n):
    m = sp.csr_matrix((np.ones(idx.size, dtype=np.float32), (idx, ranks[idx])), shape=(n, w))
    return m


def _argmax_keep(scores, cur, rng):
    """Row-wise argmax with ties broken toward the current rank, then at random."""
    s = np.asarray(scores.todense() if sp.issparse(scores) else scores, dtype=np.float64)
    best = s.max(axis=1)
    has = best > 0
    keep = s[np.arange(len(cur)), np.maximum(cur, 0)] >= best
    jit = s + 1e-6 * rng.random(s.shape)
    pick = jit.argmax(axis=1)
    out = np.where(has & ~keep, pick, cur)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument("--partition_dir", required=True)
    p.add_argument("--output_base", default="outputs/partitions")
    p.add_argument("--lp_rounds", default="1,3,10")
    p.add_argument("--seed_weights", default="3")
    a = p.parse_args()
    t0 = time.time()
    out_dir = os.path.join(a.output_base, os.path.basename(os.path.normpath(a.data_dir)))
    data = OpenEAData(a.data_dir, a.fold_dir, out_dir)
    kg1 = np.asarray(sorted(data.kg1_entities), dtype=np.int64)
    kg2 = np.asarray(sorted(data.kg2_entities), dtype=np.int64)
    n = int(max(kg1.max(), kg2.max())) + 1
    files = sorted(glob.glob(os.path.join(a.partition_dir, "[0-9]*.npy")),
                   key=lambda x: int(os.path.basename(x).split(".")[0]))
    w = len(files)
    own1 = np.full(n, -1, dtype=np.int64)
    for r, fn in enumerate(files):
        own1[np.load(fn).astype(np.int64)] = r
    with open(os.path.join(a.partition_dir, "kg2_assignments.json")) as f:
        assign = json.load(f)
    stored = np.full(n, -1, dtype=np.int64)
    for r, ids in assign.items():
        stored[np.asarray(ids, dtype=np.int64)] = int(r)
    train = np.asarray(data.load_train_alignment(), dtype=np.int64).reshape(-1, 2)
    valid = np.asarray(data.load_valid_alignment(), dtype=np.int64).reshape(-1, 2)
    test = np.asarray(data.load_test_alignment(), dtype=np.int64).reshape(-1, 2)
    a1 = _adj(data.kg1_triples, n)
    a2 = _adj(data.kg2_triples, n)
    seeds2 = np.unique(train[:, 1])
    is_seed = np.zeros(n, dtype=bool)
    is_seed[seeds2] = True
    seed_rank = np.full(n, -1, dtype=np.int64)
    seed_rank[train[:, 1]] = own1[train[:, 0]]
    nonseed = kg2[~is_seed[kg2]]
    rng = np.random.default_rng(0)
    e2 = sp.coo_matrix(a2)
    m2 = e2.row < e2.col
    eu, ev = e2.row[m2], e2.col[m2]
    print(f"[kg2eval] {a.partition_dir}: ranks={w} KG1={kg1.size} KG2={kg2.size} train={len(train)} "
          f"valid={len(valid)} test={len(test)} KG2 edges={eu.size} load {time.time() - t0:.0f}s", flush=True)

    def report(name, asg):
        cv = float(np.mean(own1[valid[:, 0]] != asg[valid[:, 1]])) if len(valid) else float("nan")
        ct = float(np.mean(own1[test[:, 0]] != asg[test[:, 1]]))
        load = np.bincount(asg[kg2], minlength=w)
        cut = float(np.mean(asg[eu] != asg[ev])) if eu.size else 0.0
        moved = float(np.mean(asg[kg2] != stored[kg2]))
        print(f"  {name:16s} cross valid {cv:.4f}  test {ct:.4f} | KG2 load max/mean {load.max() / load.mean():.2f} "
              f"min {load.min()} | KG2 edge cut {cut:.4f} | moved vs stored {moved:.4f}", flush=True)

    def lp(start, rounds, seed_w=1.0):
        cur = start.copy()
        wt = np.where(is_seed, seed_w, 1.0).astype(np.float32)
        for _ in range(rounds):
            oh = _onehot(cur, kg2, w, n)
            votes = a2 @ sp.diags(wt) @ oh
            new = _argmax_keep(votes[nonseed], cur[nonseed], rng)
            cur[nonseed] = new
        return cur

    report("stored", stored)
    rounds = [int(x) for x in a.lp_rounds.split(",") if x]
    sws = [float(x) for x in a.seed_weights.split(",") if x]
    for k in rounds:
        report(f"lp{k}", lp(stored, k))
        for sw in sws:
            report(f"lp{k}_sw{sw:g}", lp(stored, k, sw))
    # kg1vote: seed s2 -> rank distribution of {s1} U N_KG1(s1)
    a1i = (a1 + sp.identity(n, dtype=np.float32, format="csr")).tocsr()
    oh1 = _onehot(own1, kg1, w, n)
    dist1 = a1i @ oh1                       # (n, w): rank counts of each KG1 node's closed neighbourhood
    rowsum = np.asarray(dist1.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    dist1 = sp.diags(1.0 / rowsum) @ dist1  # rank distribution
    pmap = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train[:, 1], train[:, 0])), shape=(n, n))
    seed_dist = pmap @ dist1                 # (n, w): for KG2 seeds, partner's neighbourhood distribution
    score = a2[nonseed] @ seed_dist           # sum over seed neighbours
    kv = stored.copy()
    kv[nonseed] = _argmax_keep(score, stored[nonseed], rng)
    report("kg1vote", kv)
    for k in rounds:
        report(f"kg1vote_lp{k}", lp(kv, k))
    print(f"[kg2eval] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
