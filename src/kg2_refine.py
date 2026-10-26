"""
kg2_refine.py -- refine a KG2-to-rank
assignment so fewer alignment pairs straddle two ranks.

Multi-node DistEA loses most of its accuracy on pairs whose KG1 and KG2
entities are owned by different ranks (see the --eval_rank_split_dir
breakdown). The locality assignment
(counterpart_disc.assign_kg2_to_ranks_locality) decides every non-seed KG2
entity once, by plurality of whichever neighbours were already decided
when the BFS first reached it. This refinement starts from that result:

  1. KG1-side vote (kg1_vote=True): each non-seed KG2 entity scores ranks by
     its seed neighbours s2 -- each contributes the rank distribution of its
     train partner s1 and s1's KG1 neighbours, i.e. where the entity's own
     counterpart most likely sits. Ties and entities without seed
     neighbours keep their rank.
  2. `rounds` synchronous label-propagation rounds over the KG2-KG2 graph:
     each non-seed entity moves to the plurality rank of all its
     neighbours (ties keep the current rank).

Seeds (KG2 partners of train links) never move. Only train links are used.
Offline evaluation (scripts/eval_kg2_assignment.py, variant kg1vote_lp3):
cross-rank test pairs 100K N=4 10.96% -> 9.50%, DBP1M N=4 19.97% -> 17.00%,
N=8 28.25% -> 23.38%; KG2 edge cut down 17-25%; KG2 load max/mean up
0.03-0.11.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _adj(triples, n):
    t = np.asarray(list(triples), dtype=np.int64).reshape(-1, 3)
    r = np.concatenate([t[:, 0], t[:, 2]])
    c = np.concatenate([t[:, 2], t[:, 0]])
    a = sp.csr_matrix((np.ones(r.size, dtype=np.float32), (r, c)), shape=(n, n))
    a.data[:] = 1.0
    return a


def _onehot(ranks, idx, w, n):
    return sp.csr_matrix((np.ones(idx.size, dtype=np.float32), (idx, ranks[idx])), shape=(n, w))


def _argmax_keep(scores, cur, rng):
    s = np.asarray(scores.todense() if sp.issparse(scores) else scores, dtype=np.float64)
    best = s.max(axis=1)
    has = best > 0
    keep = s[np.arange(len(cur)), np.maximum(cur, 0)] >= best
    pick = (s + 1e-6 * rng.random(s.shape)).argmax(axis=1)
    return np.where(has & ~keep, pick, cur)


def refine_kg2_assignment(data, kg1_rank: np.ndarray, kg2_rank: np.ndarray, world_size: int,
                          rounds: int = 3, kg1_vote: bool = True, seed: int = 0) -> np.ndarray:
    """kg1_rank / kg2_rank: (n_global,) int arrays (-1 where not applicable).
    Returns a new (n_global,) KG2 rank array; KG1 entries are untouched."""
    kg1 = np.asarray(sorted(data.kg1_entities), dtype=np.int64)
    kg2 = np.asarray(sorted(data.kg2_entities), dtype=np.int64)
    n = int(kg2_rank.size)
    train = np.asarray(data.load_train_alignment(), dtype=np.int64).reshape(-1, 2)
    is_seed = np.zeros(n, dtype=bool)
    is_seed[train[:, 1]] = True
    nonseed = kg2[~is_seed[kg2]]
    a2 = _adj(data.kg2_triples, n)
    rng = np.random.default_rng(seed)
    cur = kg2_rank.copy()
    if kg1_vote:
        a1 = _adj(data.kg1_triples, n)
        a1i = (a1 + sp.identity(n, dtype=np.float32, format="csr")).tocsr()
        dist1 = a1i @ _onehot(kg1_rank, kg1, world_size, n)
        rowsum = np.asarray(dist1.sum(axis=1)).ravel()
        rowsum[rowsum == 0] = 1.0
        dist1 = sp.diags(1.0 / rowsum) @ dist1
        pmap = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train[:, 1], train[:, 0])), shape=(n, n))
        score = a2[nonseed] @ (pmap @ dist1)
        cur[nonseed] = _argmax_keep(score, cur[nonseed], rng)
    for _ in range(rounds):
        votes = a2 @ _onehot(cur, kg2, world_size, n)
        cur[nonseed] = _argmax_keep(votes[nonseed], cur[nonseed], rng)
    return cur
