"""
minibatch_eval.py — distributed batched inference for mini-batch training:
computes embeddings for a set of entity
ids via RREA.forward_blocks + cross-rank on-demand fetch (minibatch_ocd.py),
chunked into batches so no more than one batch's sampled frontier is ever
materialized at once — the same memory property training itself has,
unlike eval.py's distributed_eval/selective_eval which need a full
`model(g, h_init)` forward over this rank's whole local+halo graph.

Collective-safety: gather_minibatch_features (inside compute_minibatch_
embeddings) issues a dist.all_gather_object every call, so every rank
MUST call it the same number of times in the same order or the collective
desyncs (one rank blocks forever on a round another rank never issues).
Per-rank entity_ids lists can have different lengths (e.g. "this rank's
own local test-link entities"), so _synchronized_batch_count agrees on a
common batch count (all_reduce MAX) up front, and ranks that run out of
real data pad with an empty chunk (see gather_minibatch_features' own
docstring for why an empty outer_ids array still participates correctly).

Reuses eval.py's own similarity/CSLS/ranking helpers (_compute_sim,
_csls_thr, _apply_csls, _compute_metrics) rather than reimplementing them
— same metric definitions as the full-graph path, no second code path to
drift from it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from comm_stats import CommStats
from eval import (
    _apply_csls,
    _chunked_mutual_nn,
    _compute_sim,
    _csls_thr,
    _dense_limit_bytes,
    _eval_metrics_auto,
)
from minibatch_ocd import gather_minibatch_features
from rrea_minibatch import RREAMinibatchIndex, prepare_rrea_batch
from sparse_table import SparseEntityTable


def _synchronized_batch_count(
    n_local: int, batch_size: int, world_size: int, device: torch.device
) -> int:
    """Agree on one batch count across all ranks (all_reduce MAX) so every
    rank issues the same number of collective (cross-rank fetch) rounds,
    regardless of how many entities THIS rank happens to need embeddings
    for."""
    n_batches_local = max(1, (n_local + batch_size - 1) // batch_size) if n_local else 0
    if world_size <= 1:
        return n_batches_local
    t = torch.tensor([n_batches_local], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def compute_minibatch_embeddings(
    entity_ids: np.ndarray,
    model,
    index: RREAMinibatchIndex,
    table: SparseEntityTable,
    owner_map: dict[int, int],
    rank: int,
    world_size: int,
    device: torch.device,
    dim: int,
    num_layers: int,
    fanout: int | None,
    batch_size: int,
    comm_stats: CommStats | None = None,
    group=None,
) -> dict[int, np.ndarray]:
    """{global_id: embedding} for entity_ids on THIS rank, batched,
    no_grad. Collective — see module docstring; every rank must call this
    together, even with an empty/different-length entity_ids."""
    model.eval()
    emb_lookup: dict[int, np.ndarray] = {}
    n_batches = _synchronized_batch_count(len(entity_ids), batch_size, world_size, device)

    with torch.no_grad():
        for b in range(n_batches):
            chunk = entity_ids[b * batch_size : (b + 1) * batch_size] if len(entity_ids) else []
            if len(chunk):
                seed_ids = np.asarray(chunk, dtype=np.int64)
                batch = prepare_rrea_batch(seed_ids, index, num_layers=num_layers, fanout=fanout, device=device)
                outer_ids = batch.outer_ids
            else:
                batch = None
                outer_ids = np.zeros(0, dtype=np.int64)

            gathered = gather_minibatch_features(
                outer_ids, table, owner_map, rank, world_size, device, dim, comm_stats, group,
            )

            if batch is not None:
                emb = model.forward_blocks(batch, index, gathered.rows)
                for row, gid in enumerate(batch.seed_ids):
                    emb_lookup[int(gid)] = emb[row].cpu().numpy()

    return emb_lookup


def _gather_emb_dicts(
    local_emb: dict[int, np.ndarray], rank: int, world_size: int
) -> dict[int, np.ndarray]:
    """Merge every rank's {gid: emb} dict into one, identical on every
    rank. all_gather_object (not the tensor all_gather eval.py's
    _all_gather_embeddings uses) since per-rank dict sizes vary and ids
    are the natural key here -- same primitive dist_ocd.py already uses
    for its own variable-length request lists."""
    if world_size <= 1:
        return local_emb
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_emb)
    merged: dict[int, np.ndarray] = {}
    for d in gathered:
        merged.update(d)  # type: ignore[arg-type]
    return merged


def minibatch_selective_eval(
    model,
    index: RREAMinibatchIndex,
    table: SparseEntityTable,
    owner_map: dict[int, int],
    test_links: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
    dim: int,
    num_layers: int,
    fanout: int | None,
    batch_size: int,
    use_csls: bool = True,
    csls_k: int = 10,
    top_k: tuple[int, ...] = (1, 10),
    metric: str = "cosine",
    comm_stats: CommStats | None = None,
    group=None,
) -> dict[str, float] | None:
    """Mini-batch counterpart of eval.py's selective_eval: only computes
    embeddings for entities appearing in test_links (chunked, batched
    inference instead of one full-graph forward), then reuses eval.py's
    own _compute_sim/_compute_metrics for the actual ranking — same
    metric definitions as the full-graph path. Every rank computes the
    SAME global test-set embeddings (redundant per-rank inference buys a
    simple contract: no "rank 0 only" collection step, every rank agrees
    after the all_gather_object merge). NOTE test_links is NOT small
    relative to the full entity set under OpenEA's 3-7 split (70% of
    entities are test entities) — inference cost is bounded by batching
    either way, but the ranking below must never densify (see the
    comment before _eval_metrics_auto).

    Returns metrics dict identically on every rank (unlike distributed_eval/
    selective_eval, which return None on non-zero ranks) — simpler
    contract, and every rank already did equivalent work to produce it.
    """
    test_e1 = np.array(sorted({int(e1) for e1, _ in test_links}), dtype=np.int64)
    test_e2 = np.array(sorted({int(e2) for _, e2 in test_links}), dtype=np.int64)

    emb1_local = compute_minibatch_embeddings(
        test_e1, model, index, table, owner_map, rank, world_size, device,
        dim, num_layers, fanout, batch_size, comm_stats, group,
    )
    emb2_local = compute_minibatch_embeddings(
        test_e2, model, index, table, owner_map, rank, world_size, device,
        dim, num_layers, fanout, batch_size, comm_stats, group,
    )
    emb1_all = _gather_emb_dicts(emb1_local, rank, world_size)
    emb2_all = _gather_emb_dicts(emb2_local, rank, world_size)

    kg1_id2row = {gid: i for i, gid in enumerate(test_e1)}
    kg2_id2row = {gid: i for i, gid in enumerate(test_e2)}
    emb1 = torch.from_numpy(np.stack([emb1_all[int(g)] for g in test_e1]))
    emb2 = torch.from_numpy(np.stack([emb2_all[int(g)] for g in test_e2]))

    link_pairs = [
        (kg1_id2row[int(e1)], kg2_id2row[int(e2)])
        for e1, e2 in test_links
        if int(e1) in kg1_id2row and int(e2) in kg2_id2row
    ]

    # _eval_metrics_auto switches to the chunked path above the
    # dense-memory limit. Every rank runs this ranking, so a dense
    # 70000^2 fp32 sim at 100K scale (19.6 GB per rank, 4 ranks on a
    # 64 GB node) would swap-thrash the node.
    cpu = torch.device("cpu")
    return _eval_metrics_auto(
        metric, emb1, emb2, link_pairs, cpu, use_csls, csls_k, top_k,
    )


def minibatch_generate_pseudo_seeds(
    model,
    index: RREAMinibatchIndex,
    table: SparseEntityTable,
    owner_map: dict[int, int],
    local_kg1_ids: np.ndarray,
    local_kg2_ids: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
    dim: int,
    num_layers: int,
    fanout: int | None,
    batch_size: int,
    metric: str = "cosine",
    use_csls: bool = True,
    csls_k: int = 10,
) -> list[tuple[int, int]]:
    """Mini-batch counterpart of eval.py's generate_pseudo_seeds: mutual
    nearest-neighbour pseudo-seed mining over THIS rank's own local KG1 x
    all-KG2 (and vice versa) candidates, identically to that function's
    own algorithm (RREA/runner.py's iterative_training -- no threshold
    beyond mutual-NN agreement), just sourcing embeddings from batched
    mini-batch inference instead of one full-graph forward.

    Scope note: computes candidates from THIS rank's local_kg1_ids against
    the GLOBAL kg2 set gathered across ranks (kg2 is usually the smaller
    side at DistEA's target scale) -- a full N1 x N2 pairwise scan across
    the WHOLE graph is the same cost class eval.py's own generate_pseudo_
    seeds already pays for the full-graph path, not a new scaling concern
    this function introduces. Returns global (kg1_id, kg2_id) pairs,
    IDENTICAL on every rank (each rank mines its own local_kg1_ids' pairs,
    then all-gathers the result) -- matches generate_pseudo_seeds' own
    "all ranks receive the same list" contract.
    """
    # Every rank needs the FULL kg2 embedding set to rank against -- gather
    # each rank's own local_kg2_ids' embeddings once, merge, reuse for
    # every rank's own local_kg1_ids scan below (no repeated gathers).
    emb_kg2_local = compute_minibatch_embeddings(
        local_kg2_ids, model, index, table, owner_map, rank, world_size, device,
        dim, num_layers, fanout, batch_size,
    )
    emb_kg2_all = _gather_emb_dicts(emb_kg2_local, rank, world_size)
    kg2_ids = np.array(sorted(emb_kg2_all.keys()), dtype=np.int64)
    if len(kg2_ids) == 0:
        return []
    emb_kg2 = torch.from_numpy(np.stack([emb_kg2_all[int(g)] for g in kg2_ids]))

    emb_kg1_local = compute_minibatch_embeddings(
        local_kg1_ids, model, index, table, owner_map, rank, world_size, device,
        dim, num_layers, fanout, batch_size,
    )
    kg1_ids = np.array(sorted(emb_kg1_local.keys()), dtype=np.int64)

    seeds: list[tuple[int, int]] = []
    if len(kg1_ids) > 0:
        emb_kg1 = torch.from_numpy(np.stack([emb_kg1_local[int(g)] for g in kg1_ids]))
        cpu = torch.device("cpu")
        # Chunked mutual-NN above the dense limit -- same
        # per-rank dense-sim memory blowup as minibatch_selective_eval
        # (see its comment), just on the local-KG1 x global-KG2 shape.
        n1, n2 = emb_kg1.size(0), emb_kg2.size(0)
        if n1 * n2 * 4 > _dense_limit_bytes():
            best_kg2, best_kg1 = _chunked_mutual_nn(
                metric, emb_kg1, emb_kg2, cpu,
                use_csls=use_csls, csls_k=csls_k,
            )
        else:
            sim = _compute_sim(metric, emb_kg1, emb_kg2, cpu)
            if use_csls:
                row_thr = _csls_thr(sim, csls_k, cpu)
                col_thr = _csls_thr(sim.t(), csls_k, cpu)
                sim = _apply_csls(sim, row_thr, col_thr, cpu)
            best_kg2 = sim.argmax(dim=1)
            best_kg1 = sim.argmax(dim=0)
        for i, j in enumerate(best_kg2.tolist()):
            # Mutual-NN check needs kg2's #1 nearest KG1 restricted to THIS
            # rank's own local_kg1_ids scan, which only covers a subset of
            # the true global KG1 set -- best_kg1[j] is only meaningful
            # relative to kg1_ids (this rank's candidates), a real
            # approximation vs. the full-graph function's whole-KG1 scan.
            # Acceptable for a first version: still requires genuine
            # bidirectional agreement WITHIN this rank's own candidate
            # pool, not a threshold-free rubber stamp.
            if best_kg1[j].item() == i:
                seeds.append((int(kg1_ids[i]), int(kg2_ids[j])))

    gathered: list[object] = [None] * world_size if world_size > 1 else [seeds]
    if world_size > 1:
        dist.all_gather_object(gathered, seeds)
    merged: list[tuple[int, int]] = []
    for s in gathered:
        merged.extend(s)  # type: ignore[arg-type]
    return merged
