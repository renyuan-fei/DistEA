"""
counterpart_disc.py — assign each KG2 entity to a rank.

Two strategies, both writing the same kg2_assignments.json shape
(`{"0": [kg2_id, ...], "1": [...], ...}`):

  - assign_kg2_to_ranks() (round-robin, the ablation baseline): KG2
    entities that appear in train_links follow their paired KG1 entity's
    rank; unlinked KG2 entities are distributed round-robin for load
    balance, ignoring KG2-KG2 graph structure entirely.

  - assign_kg2_to_ranks_locality() (the default): same train-linked
    seeding, but unlinked entities are placed by BFS-layered majority vote
    over the KG2-KG2 graph instead of blind round-robin — see its own
    docstring. Motivation: with a 30% training split, ~70% of KG2 entities
    are "unlinked" (not in train_links) and round-robin placement puts
    47%/70% of KG2-KG2 edges across a partition boundary at world_size=2/4.
    halo_kg2 (dist_graph.py) recovers the message passing over those
    edges; locality assignment removes most of the cut in the first place.
"""

import json
import logging
import os
from collections import defaultdict

from dist_ctx_kg2 import build_kg2_adjacency


def _write_kg2_assignments(result: dict[int, list[int]], output_dir: str) -> None:
    """Shared by both assignment strategies -- identical kg2_assignments.json
    shape either way, so downstream (dist_graph.py, framework.py) never
    needs to know or care which strategy produced it."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "kg2_assignments.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in result.items()}, f)
    logging.info(f"[counterpart_disc] kg2_assignments written to {out_path}")


def assign_kg2_to_ranks(data, partitions, output_dir):
    """
    Assign KG2 entities to ranks.

    Args:
        data:        OpenEAData (provides train_links, kg2_entities)
        partitions:  list of sets/lists, partitions[rank] = {kg1_entity_id, ...}
        output_dir:  directory where kg2_assignments.json will be written

    Returns:
        dict mapping rank (int) → list of kg2 entity IDs
    """
    # Build kg1_entity → rank lookup
    ent_to_rank = {}
    for rank, part in enumerate(partitions):
        for eid in part:
            ent_to_rank[eid] = rank

    # Train-link KG2 entities follow their KG1 partner
    linked_kg2 = set()
    assignments = defaultdict(list)
    for kg1_id, kg2_id in data.load_train_alignment():
        rank = ent_to_rank.get(kg1_id, 0)
        assignments[rank].append(kg2_id)
        linked_kg2.add(kg2_id)

    # Unlinked KG2 entities distributed round-robin
    world_size = len(partitions)
    unlinked = [e for e in data.kg2_entities if e not in linked_kg2]
    for i, kg2_id in enumerate(sorted(unlinked)):
        assignments[i % world_size].append(kg2_id)

    # Convert to plain dict with int keys
    result = {rank: assignments[rank] for rank in range(world_size)}
    _write_kg2_assignments(result, output_dir)
    return result


def assign_kg2_to_ranks_locality(data, partitions, output_dir, seed_links=None):
    """
    Locality-aware alternative to assign_kg2_to_ranks(): train-linked
    entities seed the assignment exactly as there, but every other ("unlinked") KG2 entity is placed by
    BFS-layered majority vote over the KG2-KG2 graph instead of blind
    round-robin.

    `seed_links`: optional list of (kg1_id, kg2_id) pairs to seed from,
    instead of `data.load_train_alignment()` (used when `None`). Passing an
    expanded list (e.g. the fold's train_links plus pseudo-seeds
    accumulated during training) lets a re-run mid-training use a
    richer seed set than what was available at partition time -- the BFS
    layering and vote logic below are unchanged either way.

    Algorithm:
      1. Seed: train-linked KG2 entities follow their KG1 partner's rank
         (identical to assign_kg2_to_ranks).
      2. Repeatedly expand one BFS layer at a time from the current set of
         decided entities through the (undirected) KG2-KG2 adjacency
         (dist_ctx_kg2.build_kg2_adjacency). All entities newly reached in
         the same layer are decided using the frontier's state at the
         START of that layer (not one-by-one as encountered), so decisions
         within a layer don't depend on iteration order. Each is assigned
         to the rank holding a plurality of its already-decided neighbours;
         ties go to whichever tied rank currently has fewer entities
         assigned (a running load counter), for balance.
      3. Any entity never reached this way (its KG2-KG2 component has no
         train-linked seed at all) falls back to whichever rank currently
         has the fewest entities assigned -- same load-balancing intent as
         plain round-robin, just informed by actual running load instead of
         a blind positional index.

    Args:
        data:        OpenEAData (provides train_links, kg2_entities, kg2_triples)
        partitions:  list of sets/lists, partitions[rank] = {kg1_entity_id, ...}
        output_dir:  directory where kg2_assignments.json will be written
        seed_links:  optional list of (kg1_id, kg2_id) pairs; defaults to
                     data.load_train_alignment() when None

    Returns:
        dict mapping rank (int) → list of kg2 entity IDs (same shape as
        assign_kg2_to_ranks -- drop-in compatible with everything downstream).
    """
    world_size = len(partitions)
    ent_to_rank = {}
    for rank, part in enumerate(partitions):
        for eid in part:
            ent_to_rank[eid] = rank

    kg2_rank: dict[int, int] = {}
    rank_load = [0] * world_size
    links = data.load_train_alignment() if seed_links is None else seed_links
    for kg1_id, kg2_id in links:
        if kg2_id in kg2_rank:
            continue  # first mapping wins if train_links ever double-maps a kg2_id
        rank = ent_to_rank.get(kg1_id, 0)
        kg2_rank[kg2_id] = rank
        rank_load[rank] += 1

    kg2_adj = build_kg2_adjacency(data.kg2_triples)
    unassigned = set(data.kg2_entities) - set(kg2_rank.keys())

    frontier = set(kg2_rank.keys())
    while frontier and unassigned:
        candidates: set = set()
        for e in frontier:
            for nb in kg2_adj.get(e, ()):
                if nb in unassigned:
                    candidates.add(nb)
        if not candidates:
            break

        decisions: dict[int, int] = {}
        for e in candidates:
            votes: dict[int, int] = defaultdict(int)
            for nb in kg2_adj.get(e, ()):
                if nb in kg2_rank:
                    votes[kg2_rank[nb]] += 1
            if not votes:
                continue
            max_votes = max(votes.values())
            tied = [r for r, v in votes.items() if v == max_votes]
            decisions[e] = min(tied, key=lambda r: rank_load[r])

        for e, r in decisions.items():
            kg2_rank[e] = r
            rank_load[r] += 1
            unassigned.discard(e)
        frontier = set(decisions.keys())

    # Fallback: no train-linked seed reachable at all (isolated KG2-KG2
    # component, or no KG2-KG2 edges for this entity in the first place).
    for kg2_id in sorted(unassigned):
        r = min(range(world_size), key=lambda x: rank_load[x])
        kg2_rank[kg2_id] = r
        rank_load[r] += 1

    assignments = defaultdict(list)
    for kg2_id, r in kg2_rank.items():
        assignments[r].append(kg2_id)
    result = {rank: assignments[rank] for rank in range(world_size)}
    _write_kg2_assignments(result, output_dir)
    return result


def compute_kg2_locality_edge_weights(data, bonus: int = 5) -> dict:
    """
    KG1-KG1 edge weights for METIS (partition.py's --kg1_edge_weight),
    computed entirely from information available
    BEFORE any KG1 partition exists -- avoids the chicken-and-egg problem of
    needing a KG2 counterpart for the ~70% of KG1 entities that aren't
    train-linked. Only edges whose BOTH endpoints are train-linked get a
    non-default weight: `1 + bonus` if their KG2 counterparts fall in the
    same weakly-connected component of the (static) KG2-KG2 adjacency graph,
    else the unweighted default `1`. Every other edge also gets the
    default -- no signal exists for them, and inventing one would depend on
    a partition result that doesn't exist yet.

    Weights are plain Python ints, not floats: METIS's underlying C API
    (modules/networkx-metis/nxmetis/_metis.pyx) marshals edge weights into
    an `idx_t*` array -- passing floats raises an opaque
    `TypeError: 'TypeError' object is not subscriptable` from deep inside
    nxmetis's exception-translation wrapper.

    Returns a dict keyed by `tuple(sorted((h, t)))` (order-independent,
    since a KG1-KG1 edge can appear as (h, t) or (t, h) across different
    relations in kg1_triples) -- callers must normalize their own lookup key
    the same way (see partition.py's build_kg1_graph).
    """
    kg2_adj = build_kg2_adjacency(data.kg2_triples)

    # Weakly-connected components over ALL kg2 entities, including ones
    # with no KG2-KG2 edges at all (each its own singleton component).
    component: dict[int, int] = {}
    next_id = 0
    for start in data.kg2_entities:
        if start in component:
            continue
        comp_id = next_id
        next_id += 1
        stack = [start]
        component[start] = comp_id
        while stack:
            e = stack.pop()
            for nb in kg2_adj.get(e, ()):
                if nb not in component:
                    component[nb] = comp_id
                    stack.append(nb)

    kg1_to_kg2: dict[int, int] = {}
    for kg1_id, kg2_id in data.load_train_alignment():
        kg1_to_kg2.setdefault(kg1_id, kg2_id)

    weights: dict = {}
    for h, _r, t in data.kg1_triples:
        if h == t:
            continue
        h2 = kg1_to_kg2.get(h)
        t2 = kg1_to_kg2.get(t)
        if h2 is None or t2 is None:
            continue
        if component.get(h2) == component.get(t2):
            weights[tuple(sorted((h, t)))] = 1 + bonus

    return weights
