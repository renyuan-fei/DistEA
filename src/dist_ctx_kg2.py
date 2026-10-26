"""
dist_ctx_kg2.py — KG2 Counterpart Halo.

Distributed training partitions KG2 disjointly across ranks
(counterpart_disc.assign_kg2_to_ranks): every KG2 entity belongs to exactly
one rank, with no overlap. models/gcn.py's alignment_loss samples negatives
via `torch.randint(kg1_size, N, ...)`, where N is bounded by the *local*
graph — so at world_size=4 each rank's negative-sampling pool is only ~25%
of the global KG2 set, yet eval.py's selective_eval ranks against the full
globally-gathered pool. The model never receives gradient signal telling it
"this cross-rank entity is not a match" for the ~75% of KG2 it never samples,
which is why quality collapses under partitioning even though the model's
weights are kept in sync every epoch (see dist_trainer.py's all_reduce).

This module selects, per rank, a small set of *additional* KG2 entities
("halo_kg2") that are structurally close to this rank's current alignment
anchors — mirroring ELsEA's CounterpartDiscovery (DivEA/divea/components.py):
multi-hop BFS from anchors through the KG2-KG2 graph, degree-aware scoring,
top-K selection. Unlike counterpart_disc.assign_kg2_to_ranks, overlap across
ranks is expected and desired: the same KG2 entity can be a halo_kg2 pick for
several ranks simultaneously if it is structurally relevant to each.

Serving follows LeapGNN's DistCacheClient philosophy (src/leapgnn/storage/
storage_dist.py): halo_kg2 entities are *not* added as graph nodes (no
message passing, no local KG2-side edges needed) — only their last-pushed
embedding vector is fetched on demand from the same file-based cache
framework.py already uses to refresh halo_kg1 rows. See framework.py's
`--use_halo_kg2` wiring for how selection (this module) and serving
(the existing cache_dir read/write path) fit together.
"""

from __future__ import annotations

from collections import defaultdict


def build_kg2_adjacency(kg2_triples: list[tuple[int, int, int]]) -> dict[int, set[int]]:
    """
    Undirected KG2-KG2 adjacency for BFS. Relation type is ignored — ELsEA's
    CounterpartDiscovery.get_neighbours() is likewise structure-only; the
    r2f/r2if edge *weights* already computed by graph_metrics.py encode
    relation-quality information elsewhere and aren't needed here.
    """
    adj: dict[int, set[int]] = defaultdict(set)
    for h, _r, t in kg2_triples:
        adj[h].add(t)
        adj[t].add(h)
    return adj


def select_halo_kg2(
    anchors: set[int],
    kg2_adj: dict[int, set[int]],
    node_weights2: dict[int, float],
    exclude: set[int],
    max_hop: int,
    top_k: int,
    gamma: float = 0.5,
) -> list[int]:
    """
    Select up to `top_k` KG2 global entity IDs structurally close to
    `anchors`, for use as extra (non-local) negative-sampling candidates.

    Args:
        anchors:       KG2 entity IDs already aligned (train_links, incl.
                       pseudo-seeds) to a KG1 entity local to this rank —
                       mirrors ELsEA's CounterpartDiscovery.get_anchors().
        kg2_adj:       output of build_kg2_adjacency(data.kg2_triples).
        node_weights2: Data.node_weights2 (in_degree + out_degree + 1,
                       populated by OpenEAData.load_weights()) — reused
                       as-is for the "prefer lower-degree, more
                       discriminative entities" weighting ELsEA applies
                       via `1 + 1/node_weights2[e]`.
        exclude:       KG2 entity IDs already local to this rank — no point
                       adding them again as "halo".
        max_hop:       BFS depth. ELsEA sizes this to 2*num_layers (a GCN's
                       receptive field), so halo_kg2 stays within the range
                       the model can actually reason about relationally —
                       though note halo_kg2 entities don't get graph edges
                       here (see module docstring), so this mainly bounds
                       *how far from an anchor* a candidate may be, not a
                       message-passing radius.
        top_k:         candidate budget for this rank (framework.py sizes
                       this via --halo_kg2_ratio × len(local_kg2)).
        gamma:         per-hop decay factor (closer hops score higher).

    Returns:
        Up to top_k KG2 global IDs, ranked highest-score first. Empty list
        if there are no anchors yet (e.g. before any train_links exist) or
        top_k <= 0 — callers should treat that as "no halo_kg2 this round"
        and fall back to local-only negative sampling.

    Note on overlap: this function only looks at its own `anchors` argument
    and never at what other ranks picked — the same entity can independently
    score well for multiple ranks and be selected by all of them. That's
    intentional (see module docstring); nothing here enforces exclusivity.
    """
    if not anchors or top_k <= 0:
        return []

    visited = set(anchors) | exclude
    frontier = set(anchors)
    scores: dict[int, float] = {}
    for d in range(1, max_hop + 1):
        next_frontier: set[int] = set()
        for e in frontier:
            for nb in kg2_adj.get(e, ()):
                if nb not in visited:
                    next_frontier.add(nb)
        for e in next_frontier:
            deg = node_weights2.get(e, 1)
            scores[e] = (gamma ** d) * (1.0 + 1.0 / max(deg, 1e-9))
        visited |= next_frontier
        if not next_frontier:
            break
        frontier = next_frontier

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [e for e, _score in ranked[:top_k]]
