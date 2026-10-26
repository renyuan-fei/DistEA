"""
minibatch_sampler.py — full-neighborhood multi-hop sampling for mini-batch
training ("Strategy A": full neighborhoods; optional fanout cap = "Strategy B").

Pure functions only: no torch.distributed, no model, no GPU tensors. Takes
the same triples/edge-weight structures dist_graph.py already uses (so the
edge-weight convention — r2if head->tail, r2f tail->head, each floored at
0.3 — is identical to the full-graph path, not a separate reimplementation
that could silently drift from it) and produces:

  1. build_neighbor_index(): {entity_id: [(neighbor_id, edge_weight), ...]},
     built ONCE per dataset/partition from triples. Every other function
     here takes this precomputed index rather than raw triples --
     rebuilding it per call would make every mini-batch re-scan the ENTIRE
     triple set from scratch, which dominates the cost at real scale.
  2. compute_weighted_degrees(): each entity's TRUE weighted degree over its
     COMPLETE neighbor set (Â = A + I row-sum). This matters specifically
     because GCNLayer._sparse_forward's symmetric normalisation
     (D̂^-1/2 Â D̂^-1/2) needs each node's real degree — recomputing
     "degree" from only a sampled/partial neighborhood would silently
     diverge from what full-graph training computes for the same node,
     corrupting the exact numerical correctness the
     forward()-vs-forward_blocks() parity test requires. Precomputing once,
     structure-only (no embeddings involved), sidesteps this entirely — the
     same pattern the VBIP computation uses.
  3. sample_multi_hop(): given a batch of seed global IDs, expand outward
     num_layers hops, taking each frontier node's COMPLETE neighbor set (no
     fanout cap by default; see sample_multi_hop). Returns layers
     ordered OUTERMOST-first (layers[0] = the outermost hop, layers[-1] =
     the seeds themselves), matching the order forward_blocks() consumes
     them in (propagate from outermost inward).
  4. build_block(): turn one (outer_ids, inner_ids) hop into the sparse
     normalised bipartite adjacency GCNLayer-style propagation needs,
     reusing the precomputed degrees from (2) instead of recomputing them
     from the block's own (necessarily partial) edge set.
"""

from __future__ import annotations

import numpy as np
import torch

NeighborIndex = dict[int, list[tuple[int, float]]]


def _direction_weight(r: int, edge_weights: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """(w_fwd, w_bwd) for a triple's relation r, matching dist_graph.py's
    add_edge exactly: r2if head->tail, r2f tail->head, floored at 0.3."""
    r2f, r2if = edge_weights.get(r, (1.0, 1.0))
    return max(r2if, 0.3), max(r2f, 0.3)


def build_neighbor_index(
    triples: list[tuple[int, int, int]], edge_weights: dict[int, tuple[float, float]]
) -> NeighborIndex:
    """{entity_id: [(neighbor_id, edge_weight), ...]} — undirected (both
    directions of every triple recorded, each with its own directional
    weight), matching dist_graph.py's add_edge which always adds both
    (h->t, w_fwd) and (t->h, w_bwd). Build once per dataset/partition; every
    other function in this module takes the result, not raw triples."""
    nbrs: NeighborIndex = {}
    for h, r, t in triples:
        w_fwd, w_bwd = _direction_weight(r, edge_weights)
        nbrs.setdefault(h, []).append((t, w_fwd))
        nbrs.setdefault(t, []).append((h, w_bwd))
    return nbrs


def compute_weighted_degrees(neighbor_index: NeighborIndex) -> dict[int, float]:
    """deg(v) = self-loop weight (1.0) + sum of incident edge weights over
    v's COMPLETE neighbor set."""
    return {eid: 1.0 + sum(w for _, w in edges) for eid, edges in neighbor_index.items()}


def sample_multi_hop(
    seed_ids: np.ndarray,
    neighbor_index: NeighborIndex,
    num_layers: int,
    fanout: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Expand outward from seed_ids for num_layers hops. A node already seen
    at a closer hop is not re-added at a farther one (each entity appears
    in exactly one layer, its closest-to-seed occurrence) — this keeps
    block_construction's per-hop node sets disjoint, matching NodeFlow's
    own layer semantics.

    fanout=None (default, Strategy A): take each frontier node's COMPLETE
    neighbor set at every hop, no cap. On IDS15K this saturates 95.7% of a
    30K-entity graph within the first batches and runs ~238x slower per
    epoch than full-graph training (28.5s vs 0.12s per epoch) -- Strategy A
    saves neither memory nor time on graphs with realistic hub/small-world
    density. Kept as the default
    only for the correctness tests that don't care about scale (e.g.
    test_forward_blocks_matches_forward_when_unrestricted, which
    deliberately wants "no sampling at all").

    fanout=K (Strategy B, GraphSAGE/LeapGNN-AE's `expand_factor`
    convention, see LeapGNN-AE's dgl_default_dissampler.py): cap each frontier node's contribution to at most K randomly
    -chosen neighbors per hop. No importance-reweighting is applied for
    under-sampled nodes (matches LeapGNN-AE's own dgl.contrib.sampling
    usage, which doesn't rescale either) -- the aggregation becomes a
    biased-but-consistent estimate of the true full aggregation, the
    standard GraphSAGE-style tradeoff.

    Returns layers ordered OUTERMOST-first: layers[0] is the num_layers-hop
    frontier, layers[-1] is exactly seed_ids (the innermost/seed layer) —
    forward_blocks() consumes them in this order, propagating outer→inner.
    """
    if rng is None:
        rng = np.random.default_rng()

    seen = {int(s) for s in seed_ids}
    frontier = set(seen)
    layers_inner_first: list[np.ndarray] = [np.array(sorted(seen), dtype=np.int64)]

    for _ in range(num_layers):
        next_frontier: set = set()
        for eid in frontier:
            candidates = [nbr_id for nbr_id, _w in neighbor_index.get(eid, []) if nbr_id not in seen]
            if fanout is not None and len(candidates) > fanout:
                idx = rng.choice(len(candidates), size=fanout, replace=False)
                candidates = [candidates[i] for i in idx]
            next_frontier.update(candidates)
        seen |= next_frontier
        layers_inner_first.append(np.array(sorted(next_frontier), dtype=np.int64))
        frontier = next_frontier
        if not frontier:
            # No more neighbors to expand (isolated component) — remaining
            # hops are empty; stop early rather than appending empty arrays
            # forever (loop would no-op anyway, but this is clearer).
            break

    return list(reversed(layers_inner_first))


def build_block(
    outer_ids: np.ndarray,
    inner_ids: np.ndarray,
    neighbor_index: NeighborIndex,
    degrees: dict[int, float],
    device: torch.device,
    reweight: bool = False,
) -> Block:
    """
    Build the normalised bipartite adjacency for one hop: aggregate FROM
    outer_ids (this hop's source/neighbor set, plus inner_ids itself for
    the self-loop term) TO inner_ids (this hop's destination/target set,
    one hop closer to the seeds).

    Edge weight formula matches GCNLayer._sparse_forward exactly:
        w_combined(u, v) = deg(u)^-0.5 * deg(v)^-0.5 * w(u, v)
    using the PRECOMPUTED, TRUE-degree `degrees` dict (see module
    docstring for why this must not be recomputed from the block's own
    partial edge set).

    reweight: when fanout-capped sampling drops some of v's true neighbors
    from outer_ids, `degrees[v]` still reflects v's TRUE full degree (by
    design -- see module docstring), but the sum below only has terms for
    whichever true neighbors happen to survive into outer_ids. Since the
    normalisation denominator (dv**-0.5) is calibrated assuming ALL of v's
    edges are present, summing over a strict subset silently under-scales
    v's aggregated feature -- the "no importance-reweighting" bias, a
    candidate (unconfirmed) contributor to the mini-batch-vs-full-graph
    accuracy gap. When True, rescale each v's non-self-loop terms by
    (true neighbor count / surviving neighbor count) -- a standard
    Horvitz-Thompson-style correction (same family as FastGCN/VRGCN's
    importance sampling) making the summed term an unbiased estimator of
    the true full-neighborhood sum in expectation over the sampling
    procedure. Default False keeps the unscaled behaviour the existing
    tests check.
    """
    outer_pos = {int(g): i for i, g in enumerate(outer_ids)}
    inner_pos = {int(g): i for i, g in enumerate(inner_ids)}

    src_idx: list[int] = []
    dst_idx: list[int] = []
    vals: list[float] = []

    for v in inner_ids:
        v = int(v)
        dv = degrees.get(v, 1.0)
        # Self-loop term (weight 1.0, matching  = A + I) -- never rescaled,
        # it isn't subject to fanout sampling.
        if v in outer_pos:
            src_idx.append(outer_pos[v])
            dst_idx.append(inner_pos[v])
            vals.append((dv ** -0.5) * (dv ** -0.5) * 1.0)

        true_neighbors = neighbor_index.get(v, [])
        v_start = len(vals)
        for nbr_id, w in true_neighbors:
            u = outer_pos.get(nbr_id)
            if u is None:
                continue
            du = degrees.get(nbr_id, 1.0)
            src_idx.append(u)
            dst_idx.append(inner_pos[v])
            vals.append((du ** -0.5) * (dv ** -0.5) * w)

        if reweight and true_neighbors:
            surviving = len(vals) - v_start
            if 0 < surviving < len(true_neighbors):
                scale = len(true_neighbors) / surviving
                for i in range(v_start, len(vals)):
                    vals[i] *= scale

    indices = torch.tensor([src_idx, dst_idx], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        indices, values, size=(len(outer_ids), len(inner_ids)), device=device
    ).coalesce()

    return Block(outer_ids=outer_ids, inner_ids=inner_ids, adj_t=adj)


class Block:
    """One hop's bipartite computation graph: `adj_t` is (len(outer_ids),
    len(inner_ids)) sparse, normalised so that
    `torch.sparse.mm(adj_t.t(), outer_features)` produces the aggregated
    (inner_ids-ordered) output — mirrors GCNLayer._sparse_forward's
    `torch.sparse.mm(a_sym, h)` but rectangular (outer -> inner) instead of
    square (whole graph -> itself)."""

    __slots__ = ("outer_ids", "inner_ids", "adj_t")

    def __init__(self, outer_ids: np.ndarray, inner_ids: np.ndarray, adj_t: torch.Tensor):
        self.outer_ids = outer_ids
        self.inner_ids = inner_ids
        self.adj_t = adj_t
