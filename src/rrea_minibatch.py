"""
rrea_minibatch.py — RREA-specific mini-batch block construction (the
blocks consumed by RREA.forward_blocks).

Why RREA needs its own module instead of reusing minibatch_sampler.py's
build_neighbor_index/build_block (GCN's path):

1. RREA's pre-aggregation step (`ent_feature = adj_self_norm @ h_init`,
   models/rrea.py) uses a STRUCTURAL (relation-multiplicity, coalesce-
   summed) edge weight -- dist_graph.py's build_rrea_graph builds
   `adj_self_norm` from unit-weight (1.0) entries per (pair, relation)
   that get SUMMED when duplicate (lh, lt) index pairs coalesce, i.e. an
   edge's weight is "how many distinct relations connect this pair", NOT
   minibatch_sampler.py's r2f/r2if-weighted convention (which is GCN-
   Align's own edge-weight scheme, unrelated to RREA's). Reusing
   build_block here would silently apply the wrong weighting.
2. RREA's NRGraphAttention message-passing step needs the RAW,
   relation-tagged edge list (src, dst, rel_id, rel_val) for its
   per-edge reflection + attention math -- this cannot be collapsed into
   a single pre-normalised sparse adjacency the way GCN's symmetric
   propagation can (build_block's Block class only carries `adj_t`, no
   per-edge relation information).
3. RREA's attention step needs each edge's INNER-role ("self") entity's
   OWN previous-hop feature to compute attention logits
   (`att_in = cat([selfs, neighs_reflected, rel_vec])`) -- this is a real
   input, not an optional additive term the way GCN's self-loop is, so
   disjoint sample_multi_hop layers (GCN's block convention: outer_ids
   and inner_ids never overlap) cannot supply it. This module uses NESTED
   "active" sets instead (active[i] = union of sample_multi_hop's
   layers[i:]) so every hop's inner_ids is a SUBSET of that hop's
   outer_ids, making each inner entity's own prior feature available by
   simple lookup into the outer-indexed feature tensor.

Mirrors dist_graph.py's build_rrea_graph formulas exactly (same
structural-degree weighting, same directional relation-id doubling, same
rel_profile row-scaling), restricted to a sampled node set instead of the
whole partition -- see each function's docstring for the specific
full-graph line it replicates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from minibatch_csr import (
    CSRIndex,
    build_csr_index,
    build_rrea_attention_block_csr,
    build_rrea_self_block_csr,
    compute_rel_profile_features_csr,
    nested_active_sets_np,
    resolve_sampler_backend,
    sample_multi_hop_csr,
    sample_multi_hop_dgl,
)
from minibatch_sampler import Block, sample_multi_hop

RelNeighborIndex = dict[int, list[tuple[int, int, float]]]
RelProfileIndex = dict[int, dict[int, int]]
AttentionEdges = tuple[list[int], list[int], list[int], list[float]]

# Module-level singleton so prepare_rrea_batch's default is not a function
# call evaluated in the signature (ruff B008); torch.device is immutable.
_CPU_DEVICE = torch.device("cpu")


def build_rrea_relation_index(
    triples: list[tuple[int, int, int]], rel_num: int
) -> RelNeighborIndex:
    """{entity_id: [(neighbor_id, directional_rel_id, edge_val), ...]} --
    replicates dist_graph.py's build_rrea_graph pair_rels construction
    exactly: grouped by DIRECTED (h, t) pair, forward direction keeps the
    raw relation id, reverse direction uses id + rel_num (a separate
    learnable vector, matching RREA's directional relation convention),
    edge_val = 1/(relation count on that specific directed pair). Both
    directions of every triple are inserted as separate dict entries, so
    index[v] already covers v's COMPLETE (undirected) neighbor set --
    needed both for the self-block (structural degree/weight) and the
    attention block (per-edge relation info)."""
    pair_rels: dict[tuple[int, int], list[int]] = {}
    for h, r, t in triples:
        pair_rels.setdefault((h, t), []).append(r)
        pair_rels.setdefault((t, h), []).append(r + rel_num)

    index: RelNeighborIndex = {}
    for (h, t), rels in pair_rels.items():
        w = 1.0 / len(rels)
        for r in rels:
            index.setdefault(h, []).append((t, r, w))
    return index


def build_rel_profile_index(
    triples: list[tuple[int, int, int]]
) -> tuple[RelProfileIndex, RelProfileIndex]:
    """(rel_out, rel_in): {entity_id: {raw_relation_id: count}} -- RAW,
    undoubled relation ids (unlike build_rrea_relation_index's directional
    ones), matching build_rrea_graph's rel_out[lh, r] += 1 / rel_in[lt, r]
    += 1 exactly. Feeds rel_feature's per-entity relation-profile, computed
    directly from an entity's own incident triples -- no neighbor
    expansion/sampling needed, unlike ent_feature."""
    rel_out: RelProfileIndex = {}
    rel_in: RelProfileIndex = {}
    for h, r, t in triples:
        rel_out.setdefault(h, {})
        rel_out[h][r] = rel_out[h].get(r, 0) + 1
        rel_in.setdefault(t, {})
        rel_in[t][r] = rel_in[t].get(r, 0) + 1
    return rel_out, rel_in


def compute_rrea_self_degrees(relation_index: RelNeighborIndex) -> dict[int, float]:
    """deg(v) = 1.0 (self-loop) + total relation-arc count incident to v --
    matches build_rrea_graph's adj_self_norm degree exactly: `a_hat`'s
    self-loop (weight 1) plus one unit-weight term per (pair, relation)
    entry that survives coalesce-summing, which is exactly
    len(relation_index[v]) since each entry there is already one such
    term (see build_rrea_relation_index)."""
    return {v: 1.0 + len(edges) for v, edges in relation_index.items()}


def build_rrea_self_block(
    outer_ids: np.ndarray,
    inner_ids: np.ndarray,
    relation_index: RelNeighborIndex,
    self_degrees: dict[int, float],
    device: torch.device,
) -> Block:
    """Symmetric-normalised self-loop aggregation block for RREA's
    ent_feature pre-aggregation (`ent_feature = adj_self_norm @ h_init`,
    models/rrea.py) -- structural weight (relation-multiplicity,
    coalesce-summed), NOT minibatch_sampler.build_block's r2f/r2if
    weighting. Reuses the same Block/adj_t convention (outer -> inner,
    `torch.sparse.mm(adj_t.t(), h)` produces inner-indexed output) so
    callers can treat it identically to a GCN-style block."""
    outer_pos = {int(g): i for i, g in enumerate(outer_ids)}
    inner_pos = {int(g): i for i, g in enumerate(inner_ids)}

    raw: dict[tuple[int, int], float] = defaultdict(float)
    for v in inner_ids:
        v = int(v)
        if v in outer_pos:
            raw[(outer_pos[v], inner_pos[v])] += 1.0  # self-loop, weight 1
        for nbr_id, _rel_id, _val in relation_index.get(v, []):
            u = outer_pos.get(nbr_id)
            if u is None:
                continue
            raw[(u, inner_pos[v])] += 1.0  # one relation-arc, matches a_hat's coalesce-sum

    src_idx: list[int] = []
    dst_idx: list[int] = []
    vals: list[float] = []
    for (u, i), count in raw.items():
        dv = self_degrees.get(int(inner_ids[i]), 1.0)
        du = self_degrees.get(int(outer_ids[u]), 1.0)
        src_idx.append(u)
        dst_idx.append(i)
        vals.append((du ** -0.5) * (dv ** -0.5) * count)

    indices = torch.tensor([src_idx, dst_idx], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        indices, values, size=(len(outer_ids), len(inner_ids)), device=device
    ).coalesce()
    return Block(outer_ids=outer_ids, inner_ids=inner_ids, adj_t=adj)


def compute_rel_profile_features(
    entity_ids: np.ndarray,
    rel_out_index: RelProfileIndex,
    rel_in_index: RelProfileIndex,
    self_degrees: dict[int, float],
    rel_num: int,
    rel_emb: torch.Tensor,
) -> torch.Tensor:
    """rel_feature[v] = (1/deg(v)) * sum_r rel_profile[v, r] * rel_emb[r] --
    matches build_rrea_graph's rel_profile_norm @ rel_emb exactly. Note
    `rel_profile_norm`'s row scaling there is `d_inv_sqrt.unsqueeze(1) *
    rel_profile * d_inv_sqrt.unsqueeze(1)` -- the SAME per-entity
    d_inv_sqrt applied on BOTH sides of a per-row (not per-column)
    broadcast, i.e. deg(v)^-1 overall, not a two-sided bipartite
    normalisation (relations have no "degree" of their own in this
    formula). rel_in occupies rel_emb's first rel_num rows, rel_out the
    next rel_num rows (raw, undoubled relation ids) -- matches
    `np.concatenate([rel_in, rel_out], axis=1)` @ rel_emb's row order.
    No neighbor expansion needed -- entirely computable from each
    entity's own incident-triple counts.

    Vectorized: the Python loop only appends plain ints/floats to flat
    lists (it runs over the large active[1] set once per batch); all
    tensor arithmetic happens in one batched gather + index_add_,
    numerically identical up to float summation order (within the parity
    test's 1e-5 tolerance)."""
    device = rel_emb.device
    dim = rel_emb.size(1)
    n = len(entity_ids)

    ent_rows: list[int] = []
    emb_rows: list[int] = []
    counts: list[float] = []
    for i, v in enumerate(entity_ids):
        v = int(v)
        for r, cnt in rel_in_index.get(v, {}).items():
            ent_rows.append(i)
            emb_rows.append(r)
            counts.append(cnt)
        for r, cnt in rel_out_index.get(v, {}).items():
            ent_rows.append(i)
            emb_rows.append(rel_num + r)
            counts.append(cnt)

    out = torch.zeros(n, dim, device=device, dtype=rel_emb.dtype)
    if ent_rows:
        emb_idx = torch.tensor(emb_rows, dtype=torch.long, device=device)
        cnt_t = torch.tensor(counts, dtype=rel_emb.dtype, device=device).unsqueeze(1)
        msg = rel_emb[emb_idx] * cnt_t
        out.index_add_(0, torch.tensor(ent_rows, dtype=torch.long, device=device), msg)

    deg = torch.tensor(
        [self_degrees.get(int(v), 1.0) for v in entity_ids],
        dtype=rel_emb.dtype, device=device,
    ).unsqueeze(1)
    return out / deg


def build_rrea_attention_block(
    outer_ids: np.ndarray,
    inner_ids: np.ndarray,
    relation_index: RelNeighborIndex,
) -> tuple[list[int], list[int], list[int], list[float]]:
    """Bipartite edge list for one NRGraphAttention hop, restricted to
    (outer_ids -> inner_ids). Role mapping vs. models/rrea.py's full-graph
    `src, dst = edge_index[:, 0], edge_index[:, 1]`: `dst` (softmax-group
    key, `neighs` source) -> OUTER; `src` (index_add_ output target,
    `selfs` source) -> INNER -- same outer=info-source/inner=being-updated
    convention minibatch_sampler.build_block already uses. Requires
    inner_ids to be a SUBSET of outer_ids (see nested_active_sets) so each
    inner entity's own previous feature -- needed for `selfs` -- can be
    read out of the outer-indexed feature tensor by the caller.

    Returns (edge_outer_idx, edge_inner_idx, edge_rel_id, edge_rel_val) as
    plain Python lists (tensor construction is the caller's job, so an
    empty block can be detected before allocating zero-sized tensors)."""
    outer_pos = {int(g): i for i, g in enumerate(outer_ids)}
    inner_pos = {int(g): i for i, g in enumerate(inner_ids)}

    e_outer: list[int] = []
    e_inner: list[int] = []
    e_rel: list[int] = []
    e_val: list[float] = []
    for v in inner_ids:
        v = int(v)
        for nbr_id, rel_id, val in relation_index.get(v, []):
            u = outer_pos.get(nbr_id)
            if u is None:
                continue
            e_outer.append(u)
            e_inner.append(inner_pos[v])
            e_rel.append(rel_id)
            e_val.append(val)
    return e_outer, e_inner, e_rel, e_val


@dataclass
class RREAMinibatchIndex:
    """Structure-only, built ONCE per partition/dataset -- reused across
    every batch/epoch (mirrors minibatch_sampler.py's build_neighbor_index
    + compute_weighted_degrees being precomputed once, not per batch)."""

    relation_index: RelNeighborIndex
    rel_out_index: RelProfileIndex
    rel_in_index: RelProfileIndex
    self_degrees: dict[int, float]
    rel_num: int
    # --minibatch_sampler backend: "python" (this module's dict-of-lists
    # loops, the reference implementation), "vectorized" (minibatch_csr.py,
    # numpy/CSR, same semantics) or "dgl" (DGL C++ neighbour sampling +
    # CSR block construction). `csr` is populated for the two non-python
    # backends. Excluded from compare: structure-only derived data.
    backend: str = "python"
    csr: CSRIndex | None = field(default=None, repr=False, compare=False)
    # Memoized plain_neighbor_index() result -- see that method. Not part
    # of the logical value (excluded from compare), populated lazily.
    _plain_index: dict[int, list[tuple[int, float]]] | None = None

    @classmethod
    def build(
        cls,
        triples: list[tuple[int, int, int]],
        rel_num: int,
        backend: str = "python",
        n_entities: int | None = None,
    ) -> RREAMinibatchIndex:
        relation_index = build_rrea_relation_index(triples, rel_num)
        rel_out_index, rel_in_index = build_rel_profile_index(triples)
        self_degrees = compute_rrea_self_degrees(relation_index)
        backend = resolve_sampler_backend(backend)
        csr = None
        if backend != "python":
            csr = build_csr_index(
                relation_index, rel_out_index, rel_in_index, self_degrees, rel_num,
                n_entities=n_entities,
            )
        return cls(relation_index, rel_out_index, rel_in_index, self_degrees, rel_num,
                   backend=backend, csr=csr)

    def rel_profile_features(self, entity_ids: np.ndarray, rel_emb: torch.Tensor) -> torch.Tensor:
        """Backend-dispatched compute_rel_profile_features (used by
        RREA.forward_blocks)."""
        if self.csr is not None:
            return compute_rel_profile_features_csr(entity_ids, self.csr, rel_emb)
        return compute_rel_profile_features(
            entity_ids, self.rel_out_index, self.rel_in_index, self.self_degrees,
            self.rel_num, rel_emb,
        )

    def plain_neighbor_index(self) -> dict[int, list[tuple[int, float]]]:
        """Adapter for minibatch_sampler.sample_multi_hop, which only reads
        neighbor id + an unused second element (see its own unpacking:
        `for nbr_id, _w in ...`) -- reuses relation_index's connectivity
        without needing a second, separately-built index.

        MEMOIZED: prepare_rrea_batch calls it once per batch, and
        rebuilding this dict-of-lists over the entire entity set each time
        would re-materialize the whole adjacency (200K entries at 100K
        scale) for every batch. The underlying relation_index is immutable
        after build(), so caching is safe."""
        if self._plain_index is None:
            self._plain_index = {
                v: [(nbr_id, val) for nbr_id, _rel_id, val in edges]
                for v, edges in self.relation_index.items()
            }
        return self._plain_index


@dataclass
class RREAMinibatchBatch:
    """Per-batch bundle: everything RREA.forward_blocks needs for one
    forward pass, built fresh from an RREAMinibatchIndex + this batch's
    seed ids. `outer_ids` is the outermost frontier (active[0]) -- the
    caller must gather h_init for exactly these ids (e.g. via
    SparseEntityTable.gather) before calling forward_blocks."""

    outer_ids: np.ndarray
    pre_block: Block
    active: list[np.ndarray]
    attention_edges: list[AttentionEdges]
    seed_ids: np.ndarray  # == active[-1], exposed directly for caller convenience


def prepare_rrea_batch(
    seed_ids: np.ndarray,
    index: RREAMinibatchIndex,
    num_layers: int,
    fanout: int | None = None,
    rng: np.random.Generator | None = None,
    device: torch.device = _CPU_DEVICE,
) -> RREAMinibatchBatch:
    """Builds everything RREA.forward_blocks needs for one mini-batch.

    Samples num_layers+1 raw hops (1 extra for the pre-aggregation step
    ent_feature/rel_feature needs on top of the num_layers attention
    hops -- see module docstring), then derives NESTED active sets
    (active[i] = union of layers[i:], monotonically shrinking) instead of
    sample_multi_hop's own disjoint layers, so every attention hop's
    inner_ids is a subset of that hop's outer_ids (required for the
    `selfs` lookup -- see build_rrea_attention_block's docstring)."""
    if index.csr is not None:
        # --minibatch_sampler vectorized / dgl (minibatch_csr.py): same
        # structural semantics as the python branch below, numpy/CSR
        # implementation; tests/test_stage_v.py enforces the parity.
        sampler = sample_multi_hop_dgl if index.backend == "dgl" else sample_multi_hop_csr
        layers = sampler(seed_ids, index.csr, num_layers=num_layers + 1, fanout=fanout, rng=rng)
        active = nested_active_sets_np(layers)
        pre_block = build_rrea_self_block_csr(active[0], active[1], index.csr, device)
        attention_edges: list[AttentionEdges] = [
            build_rrea_attention_block_csr(active[k + 1], active[k + 2], index.csr)  # type: ignore[misc]
            for k in range(num_layers)
        ]
    else:
        layers = sample_multi_hop(
            seed_ids, index.plain_neighbor_index(), num_layers=num_layers + 1,
            fanout=fanout, rng=rng,
        )
        active = nested_active_sets(layers)

        pre_block = build_rrea_self_block(
            active[0], active[1], index.relation_index, index.self_degrees, device
        )

        attention_edges = [
            build_rrea_attention_block(active[k + 1], active[k + 2], index.relation_index)
            for k in range(num_layers)
        ]

    return RREAMinibatchBatch(
        outer_ids=active[0],
        pre_block=pre_block,
        active=active,
        attention_edges=attention_edges,
        seed_ids=active[-1],
    )


def nested_active_sets(layers: list[np.ndarray]) -> list[np.ndarray]:
    """layers: outermost-first, disjoint (sample_multi_hop's own output,
    length K+1, layers[-1] == seeds). Returns active[i] = union(layers[i],
    layers[i+1], ..., layers[-1]) for each i -- monotonically SHRINKING,
    active[-1] == layers[-1] == seeds, active[0] == every sampled entity.
    Unlike GCN's disjoint (outer_ids, inner_ids) block pairs, RREA's
    attention hops need inner_ids ⊆ outer_ids (see build_rrea_attention_block
    docstring) -- consecutive active[i]/active[i+1] pairs give exactly
    that nesting."""
    active: list[np.ndarray] = [None] * len(layers)  # type: ignore[list-item]
    acc: set = set()
    for i in range(len(layers) - 1, -1, -1):
        acc = acc | {int(g) for g in layers[i]}
        active[i] = np.array(sorted(acc), dtype=np.int64)
    return active
