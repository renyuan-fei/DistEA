"""
minibatch_csr.py — vectorized (CSR/numpy) backend for RREA mini-batch
sampling and block construction, selected by `--minibatch_sampler
vectorized` (default) or `dgl`; `python` keeps rrea_minibatch.py's reference
dict-of-lists implementation byte-for-byte.

Why: after the SparseEntityTable and rel-profile vectorizations, the
remaining per-batch time is dominated by
`sample_multi_hop` + `build_rrea_attention_block` / `build_rrea_self_block`
— all pure-Python loops over dict-of-lists adjacency, executed once per
batch (~70 batches/epoch at IDS15K, ~98 at IDS100K). This module keeps the
SAME structural semantics (see each function's docstring for the exact
rrea_minibatch.py function it mirrors) but stores the adjacency once as
CSR arrays and expresses every per-batch step as numpy array operations.

Semantics preserved exactly (tests/test_stage_v.py enforces each):
  - sample_multi_hop: identical layer sets for fanout=None; for fanout=K,
    the same sampling DISTRIBUTION (uniform K-subset of each frontier
    node's unseen relation-ARCS — multi-relation neighbours are weighted
    by arc count, as in the python version) but not the same draws.
  - nested active sets: identical.
  - self block: identical coalesced sparse matrix (self-loop 1.0, one
    unit per relation-arc, D^-1/2 (A+I) D^-1/2 with the TRUE structural
    degrees).
  - attention block: identical edge multiset (as numpy arrays instead of
    Python lists; RREA.forward_block_hop accepts both).
  - rel-profile features: identical up to float summation order.

The `dgl` backend replaces ONLY the neighbour-sampling step with
`dgl.sampling.sample_neighbors` (C++ sampler); block construction stays on
this module's CSR path. DGL samples K arcs among ALL of a node's arcs and
we then drop already-seen endpoints, so a node's NEW frontier can be
smaller than K — a GraphSAGE-style difference from the python/vectorized
"K among unseen arcs" rule, documented rather than hidden. Falls back to
`vectorized` with a warning when DGL cannot be imported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from minibatch_sampler import Block

SAMPLER_BACKENDS = ("python", "vectorized", "dgl")


def import_dgl():
    """`import dgl`, tolerating the graphbolt import-chain breakage of newer
    environments (DGL 2.1.0 wheels ship libgraphbolt only for
    torch <= 2.2.1 and need torchdata<0.8's datapipes, while e.g. torch
    2.8 + torchdata 0.11 is common). `import dgl` pulls dgl.dataloading ->
    dgl.distributed -> dgl.graphbolt unconditionally, so a plain import
    dies in graphbolt/base.py even though the core C library (dgl.graph,
    dgl.sampling.sample_neighbors -- all this module uses) loads fine.
    On that specific failure a stub `dgl.graphbolt` package is registered
    and the import retried; graphbolt/distributed features are then
    unavailable, which nothing here needs. Tested with torch 2.8.0+cu128
    and dgl 2.1.0. Raises ImportError when
    DGL is genuinely absent or fails for another reason."""
    try:
        import dgl  # noqa: PLC0415

        return dgl
    except ModuleNotFoundError as exc:
        if exc.name == "dgl":
            raise
        graphbolt_chain = exc.name in ("torchdata.datapipes", "torchdata") or "graphbolt" in str(exc)
    except (ImportError, FileNotFoundError, OSError) as exc:
        graphbolt_chain = "graphbolt" in str(exc).lower()
    if not graphbolt_chain:
        raise ImportError("dgl import failed for a reason other than the graphbolt chain")
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    for mod in [m for m in sys.modules if m == "dgl" or m.startswith("dgl.")]:
        del sys.modules[mod]
    stub = types.ModuleType("dgl.graphbolt")
    stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["dgl.graphbolt"] = stub
    import dgl  # noqa: PLC0415

    logging.warning(
        "dgl imported with a stubbed dgl.graphbolt (torch/torchdata newer than this "
        "DGL build supports); graphbolt/distributed features unavailable, core sampling OK"
    )
    return dgl


@dataclass
class CSRIndex:
    """Structure-only CSR view of rrea_minibatch's dict indexes, built once
    per partition/dataset. Entity ids >= n (entities with no triples at
    all) are legal lookups everywhere below: they have no arcs and
    structural degree 1.0 (self-loop only), matching the python version's
    `.get(v, [])` / `.get(v, 1.0)` defaults."""

    n: int
    indptr: np.ndarray      # (n+1,) int64 — arcs of node v are [indptr[v], indptr[v+1])
    nbr: np.ndarray         # (E,) int64 neighbour id
    rel: np.ndarray         # (E,) int64 directional relation id (reverse = r + rel_num)
    val: np.ndarray         # (E,) float32 1 / (#relations on that directed pair)
    self_deg: np.ndarray    # (n,) float64 1 + #incident arcs
    rp_indptr: np.ndarray   # (n+1,) int64 relation-profile CSR
    rp_col: np.ndarray      # (P,) int64 rel_emb row: rel_in -> r, rel_out -> rel_num + r
    rp_cnt: np.ndarray      # (P,) float32
    rel_num: int
    _dgl_graph: object = field(default=None, repr=False, compare=False)

    # -- degree / membership helpers ------------------------------------
    def degrees(self, ids: np.ndarray) -> np.ndarray:
        out = np.ones(len(ids), dtype=np.float64)
        m = ids < self.n
        out[m] = self.self_deg[ids[m]]
        return out

    def arcs_of(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(arc_positions, owner_local_index) — every arc of every id in
        `ids` (ids >= n contribute nothing), owner_local_index indexes
        back into `ids`."""
        ids = np.asarray(ids, dtype=np.int64)
        in_range = ids < self.n
        local = np.nonzero(in_range)[0]
        v = ids[in_range]
        starts = self.indptr[v]
        counts = self.indptr[v + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        excl = np.cumsum(counts) - counts
        pos = np.repeat(starts - excl, counts) + np.arange(total, dtype=np.int64)
        owner = np.repeat(local, counts)
        return pos, owner


def _csr_from_dict(d: dict, n: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Generic {node: list[tuple]} -> (indptr, rows) with rows sorted by node."""
    counts = np.zeros(n, dtype=np.int64)
    for v, entries in d.items():
        counts[v] = len(entries)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    rows = np.empty((int(indptr[-1]), width), dtype=np.float64)
    for v, entries in d.items():
        s = indptr[v]
        rows[s : s + len(entries)] = entries
    return indptr, rows


def build_csr_index(
    relation_index: dict[int, list[tuple[int, int, float]]],
    rel_out_index: dict[int, dict[int, int]],
    rel_in_index: dict[int, dict[int, int]],
    self_degrees: dict[int, float],
    rel_num: int,
    n_entities: int | None = None,
) -> CSRIndex:
    """Build from rrea_minibatch's own dict indexes so the structural
    semantics (directional relation ids, 1/multiplicity values, self
    degrees, raw relation-profile counts) have exactly one definition."""
    keys = [max(d) for d in (relation_index, rel_out_index, rel_in_index, self_degrees) if d]
    n = max(keys) + 1 if keys else 0
    if n_entities is not None:
        n = max(n, int(n_entities))

    indptr, rows = _csr_from_dict(relation_index, n, 3)
    nbr = rows[:, 0].astype(np.int64)
    rel = rows[:, 1].astype(np.int64)
    val = rows[:, 2].astype(np.float32)

    self_deg = np.ones(n, dtype=np.float64)
    for v, dgr in self_degrees.items():
        self_deg[v] = dgr

    # relation profile: rel_in first (row r), rel_out second (row rel_num + r)
    rp: dict[int, list[tuple[int, int]]] = {}
    for v, cnts in rel_in_index.items():
        rp.setdefault(v, []).extend((r, c) for r, c in cnts.items())
    for v, cnts in rel_out_index.items():
        rp.setdefault(v, []).extend((rel_num + r, c) for r, c in cnts.items())
    rp_indptr, rp_rows = _csr_from_dict(rp, n, 2)
    rp_col = rp_rows[:, 0].astype(np.int64)
    rp_cnt = rp_rows[:, 1].astype(np.float32)

    return CSRIndex(
        n=n, indptr=indptr, nbr=nbr, rel=rel, val=val, self_deg=self_deg,
        rp_indptr=rp_indptr, rp_col=rp_col, rp_cnt=rp_cnt, rel_num=rel_num,
    )


# ---------------------------------------------------------------------------
#  Sampling
# ---------------------------------------------------------------------------


def _cap_per_owner(cand: np.ndarray, owner: np.ndarray, fanout: int, rng: np.random.Generator) -> np.ndarray:
    """Keep at most `fanout` uniformly-chosen arcs per owner (uniform
    K-subset without replacement, same distribution as rng.choice over
    the owner's candidate arc list in the python version)."""
    if len(cand) == 0:
        return cand
    keys = rng.random(len(cand))
    order = np.lexsort((keys, owner))
    owner_sorted = owner[order]
    first = np.searchsorted(owner_sorted, owner_sorted, side="left")
    rank = np.arange(len(order)) - first
    return cand[order[rank < fanout]]


def sample_multi_hop_csr(
    seed_ids: np.ndarray,
    csr: CSRIndex,
    num_layers: int,
    fanout: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Vectorized twin of minibatch_sampler.sample_multi_hop: returns
    layers OUTERMOST-first, layers[-1] == unique sorted seeds, each entity
    in exactly one layer (its closest-to-seed hop), early stop on an empty
    frontier."""
    if rng is None:
        rng = np.random.default_rng()
    seeds = np.unique(np.asarray(seed_ids, dtype=np.int64))
    size = max(csr.n, int(seeds.max()) + 1 if len(seeds) else 0)
    seen = np.zeros(size, dtype=bool)
    seen[seeds] = True
    layers_inner_first = [seeds]
    frontier = seeds
    for _ in range(num_layers):
        pos, owner = csr.arcs_of(frontier)
        cand = csr.nbr[pos]
        keep = ~seen[cand]
        cand, owner = cand[keep], owner[keep]
        if fanout is not None:
            cand = _cap_per_owner(cand, owner, fanout, rng)
        nxt = np.unique(cand)
        seen[nxt] = True
        layers_inner_first.append(nxt)
        frontier = nxt
        if len(frontier) == 0:
            break
    return list(reversed(layers_inner_first))


def sample_multi_hop_dgl(
    seed_ids: np.ndarray,
    csr: CSRIndex,
    num_layers: int,
    fanout: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Same layer contract as sample_multi_hop_csr, but each hop's arc
    sampling is done by dgl.sampling.sample_neighbors (fanout arcs among
    ALL arcs of the node, then already-seen endpoints dropped — see module
    docstring). Requires DGL; callers should have resolved the backend
    with resolve_sampler_backend() first."""
    dgl = import_dgl()

    if csr._dgl_graph is None:
        owner = np.repeat(np.arange(csr.n, dtype=np.int64), np.diff(csr.indptr))
        csr._dgl_graph = dgl.graph(
            (torch.from_numpy(csr.nbr), torch.from_numpy(owner)), num_nodes=csr.n
        )
    g = csr._dgl_graph
    if rng is not None:
        dgl.seed(int(rng.integers(0, 2**31 - 1)))

    seeds = np.unique(np.asarray(seed_ids, dtype=np.int64))
    size = max(csr.n, int(seeds.max()) + 1 if len(seeds) else 0)
    seen = np.zeros(size, dtype=bool)
    seen[seeds] = True
    layers_inner_first = [seeds]
    frontier = seeds
    k = -1 if fanout is None else int(fanout)
    for _ in range(num_layers):
        fr = frontier[frontier < csr.n]
        if len(fr) == 0:
            break
        sg = dgl.sampling.sample_neighbors(g, torch.from_numpy(fr), k, edge_dir="in")
        u, _v = sg.edges()
        cand = u.numpy().astype(np.int64)
        cand = cand[~seen[cand]]
        nxt = np.unique(cand)
        seen[nxt] = True
        layers_inner_first.append(nxt)
        frontier = nxt
        if len(frontier) == 0:
            break
    return list(reversed(layers_inner_first))


def nested_active_sets_np(layers: list[np.ndarray]) -> list[np.ndarray]:
    """Vectorized twin of rrea_minibatch.nested_active_sets."""
    active: list[np.ndarray] = [None] * len(layers)  # type: ignore[list-item]
    acc = np.empty(0, dtype=np.int64)
    for i in range(len(layers) - 1, -1, -1):
        acc = np.union1d(acc, np.asarray(layers[i], dtype=np.int64))
        active[i] = acc
    return active


# ---------------------------------------------------------------------------
#  Block construction
# ---------------------------------------------------------------------------


def _positions_in(sorted_ids: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position, valid_mask): position of each query id in the sorted
    unique array `sorted_ids`, valid where the id is actually present."""
    if len(sorted_ids) == 0:
        return np.zeros(len(query), dtype=np.int64), np.zeros(len(query), dtype=bool)
    pos = np.searchsorted(sorted_ids, query)
    pos_c = np.minimum(pos, len(sorted_ids) - 1)
    valid = sorted_ids[pos_c] == query
    return pos_c, valid


def _inner_arcs_in_outer(outer_ids, inner_ids, csr: CSRIndex):
    """Every arc (v -> nbr) with v in inner_ids and nbr in outer_ids, as
    (outer_local, inner_local, arc_position)."""
    pos, owner = csr.arcs_of(inner_ids)
    nbr = csr.nbr[pos]
    o_pos, valid = _positions_in(outer_ids, nbr)
    return o_pos[valid], owner[valid], pos[valid]


def build_rrea_self_block_csr(
    outer_ids: np.ndarray, inner_ids: np.ndarray, csr: CSRIndex, device: torch.device,
) -> Block:
    """Vectorized twin of rrea_minibatch.build_rrea_self_block. outer_ids
    and inner_ids must be sorted unique (nested_active_sets output)."""
    outer_ids = np.asarray(outer_ids, dtype=np.int64)
    inner_ids = np.asarray(inner_ids, dtype=np.int64)
    src, dst, _pos = _inner_arcs_in_outer(outer_ids, inner_ids, csr)

    self_pos, self_valid = _positions_in(outer_ids, inner_ids)
    self_src = self_pos[self_valid]
    self_dst = np.nonzero(self_valid)[0]

    src = np.concatenate([self_src, src])
    dst = np.concatenate([self_dst, dst])
    d_out = csr.degrees(outer_ids)
    d_in = csr.degrees(inner_ids)
    vals = (d_out[src] ** -0.5) * (d_in[dst] ** -0.5)

    indices = torch.from_numpy(np.stack([src, dst])).to(device)
    values = torch.from_numpy(vals.astype(np.float32)).to(device)
    adj = torch.sparse_coo_tensor(
        indices, values, size=(len(outer_ids), len(inner_ids)), device=device
    ).coalesce()
    return Block(outer_ids=outer_ids, inner_ids=inner_ids, adj_t=adj)


def build_rrea_attention_block_csr(
    outer_ids: np.ndarray, inner_ids: np.ndarray, csr: CSRIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized twin of rrea_minibatch.build_rrea_attention_block —
    returns numpy arrays (edge_outer_idx, edge_inner_idx, edge_rel_id,
    edge_rel_val)."""
    outer_ids = np.asarray(outer_ids, dtype=np.int64)
    inner_ids = np.asarray(inner_ids, dtype=np.int64)
    e_outer, e_inner, pos = _inner_arcs_in_outer(outer_ids, inner_ids, csr)
    return e_outer, e_inner, csr.rel[pos], csr.val[pos]


def compute_rel_profile_features_csr(
    entity_ids: np.ndarray, csr: CSRIndex, rel_emb: torch.Tensor,
) -> torch.Tensor:
    """Vectorized twin of rrea_minibatch.compute_rel_profile_features:
    rel_feature[v] = (1/deg(v)) * sum_r count[v, r] * rel_emb[row(r)]."""
    device = rel_emb.device
    ids = np.asarray(entity_ids, dtype=np.int64)
    n = len(ids)
    out = torch.zeros(n, rel_emb.size(1), device=device, dtype=rel_emb.dtype)

    in_range = ids < csr.n
    local = np.nonzero(in_range)[0]
    v = ids[in_range]
    starts = csr.rp_indptr[v]
    counts = csr.rp_indptr[v + 1] - starts
    total = int(counts.sum())
    if total > 0:
        excl = np.cumsum(counts) - counts
        pos = np.repeat(starts - excl, counts) + np.arange(total, dtype=np.int64)
        rows = torch.from_numpy(np.repeat(local, counts)).to(device)
        cols = torch.from_numpy(csr.rp_col[pos]).to(device)
        cnt = torch.from_numpy(csr.rp_cnt[pos]).to(device=device, dtype=rel_emb.dtype).unsqueeze(1)
        out.index_add_(0, rows, rel_emb[cols] * cnt)

    deg = torch.from_numpy(csr.degrees(ids)).to(device=device, dtype=rel_emb.dtype).unsqueeze(1)
    return out / deg


# ---------------------------------------------------------------------------
#  Backend resolution
# ---------------------------------------------------------------------------


def resolve_sampler_backend(name: str) -> str:
    """Validate a --minibatch_sampler value; 'dgl' degrades to 'vectorized'
    (with a warning) when DGL is not importable in this environment."""
    if name not in SAMPLER_BACKENDS:
        raise ValueError(f"unknown minibatch sampler backend {name!r}; choose from {SAMPLER_BACKENDS}")
    if name == "dgl":
        try:
            import_dgl()
        except Exception as exc:  # ImportError, FileNotFoundError (DGL/torch ABI), …
            logging.warning(f"--minibatch_sampler dgl requested but DGL is unavailable ({exc!r}); using 'vectorized'")
            return "vectorized"
    return name
