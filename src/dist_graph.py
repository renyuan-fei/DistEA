"""
dist_graph.py — DistGraphLoader: build per-rank local graph with Halo nodes.

Node ordering within each rank's graph:
    [local_kg1 | halo_kg1 | local_kg2 | halo_kg2]

- local_kg1: KG1 entities in this rank's METIS partition
- halo_kg1:  1-hop KG1 neighbours from other partitions (read-only in training)
- local_kg2: KG2 entities assigned to this rank
- halo_kg2:  1-hop KG2 neighbours assigned to other ranks (read-only in
             training) — mirrors halo_kg1, so KG2-KG2 cross-partition edges
             are retained instead of silently dropped (counterpart_disc's
             round-robin assignment for non-train-linked KG2 entities means
             most KG2-KG2 edges would otherwise cross a partition boundary).

Returns:
    (g, global2local, all_global_ids, local_kg1_idx, halo_kg1_idx,
     local_kg2_idx, halo_kg2_idx)

Where:
    g               — DGL graph (or sparse adj matrix if DGL unavailable)
    global2local    — dict: global_entity_id → local_node_index
    all_global_ids  — np.ndarray, shape (N_local_nodes,), global IDs in node order
    local_kg1_idx   — np.ndarray, local indices of local_kg1 nodes
    halo_kg1_idx    — np.ndarray, local indices of halo_kg1 nodes
    local_kg2_idx   — np.ndarray, local indices of local_kg2 nodes
    halo_kg2_idx    — np.ndarray, local indices of halo_kg2 nodes
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

try:
    import dgl

    _HAS_DGL = True
except (ImportError, FileNotFoundError):
    _HAS_DGL = False


def _vector_graph_enabled() -> bool:
    """DISTEA_VECTOR_GRAPH=1 switches DistGraphLoader.load()'s
    halo/edge construction and build_rrea_graph() to numpy-vectorised code
    paths that reproduce the Python loops' outputs exactly (same halo sets,
    same edge ORDER -- dict insertion order + list append order -- same
    weights), measured 30-70 s + 15-25 s per call at DBP1M, and avoid the
    dense (N, 2*rel_num) relation-profile intermediate (8.5 GB host + GPU at
    N=1). Read at call time; default off keeps the Python-loop code path.
    Parity: tests/test_vector_graph.py."""
    return os.environ.get("DISTEA_VECTOR_GRAPH", "0").lower() in ("1", "true", "yes")


def _data_cache_enabled() -> bool:
    """DISTEA_DATA_CACHE=1 (the parsed-triples cache, dataload.py): data
    objects then carry the (T, 3) arrays from construction, and load() does
    not build the int-tuple lists on the vectorised path. Read at call time."""
    return os.environ.get("DISTEA_DATA_CACHE", "0").lower() in ("1", "true", "yes")


def _triples_array(data, attr: str) -> np.ndarray:
    """(T, 3) int64 view of data.<attr> (kg1_triples / kg2_triples), cached
    on the data object under _<attr>_np so the loader and build_rrea_graph
    (called once per iteration each) convert the list of tuples once."""
    cache = f"_{attr}_np"
    arr = getattr(data, cache, None)
    if arr is None:
        triples = getattr(data, attr)
        arr = np.asarray(list(triples), dtype=np.int64).reshape(-1, 3)
        try:
            setattr(data, cache, arr)
        except AttributeError:
            pass
    return arr


def _weight_arrays(edge_weights: dict, n_rel: int) -> tuple[np.ndarray, np.ndarray]:
    """edge_weights {r: (r2f, r2if)} -> (r2f[r], r2if[r]) arrays, 1.0 where
    missing -- the `.get(r, (1.0, 1.0))` default of the Python path."""
    r2f = np.ones(n_rel, dtype=np.float64)
    r2if = np.ones(n_rel, dtype=np.float64)
    for r, (a, b) in (edge_weights or {}).items():
        if 0 <= int(r) < n_rel:
            r2f[int(r)] = float(a)
            r2if[int(r)] = float(b)
    return r2f, r2if


class DistGraphLoader:
    """Load the local subgraph for a single rank with Halo nodes."""

    def __init__(
        self, data, partition_dir: str, rank: int, world_size: int, device: torch.device,
        no_halo_kg1: bool = False, no_halo_kg2: bool = False,
        full_graph: bool = False,
    ):
        self.data = data
        self.partition_dir = partition_dir
        self.rank = rank
        self.world_size = world_size
        self.device = device
        # no_halo_kg1 / no_halo_kg2 ablate the halo mechanism itself (the
        # "locality, no halo" configuration). Forcing halo_kg1/
        # halo_kg2 to empty sets below reuses that same already-correct
        # empty-halo codepath (indices, edge filtering) rather than adding a
        # second code path.
        self.no_halo_kg1 = no_halo_kg1
        self.no_halo_kg2 = no_halo_kg2
        # Multi-node loss ablation: full_graph=True gives EVERY
        # rank the whole graph: the 'halo' becomes the complement of this
        # rank's owned KG1 / KG2 sets (all edges kept, KG2 included), while
        # ownership (local_kg1_idx / local_kg2_idx, hence eval rows and --
        # via framework._map_links_to_local(owned_kg1_n=...) -- the train
        # pairs) stays exactly the partition's. Isolates graph truncation
        # from pair sharding / per-rank h_init. Default off.
        self.full_graph = full_graph

    def load(self):
        """
        Build the local graph for this rank.

        Returns:
            (g, global2local, all_global_ids, local_kg1_idx, halo_kg1_idx,
             local_kg2_idx, halo_kg2_idx)
        """
        data = self.data
        rank = self.rank

        # ── 1. Load local KG1 partition ──────────────────────────────────────
        part_file = os.path.join(self.partition_dir, f"{rank}.npy")
        local_kg1 = list(np.load(part_file).astype(int))
        local_kg1_set = set(local_kg1)

        # ── 2. Load KG2 assignments ──────────────────────────────────────────
        kg2_file = os.path.join(self.partition_dir, "kg2_assignments.json")
        with open(kg2_file) as f:
            kg2_assignments = json.load(f)
        local_kg2 = [int(e) for e in kg2_assignments.get(str(rank), [])]

        vec = _vector_graph_enabled()
        # Lists of (h, r, t) int IDs, read by the Python path only. Under
        # DISTEA_DATA_CACHE=1 the vectorised path skips them -- it reads the
        # (T, 3) arrays (_triples_array), and building the lists as well cost
        # ~6 s per call at DBP1M; off keeps the Python-loop code path.
        _skip_lists = vec and _data_cache_enabled()
        kg1_triples = None if _skip_lists else data.kg1_triples
        kg2_triples = None if _skip_lists else data.kg2_triples
        local_kg2_set = set(local_kg2)

        if vec:
            # ── 3/3b (vectorised): halo = 1-hop neighbours (either
            # direction) of the local set that are not local themselves;
            # np.unique gives the same sorted list as sorted(set(...)).
            tri1 = _triples_array(data, "kg1_triples")
            tri2 = _triples_array(data, "kg2_triples")
            halo_kg1 = self._halo_np(tri1, np.asarray(local_kg1, dtype=np.int64), self.no_halo_kg1)
            halo_kg2 = self._halo_np(tri2, np.asarray(local_kg2, dtype=np.int64), self.no_halo_kg2)
        else:
            # ── 3. Find 1-hop halo KG1 nodes ──────────────────────────────────
            # Build neighbor sets for KG1 (undirected halo detection)
            kg1_nbrs: dict[int, set] = {}
            for h, _r, t in kg1_triples:
                kg1_nbrs.setdefault(h, set()).add(t)
                kg1_nbrs.setdefault(t, set()).add(h)

            halo_kg1_set: set = set()
            if not self.no_halo_kg1:
                for eid in local_kg1:
                    for nbr in kg1_nbrs.get(eid, []):
                        if nbr not in local_kg1_set:
                            halo_kg1_set.add(nbr)

            halo_kg1 = sorted(halo_kg1_set)

            # ── 3b. Find 1-hop halo KG2 nodes ───────────────────────────────
            # Mirrors halo_kg1 above: counterpart_disc.assign_kg2_to_ranks
            # scatters non-train-linked KG2 entities round-robin (ignoring
            # KG2-KG2 graph locality), so most real KG2-KG2 edges would
            # otherwise cross a partition boundary and get dropped by the
            # `included` check below.
            kg2_nbrs: dict[int, set] = {}
            for h, _r, t in kg2_triples:
                kg2_nbrs.setdefault(h, set()).add(t)
                kg2_nbrs.setdefault(t, set()).add(h)

            halo_kg2_set: set = set()
            if not self.no_halo_kg2:
                for eid in local_kg2:
                    for nbr in kg2_nbrs.get(eid, []):
                        if nbr not in local_kg2_set:
                            halo_kg2_set.add(nbr)

            halo_kg2 = sorted(halo_kg2_set)

        # ── 4. Assign local indices ──────────────────────────────────────────
        if self.full_graph:
            halo_kg1, halo_kg2 = self._complement_halo(local_kg1_set, local_kg2_set)
        all_global_ids = np.array(
            local_kg1 + halo_kg1 + local_kg2 + halo_kg2, dtype=np.int64
        )
        global2local: dict[int, int] = {gid: i for i, gid in enumerate(all_global_ids)}

        n_local_kg1 = len(local_kg1)
        n_halo_kg1 = len(halo_kg1)
        n_local_kg2 = len(local_kg2)
        n_halo_kg2 = len(halo_kg2)

        local_kg1_idx = np.arange(0, n_local_kg1, dtype=np.int64)
        halo_kg1_idx = np.arange(n_local_kg1, n_local_kg1 + n_halo_kg1, dtype=np.int64)
        local_kg2_idx = np.arange(
            n_local_kg1 + n_halo_kg1,
            n_local_kg1 + n_halo_kg1 + n_local_kg2,
            dtype=np.int64,
        )
        halo_kg2_idx = np.arange(
            n_local_kg1 + n_halo_kg1 + n_local_kg2,
            n_local_kg1 + n_halo_kg1 + n_local_kg2 + n_halo_kg2,
            dtype=np.int64,
        )

        # ── 5. Build edges (local index space) ──────────────────────────────
        N = len(all_global_ids)
        if vec:
            # Same triples, same filters, same (fwd, bwd) interleaving and
            # weights as the Python path below, computed on arrays.
            g2l = self._g2l_array(all_global_ids, tri1, tri2)
            src_a, dst_a, w_a = self._edges_np(
                tri1, g2l, getattr(data, "edge_weights1", {}) or {}, None)
            in_local_kg2 = np.zeros(g2l.shape[0], dtype=bool)
            if len(local_kg2):
                in_local_kg2[np.asarray(local_kg2, dtype=np.int64)] = True
            kg2_mask = in_local_kg2[tri2[:, 0]] | in_local_kg2[tri2[:, 2]]
            if self.full_graph:
                kg2_mask = None  # every KG2 edge, not just those touching owned KG2
            src_b, dst_b, w_b = self._edges_np(
                tri2, g2l, getattr(data, "edge_weights2", {}) or {}, kg2_mask)
            src_t = torch.from_numpy(np.concatenate([src_a, src_b]))
            dst_t = torch.from_numpy(np.concatenate([dst_a, dst_b]))
            w_t = torch.from_numpy(np.concatenate([w_a, w_b]).astype(np.float32))
        else:
            included = set(all_global_ids.tolist())

            src_list, dst_list, weight_list = [], [], []

            def add_edge(h, r, t, edge_weights):
                if h in included and t in included:
                    # ELsEA's get_weighted_adj (DivEA/GCN_Align/utils.py)
                    # weights the two directions of a triple *differently*
                    # — r2if for head->tail, r2f for tail->head — each
                    # floored at 0.3 so no edge collapses to zero weight.
                    # `.get(r, (1.0, 1.0))` keeps the uniform-weight
                    # fallback when load_weights() was never called (empty
                    # dict).
                    r2f, r2if = edge_weights.get(r, (1.0, 1.0))
                    w_fwd = max(r2if, 0.3)  # head -> tail
                    w_bwd = max(r2f, 0.3)   # tail -> head
                    lh, lt = global2local[h], global2local[t]
                    src_list.append(lh)
                    dst_list.append(lt)
                    weight_list.append(w_fwd)
                    src_list.append(lt)
                    dst_list.append(lh)
                    weight_list.append(w_bwd)

            for h, r, t in kg1_triples:
                add_edge(h, r, t, data.edge_weights1)

            # KG2 triples (local KG2 or halo_kg2 endpoint involved — the
            # other endpoint, if on a different rank and not a halo_kg2
            # pick, is filtered out by add_edge's `included` check above)
            for h, r, t in kg2_triples:
                if self.full_graph or h in local_kg2_set or t in local_kg2_set:
                    add_edge(h, r, t, data.edge_weights2)

            src_t = torch.tensor(src_list, dtype=torch.long)
            dst_t = torch.tensor(dst_list, dtype=torch.long)
            w_t = torch.tensor(weight_list, dtype=torch.float32)

        # ── 6. Build graph object ────────────────────────────────────────────

        if _HAS_DGL:
            g = dgl.graph((src_t, dst_t), num_nodes=N)
            g.edata["w"] = w_t
            g = g.to(self.device)
        else:
            # Fallback: raw weighted sparse adjacency, UNNORMALISED.
            # GCNLayer._sparse_forward does the single D^-1/2 (A+I) D^-1/2
            # symmetric-normalisation pass (matching ELsEA's preprocess_adj /
            # normalize_adj). Row-normalising here too would stack two
            # different, non-equivalent normalisations on top of each other,
            # which is not what ELsEA computes.
            indices = torch.stack([src_t, dst_t], dim=0)
            g = torch.sparse_coo_tensor(
                indices, w_t, size=(N, N), dtype=torch.float32
            ).coalesce().to(self.device)

        logging.info(
            f"[rank {rank}] DistGraphLoader: "
            f"local_kg1={n_local_kg1}, halo_kg1={n_halo_kg1}, "
            f"local_kg2={n_local_kg2}, halo_kg2={n_halo_kg2}, "
            f"edges={int(src_t.numel()) // 2}"
        )

        return (
            g,
            global2local,
            all_global_ids,
            local_kg1_idx,
            halo_kg1_idx,
            local_kg2_idx,
            halo_kg2_idx,
        )

    # ── vectorised helpers (DISTEA_VECTOR_GRAPH=1) ───────────────────────────
    def _complement_halo(self, local_kg1_set: set, local_kg2_set: set):
        """full_graph: all KG1 ids owned by OTHER ranks ({r}.npy, r != rank) and
        all KG2 ids assigned to other ranks (kg2_assignments.json), sorted --
        the partition covers every entity, so local + this = the whole KG."""
        others_kg1: set = set()
        for r in range(self.world_size):
            if r == self.rank:
                continue
            others_kg1.update(int(x) for x in np.load(os.path.join(self.partition_dir, f"{r}.npy")))
        with open(os.path.join(self.partition_dir, "kg2_assignments.json")) as f:
            assign = json.load(f)
        others_kg2: set = set()
        for r, ids in assign.items():
            if int(r) != self.rank:
                others_kg2.update(int(e) for e in ids)
        return sorted(others_kg1 - local_kg1_set), sorted(others_kg2 - local_kg2_set)

    @staticmethod
    def _halo_np(tri: np.ndarray, local: np.ndarray, disabled: bool) -> list:
        """1-hop neighbours (either direction) of `local` that are not local:
        the sorted(set(...)) of the Python path, as a list of ints."""
        if disabled or tri.shape[0] == 0 or local.shape[0] == 0:
            return []
        h, t = tri[:, 0], tri[:, 2]
        size = int(max(h.max(), t.max(), local.max())) + 1
        in_local = np.zeros(size, dtype=bool)
        in_local[local] = True
        cand = np.concatenate([t[in_local[h] & ~in_local[t]], h[in_local[t] & ~in_local[h]]])
        return np.unique(cand).tolist()

    @staticmethod
    def _g2l_array(all_global_ids: np.ndarray, *triple_arrays: np.ndarray) -> np.ndarray:
        """global id -> local index, -1 where not included (the Python
        path's `h in included` test)."""
        size = int(all_global_ids.max()) + 1 if all_global_ids.size else 1
        for tri in triple_arrays:
            if tri.shape[0]:
                size = max(size, int(tri[:, [0, 2]].max()) + 1)
        g2l = np.full(size, -1, dtype=np.int64)
        g2l[all_global_ids] = np.arange(all_global_ids.shape[0], dtype=np.int64)
        return g2l

    @staticmethod
    def _edges_np(tri: np.ndarray, g2l: np.ndarray, edge_weights: dict,
                  extra_mask: np.ndarray | None):
        """The Python path's add_edge over every triple of `tri` (in order),
        on arrays: both endpoints included (and extra_mask, if given), then
        (lh->lt, max(r2if, 0.3)) followed by (lt->lh, max(r2f, 0.3))."""
        if tri.shape[0] == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), np.zeros(0, dtype=np.float64)
        lh = g2l[tri[:, 0]]
        lt = g2l[tri[:, 2]]
        r = tri[:, 1]
        keep = (lh >= 0) & (lt >= 0)
        if extra_mask is not None:
            keep &= extra_mask
        lh, lt, r = lh[keep], lt[keep], r[keep]
        n_rel = int(r.max()) + 1 if r.size else 1
        r2f, r2if = _weight_arrays(edge_weights, n_rel)
        w_fwd = np.maximum(r2if[r], 0.3)
        w_bwd = np.maximum(r2f[r], 0.3)
        n = lh.shape[0]
        src = np.empty(2 * n, dtype=np.int64)
        dst = np.empty(2 * n, dtype=np.int64)
        w = np.empty(2 * n, dtype=np.float64)
        src[0::2], src[1::2] = lh, lt
        dst[0::2], dst[1::2] = lt, lh
        w[0::2], w[1::2] = w_fwd, w_bwd
        return src, dst, w


def build_rrea_graph(data, global2local, all_global_ids, rel_num, device, local_mask=None):
    """
    Build the RREAGraph bundle (models/rrea.py) for this rank's local
    subgraph — RREA/utils.py::get_matrix, ported to local (partitioned)
    index space. Relations are directional for the attention layer's edge
    lookup (id + rel_num for the reverse direction, a separate learnable
    vector — not shared/transposed), but rel_in/rel_out (the per-entity
    relation-count profile feeding rel_feature) use the RAW, undoubled
    relation id, matching utils.py::get_matrix exactly — these are two
    separate uses of "relation id", not the same range.

    local_mask (--drop_halo_halo_edges, default None = every triple with
    both ends in the node set): (N,) bool, True for the rank's
    own rows; triples whose two ends are both halo rows are left out, so
    the graph holds exactly the triples touching an owned entity (what the
    KG2 side of DistGraphLoader already keeps). At DBP1M N=8 those
    halo-halo triples are 28% of each rank's RREA graph and the heaviest
    rank's load drops 27% (2.17x -> 1.58x the even 1/N share); halo rows
    then aggregate only from owned neighbours.
    """
    from models.rrea import RREAGraph

    N = len(all_global_ids)
    vec = _vector_graph_enabled()
    prof_rows = prof_cols = None  # vectorised path: sparse relation profile entries

    if vec:
        # Sequence of directed entries exactly as the Python path appends
        # them: per included triple (kg1 then kg2, in order) the forward
        # (lh, lt, r) then the reverse (lt, lh, r + rel_num). The final
        # edge order is the Python dict's: pairs by first appearance, rels
        # within a pair in appearance order -> a stable sort by the pair's
        # first index. edge_val = 1 / (#entries on that directed pair).
        tri = np.concatenate(
            [_triples_array(data, "kg1_triples"), _triples_array(data, "kg2_triples")], axis=0
        )
        g2l = DistGraphLoader._g2l_array(np.asarray(all_global_ids, dtype=np.int64), tri)
        lh = g2l[tri[:, 0]]
        lt = g2l[tri[:, 2]]
        r = tri[:, 1]
        keep = (lh >= 0) & (lt >= 0)
        if local_mask is not None:
            lm = np.asarray(local_mask, dtype=bool)
            keep[keep] = lm[lh[keep]] | lm[lt[keep]]
        lh, lt, r = lh[keep], lt[keep], r[keep]
        n = lh.shape[0]
        src = np.empty(2 * n, dtype=np.int64)
        dst = np.empty(2 * n, dtype=np.int64)
        rel = np.empty(2 * n, dtype=np.int64)
        src[0::2], src[1::2] = lh, lt
        dst[0::2], dst[1::2] = lt, lh
        rel[0::2], rel[1::2] = r, r + rel_num
        if n:
            key = src * N + dst
            _, first_idx, inverse, counts = np.unique(
                key, return_index=True, return_inverse=True, return_counts=True)
            inverse = inverse.reshape(-1)
            order = np.argsort(first_idx[inverse], kind="stable")
            edge_src_np, edge_dst_np, edge_rel_np = src[order], dst[order], rel[order]
            edge_val_np = (1.0 / counts[inverse][order]).astype(np.float32)
        else:
            edge_src_np = edge_dst_np = edge_rel_np = np.zeros(0, dtype=np.int64)
            edge_val_np = np.zeros(0, dtype=np.float32)
        edge_index = torch.from_numpy(np.stack([edge_src_np, edge_dst_np], axis=1)).to(device) \
            if n else torch.zeros((0, 2), dtype=torch.long, device=device)
        edge_rel_t = torch.from_numpy(edge_rel_np).to(device)
        edge_val_t = torch.from_numpy(edge_val_np).to(device)
        # rel_in counts at (lt, r), rel_out counts at (lh, rel_num + r):
        # the columns of concat([rel_in, rel_out]) below, kept sparse.
        prof_rows = np.concatenate([lt, lh])
        prof_cols = np.concatenate([r, r + rel_num])
    else:
        included = set(all_global_ids.tolist())

        # edges keyed by DIRECTED (lh, lt) pair -> list of directional relation
        # ids seen on that pair (>1 only if a multigraph has parallel edges).
        pair_rels: dict[tuple, list] = {}
        rel_in = np.zeros((N, rel_num), dtype=np.float32)
        rel_out = np.zeros((N, rel_num), dtype=np.float32)

        lm = None if local_mask is None else np.asarray(local_mask, dtype=bool)

        def add(h, r, t):
            if h in included and t in included:
                lh, lt = global2local[h], global2local[t]
                if lm is not None and not (lm[lh] or lm[lt]):
                    return  # halo-halo triple (--drop_halo_halo_edges)
                pair_rels.setdefault((lh, lt), []).append(r)
                pair_rels.setdefault((lt, lh), []).append(r + rel_num)
                rel_out[lh, r] += 1
                rel_in[lt, r] += 1

        for h, r, t in data.kg1_triples:
            add(h, r, t)
        for h, r, t in data.kg2_triples:
            add(h, r, t)

        edge_src, edge_dst, edge_rel, edge_val = [], [], [], []
        for (lh, lt), rels in pair_rels.items():
            w = 1.0 / len(rels)
            for r in rels:
                edge_src.append(lh)
                edge_dst.append(lt)
                edge_rel.append(r)
                edge_val.append(w)

        edge_index = torch.tensor(list(zip(edge_src, edge_dst, strict=False)), dtype=torch.long, device=device) \
            if edge_src else torch.zeros((0, 2), dtype=torch.long, device=device)
        edge_rel_t = torch.tensor(edge_rel, dtype=torch.long, device=device) \
            if edge_rel else torch.zeros((0,), dtype=torch.long, device=device)
        edge_val_t = torch.tensor(edge_val, dtype=torch.float32, device=device) \
            if edge_val else torch.zeros((0,), dtype=torch.float32, device=device)

    # adj_self_norm: D^-1/2 (A + I) D^-1/2, undirected unweighted A from the
    # same edge set (self-loop weight 1) — feeds ent_feature's "avg" step.
    n_edges = int(edge_index.size(0))
    eye_idx = torch.arange(N, device=device)
    a_src = torch.cat([edge_index[:, 0], eye_idx]) if n_edges else eye_idx
    a_dst = torch.cat([edge_index[:, 1], eye_idx]) if n_edges else eye_idx
    a_val = torch.cat([torch.ones(n_edges, device=device), torch.ones(N, device=device)]) \
        if n_edges else torch.ones(N, device=device)
    a_hat = torch.sparse_coo_tensor(torch.stack([a_src, a_dst]), a_val, (N, N)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    idx = a_hat.indices()
    vals = a_hat.values() * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
    adj_self_norm = torch.sparse_coo_tensor(idx, vals, (N, N), device=device).coalesce()

    # rel_profile_norm: per-entity in/out relation-count profile, row+col
    # normalised the same D^-1/2 (.) D^-1/2 way RREA/utils.py::normalize_adj
    # applies it (entity-degree on both sides) — feeds rel_feature.
    if vec:
        # Sparse from the start: coalesce() sums the per-entry ones into the
        # exact integer counts the dense rel_in/rel_out held, then the same
        # (d * count) * d association as the dense path, on the same rows.
        if prof_rows is not None and prof_rows.shape[0]:
            p_idx = torch.from_numpy(np.stack([prof_rows, prof_cols])).to(device)
            prof = torch.sparse_coo_tensor(
                p_idx, torch.ones(prof_rows.shape[0], dtype=torch.float32, device=device),
                (N, 2 * rel_num),
            ).coalesce()
            rows = prof.indices()[0]
            vals = (d_inv_sqrt[rows] * prof.values()) * d_inv_sqrt[rows]
            rel_profile_norm = torch.sparse_coo_tensor(prof.indices(), vals, (N, 2 * rel_num), device=device).coalesce()
        else:
            rel_profile_norm = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.float32, device=device), (N, 2 * rel_num),
            ).coalesce()
    else:
        rel_profile = torch.from_numpy(np.concatenate([rel_in, rel_out], axis=1)).to(device)
        rel_profile = (d_inv_sqrt.unsqueeze(1) * rel_profile) * d_inv_sqrt.unsqueeze(1)
        rel_profile_norm = rel_profile.to_sparse()

    return RREAGraph(adj_self_norm, rel_profile_norm, edge_index, edge_rel_t,
                     edge_val_t, rel_num)


def sample_halo_rrea_graph(g, halo_kg1_idx, halo_kg2_idx, p_kg1: float, p_kg2: float):
    """
    BNS-GCN-style per-epoch boundary (halo) node sampling.

    Each call draws a fresh subsample: ceil(p * |halo|) halo nodes are kept
    (independently for halo_kg1 / halo_kg2, uniformly without replacement);
    every edge incident to a dropped halo node is removed for this epoch,
    and adj_self_norm is renormalised on the surviving edge set with the
    exact D^-1/2 (A + I) D^-1/2 recipe build_rrea_graph uses (dropped
    nodes keep only their self-loop, so tensor shapes never change --
    their embedding rows simply receive no neighbour information this
    epoch). Following BNS-GCN itself, kept edges are NOT rescaled by 1/p;
    the recomputed degree normalisation is the only compensation.

    rel_profile_norm is left untouched: it is a per-entity count profile,
    so a kept node's row never depends on any other (dropped) node.

    Returns `g` itself when nothing would be dropped (both p >= 1 or no
    halo nodes) -- callers can pass the result straight to train_epoch.
    Eval must keep using the FULL graph; only training epochs sample.
    """
    from models.rrea import RREAGraph

    N = g.adj_self_norm.size(0)
    device = g.edge_index.device
    drop = torch.zeros(N, dtype=torch.bool, device=device)
    for idx, p in ((halo_kg1_idx, p_kg1), (halo_kg2_idx, p_kg2)):
        n_halo = 0 if idx is None else len(idx)
        if p >= 1.0 or n_halo == 0:
            continue
        keep_n = int(np.ceil(p * n_halo))
        perm = torch.randperm(n_halo, device=device)
        dropped = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device)[perm[keep_n:]]
        drop[dropped] = True
    if not bool(drop.any()):
        return g

    src, dst = g.edge_index[:, 0], g.edge_index[:, 1]
    keep_e = ~(drop[src] | drop[dst])
    edge_index = g.edge_index[keep_e]
    edge_rel = g.edge_rel[keep_e]
    edge_val = g.edge_rel_val[keep_e]

    # Same normalisation as build_rrea_graph, on the sampled edge set.
    eye_idx = torch.arange(N, device=device)
    a_src = torch.cat([edge_index[:, 0], eye_idx])
    a_dst = torch.cat([edge_index[:, 1], eye_idx])
    a_val = torch.ones(edge_index.size(0) + N, device=device)
    a_hat = torch.sparse_coo_tensor(torch.stack([a_src, a_dst]), a_val, (N, N)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    idx2 = a_hat.indices()
    vals = a_hat.values() * d_inv_sqrt[idx2[0]] * d_inv_sqrt[idx2[1]]
    adj_self_norm = torch.sparse_coo_tensor(idx2, vals, (N, N), device=device).coalesce()

    return RREAGraph(adj_self_norm, g.rel_profile_norm, edge_index, edge_rel,
                     edge_val, g.rel_num)
