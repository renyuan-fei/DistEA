"""
gas.py — GNNAutoScale-style historical-embedding (cluster-wise)
training for RREA (Fey et al., "GNNAutoScale: Scalable and Expressive
Graph Neural Networks via Historical Embeddings", ICML 2021).

Idea: partition the local graph into clusters; each training batch is one
cluster. A batch node's layer-l feature is computed EXACTLY over its full
1-hop in-neighborhood (no sampling, unlike --minibatch's fanout path);
neighbors outside the batch contribute their layer-(l-1) feature from a
CPU-resident history table (stale, no gradient), refreshed whenever their
own cluster is the batch. Peak GPU memory scales with the cluster, not
the graph.

Two RREA-specific wrinkles this file handles:

1. **Softmax closure.** RREA's attention softmax is grouped by the edge's
   dst role while messages accumulate into src (see models/rrea.py's
   forward and its role-mapping notes). A batch-src edge's softmax group
   therefore also contains edges whose src lies OUTSIDE the batch. Each
   cluster's edge set is closed over those groups: for D1 = dst(edges
   with src in batch), ALL edges with dst in D1 are included, so every
   softmax group touching a batch edge is complete. Out-of-batch srcs on
   closure edges read their features from history — that staleness is
   the GAS trade, not an approximation of the normalization itself.

2. **Layer 0 is exact.** RREA's layer-0 inputs (ent_feature =
   adj_self_norm @ h_init, rel_feature = rel_profile_norm @ rel_emb) are
   linear in the always-resident learnable tables, so the needed ROWS are
   recomputed exactly per batch from row-sliced sparse matrices — history
   tables exist only for attention layers l >= 1 (per channel).

Parity gate (tests/test_stage_w.py): with history pre-filled by one
whole-graph-as-single-cluster pass, a multi-cluster forward reproduces
RREA.forward() exactly.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import breadth_first_order, connected_components
from torch.utils.checkpoint import checkpoint

from models.rrea import RREA, RREAGraph, _sparse_softmax_by_dst, edge_attention_core


class GASHistory:
    """Per-(channel, layer) historical feature tables.

    channels: "ent" / "rel"; layers 1..depth (layer 0 needs no history —
    see module docstring). CPU-resident by default (the GAS paper's
    setting); `device=` keeps them on the compute device instead
    (--gas_history_on_device) — 2 channels x (depth-1) layers x N x dim
    fp32 is ~660 MB for a 644K-node DBP1M rank graph, small next to the
    training peak, and it removes a CPU gather + host-to-device copy of
    every cluster's `needed` rows per channel per layer per step. Pulled
    rows are detached copies on the compute device; pushes store detached
    copies on the table's device."""

    def __init__(self, n_nodes: int, dim: int, depth: int,
                 device: torch.device | str | None = None,
                 edge_dst: torch.Tensor | None = None, attn_heads: int = 1,
                 layer0: bool = False):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        # layer0=True (--gas_layer0_history): layer-0 rows of
        # out-of-batch needed nodes come from a history table too, instead
        # of the exact per-cluster sparse.mm over their full in-edge lists
        # (a hub destination drags its whole in-edge list -- up to 85K rows
        # of adj_self_norm -- into every cluster that touches it, so the
        # exact layer-0 spmm processed ~10x E per epoch at DBP1M even with
        # the softmax history; its backward transposes and coalesces the
        # same slices). Same trade as layers >= 1: at most one epoch stale,
        # exact at the fixed point. warm_layer0() fills the tables once.
        self.layer0 = bool(layer0)
        self.tables: dict[tuple[str, int], torch.Tensor] = {
            (c, layer): torch.zeros(n_nodes, dim, device=self.device)
            for c in ("ent", "rel")
            for layer in range((0 if self.layer0 else 1), depth + 1)
        }
        # Softmax-denominator history (--gas_softmax_history):
        # per (channel, layer, head) one fp64 row per EDGE holding the last
        # computed exp(attention logit), plus its per-destination sum. With
        # these a cluster only processes its own batch-src edges (the ones
        # that deliver messages): a dst's denominator is the fresh sum over
        # those edges plus (stale per-dst total minus the stale values of
        # those same edges) -- every edge is recomputed exactly once per
        # epoch, when its src's cluster is the batch, so no denominator is
        # more than one epoch stale, the same trade GAS makes for features.
        # fp64 with no max-shift: logits are clamped to +-300 in the head,
        # far outside any range the exact softmax tolerates in practice, and
        # exp() stays representable there. refresh_dst_sums() rebuilds the
        # per-dst totals from the edge tables (end of every epoch) so the
        # incremental updates cannot drift.
        self.softmax = edge_dst is not None
        self.edge_exp: dict[tuple[str, int, int], torch.Tensor] = {}
        self.dst_sum: dict[tuple[str, int, int], torch.Tensor] = {}
        if self.softmax:
            self.edge_dst = edge_dst.to(self.device).long()
            n_edges = int(self.edge_dst.numel())
            for c in ("ent", "rel"):
                for layer in range(depth):
                    for head in range(attn_heads):
                        self.edge_exp[(c, layer, head)] = torch.zeros(
                            n_edges, dtype=torch.float64, device=self.device)
                        self.dst_sum[(c, layer, head)] = torch.zeros(
                            n_nodes, dtype=torch.float64, device=self.device)

    def pull(self, channel: str, layer: int, ids: torch.Tensor,
             device: torch.device) -> torch.Tensor:
        table = self.tables[(channel, layer)]
        return table[ids.to(table.device)].to(device)

    def push(self, channel: str, layer: int, ids: torch.Tensor,
             values: torch.Tensor) -> None:
        table = self.tables[(channel, layer)]
        table[ids.to(table.device)] = values.detach().to(table.device)

    @torch.no_grad()
    def warm_layer0(self, g: RREAGraph, h_init: torch.Tensor, rel_emb: torch.Tensor,
                    device: torch.device) -> None:
        """Fill the layer-0 tables with the exact whole-graph layer-0 features
        (one spmm per channel, no grad) so the first epoch of
        --gas_layer0_history starts from exact rows instead of zeros."""
        if not self.layer0:
            return
        with torch.autocast(device_type=device.type, enabled=False):
            f_ent = torch.sparse.mm(g.adj_self_norm.to(device), h_init.detach().to(device))
            f_rel = torch.sparse.mm(g.rel_profile_norm.to(device), rel_emb.detach().to(device))
        self.tables[("ent", 0)].copy_(f_ent.to(self.device))
        self.tables[("rel", 0)].copy_(f_rel.to(self.device))

    # ── softmax-denominator history ──────────────────────────────────────
    def pull_denominators(self, key: tuple[str, int, int], e_gid: torch.Tensor,
                          needed: torch.Tensor, device: torch.device
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """(stale exp per cluster edge, stale per-dst total per needed row)."""
        ee, ds = self.edge_exp[key], self.dst_sum[key]
        return (ee[e_gid.to(ee.device)].to(device), ds[needed.to(ds.device)].to(device))

    def push_denominators(self, key: tuple[str, int, int], e_gid: torch.Tensor,
                          e_fresh: torch.Tensor, needed: torch.Tensor,
                          denom: torch.Tensor) -> None:
        ee, ds = self.edge_exp[key], self.dst_sum[key]
        ee[e_gid.to(ee.device)] = e_fresh.detach().to(ee.device, torch.float64)
        ds[needed.to(ds.device)] = denom.detach().to(ds.device, torch.float64)

    def refresh_dst_sums(self) -> float:
        """Rebuild every per-dst total from its edge table; returns the
        largest absolute change (drift of the incremental updates)."""
        worst = 0.0
        for key, ee in self.edge_exp.items():
            fresh = torch.zeros_like(self.dst_sum[key]).index_add_(0, self.edge_dst, ee)
            worst = max(worst, float((fresh - self.dst_sum[key]).abs().max()))
            self.dst_sum[key] = fresh
        return worst


class GASCluster:
    """One batch: node set + closure edge set + row-sliced layer-0 mats.

    All index tensors are precomputed positions (into `needed` / `batch`)
    so the forward is pure tensor ops. Built once per iteration by
    build_gas_clusters; graph structure doesn't change within one."""

    __slots__ = (
        "batch", "needed", "batch_pos_in_needed",
        "e_dst_pos", "e_src_pos", "e_rel", "e_val",
        "e_src_batch_mask", "e_src_batch_pos",
        "e_w",  # per-edge softmax weights (closure_cap > 0) or None (exact)
        "e_gid",  # global (local-graph) edge ids of the cluster's edges (softmax_history) or None
        "adj_rows", "relprof_rows",
        "l0_batch_only",  # adj_rows/relprof_rows sliced for `batch` (layer0_history) instead of `needed`
        "adj_rows_t", "relprof_rows_t",  # precomputed CSR transposes for the layer-0 backward (csr_spmm) or None
        "e_val_pos",  # True when every e_val > 0: rel_vec may be normalised per relation (see _gas_attention_head)
        # csr_gather: incidence matrices (CSR) + transposes so the per-edge
        # gathers and the message scatter are spmm's with sort-free backward
        "g_dst", "g_dst_t", "g_src", "g_src_t", "s_mat", "s_mat_t",
        "src_seg_counts",  # edges sorted by src (src_sorted_edges): length of each src's run, else None
        "src_seg_pos", "src_seg_bpos",  # each run's src as a needed / batch position (bpos: all srcs in batch only)
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def to_(self, device: torch.device | str) -> GASCluster:
        """Move every tensor field (index vectors and the row-sliced sparse
        layer-0 matrices) to `device` in place (--gas_clusters_on_device).
        gas_forward_batch's per-step `.to(device)` calls then become no-ops
        (otherwise ~0.5 s/epoch of pageable host-to-device copies, 16 per
        cluster, at DBP1M/N=1)."""
        for k in self.__slots__:
            v = getattr(self, k, None)
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


class _SpMMFixedT(torch.autograd.Function):
    """Y = A @ X for a constant sparse A whose transpose is precomputed.

    torch.sparse.mm's backward for the dense operand forms A.t() @ grad by
    transposing the COO matrix and coalescing it (a sort over every stored
    entry) on EVERY call; with the exact layer-0 slices that is ~5M entries
    per cluster per channel at DBP1M -- about 0.9 s of a 5.4 s epoch
    (SparseAddmmBackward0 in a kernel profile). Here the
    transpose is built once per cluster (CSR, build_gas_clusters(
    csr_spmm=True)) and the backward is a single spmm. Same values, same
    gradient (fp32 summation order aside); grads flow only to X."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        ctx.a_t = a_t
        return torch.sparse.mm(a, x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return torch.sparse.mm(ctx.a_t, grad.contiguous()), None, None


def _segment_mp_enabled() -> bool:
    """DISTEA_GAS_SEGMENT_MP=1: on clusters whose edges are
    sorted by src (DISTEA_GAS_SEGMENT_SOFTMAX), the per-edge src gather and
    the message scatter run as row gathers and contiguous-segment sums
    instead of incidence spmm's, and the dst gather's forward is a row
    gather (its backward keeps the incidence transpose). Off by default and
    not recommended: torch.segment_reduce over (E, dim) rows takes ~4 ms a
    call on an A100 (DBP1M epoch 2.03 s -> 2.94 s), far slower than the
    cuSPARSE spmm it replaces. Values are correct (same accuracy)."""
    import os

    return os.environ.get("DISTEA_GAS_SEGMENT_MP", "0").lower() in ("1", "true", "yes")


class _GatherRowsCsrBwd(torch.autograd.Function):
    """out = x[idx]; backward = a_t @ grad with the precomputed CSR transpose
    of the 0/1 incidence matrix (segmented sums, no atomics on hub rows).
    Same values as _SpMMFixedT.apply(x, a, a_t) for a with one entry per
    row; the forward is an index_select instead of a cuSPARSE spmm."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, idx: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        ctx.a_t = a_t
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return torch.sparse.mm(ctx.a_t, grad.contiguous()), None, None


class _SegRepeatRows(torch.autograd.Function):
    """out[e] = x[idx[e]] for edges sorted into contiguous runs of equal idx
    (run r: counts[r] edges, all with idx == seg_pos[r]); the backward sums
    each run (segment_reduce) and writes the sums to the run rows -- no
    scatter onto repeated rows."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, idx: torch.Tensor, seg_pos: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(seg_pos, counts)
        ctx.n_rows = x.size(0)
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        seg_pos, counts = ctx.saved_tensors
        sums = torch.segment_reduce(grad.contiguous(), "sum", lengths=counts, axis=0)
        gx = torch.zeros(ctx.n_rows, grad.size(1), dtype=grad.dtype, device=grad.device)
        gx.index_copy_(0, seg_pos, sums)
        return gx, None, None, None


class _SegSumToRows(torch.autograd.Function):
    """out[out_pos[r]] = sum of msg over run r (contiguous runs of counts[r]
    edges; distinct out_pos), other rows 0 -- the message aggregation into
    the batch rows when every edge's src is a batch node. The backward is a
    row gather by each edge's output row (edge_out_pos)."""

    @staticmethod
    def forward(ctx, msg: torch.Tensor, counts: torch.Tensor, out_pos: torch.Tensor,
                edge_out_pos: torch.Tensor, n_out: int) -> torch.Tensor:
        sums = torch.segment_reduce(msg.contiguous(), "sum", lengths=counts, axis=0)
        out = torch.zeros(n_out, msg.size(1), dtype=msg.dtype, device=msg.device)
        out.index_copy_(0, out_pos, sums)
        ctx.save_for_backward(edge_out_pos)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (edge_out_pos,) = ctx.saved_tensors
        return grad.index_select(0, edge_out_pos), None, None, None, None


def _layer0_mm(cluster: GASCluster, which: str, x: torch.Tensor,
               device: torch.device) -> torch.Tensor:
    """Row-sliced layer-0 product (adj_rows @ h_init or relprof_rows @
    rel_emb) through _SpMMFixedT when the cluster carries the precomputed
    CSR transpose, else plain torch.sparse.mm."""
    a = getattr(cluster, which).to(device)
    a_t = getattr(cluster, which + "_t", None)
    if a_t is None:
        return torch.sparse.mm(a, x)
    return _SpMMFixedT.apply(x, a, a_t.to(device))


def _incidence_pair(rows: torch.Tensor, cols: torch.Tensor, shape: tuple[int, int],
                    device: torch.device | str | None = None,
                    dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """0/1 incidence matrix with one entry per (rows[k], cols[k]) as CSR, plus
    its CSR transpose (built on `device` when given). With `rows = arange(E)`
    and `cols = e_dst_pos` the product `M @ X` is the per-edge gather
    X[e_dst_pos] and `M.t() @ G` is its scatter-add -- a cuSPARSE spmm with
    a segmented reduction instead of the sort-based index_put_/embedding
    backward (the gathers, their backward and the message index_add_ were
    ~1.3 s of a 2.3 s DBP1M epoch)."""
    idx = torch.stack([rows.to(torch.int64), cols.to(torch.int64)])
    vals = torch.ones(idx.size(1), dtype=dtype)
    if device is not None and _fast_build():
        # Coalesce on the device, not the CPU. Every caller passes distinct
        # (row, col) pairs (rows = arange(E), or cols = distinct edge ids)
        # and every value is 1, so the coalesce is a pure sort of unique
        # keys: the same matrix on any device. The CPU sort of three
        # ~0.5M-entry matrices per cluster was ~3.5 s of the 34 s DBP1M/N=1
        # cluster build.
        idx, vals = idx.to(device), vals.to(device)
        coo = torch.sparse_coo_tensor(idx, vals, shape).coalesce()
    else:
        coo = torch.sparse_coo_tensor(idx, vals, shape).coalesce()
        if device is not None:
            coo = coo.to(device)
    return coo.to_sparse_csr(), coo.t().coalesce().to_sparse_csr()


def _csr_pair(coo: torch.Tensor, device: torch.device | str | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """(CSR copy of a coalesced COO matrix, CSR of its transpose). With
    `device` both are built there: the transpose's coalesce is a sort over
    every stored entry (~5M per cluster at DBP1M/N=1, 27 s of CPU time per
    build across the 32 clusters), milliseconds on the GPU."""
    if device is not None:
        coo = coo.to(device)
    return coo.to_sparse_csr(), coo.t().coalesce().to_sparse_csr()


class _RowSliceIndex:
    """Row-range index over a coalesced COO matrix (whose entries are sorted
    by row, then column), so that slicing a sorted set of rows is O(entries
    in the slice) instead of an np.isin over every stored entry per call:
    the per-cluster slicing in build_gas_clusters (2 x K calls, 18M entries
    at DBP1M/N=1) otherwise costs about half of the cluster build."""

    def __init__(self, mat: torch.Tensor):
        mat = mat.coalesce()
        idx = mat.indices().cpu()
        self.row = idx[0].numpy()
        self.col = idx[1]
        self.vals = mat.values().cpu()
        self.n_rows, self.n_cols = int(mat.size(0)), int(mat.size(1))
        # ptr[r] .. ptr[r+1] are the entries of row r (rows are sorted).
        self.ptr = np.searchsorted(self.row, np.arange(self.n_rows + 1))

    def slice(self, rows: np.ndarray) -> torch.Tensor:
        rows = np.asarray(rows, dtype=np.int64)
        starts, ends = self.ptr[rows], self.ptr[rows + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.int64), torch.zeros(0, dtype=self.vals.dtype),
                (len(rows), self.n_cols), is_coalesced=True)
        rep_starts = np.repeat(starts, lengths)
        offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        src = torch.from_numpy(rep_starts + offsets)
        new_row = torch.from_numpy(np.repeat(np.arange(len(rows), dtype=np.int64), lengths))
        # Rows were visited in increasing order and columns are sorted within
        # a row, so the result is already coalesced.
        return torch.sparse_coo_tensor(
            torch.stack([new_row, self.col[src]]), self.vals[src],
            (len(rows), self.n_cols), is_coalesced=True)


def _fast_build() -> bool:
    """DISTEA_GAS_FAST_BUILD=1 turns on the two exact
    cluster-build speedups below: _incidence_pair coalescing on the device
    and _assign_clusters_cached. Off by default (opt-in; same clusters
    either way)."""
    return os.environ.get("DISTEA_GAS_FAST_BUILD", "0").lower() in ("1", "true", "yes")


def _row_slice_sparse(mat: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    """(len(rows), n_cols) sparse slice of a coalesced sparse COO matrix
    (`rows` sorted). One-off convenience wrapper over _RowSliceIndex."""
    return _RowSliceIndex(mat).slice(rows)


def build_gas_clusters(
    g: RREAGraph,
    n_clusters: int,
    method: str = "metis",
    closure_cap: int = 0,
    closure_seed: int = 0,
    hub_degree: int = 1000,
    softmax_history: bool = False,
    layer0_history: bool = False,
    csr_spmm: bool = False,
    csr_device: torch.device | str | None = None,
    csr_gather: bool = False,
    src_sorted_edges: bool = False,
) -> list[GASCluster]:
    """Partition local nodes into n_clusters batches and precompute each
    batch's closure edge set + positional indices + row-sliced layer-0
    matrices. method="metis" (nxmetis via networkx, minimizes closure size
    via edge-cut) with a BFS-chunk fallback when nxmetis is unavailable or
    the graph exceeds the size guard; method="metis_csr" calls METIS on a
    scipy-built CSR with hubs of degree > hub_degree removed first
    (_metis_csr_assign, DBP1M-scale); method="bfs" is the chunk heuristic.

    closure_cap > 0 (default 0 = exact closure, behaviour unchanged) bounds
    the softmax closure per destination node. Every edge whose src is in
    the batch is kept — those are the only edges that deliver messages to
    batch nodes (gas_forward_batch's `msg[src_b_mask]`). Of the OTHER
    in-edges of each dst, which exist purely to complete that dst's softmax
    denominator, at most `closure_cap` are kept, drawn uniformly without
    replacement, each carrying the weight n_other / n_kept so that the
    weighted denominator is an unbiased estimate of the exact one
    (_cap_closure; applied in _gas_attention_head's weighted softmax).

    Why: the exact closure is dominated by hub destinations (type/country
    entities with in-degrees in the tens of thousands) whose COMPLETE
    in-edge set lands in every cluster that has even one src pointing at
    them — on a DBP1M/N=4 rank one cluster alone carried 3.8M of the
    7.04M local edges, so K=32 clusters re-process the hub
    edges ~32x per epoch, and no clustering quality can change that. With
    a cap the per-cluster edge set is ~ batch out-edges + sum_d
    min(indeg_d, cap). Approximate (stochastic denominators for dsts with
    more than `closure_cap` other in-edges; the sample is fixed for the
    life of the cluster set, i.e. one outer iteration). Parity: cap=0 and
    cap >= max in-degree are identical to the exact path, and the estimator
    is unbiased — tests/test_gas_closure_cap.py.

    softmax_history=True (--gas_softmax_history) drops the closure entirely:
    a cluster's edge set is exactly its batch-src edges and the softmax
    denominators come from GASHistory's per-edge exp tables (see there), so
    the per-epoch edge work is exactly 1x the local edge count. closure_cap
    is ignored in that mode. tests/test_gas_softmax_history.py.

    layer0_history=True (--gas_layer0_history) slices the layer-0 matrices
    for the BATCH rows only; the other needed rows' layer-0 features are
    pulled from GASHistory's layer-0 tables at forward time (see
    GASHistory). Per-epoch layer-0 spmm work drops to exactly 1x the local
    edge count (each row's in-edges once, when its cluster is the batch)
    and the cluster build no longer slices hub rows per cluster.
    tests/test_gas_layer0_history.py.

    csr_spmm=True (--gas_csr_spmm) stores the layer-0 slices as CSR together
    with their precomputed CSR transposes, so the layer-0 backward is one
    spmm instead of a transpose + coalesce per call (_SpMMFixedT). Exact.
    tests/test_gas_csr_spmm.py. csr_device: build the CSR pairs there (the
    GPU when the clusters live on it anyway, --gas_clusters_on_device):
    the transposes' sorts then take milliseconds instead of ~27 s of CPU
    per DBP1M/N=1 build.

    csr_gather=True (--gas_csr_gather) adds per-cluster incidence matrices
    (edge x needed for the dst and src gathers, batch x edge for the message
    scatter) with their CSR transposes, so _gas_attention_head does those
    three index ops as spmm's whose backward is a segmented reduction
    instead of a sort (_incidence_pair). Exact. tests/test_gas_csr_gather.py."""
    N = g.adj_self_norm.size(0)
    edges = g.edge_index.cpu().numpy()
    e_rel_np = g.edge_rel.cpu().numpy()
    e_val_np = g.edge_rel_val.cpu().numpy()
    src_all, dst_all = edges[:, 0], edges[:, 1]
    n_edges_total = int(src_all.shape[0])
    rng = np.random.default_rng(closure_seed)

    assign = _assign_clusters_cached(N, edges, n_clusters, method, hub_degree)
    adj_index = _RowSliceIndex(g.adj_self_norm)
    relprof_index = _RowSliceIndex(g.rel_profile_norm)

    clusters = []
    sum_closure = sum_needed = max_closure = 0
    for c in range(n_clusters):
        batch = np.sort(np.nonzero(assign == c)[0])
        if len(batch) == 0:
            continue
        in_batch_src = np.isin(src_all, batch)
        e_w = None
        e_gid = None
        if softmax_history:
            gid = np.nonzero(in_batch_src)[0]
            ce_src, ce_dst = src_all[gid], dst_all[gid]
            ce_rel, ce_val = e_rel_np[gid], e_val_np[gid]
            src_in_batch = np.ones(gid.shape[0], dtype=bool)
            e_gid = torch.from_numpy(gid.astype(np.int64))
        else:
            d1 = np.unique(dst_all[in_batch_src])
            closure = np.isin(dst_all, d1)
            gid = np.nonzero(closure)[0]
            ce_src = src_all[closure]
            ce_dst = dst_all[closure]
            ce_rel = e_rel_np[closure]
            ce_val = e_val_np[closure]
            src_in_batch = np.isin(ce_src, batch)
            if closure_cap > 0:
                keep, w = _cap_closure(ce_dst, src_in_batch, closure_cap, rng)
                ce_src, ce_dst = ce_src[keep], ce_dst[keep]
                ce_rel, ce_val = ce_rel[keep], ce_val[keep]
                src_in_batch = src_in_batch[keep]
                gid = gid[keep]
                e_w = torch.from_numpy(w[keep].astype(np.float32))
            # Edge ids are recorded for closure clusters too: with a
            # softmax-history GASHistory a closure cluster is still exact
            # (every in-edge of each touched dst is processed, so the stale
            # terms cancel) AND fills the edge tables -- a single-cluster
            # closure pass is the exact warm-up for the history mode.
            e_gid = torch.from_numpy(gid.astype(np.int64))
        src_seg_counts = None
        if src_sorted_edges:
            # --attn_softmax src groups the softmax by src: with the
            # edges in src order each group is a contiguous run and the head
            # takes _SegmentSoftmax (segment_reduce, no index gathers whose
            # backward sorts every call -- 8 of the 12 embedding backwards
            # per cluster otherwise). Same edges, same groups; only
            # the order of the floating-point sums changes.
            order = np.argsort(ce_src, kind="stable")
            ce_src, ce_dst = ce_src[order], ce_dst[order]
            ce_rel, ce_val = ce_rel[order], ce_val[order]
            src_in_batch = src_in_batch[order]
            gid = gid[order]
            e_gid = torch.from_numpy(gid.astype(np.int64))
            if e_w is not None:
                e_w = e_w[torch.from_numpy(order)]
            if ce_src.shape[0] > 0:
                src_seg_counts = torch.from_numpy(np.unique(ce_src, return_counts=True)[1].astype(np.int64))
        needed = np.unique(np.concatenate([batch, ce_src, ce_dst]))
        src_seg_pos = src_seg_bpos = None
        if src_seg_counts is not None:
            seg_src = np.unique(ce_src)  # the runs' srcs, in run order
            src_seg_pos = torch.from_numpy(np.searchsorted(needed, seg_src))
            if bool(src_in_batch.all()):
                src_seg_bpos = torch.from_numpy(np.searchsorted(batch, seg_src))
        sum_closure += int(ce_src.shape[0])
        sum_needed += int(needed.shape[0])
        max_closure = max(max_closure, int(ce_src.shape[0]))

        l0_rows = batch if layer0_history else needed
        adj_rows = adj_index.slice(l0_rows)
        relprof_rows = relprof_index.slice(l0_rows)
        adj_rows_t = relprof_rows_t = None
        if csr_spmm:
            try:
                adj_rows, adj_rows_t = _csr_pair(adj_rows, csr_device)
                relprof_rows, relprof_rows_t = _csr_pair(relprof_rows, csr_device)
            except Exception as exc:  # CSR support varies across torch builds
                if c == 0:
                    logging.warning(f"[GAS] csr_spmm unavailable ({exc}); COO layer-0 path")
                adj_rows_t = relprof_rows_t = None
        g_dst = g_dst_t = g_src = g_src_t = s_mat = s_mat_t = None
        if csr_gather:
            try:
                n_e, n_nd, n_b = int(ce_src.shape[0]), int(needed.shape[0]), int(batch.shape[0])
                e_ar = torch.arange(n_e)
                dst_pos = torch.from_numpy(np.searchsorted(needed, ce_dst))
                src_pos = torch.from_numpy(np.searchsorted(needed, ce_src))
                g_dst, g_dst_t = _incidence_pair(e_ar, dst_pos, (n_e, n_nd), csr_device)
                g_src, g_src_t = _incidence_pair(e_ar, src_pos, (n_e, n_nd), csr_device)
                sel = torch.from_numpy(np.nonzero(src_in_batch)[0])
                b_pos = torch.from_numpy(np.searchsorted(batch, ce_src[src_in_batch]))
                s_mat, s_mat_t = _incidence_pair(b_pos, sel, (n_b, n_e), csr_device)
            except Exception as exc:
                if c == 0:
                    logging.warning(f"[GAS] csr_gather unavailable ({exc}); index path")
                g_dst = g_dst_t = g_src = g_src_t = s_mat = s_mat_t = None
        clusters.append(GASCluster(
            batch=torch.from_numpy(batch),
            needed=torch.from_numpy(needed),
            batch_pos_in_needed=torch.from_numpy(
                np.searchsorted(needed, batch)),
            e_dst_pos=torch.from_numpy(np.searchsorted(needed, ce_dst)),
            e_src_pos=torch.from_numpy(np.searchsorted(needed, ce_src)),
            e_rel=torch.from_numpy(ce_rel.astype(np.int64)),
            e_val=torch.from_numpy(ce_val.astype(np.float32)),
            e_src_batch_mask=torch.from_numpy(src_in_batch),
            e_src_batch_pos=torch.from_numpy(
                np.searchsorted(batch, ce_src[src_in_batch])),
            e_w=e_w,
            e_gid=e_gid,
            adj_rows=adj_rows,
            relprof_rows=relprof_rows,
            l0_batch_only=bool(layer0_history),
            adj_rows_t=adj_rows_t,
            relprof_rows_t=relprof_rows_t,
            e_val_pos=bool(ce_val.shape[0] == 0 or np.all(ce_val > 0)),
            g_dst=g_dst, g_dst_t=g_dst_t, g_src=g_src, g_src_t=g_src_t,
            s_mat=s_mat, s_mat_t=s_mat_t,
            src_seg_counts=src_seg_counts,
            src_seg_pos=src_seg_pos,
            src_seg_bpos=src_seg_bpos,
        ))
    # The redundancy factor is the per-epoch cost driver of this path: every
    # closure edge is gathered/attended once per channel per layer per head.
    logging.info(
        f"[GAS] {len(clusters)} clusters, closure_cap={closure_cap}, "
        f"softmax_history={softmax_history}: closure edges total {sum_closure} = "
        f"{sum_closure / max(n_edges_total, 1):.2f}x the {n_edges_total} local edges "
        f"(max cluster {max_closure}); needed rows total {sum_needed} = "
        f"{sum_needed / max(N, 1):.2f}x N={N}"
    )
    return clusters


def _cap_closure(
    ce_dst: np.ndarray, src_in_batch: np.ndarray, cap: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dst bounded sampling of a cluster's closure edges.

    ce_dst / src_in_batch: the closure's edge dsts and the mask of edges
    whose src is in the batch. Returns (keep, w): keep[e] is True for every
    batch-src edge and for at most `cap` of each dst's other in-edges
    (uniform without replacement); w[e] is 1.0 for batch-src edges and
    n_other(dst) / n_kept(dst) for the kept others, so that for each dst
      sum_{kept} w_e * exp(z_e)  is an unbiased estimate of
      sum_{all other} exp(z_e)
    (the batch-src terms are summed exactly). Vectorized: sort the other
    edges by (dst, random key) and keep the first `cap` of each dst run."""
    n = int(ce_dst.shape[0])
    keep = src_in_batch.copy()
    w = np.ones(n, dtype=np.float32)
    others = np.nonzero(~src_in_batch)[0]
    if others.size == 0:
        return keep, w
    d_o = ce_dst[others]
    key = rng.random(others.size)
    order = np.lexsort((key, d_o))  # primary: dst, secondary: random key
    d_sorted = d_o[order]
    first = np.concatenate([[0], np.nonzero(np.diff(d_sorted))[0] + 1])
    counts = np.diff(np.concatenate([first, [d_sorted.size]]))
    rank_in_group = np.arange(d_sorted.size) - np.repeat(first, counts)
    kept_sorted = rank_in_group < cap
    group_count = np.repeat(counts, counts)
    scale = group_count / np.minimum(group_count, cap)
    idx_kept = others[order[kept_sorted]]
    keep[idx_kept] = True
    w[idx_kept] = scale[kept_sorted].astype(np.float32)
    return keep, w


def _sparse_softmax_by_dst_weighted(
    logits: torch.Tensor, dst: torch.Tensor, N: int, w: torch.Tensor
) -> torch.Tensor:
    """_sparse_softmax_by_dst with per-edge weights in the denominator:
    att_e = exp(z_e) * w_e / sum_{e' in group(dst_e)} exp(z_e') * w_e'.
    With w from _cap_closure this is the exact softmax for every dst whose
    other-in-edges all fit under the cap, and an unbiased-denominator
    estimate for the hubs; batch-src edges (w=1) are the only ones whose
    att is consumed."""
    max_per_dst = torch.full((N,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_per_dst.index_reduce_(0, dst, logits, "amax", include_self=True)
    exp = (logits - max_per_dst[dst]).exp() * w.to(logits.dtype)
    sum_per_dst = torch.zeros(N, device=logits.device, dtype=exp.dtype)
    sum_per_dst.index_add_(0, dst, exp)
    return exp / sum_per_dst[dst].clamp_min(1e-12)


def _assign_clusters_cached(
    N: int, edges: np.ndarray, n_clusters: int, method: str, hub_degree: int = 0
) -> np.ndarray:
    """_assign_clusters with an on-disk cache under DISTEA_GAS_FAST_BUILD=1.
    The METIS clustering of a rank's local graph is a pure function of that
    graph, K, the method and the hub threshold (the main partition is fixed,
    so every run of a configuration rebuilds the same one: pymetis takes
    ~8 s of the DBP1M/N=1 build). Keyed on a hash of the
    edge array plus those parameters, stored as
    <DISTEA_DATA_CACHE_DIR or ~/.cache/distea>/gas_assign_<key>.npy; a hit
    returns the array an earlier run computed, a miss computes and writes it
    (atomic rename; an OSError only skips the write). Only metis_csr is
    cached, and only a METIS result: a failed call takes _assign_clusters'
    BFS-chunk fallback uncached, so a later run never reads a fallback
    clustering under the METIS key. A run that hits the cache does not pay
    for that clustering, like the main partition computed once by
    partition.py."""
    import hashlib
    import socket

    if not _fast_build() or method != "metis_csr" or n_clusters <= 1:
        return _assign_clusters(N, edges, n_clusters, method, hub_degree)
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(("gas_assign_v1", int(N), int(n_clusters), method, int(hub_degree),
                   tuple(edges.shape), str(edges.dtype))).encode())
    h.update(np.ascontiguousarray(edges).tobytes())
    folder = os.environ.get("DISTEA_DATA_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "distea")
    path = os.path.join(folder, f"gas_assign_{h.hexdigest()}.npy")
    try:
        assign = np.load(path, allow_pickle=False).astype(np.int64)
        if assign.shape == (N,) and (N == 0 or (assign.min() >= 0 and assign.max() < n_clusters)):
            logging.info(f"[GAS] cluster assignment from cache {path}")
            return assign
    except (OSError, ValueError):
        pass
    try:
        assign = _metis_csr_assign(N, edges, n_clusters, hub_degree)
    except Exception as e:  # noqa: BLE001 -- same contract as _assign_clusters
        logging.warning(f"[GAS] metis_csr failed ({type(e).__name__}: {e}); BFS-chunk fallback")
        return _assign_clusters(N, edges, n_clusters, "bfs", hub_degree)
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    try:
        os.makedirs(folder, exist_ok=True)
        with open(tmp, "wb") as f:
            np.save(f, assign.astype(np.int16 if n_clusters < 2**15 else np.int64))
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return assign


def _assign_clusters(
    N: int, edges: np.ndarray, n_clusters: int, method: str, hub_degree: int = 0
) -> np.ndarray:
    if n_clusters <= 1:
        return np.zeros(N, dtype=np.int64)
    n_edges = int(edges.shape[0])
    if method == "metis_csr":
        # METIS via the vendored package's C-level CSR entry
        # point -- no networkx graph, no size guard. Any failure (ImportError
        # on a machine without the Linux .so, a METIS error) degrades to the
        # BFS-chunk heuristic exactly like the networkx path does.
        try:
            return _metis_csr_assign(N, edges, n_clusters, hub_degree)
        except Exception as e:  # noqa: BLE001 -- same contract as the nxmetis branch
            logging.warning(
                f"[GAS] metis_csr failed ({type(e).__name__}: {e}); BFS-chunk fallback"
            )
            method = "bfs"
    if method == "rcm":
        return _rcm_assign(N, edges, n_clusters, hub_degree)
    if method == "metis" and _metis_too_large(N, n_edges):
        # nxmetis (networkx graph built by a pure-Python generator over every
        # edge, then METIS) takes ~1 min for a 62K-node local+halo graph
        # (100K/N=4), ~25 min for a DBP1M/N=16 rank graph, and does not
        # return within 48 min on a 644K-node / 3.38M-edge DBP1M/N=4 rank
        # graph -- and the vendored nxmetis fails outright on local graphs
        # of ~300K+ nodes anyway (see the except branch).
        # Above the (env-tunable) size guard, go straight to BFS-chunk.
        logging.info(
            f"[GAS] local graph N={N} E={n_edges} exceeds the nxmetis size guard "
            f"(DISTEA_GAS_METIS_MAX_NODES / DISTEA_GAS_METIS_MAX_EDGES); "
            "using BFS-chunk directly"
        )
        method = "bfs"
    if method == "metis":
        try:
            import networkx as nx
            import nxmetis
            logging.info(f"[GAS] nxmetis.partition on N={N} E={n_edges} K={n_clusters}")
            G = nx.Graph()
            G.add_nodes_from(range(N))
            G.add_edges_from((int(a), int(b)) for a, b in edges if a != b)
            _, parts = nxmetis.partition(G, n_clusters, recursive=True)
            assign = np.zeros(N, dtype=np.int64)
            for c, nodes in enumerate(parts):
                assign[np.asarray(nodes, dtype=np.int64)] = c
            return assign
        except ImportError:
            logging.warning("[GAS] nxmetis unavailable; BFS-chunk fallback")
        except Exception as e:
            # The vendored `modules/networkx-metis` has its own Python
            # 2->3 bug independent of anything here: its
            # `_convert_exceptions` decorator does
            #   except catch_types as e: exc = e
            #   ...
            #   six.reraise(convert_type, convert_type(exc[1]), exc[2])
            # binding `exc` to the exception INSTANCE (not sys.exc_info()'s
            # 3-tuple) whenever nxmetis.partition's own C call raises
            # MetisError/ValueError/TypeError directly -- so `exc[1]`
            # crashes with "'MetisError' object is not subscriptable",
            # masking whatever real METIS failure triggered it. Not observed
            # at 100K scale; at DBP1M N=16 about half of the ranks'
            # local+halo graphs (up to ~300K nodes) hit this. Not
            # worth patching the submodule for a message string; every
            # cluster-count/graph-shape METIS could plausibly reject
            # degrades to the existing BFS-chunk heuristic instead of
            # crashing the whole run.
            logging.warning(
                f"[GAS] nxmetis.partition failed ({type(e).__name__}: {e}); "
                "BFS-chunk fallback"
            )
    # BFS-ordered contiguous chunks: cheap locality proxy.
    #
    # Vectorized (scipy CSR + compiled BFS) instead of a pure-Python
    # adjacency dict + manual stack, which does not finish within 45 minutes
    # on DBP1M's ~640K-node/3.4M-edge per-rank graphs. With
    # `method="metis"` this is the path taken at DBP1M scale (the vendored
    # nxmetis's own bug means it falls through here for local+halo graphs
    # this large, see the `except Exception` branch above).
    # `connected_components` finds every component's membership in one C
    # call instead of N Python iterations;
    # `breadth_first_order` is scipy's own compiled BFS. The Python loop
    # left below only runs once per CONNECTED COMPONENT (typically O(1)
    # for a real knowledge graph), not once per node.
    logging.info(f"[GAS] BFS-chunk clustering on N={N} E={n_edges} K={n_clusters}")
    order = np.arange(N, dtype=np.int64) if len(edges) == 0 else _bfs_chunk_order(N, edges)
    assign = np.zeros(N, dtype=np.int64)
    size = (N + n_clusters - 1) // n_clusters
    assign[order] = np.minimum(np.arange(N, dtype=np.int64) // size, n_clusters - 1)
    return assign


def _metis_csr_assign(
    N: int, edges: np.ndarray, n_clusters: int, hub_degree: int
) -> np.ndarray:
    """METIS clustering of the local graph at DBP1M scale.

    Calls pymetis or the vendored networkx-metis package's C-level entry
    point (nxmetis.metis.part_graph on xadj/adjncy CSR arrays, the same call
    nxmetis.partition() makes after its own conversion) on a symmetric CSR
    built with scipy -- no networkx Graph and no per-edge Python generator,
    which is what makes the `method="metis"` path take >48 min on a
    644K-node / 7.04M-edge DBP1M/N=4 rank graph while partition.py's METIS
    on the full 1.12M-node KG1 completes.

    hub_degree > 0 drops every node of (undirected, deduplicated) degree
    above it from the graph METIS sees and assigns those hubs round-robin
    afterwards. Hubs (type/country entities, in-degree up to 85K on DBP1M)
    are adjacent to every cluster, so leaving them in makes every bisection
    cut through them and hides the real locality structure; their own
    placement barely matters for the closure (their out-edges are few and
    their in-edges are the capped `others`, see build_gas_clusters).

    Why the cut matters for GAS: an edge (s->d) lands in the closure of
    every cluster containing any in-neighbour of d, so non-hub edges are
    replicated once per distinct cluster among d's in-neighbours -- with
    BFS-chunk clusters on DBP1M/N=4 that replication alone leaves the
    per-epoch edge work at 4.5x the local edge count even after capping
    hubs (K=32: cap=0 7.34x, cap=64 4.48x). A low-cut
    partition keeps d's in-neighbours together and drives it toward 1x."""
    import time as _time

    t0 = _time.perf_counter()
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adj = sp.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(N, N)
    )
    adj.sum_duplicates()
    deg = np.diff(adj.indptr)
    hub = (deg > hub_degree) if hub_degree > 0 else np.zeros(N, dtype=bool)
    n_hub = int(hub.sum())
    if n_hub:
        e_keep = ~hub[rows] & ~hub[cols]
        adj = sp.csr_matrix(
            (np.ones(int(e_keep.sum()), dtype=np.int8), (rows[e_keep], cols[e_keep])),
            shape=(N, N),
        )
        adj.sum_duplicates()
    xadj = adj.indptr.astype(np.int64).tolist()
    adjncy = adj.indices.astype(np.int64).tolist()
    # Prefer pymetis (a release build of METIS 5): the vendored
    # networkx-metis extension runs for >30 min without returning on a
    # DBP1M rank graph -- its METIS is built with the assertion checks on
    # (it can raise a libmetis__CheckBnd2 assertion), so every refinement
    # pass re-validates the whole boundary. Fall back to the vendored entry
    # point only where pymetis is not importable (small graphs, tests).
    try:
        import pymetis

        objval, part = pymetis.part_graph(
            int(n_clusters), xadj=xadj, adjncy=adjncy, recursive=True
        )
        backend = "pymetis"
    except ImportError:
        from nxmetis import metis as _nxm  # vendored C extension (Linux .so)

        # options=None -> METIS defaults (zero-based numbering); recursive
        # bisection, the mode partition.py found robust on disconnected KGs.
        objval, part = _nxm.part_graph(xadj, adjncy, int(n_clusters), recursive=True)
        backend = "nxmetis"
    assign = np.asarray(part, dtype=np.int64)
    if n_hub:
        hub_idx = np.nonzero(hub)[0]
        assign[hub_idx] = np.arange(n_hub, dtype=np.int64) % n_clusters
    sizes = np.bincount(assign, minlength=n_clusters)
    logging.info(
        f"[GAS] metis_csr ({backend}): N={N}, undirected edges {adj.nnz // 2} after "
        f"dropping {n_hub} hubs (degree > {hub_degree}), K={n_clusters}, edge-cut "
        f"{objval}, part sizes {int(sizes.min())}-{int(sizes.max())}, "
        f"{_time.perf_counter() - t0:.1f}s"
    )
    return assign


def _rcm_assign(N: int, edges: np.ndarray, n_clusters: int, hub_degree: int) -> np.ndarray:
    """Zero-dependency alternative to METIS: order the nodes by
    scipy's reverse Cuthill-McKee (bandwidth minimisation -- a BFS that
    visits low-degree neighbours first, so that CONTIGUOUS index ranges have
    few edges leaving them) with hub edges (degree > hub_degree) removed,
    then cut the order into equal-size chunks. Same chunking as the
    BFS-chunk fallback, better locality; seconds at DBP1M scale."""
    import time as _time

    from scipy.sparse.csgraph import reverse_cuthill_mckee

    t0 = _time.perf_counter()
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    deg = np.bincount(rows, minlength=N)
    hub = (deg > hub_degree) if hub_degree > 0 else np.zeros(N, dtype=bool)
    e_keep = ~hub[rows] & ~hub[cols]
    adj = sp.csr_matrix(
        (np.ones(int(e_keep.sum()), dtype=np.int8), (rows[e_keep], cols[e_keep])),
        shape=(N, N),
    )
    order = np.asarray(reverse_cuthill_mckee(adj, symmetric_mode=True), dtype=np.int64)
    assign = np.zeros(N, dtype=np.int64)
    size = (N + n_clusters - 1) // n_clusters
    assign[order] = np.minimum(np.arange(N, dtype=np.int64) // size, n_clusters - 1)
    logging.info(
        f"[GAS] rcm: N={N}, {int(hub.sum())} hubs (degree > {hub_degree}) excluded from "
        f"the ordering, K={n_clusters}, {_time.perf_counter() - t0:.1f}s"
    )
    return assign


def _metis_too_large(n_nodes: int, n_edges: int) -> bool:
    """Size guard for the nxmetis attempt in _assign_clusters. Read from the
    environment at call time so a job can retune it without a code change:
    DISTEA_GAS_METIS_MAX_NODES (default 200000) / DISTEA_GAS_METIS_MAX_EDGES
    (default 1500000). Defaults sit between the largest local graph nxmetis
    is known to handle (100K/N=4: 62K nodes, 52 s) and the smallest known to
    stall for tens of minutes or fail (DBP1M: ~300K+ nodes)."""
    import os

    max_nodes = int(os.environ.get("DISTEA_GAS_METIS_MAX_NODES", "200000"))
    max_edges = int(os.environ.get("DISTEA_GAS_METIS_MAX_EDGES", "1500000"))
    return n_nodes > max_nodes or n_edges > max_edges


def _bfs_chunk_order(N: int, edges: np.ndarray) -> np.ndarray:
    """Node visiting order for `_assign_clusters`'s BFS-chunk fallback:
    concatenate each connected component's own breadth_first_order (root =
    that component's lowest-index node), processed in ascending component
    order. Same locality intent as a literal from-scratch BFS over every
    node, computed without a Python-level loop over edges or nodes."""
    src, dst = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adj = sp.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(N, N)
    )
    n_comp, labels = connected_components(adj, directed=False)
    comp_order = np.argsort(labels, kind="stable")
    boundaries = np.searchsorted(labels[comp_order], np.arange(n_comp + 1))
    order = np.empty(N, dtype=np.int64)
    pos = 0
    for c in range(n_comp):
        nodes = comp_order[boundaries[c]:boundaries[c + 1]]
        bfs = breadth_first_order(
            adj, int(nodes[0]), directed=False, return_predecessors=False
        )
        order[pos:pos + bfs.shape[0]] = bfs
        pos += bfs.shape[0]
    return order


def _gas_profile_level() -> int:
    """DISTEA_GAS_PROFILE=1: per-epoch wall-time breakdown of the GAS step
    (history pull / layer-0 sparse mm / attention heads / history push in
    the forward; loss / backward / optimizer step / final_hist update in the
    epoch loop), logged by rank 0. Inserts torch.cuda.synchronize() at every
    boundary, so it perturbs the timing it measures by a few percent -- for
    attribution runs only, never for the reported numbers.
    =2 additionally wraps ONE epoch (the third call of gas_train_epoch, past
    the allocator warm-up) in torch.profiler on rank 0 and logs the kernel
    table sorted by device time -- the level the per-phase numbers cannot
    answer (which kernels make up "backward"). Default 0 = off."""
    import os

    v = os.environ.get("DISTEA_GAS_PROFILE", "0").strip().lower()
    if v in ("1", "true", "yes"):
        return 1
    try:
        return max(int(v), 0)
    except ValueError:
        return 0


def _gas_profile_enabled() -> bool:
    return _gas_profile_level() >= 1


_GAS_EPOCH_CALLS = {"n": 0}  # gas_train_epoch call counter (profiler epoch pick)


def _prof_mark(acc: dict | None, key: str, t0: float, device: torch.device) -> float:
    """Accumulate the time since t0 into acc[key] (after a device sync) and
    return a fresh timestamp; no-op returning t0 when acc is None."""
    if acc is None:
        return t0
    import time as _time

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    now = _time.perf_counter()
    acc[key] = acc.get(key, 0.0) + (now - t0)
    return now


def _gas_checkpoint_enabled() -> bool:
    """DISTEA_GAS_CHECKPOINT=1 recomputes each attention head's per-edge
    intermediates during backward instead of keeping them all alive until
    the cluster's loss.backward(). Read at call time; default off (trades
    recomputation time for peak memory)."""
    import os

    return os.environ.get("DISTEA_GAS_CHECKPOINT", "0").lower() in ("1", "true", "yes")


def gather_rows(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """table[idx] for a per-edge gather whose BACKWARD must scatter-add into
    rows with extremely skewed multiplicity.

    Forward-identical to `table[idx]`. The difference is the backward kernel:
    advanced indexing's backward is `index_put_(accumulate=True)`, whose CUDA
    sort-based implementation walks each run of equal indices serially, so a
    relation id that labels millions of DBP1M edges, or a hub entity with
    85K in-edges, becomes a single serial chain of row additions.
    `F.embedding`'s backward (`embedding_dense_backward_cuda`) sorts the
    same indices but reduces each run with parallel partial sums (with
    plain indexing the backward dominated the epoch, ~16.5 s of 19 s at
    DBP1M). Gradient values are the same sums in
    a different order (fp32 summation noise only); forward bit-identical.
    `table` may be any 2-D tensor (a parameter or an intermediate)."""
    return torch.nn.functional.embedding(idx, table)


class _SegmentSoftmax(torch.autograd.Function):
    """Softmax over CONTIGUOUS segments of a 1-D logit vector (segment i =
    the next counts[i] entries), with an analytic backward that uses only
    segmented reductions and broadcasts -- no index_add_/index_reduce_
    atomics. Under --attn_softmax src a cluster's edges are sorted by src and
    a DBP1M hub has ~85K out-edges in one segment: the index-based
    _sparse_softmax_by_dst serialises its backward on that address (about
    2x slower backward), this path does not."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        maxes = torch.segment_reduce(logits, "max", lengths=counts)
        exp = (logits - torch.repeat_interleave(maxes, counts)).exp()
        sums = torch.segment_reduce(exp, "sum", lengths=counts)
        y = exp / torch.repeat_interleave(sums, counts).clamp_min(1e-12)
        ctx.save_for_backward(y, counts)
        return y

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        y, counts = ctx.saved_tensors
        t = torch.segment_reduce(g * y, "sum", lengths=counts)
        return y * (g - torch.repeat_interleave(t, counts)), None


_USE_SEGMENT_SOFTMAX = True


def _segment_softmax_sorted(logits: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """Segment softmax for a SORTED group-id vector `seg` (contiguous
    groups); same values and gradients as _sparse_softmax_by_dst(logits,
    seg, N), see _SegmentSoftmax."""
    _, counts = torch.unique_consecutive(seg, return_counts=True)
    return _SegmentSoftmax.apply(logits, counts)


def _gas_attention_head(
    prev_needed: torch.Tensor,
    rel_emb: torch.Tensor,
    kernel: torch.Tensor,
    e_rel: torch.Tensor,
    e_val: torch.Tensor,
    e_dst_pos: torch.Tensor,
    e_src_pos: torch.Tensor,
    src_b_mask: torch.Tensor,
    src_b_pos: torch.Tensor,
    n_needed: int,
    n_batch: int,
    dim: int,
    e_w: torch.Tensor | None = None,
    e_stale: torch.Tensor | None = None,
    dst_sum_rows: torch.Tensor | None = None,
    e_val_pos: bool = False,
    csr_gather: tuple | None = None,
    softmax_axis: str = "dst",
    src_seg_counts: torch.Tensor | None = None,
    seg_mp: tuple | None = None,
):
    """One RREA attention head over a cluster's closure edge set; returns the
    (n_batch, dim) aggregated messages for the batch nodes. e_w (closure_cap
    > 0 only) weights the softmax denominators, see
    _sparse_softmax_by_dst_weighted; None keeps the exact path bit-identical.

    e_stale / dst_sum_rows (--gas_softmax_history): the cluster's edges are
    its batch-src edges only; the denominator of dst d is
        fresh sum over this cluster's edges into d
        + (stale per-dst total of d - stale values of those same edges),
    computed in fp64 from logits clamped to +-300. In that mode the return
    value is (new_features, fresh exp per edge, new per-dst totals) so the
    caller can write the history back OUTSIDE any checkpointed region
    (a checkpointed function is re-run during backward and must not mutate
    the tables itself).

    Every intermediate here is (E_closure, dim) or (E_closure, 3*dim):
    rel_vec, neighs, selfs, bias, neighs_reflected, att_in, msg. On
    DBP1M/N=4 a single cluster's closure can reach ~3.8M edges,
    so with 2 channels x depth 2 x heads x ~6 saved tensors each, holding
    all of them until loss.backward() is what exhausted an 80 GB card
    (59 GB PyTorch-allocated at the `att_in` cat). Factored out so
    gas_forward_batch can run it under torch.utils.checkpoint: the block
    then keeps only its (n_batch, dim) output alive and recomputes the
    per-edge tensors one head at a time during backward -- numerically the
    same ops (tests/test_gas_checkpoint.py), peak memory ~one head's
    worth instead of all of them."""
    # Per-edge gathers go through gather_rows (F.embedding): same values as
    # `x[idx]`, but a backward that does not serialise on hub entities /
    # frequent relations -- see gather_rows.
    if e_val_pos:
        # normalize(v * r) == normalize(r) for any scalar v > 0, so with
        # all-positive e_val (1/count soft weights) the per-edge scale is a
        # no-op after the L2 normalisation: normalise the (n_rel, dim) table
        # once and gather, instead of an (E, dim) multiply + norm + divide
        # per head call (forward and backward) -- the normalize's div/norm
        # and their backward were ~0.9 s of a 4.7 s DBP1M epoch.
        # Identical values and gradients.
        rel_vec = gather_rows(torch.nn.functional.normalize(rel_emb, p=2, dim=1), e_rel)
    else:
        rel_vec = gather_rows(rel_emb, e_rel) * e_val.unsqueeze(1)
        rel_vec = torch.nn.functional.normalize(rel_vec, p=2, dim=1)
    if seg_mp is not None:
        # (src_seg_pos, src_seg_bpos, g_dst_t), edges sorted by src: row
        # gathers forward, run sums / the dst incidence transpose backward
        src_seg_pos, src_seg_bpos, g_dst_t = seg_mp
        neighs = _GatherRowsCsrBwd.apply(prev_needed, e_dst_pos, g_dst_t)
        selfs = _SegRepeatRows.apply(prev_needed, e_src_pos, src_seg_pos, src_seg_counts)
    elif csr_gather is not None:
        # (g_dst, g_dst_t, g_src, g_src_t, s_mat, s_mat_t): the gathers as
        # incidence spmm's -- same values, sort-free segmented backward.
        g_dst, g_dst_t, g_src, g_src_t, s_mat, s_mat_t = csr_gather
        neighs = _SpMMFixedT.apply(prev_needed, g_dst, g_dst_t)
        selfs = _SpMMFixedT.apply(prev_needed, g_src, g_src_t)
    else:
        neighs = gather_rows(prev_needed, e_dst_pos)
        selfs = gather_rows(prev_needed, e_src_pos)
    neighs_reflected, att_logits = edge_attention_core(selfs, neighs, rel_vec, kernel)
    if softmax_axis == "src":
        # Original RREA grouping: softmax over each SELF node's
        # own edges. Every edge whose src is a batch node is in the cluster
        # (batch-src edge set), so the group is complete locally and no
        # denominator history is needed; gas_forward_batch skips the
        # history under src grouping, this branch only keeps the return
        # shape well-formed if a caller still passes the history tensors.
        if src_seg_counts is not None:
            # edges sorted by src at build time (src_sorted_edges): the run
            # lengths are precomputed, no per-call sortedness check or
            # unique_consecutive (each a host sync)
            att = _SegmentSoftmax.apply(att_logits, src_seg_counts)
        elif _USE_SEGMENT_SOFTMAX and e_src_pos.numel() > 1 and bool((e_src_pos[1:] >= e_src_pos[:-1]).all()):
            att = _segment_softmax_sorted(att_logits, e_src_pos)
        else:
            att = _sparse_softmax_by_dst(att_logits, e_src_pos, n_needed)
        if e_stale is not None:
            e_fresh = att_logits.detach().double()
            denom = dst_sum_rows
    elif e_stale is not None:
        z = att_logits.double().clamp(-300.0, 300.0)
        e_fresh = z.exp()
        s_fresh = torch.zeros(n_needed, dtype=torch.float64, device=z.device)
        s_fresh.index_add_(0, e_dst_pos, e_fresh)
        s_stale = torch.zeros(n_needed, dtype=torch.float64, device=z.device)
        s_stale.index_add_(0, e_dst_pos, e_stale)
        denom = (dst_sum_rows - s_stale).clamp_min(0.0) + s_fresh
        denom_e = gather_rows(denom.unsqueeze(1), e_dst_pos).squeeze(1)
        att = (e_fresh / denom_e.clamp_min(1e-300)).to(att_logits.dtype)
    elif e_w is None:
        att = _sparse_softmax_by_dst(att_logits, e_dst_pos, n_needed)
    else:
        att = _sparse_softmax_by_dst_weighted(att_logits, e_dst_pos, n_needed, e_w)
    msg = neighs_reflected * att.unsqueeze(1)
    if seg_mp is not None:
        new_features = _SegSumToRows.apply(msg, src_seg_counts, src_seg_bpos, src_b_pos, n_batch)
    elif csr_gather is not None:
        new_features = _SpMMFixedT.apply(msg, s_mat, s_mat_t)  # (n_batch, dim)
    else:
        new_features = torch.zeros(n_batch, dim, device=msg.device, dtype=msg.dtype)
        new_features.index_add_(0, src_b_pos, msg[src_b_mask])
    if e_stale is not None:
        return new_features, e_fresh.detach(), denom.detach()
    return new_features


def gas_forward_batch(
    model: RREA,
    cluster: GASCluster,
    h_init: torch.Tensor,
    history: GASHistory,
    device: torch.device,
) -> torch.Tensor:
    """
    Exact-neighborhood forward for one cluster's batch nodes; returns
    (len(batch), model.output_width) matching RREA.forward()'s layout
    (per-channel dense concat of layer 0..depth, ent channel then rel).
    Pushes each computed attention layer's batch rows into `history`.
    """
    prof = getattr(history, "_prof", None)  # dict when DISTEA_GAS_PROFILE=1
    if prof is not None:
        import time as _time

        _pt = _time.perf_counter()
    enc = model.encoder
    rel_emb = model.rel_emb.to(device)
    needed = cluster.needed.to(device)
    batch = cluster.batch.to(device)
    batch_pos = cluster.batch_pos_in_needed.to(device)
    e_dst_pos = cluster.e_dst_pos.to(device)
    e_src_pos = cluster.e_src_pos.to(device)
    e_rel = cluster.e_rel.to(device)
    e_val = cluster.e_val.to(device)
    src_b_mask = cluster.e_src_batch_mask.to(device)
    src_b_pos = cluster.e_src_batch_pos.to(device)
    e_w = cluster.e_w.to(device) if cluster.e_w is not None else None
    use_smh = history.softmax and cluster.e_gid is not None
    e_gid = cluster.e_gid.to(device) if use_smh else None
    n_needed = needed.size(0)
    n_batch = batch.size(0)
    use_ckpt = _gas_checkpoint_enabled() and torch.is_grad_enabled()
    if prof is not None:
        _pt = _prof_mark(prof, "fwd_index_h2d", _pt, device)

    # Layer-0 rows: exact for the whole needed set by default (grads flow
    # into the learnable tables, module docstring point 2). With
    # --gas_layer0_history the exact rows are the batch's only; the other
    # needed rows are pulled from the layer-0 history (stale <= 1 epoch, no
    # grad) and the fresh batch rows are pushed back, exactly as layers
    # >= 1 are handled below.
    l0_batch_only = bool(getattr(cluster, "l0_batch_only", False))
    if l0_batch_only and not history.layer0:
        raise ValueError("cluster was built with layer0_history=True but the "
                         "GASHistory has no layer-0 tables (layer0=False)")
    with torch.autocast(device_type=device.type, enabled=False):
        f0 = {
            "ent": _layer0_mm(cluster, "adj_rows", h_init, device),
            "rel": _layer0_mm(cluster, "relprof_rows", rel_emb, device),
        }
    if l0_batch_only:
        for channel in ("ent", "rel"):
            f0_batch = f0[channel]
            history.push(channel, 0, batch, f0_batch)
            f0[channel] = history.pull(channel, 0, needed, device).index_copy(
                0, batch_pos, f0_batch)
    if getattr(enc, "input_relu", False):
        # mirrors NRGraphAttention.forward's input activation (original
        # layer.py); applied after any layer-0 history round trip so pulled
        # rows (stored pre-activation) are treated the same as fresh ones
        f0 = {k: torch.relu(v) for k, v in f0.items()}
    if prof is not None:
        _pt = _prof_mark(prof, "fwd_layer0_spmm", _pt, device)

    outputs = []
    for channel in ("ent", "rel"):
        chan_outs = [f0[channel][batch_pos]]
        h_batch_prev = None  # fresh layer-(l-1) batch rows, with grad
        for layer in range(enc.depth):
            if layer == 0:
                prev_needed = f0[channel]
            else:
                pulled = history.pull(channel, layer, needed, device)
                prev_needed = pulled.index_copy(0, batch_pos, h_batch_prev)
            if prof is not None:
                _pt = _prof_mark(prof, "fwd_history_pull", _pt, device)

            head_outs = []
            for head in range(enc.attn_heads):
                head_args = (
                    prev_needed, rel_emb, enc._kernel(layer, head),
                    e_rel, e_val, e_dst_pos, e_src_pos, src_b_mask, src_b_pos,
                    n_needed, n_batch, enc.dim, e_w,
                )
                key = (channel, layer, head)
                # Under src grouping the softmax is complete
                # within the cluster's own edge set: the per-dst denominator
                # history (pull/push + fp64 sums) is dead work, skip it.
                smh_here = use_smh and getattr(enc, "softmax_axis", "dst") != "src"
                if smh_here:
                    # Read the stale denominators before, write the fresh
                    # ones after -- never inside -- the (possibly
                    # checkpointed) head, see _gas_attention_head.
                    e_stale, dst_sum_rows = history.pull_denominators(key, e_gid, needed, device)
                    head_args = head_args + (e_stale, dst_sum_rows)
                head_kw = {"e_val_pos": bool(getattr(cluster, "e_val_pos", False)),
                           "softmax_axis": getattr(enc, "softmax_axis", "dst")}
                if (head_kw["softmax_axis"] == "src"
                        and getattr(cluster, "src_seg_counts", None) is not None):
                    head_kw["src_seg_counts"] = cluster.src_seg_counts.to(device)
                if getattr(cluster, "g_dst", None) is not None:
                    head_kw["csr_gather"] = tuple(
                        getattr(cluster, k).to(device)
                        for k in ("g_dst", "g_dst_t", "g_src", "g_src_t", "s_mat", "s_mat_t"))
                    if ("src_seg_counts" in head_kw and _segment_mp_enabled()
                            and getattr(cluster, "src_seg_bpos", None) is not None):
                        head_kw["seg_mp"] = (cluster.src_seg_pos.to(device),
                                             cluster.src_seg_bpos.to(device),
                                             head_kw["csr_gather"][1])
                if use_ckpt:
                    out = checkpoint(_gas_attention_head, *head_args, use_reentrant=False, **head_kw)
                else:
                    out = _gas_attention_head(*head_args, **head_kw)
                if prof is not None:
                    _pt = _prof_mark(prof, "fwd_attention_heads", _pt, device)
                if smh_here:
                    new_features, e_fresh, denom = out
                    history.push_denominators(key, e_gid, e_fresh, needed, denom)
                else:
                    new_features = out
                head_outs.append(new_features)

            h_batch_prev = torch.relu(torch.stack(head_outs, dim=0).mean(dim=0))
            chan_outs.append(h_batch_prev)
            history.push(channel, layer + 1, batch, h_batch_prev)
            if prof is not None:
                _pt = _prof_mark(prof, "fwd_history_push", _pt, device)
        outputs.extend(chan_outs)

    return torch.cat(outputs, dim=1)


def _compact_loss_rows(
    final_hist: torch.Tensor,
    emb_batch: torch.Tensor,
    batch_dev: torch.Tensor,
    in_batch: torch.Tensor,
    pairs: torch.Tensor,
    neg1: torch.Tensor,
    neg2: torch.Tensor,
    kg1_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compact embedding table for the loss: the unique node ids touched by
    `pairs` / `neg1` / `neg2` (sorted), with batch rows taken from
    `emb_batch` (grad) and the others from `final_hist` (detached), plus the
    same three index tensors remapped into it and the number of KG1 rows
    (ids < kg1_size come first, so alignment_loss's KG-boundary guard keeps
    its meaning). `batch_dev` must be sorted (build_gas_clusters sorts it)."""
    ids = torch.unique(torch.cat([pairs.reshape(-1), neg1.reshape(-1), neg2.reshape(-1)]))
    rows = final_hist.detach()[ids]
    sel = in_batch[ids]
    if bool(sel.any()):
        pos_c = torch.nonzero(sel).squeeze(1)
        pos_b = torch.searchsorted(batch_dev, ids[sel])
        rows = rows.index_copy(0, pos_c, emb_batch[pos_b])
    pairs_c = torch.searchsorted(ids, pairs)
    neg1_c = torch.searchsorted(ids, neg1)
    neg2_c = torch.searchsorted(ids, neg2)
    kg1_c = int((ids < kg1_size).sum())
    return rows, pairs_c, neg1_c, neg2_c, kg1_c



def neg_pool_subset(trainer, n_nodes: int, frac: float, device):
    """Fixed random subset of row ids for `neg_pool_frac` (None when frac>=1).

    Chosen once per (trainer, n_nodes, frac) with a generator seeded by
    `12345 + rank`, so every epoch of a run samples negatives from the same
    subset -- the way a real partition pins each rank's negative pool."""
    frac = float(frac) if frac is not None else 1.0
    if frac >= 1.0 or n_nodes <= 1:
        return None
    m = max(int(round(n_nodes * frac)), 1)
    key = (int(n_nodes), m)
    cached = getattr(trainer, "_neg_pool_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345 + int(getattr(trainer, "rank", 0) or 0))
    pool = torch.randperm(n_nodes, generator=gen)[:m].to(device)
    trainer._neg_pool_cache = (key, pool)
    return pool


def sample_uniform_negatives(n_nodes: int, shape, device, pool=None):
    """torch.randint over the table, or over `pool`'s rows when given."""
    if pool is None:
        return torch.randint(0, n_nodes, shape, device=device)
    return pool[torch.randint(0, pool.numel(), shape, device=device)]


def gas_train_epoch(
    trainer,
    clusters: list[GASCluster],
    history: GASHistory,
    final_hist: torch.Tensor,
    h_init: torch.Tensor,
    train_pairs: torch.Tensor,
    kg1_size: int,
    device: torch.device,
    hard_neg=None,
    hard_neg_k: int = 0,
    nhsm_negs: str = "full",
    nhsm_pool: int = 10000,
    neg_pool_frac: float = 1.0,
) -> float:
    """
    One GAS training epoch: clusters in random order; per cluster, the
    full embedding table is assembled as final-history (detached, stale)
    with the batch's rows replaced by the fresh forward, the loss runs
    over the train pairs touching the batch, and one optimizer step is
    taken. `final_hist` ((N, output_width), device-resident) is updated
    with the fresh batch rows after each step.

    Reuses the DistTrainer's loss configuration (loss_fn margin/nhsm,
    use_infonce + loss_weight, margin, neg_k). Two divergences from
    DistTrainer.train_epoch: negatives
    are re-drawn per cluster per epoch (neg_resample_every is not
    honored -- the pair subset changes with the batch, so cached (P, k)
    tensors don't line up), and the consistency/pair_weights extras are
    not wired.

    `hard_neg` (hard_neg.HardNegPools mined from
    `final_hist` for exactly these `train_pairs`, same row order) +
    `hard_neg_k` > 0: each pair's negatives become the uniform (p, neg_k)
    draw plus `hard_neg_k` columns sampled from its pool (hard_neg.
    merge_negatives). None / 0 reproduces the default path bit-for-bit
    (no extra RNG draws).

    `nhsm_negs="compact"` (with `--loss nhsm`): instead of
    the whole (N, width) table, NHSM's negatives are a per-cluster uniform
    pool of `nhsm_pool` rows (ClusterEA's in-batch shape) united with the
    pair rows, assembled by `_compact_loss_rows` (batch rows with grad). At
    DBP1M the full form is ~15 s/epoch and memory-unsafe; the compact form
    is +0.3-1.0 s/epoch. "full" keeps the default behaviour.
    """
    # DISTEA_GAS_TF32=1: TensorFloat-32 for the cuBLAS matmuls of the training
    # step only (the InfoNCE (P, P) similarity and its backward -- ~0.5 s of a
    # 2.7 s DBP1M epoch); restored before returning so the
    # evaluation's similarity matrices stay full fp32. Approximate (10-bit
    # mantissa products). Default off.
    import os as _os

    _tf32 = (device.type == "cuda"
             and _os.environ.get("DISTEA_GAS_TF32", "0").lower() in ("1", "true", "yes"))
    _tf32_prev = torch.backends.cuda.matmul.allow_tf32 if _tf32 else None
    if _tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return _gas_train_epoch_impl(trainer, clusters, history, final_hist, h_init,
                                     train_pairs, kg1_size, device,
                                     hard_neg=hard_neg, hard_neg_k=hard_neg_k,
                                     nhsm_negs=nhsm_negs, nhsm_pool=nhsm_pool,
                                     neg_pool_frac=neg_pool_frac)
    finally:
        if _tf32:
            torch.backends.cuda.matmul.allow_tf32 = _tf32_prev


def _gas_train_epoch_impl(
    trainer,
    clusters: list[GASCluster],
    history: GASHistory,
    final_hist: torch.Tensor,
    h_init: torch.Tensor,
    train_pairs: torch.Tensor,
    kg1_size: int,
    device: torch.device,
    hard_neg=None,
    hard_neg_k: int = 0,
    nhsm_negs: str = "full",
    nhsm_pool: int = 10000,
    neg_pool_frac: float = 1.0,
) -> float:
    from hard_neg import merge_negatives
    from models.gcn import alignment_loss as _align_loss
    from models.gcn import combined_align_loss, nhsm_loss

    model = trainer.model
    model.train()
    n_nodes = final_hist.size(0)
    pair_dev = train_pairs.to(device)
    # Ablation knob: neg_pool_frac < 1
    # draws the uniform negatives from a FIXED random subset of the local
    # table (chosen once per run, seeded by rank) instead of the whole
    # table -- a single-GPU stand-in for the smaller per-rank negative pool
    # a world_size=1/frac partition gives, with the graph itself untouched.
    # 1.0 (default) leaves the default path byte-for-byte unchanged.
    _neg_pool = neg_pool_subset(trainer, n_nodes, neg_pool_frac, device)
    # Original RREA output dropout, applied to the rows
    # the loss reads only -- gas_forward_batch's output (which feeds
    # final_hist and the history push) stays clean.
    _p_drop = float(getattr(model, "rrea_dropout", 0.0) or 0.0)
    _hn_active = hard_neg is not None and hard_neg_k > 0
    if _hn_active and len(hard_neg) != pair_dev.size(0):
        raise ValueError(
            f"hard_neg pools cover {len(hard_neg)} pairs but train_pairs has "
            f"{pair_dev.size(0)} rows -- mine them from the same pair tensor"
        )
    _hn1 = hard_neg.neg_kg1.to(device) if _hn_active else None
    _hn2 = hard_neg.neg_kg2.to(device) if _hn_active else None
    total, n_steps = 0.0, 0
    prof: dict | None = {} if _gas_profile_enabled() else None
    history._prof = prof  # gas_forward_batch adds its own phases when not None
    if prof is not None:
        import time as _time

        _t_epoch = _time.perf_counter()
        _pt = _t_epoch
    _GAS_EPOCH_CALLS["n"] += 1
    torch_prof = None
    if (prof is not None and _gas_profile_level() >= 2
            and _GAS_EPOCH_CALLS["n"] == 3 and getattr(trainer, "rank", 0) == 0):
        # Entered/exited by hand so the cluster loop below keeps its shape;
        # one epoch only, rank 0 only.
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as _torch_profile

        acts = [ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(ProfilerActivity.CUDA)
        torch_prof = _torch_profile(activities=acts)
        torch_prof.__enter__()

    for ci in torch.randperm(len(clusters)).tolist():
        c = clusters[ci]
        batch_dev = c.batch.to(device)
        trainer.optimizer.zero_grad()
        if prof is not None:
            _pt = _prof_mark(prof, "zero_grad+h2d", _pt, device)
        emb_batch = gas_forward_batch(model, c, h_init, history, device)
        if prof is not None:
            _pt = _prof_mark(prof, "fwd_total", _pt, device)

        in_batch = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        in_batch[batch_dev] = True
        sel = in_batch[pair_dev[:, 0]] | in_batch[pair_dev[:, 1]]
        pairs_sub = pair_dev[sel]

        if pairs_sub.size(0) > 0:
            if trainer.loss_fn == "nhsm" and nhsm_negs == "compact":
                # Negatives = a uniform pool of nhsm_pool rows (+ the pair
                # rows), on the compact table -- never the (N, width) view.
                pool = torch.randint(0, n_nodes, (max(int(nhsm_pool), 1),), device=device)
                rows, pairs_c, _, _, _ = _compact_loss_rows(
                    final_hist, emb_batch, batch_dev, in_batch, pairs_sub,
                    pool, pool[:0], kg1_size)
                if _p_drop > 0.0:
                    rows = torch.nn.functional.dropout(rows, _p_drop, training=True)
                loss = nhsm_loss(rows, pairs_c, gamma=trainer.nhsm_gamma)
            elif trainer.loss_fn == "nhsm":
                # NHSM's negatives are the WHOLE table, so it needs the full
                # (N, width) view.
                emb_full = final_hist.detach().index_copy(0, batch_dev, emb_batch)
                if _p_drop > 0.0:
                    emb_full = torch.nn.functional.dropout(emb_full, _p_drop, training=True)
                loss = nhsm_loss(emb_full, pairs_sub, gamma=trainer.nhsm_gamma)
            else:
                p = pairs_sub.size(0)
                neg1 = sample_uniform_negatives(n_nodes, (p, trainer.neg_k), device, _neg_pool)
                neg2 = sample_uniform_negatives(n_nodes, (p, trainer.neg_k), device, _neg_pool)
                if _hn_active:
                    # + hard_neg_k columns from each pair's mined pool
                    neg1 = merge_negatives(neg1, _hn1[sel], hard_neg_k)
                    neg2 = merge_negatives(neg2, _hn2[sel], hard_neg_k)
                # InfoNCE + margin only read the pair and negative rows, so
                # evaluate them on a compact table of exactly those rows
                # (fresh batch rows with grad, the rest from final_hist)
                # instead of an (N, width) index_copy view: on DBP1M/N=1
                # every emb_full[idx] inside the loss made autograd allocate
                # and zero a 2.6 GB (N, 384) gradient and add them up,
                # 12 times per cluster -- ~2 s of a 5.5 s backward.
                # Same values, same gradient paths.
                rows, pairs_c, neg1_c, neg2_c, kg1_c = _compact_loss_rows(
                    final_hist, emb_batch, batch_dev, in_batch, pairs_sub,
                    neg1, neg2, kg1_size)
                if _p_drop > 0.0:
                    rows = torch.nn.functional.dropout(rows, _p_drop, training=True)
                if trainer.use_infonce and trainer.loss_weight is not None:
                    loss = combined_align_loss(
                        rows, pairs_c, kg1_c, trainer.loss_weight,
                        neg_k=trainer.neg_k, margin=trainer.margin,
                        infonce_temp=trainer.infonce_temp,
                        neg_kg1=neg1_c, neg_kg2=neg2_c,
                    )
                else:
                    loss = _align_loss(
                        rows, pairs_c, kg1_c, neg_k=trainer.neg_k,
                        margin=trainer.margin, neg_kg1=neg1_c, neg_kg2=neg2_c,
                    )
            if prof is not None:
                _pt = _prof_mark(prof, "loss", _pt, device)
            loss.backward()
            if prof is not None:
                _pt = _prof_mark(prof, "backward", _pt, device)
            trainer.optimizer.step()
            total += float(loss.item())
            n_steps += 1
            if prof is not None:
                _pt = _prof_mark(prof, "optimizer_step", _pt, device)

        final_hist[batch_dev] = emb_batch.detach()
        if prof is not None:
            _pt = _prof_mark(prof, "final_hist", _pt, device)

    if torch_prof is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        torch_prof.__exit__(None, None, None)
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        try:
            table = torch_prof.key_averages().table(sort_by=sort_key, row_limit=40)
        except Exception as exc:  # key name differs across torch versions
            table = torch_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
            table = f"(sort_by={sort_key} failed: {exc}; sorted by self cpu time)\n" + table
        logging.info(f"[GAS torch.profiler] epoch call {_GAS_EPOCH_CALLS['n']}, "
                     f"{len(clusters)} clusters:\n{table}")
        _pt = _time.perf_counter()  # the trace post-processing is not refresh_dst_sums' time

    if history.softmax:
        drift = history.refresh_dst_sums()
        if drift > 1e-6:
            logging.debug(f"[GAS] softmax-history dst sums drifted by {drift:.3e}; rebuilt")
    if prof is not None:
        _pt = _prof_mark(prof, "refresh_dst_sums", _pt, device)
    if trainer.world_size > 1:
        trainer._allreduce_sync()
    if prof is not None:
        _pt = _prof_mark(prof, "allreduce", _pt, device)
        if getattr(trainer, "rank", 0) == 0:
            tot_s = _time.perf_counter() - _t_epoch
            parts = ", ".join(f"{k} {v:.2f}s" for k, v in sorted(prof.items(), key=lambda kv: -kv[1]))
            logging.info(
                f"[GAS profile] epoch {tot_s:.2f}s over {len(clusters)} clusters: {parts}"
            )
    history._prof = None
    return total / max(n_steps, 1)
