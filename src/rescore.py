"""
rescore.py — offline re-scoring of an exported embedding
bundle with alternative INFERENCE protocols, zero training cost.

The export bundle (`--export_dir`, written by eval.py::_save_export_bundle
from the final test eval) holds exactly the rows the online evaluation
scored: the test-side KG1 rows and every KG2 candidate row, raw embeddings.
Every variant below re-ranks those same rows, so differences are purely the
inference protocol:

  cosine_csls   the reference protocol (eval.py::_eval_metrics_auto, cosine +
                CSLS-10) -- the PARITY gate: must reproduce metrics.json
                (equal Hit@k, |dMRR| < 1e-9) in both the dense and the
                chunked path before any other variant's number is believed.
  l1_csls       the GCN-Align/ELsEA metric (negative cityblock) + CSLS.
  bidir         ClusterEA's SPARSE bidirectional top-K: row top-K of the
                CSLS matrix merged with column top-K (entries that are
                mutual top-K are summed, i.e. doubled), then a CSLS pass
                over the sparse scores. The dense "bidirectional CSLS" is an
                identity (CSLS(S^T)^T == CSLS(S)) -- only the sparse form
                changes anything. Ranks are within each row's kept entries,
                so Hit@k (k <= K) and a TRUNCATED MRR@K are reported, the
                way ClusterEA reports them.
  sinkhorn      label-FREE block-wise Sinkhorn: blocks come from DistEA's own
                partition files ({r}.npy KG1 ids + kg2_assignments.json) or
                MiniBatchKMeans on the normalised embeddings -- never from
                test labels (ClusterEA's released eval filters blocks by the
                test map). Log-domain Sinkhorn per block with
                uniform marginals a = 1/m, b = 1/n (blocks are not square),
                row top-K of the transport plan as scores; oversize blocks
                fall back to cosine top-K; `block_recall` (fraction of test
                pairs whose two sides share a block) is always reported as
                the hard ceiling of this component.
  fuse_sum      ClusterEA-style fusion of the sparse variants: each variant's
                scores min-shifted and min-max scaled, summed over the union.
  fuse_rrf      reciprocal-rank fusion, sum of 1 / (60 + rank).

All metric arithmetic for the dense variants is eval.py's own; the sparse
ranking helper mirrors _compute_metrics' tie rule (rank = 1 + #strictly
greater). CPU-testable (tests/test_rescore_variants.py).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from eval import (  # noqa: E402  (src/ on sys.path, as framework.py does)
    _eval_metrics_auto,
    _sim_rows,
)

# ---------------------------------------------------------------------------
#  Bundle I/O
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    kg1_emb: torch.Tensor            # (n1, D) fp32
    kg2_emb: torch.Tensor            # (n2, D) fp32
    kg1_ids: np.ndarray              # (n1,) global ids
    kg2_ids: np.ndarray              # (n2,) global ids
    link_pairs: list[tuple[int, int]]  # (row in kg1, col in kg2)
    test_links: list[tuple[int, int]]  # global ids
    meta: dict = field(default_factory=dict)

    @property
    def n1(self) -> int:
        return int(self.kg1_emb.size(0))

    @property
    def n2(self) -> int:
        return int(self.kg2_emb.size(0))


def load_bundle(export_dir: str) -> Bundle:
    """Read embeddings.npy / ids.npy / kg_labels.npy / test_links.json
    (+ meta.json if present) and split into the KG1 rows and KG2 columns
    the online eval used, in the same order."""
    emb = np.load(os.path.join(export_dir, "embeddings.npy"))
    ids = np.load(os.path.join(export_dir, "ids.npy")).astype(np.int64)
    kg = np.load(os.path.join(export_dir, "kg_labels.npy"))
    with open(os.path.join(export_dir, "test_links.json")) as f:
        test_links = [(int(a), int(b)) for a, b in json.load(f)]
    meta = {}
    mp = os.path.join(export_dir, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    m1 = kg == 0
    m2 = kg == 1
    kg1_ids, kg2_ids = ids[m1], ids[m2]
    emb_t = torch.from_numpy(np.ascontiguousarray(emb)).float()
    kg1_emb = emb_t[torch.from_numpy(m1)]
    kg2_emb = emb_t[torch.from_numpy(m2)]
    r1 = {int(g): i for i, g in enumerate(kg1_ids)}
    r2 = {int(g): i for i, g in enumerate(kg2_ids)}
    link_pairs = []
    for e1, e2 in test_links:
        a, b = r1.get(e1), r2.get(e2)
        if a is not None and b is not None:
            link_pairs.append((a, b))
    return Bundle(kg1_emb, kg2_emb, kg1_ids, kg2_ids, link_pairs, test_links, meta)


# ---------------------------------------------------------------------------
#  Sparse score container + ranking metrics
# ---------------------------------------------------------------------------


@dataclass
class SparseScores:
    """Per-row candidate lists: cols[i] / vals[i] are 1-D tensors of equal
    length (any length, possibly empty). Rows are KG1 bundle rows, columns
    KG2 bundle columns."""

    cols: list[torch.Tensor]
    vals: list[torch.Tensor]

    @property
    def n_rows(self) -> int:
        return len(self.cols)

    def nnz(self) -> int:
        return int(sum(int(c.numel()) for c in self.cols))

    def row_dict(self, i: int) -> dict[int, float]:
        return {int(c): float(v) for c, v in zip(self.cols[i].tolist(), self.vals[i].tolist(), strict=False)}


def sparse_rank_metrics(
    scores: SparseScores,
    link_pairs: list[tuple[int, int]],
    top_k: tuple[int, ...] = (1, 5, 10, 50),
    k_cap: int | None = None,
) -> dict[str, float]:
    """Hit@k and truncated MRR over sparse candidate lists.

    rank = 1 + #entries in the row with a STRICTLY greater score (same tie
    rule as eval._compute_metrics); a target absent from its row's list
    counts as a miss for every k and contributes 0 to MRR. With k_cap=K the
    MRR is reported as "MRR@K" (ClusterEA's truncated convention) and Hit@k
    is only emitted for k <= K."""
    if not link_pairs:
        return {f"Hit@{k}": 0.0 for k in top_k} | {"MRR": 0.0}
    ks = tuple(k for k in top_k if k_cap is None or k <= k_cap)
    hits = dict.fromkeys(ks, 0)
    rr = 0.0
    found = 0
    for r, c in link_pairs:
        cols, vals = scores.cols[r], scores.vals[r]
        if cols.numel() == 0:
            continue
        hit = cols == c
        if not bool(hit.any()):
            continue
        found += 1
        tv = vals[hit][0]
        rank = int((vals > tv).sum().item()) + 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
        rr += 1.0 / rank
    n = len(link_pairs)
    out = {f"Hit@{k}": hits[k] / n for k in ks}
    out["MRR@K" if k_cap is not None else "MRR"] = rr / n
    out["candidate_recall"] = found / n
    return out


# ---------------------------------------------------------------------------
#  Dense protocols (parity + l1)
# ---------------------------------------------------------------------------


def dense_protocol_metrics(
    b: Bundle, metric: str, use_csls: bool, csls_k: int, device: torch.device,
    top_k: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Exactly eval.py's online path (dense below the memory limit, chunked
    above it; env DISTEA_EVAL_DENSE_LIMIT_GB=0 forces chunked)."""
    return _eval_metrics_auto(metric, b.kg1_emb, b.kg2_emb, b.link_pairs, device,
                              use_csls, csls_k, top_k)


def parity_check(online: dict, offline: dict, top_k=(1, 5, 10, 50)) -> dict:
    """Compare an offline cosine_csls result with metrics.json's numbers."""
    res = {"ok": True, "diffs": {}}
    for k in top_k:
        key = f"Hit@{k}"
        if key in online and key in offline:
            d = abs(float(online[key]) - float(offline[key]))
            res["diffs"][key] = d
            if d > 1e-12:
                res["ok"] = False
    if "MRR" in online and "MRR" in offline:
        d = abs(float(online["MRR"]) - float(offline["MRR"]))
        res["diffs"]["MRR"] = d
        if d > 1e-9:
            res["ok"] = False
    return res


# ---------------------------------------------------------------------------
#  Sparse bidirectional top-K over the CSLS matrix (ClusterEA's shape)
# ---------------------------------------------------------------------------


def _prep(metric: str, e1: torch.Tensor, e2: torch.Tensor, device: torch.device):
    e1 = e1.to(device)
    e2 = e2.to(device)
    if metric == "cosine":
        e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
    return e1, e2


def chunked_topk_both(
    metric: str,
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    K: int,
    device: torch.device,
    use_csls: bool = True,
    csls_k: int = 10,
    chunk: int = 512,
):
    """Row top-K and column top-K of the (CSLS-adjusted) similarity without
    materialising it.

    Returns (row_vals (n1,K), row_cols (n1,K), col_vals (n2,K), col_rows (n2,K)),
    all on the adjusted scale s' = 2 s - row_thr - col_thr. Pass A streams
    row blocks keeping (i) the running column top-csls_k of the RAW sims
    (-> col_thr) and (ii) the running column top-K of 2 s - row_thr with row
    indices (col_thr is constant within a column, so column order is already
    final; the value is corrected by - col_thr at the end). Pass B
    recomputes each row block, adjusts it, and takes the row top-K.
    Without CSLS both passes use the raw similarity."""
    e1, e2 = _prep(metric, embed1, embed2, device)
    n1, n2 = e1.size(0), e2.size(0)
    K = min(K, n2)
    kk = min(csls_k, n1) if use_csls else 0
    run_thr = None                       # (<=csls_k, n2) raw
    run_val = torch.full((0, n2), float("-inf"), device=device)
    run_row = torch.zeros((0, n2), dtype=torch.long, device=device)
    for i in range(0, n1, chunk):
        blk = _sim_rows(metric, e1[i:i + chunk], e2)       # (B, n2) raw
        if use_csls:
            cat = blk if run_thr is None else torch.cat([run_thr, blk], 0)
            run_thr = cat.topk(min(kk, cat.size(0)), dim=0).values
            row_thr = blk.topk(min(csls_k, n2), dim=1).values.mean(dim=1)
            adj = 2 * blk - row_thr.unsqueeze(1)
        else:
            adj = blk
        rows = torch.arange(i, i + blk.size(0), device=device).unsqueeze(1).expand_as(adj)
        cv = torch.cat([run_val, adj], 0)
        cr = torch.cat([run_row, rows], 0)
        kcol = min(K, cv.size(0))
        top = cv.topk(kcol, dim=0)
        run_val = top.values
        run_row = cr.gather(0, top.indices)
    col_thr = run_thr.mean(dim=0) if use_csls else torch.zeros(n2, device=device)
    col_vals = (run_val - col_thr.unsqueeze(0)).t().contiguous()   # (n2, K)
    col_rows = run_row.t().contiguous()
    row_vals = torch.empty((n1, K), device=device)
    row_cols = torch.empty((n1, K), dtype=torch.long, device=device)
    for i in range(0, n1, chunk):
        blk = _sim_rows(metric, e1[i:i + chunk], e2)
        if use_csls:
            row_thr = blk.topk(min(csls_k, n2), dim=1).values.mean(dim=1)
            blk = 2 * blk - row_thr.unsqueeze(1) - col_thr.unsqueeze(0)
        top = blk.topk(K, dim=1)
        row_vals[i:i + blk.size(0)] = top.values
        row_cols[i:i + blk.size(0)] = top.indices
    return row_vals, row_cols, col_vals, col_rows


def bidirectional_sparse(
    row_vals: torch.Tensor, row_cols: torch.Tensor,
    col_vals: torch.Tensor, col_rows: torch.Tensor,
    n1: int, n2: int, csls_k: int = 10, apply_csls: bool = True,
) -> SparseScores:
    """row top-K (+) column top-K^T on one sparse matrix (shared entries are
    summed, i.e. mutual top-K doubles), then an optional CSLS pass over the
    sparse scores (row/column means of the top-csls_k kept entries)."""
    device = row_vals.device
    r_idx = torch.arange(n1, device=device).unsqueeze(1).expand_as(row_cols).reshape(-1)
    c_idx = row_cols.reshape(-1)
    v = row_vals.reshape(-1)
    r2 = col_rows.reshape(-1)
    c2 = torch.arange(n2, device=device).unsqueeze(1).expand_as(col_rows).reshape(-1)
    v2 = col_vals.reshape(-1)
    rows = torch.cat([r_idx, r2])
    cols = torch.cat([c_idx, c2])
    vals = torch.cat([v, v2])
    key = rows * n2 + cols
    uk, inv = torch.unique(key, return_inverse=True)
    summed = torch.zeros(uk.numel(), device=device).index_add_(0, inv, vals.float())
    rows_u = uk // n2
    cols_u = uk % n2
    if apply_csls:
        # sparse CSLS: thresholds = mean of each row's / column's top-csls_k kept scores
        row_thr = _group_topk_mean(rows_u, summed, n1, csls_k)
        col_thr = _group_topk_mean(cols_u, summed, n2, csls_k)
        summed = 2 * summed - row_thr[rows_u] - col_thr[cols_u]
    out_cols: list[torch.Tensor] = []
    out_vals: list[torch.Tensor] = []
    order = torch.argsort(rows_u)
    rows_s, cols_s, vals_s = rows_u[order], cols_u[order], summed[order]
    counts = torch.bincount(rows_s, minlength=n1).tolist()
    pos = 0
    for i in range(n1):
        c = counts[i]
        out_cols.append(cols_s[pos:pos + c].cpu())
        out_vals.append(vals_s[pos:pos + c].cpu())
        pos += c
    return SparseScores(out_cols, out_vals)


def _group_topk_mean(groups: torch.Tensor, vals: torch.Tensor, n_groups: int, k: int) -> torch.Tensor:
    """Per-group mean of the top-k values (groups with < k entries average
    what they have; empty groups get 0)."""
    order = torch.argsort(vals, descending=True)
    g = groups[order]
    v = vals[order]
    # rank within group after a stable sort by group
    order2 = torch.argsort(g, stable=True)
    g = g[order2]
    v = v[order2]
    counts = torch.bincount(g, minlength=n_groups)
    starts = torch.cumsum(counts, 0) - counts
    idx = torch.arange(g.numel(), device=vals.device) - starts[g]
    keep = idx < k
    s = torch.zeros(n_groups, device=vals.device).index_add_(0, g[keep], v[keep])
    c = torch.zeros(n_groups, device=vals.device).index_add_(0, g[keep], torch.ones_like(v[keep]))
    return s / c.clamp_min(1.0)


# ---------------------------------------------------------------------------
#  Label-free blocks + Sinkhorn
# ---------------------------------------------------------------------------


def blocks_from_partition(b: Bundle, partition_dir: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Block id per KG1 row / KG2 column from DistEA's partition files:
    {r}.npy holds rank r's KG1 global ids, kg2_assignments.json maps
    str(rank) -> KG2 global ids. Rows/cols not covered get -1."""
    with open(os.path.join(partition_dir, "kg2_assignments.json")) as f:
        kg2a = json.load(f)
    n_parts = len(kg2a)
    kg1_block = {}
    for r in range(n_parts):
        p = os.path.join(partition_dir, f"{r}.npy")
        if os.path.exists(p):
            for g in np.load(p).tolist():
                kg1_block[int(g)] = r
    kg2_block = {int(g): int(r) for r, lst in kg2a.items() for g in lst}
    b1 = np.array([kg1_block.get(int(g), -1) for g in b.kg1_ids], dtype=np.int64)
    b2 = np.array([kg2_block.get(int(g), -1) for g in b.kg2_ids], dtype=np.int64)
    return b1, b2, n_parts


def blocks_from_kmeans(b: Bundle, n_blocks: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Label-free alternative: MiniBatchKMeans on the L2-normalised
    embeddings of both sides together."""
    from sklearn.cluster import MiniBatchKMeans

    x = torch.cat([b.kg1_emb, b.kg2_emb], 0)
    x = torch.nn.functional.normalize(x, dim=1).numpy()
    km = MiniBatchKMeans(n_clusters=n_blocks, random_state=seed, batch_size=4096, n_init=3)
    lab = km.fit_predict(x)
    return lab[: b.n1].astype(np.int64), lab[b.n1:].astype(np.int64), n_blocks


def block_recall(b: Bundle, b1: np.ndarray, b2: np.ndarray) -> float:
    """Fraction of scored test pairs whose two sides share a block -- the
    hard ceiling of any block-wise method."""
    if not b.link_pairs:
        return 0.0
    same = sum(1 for r, c in b.link_pairs if b1[r] >= 0 and b1[r] == b2[c])
    return same / len(b.link_pairs)


def log_sinkhorn(sim: torch.Tensor, tau: float = 0.05, iters: int = 100) -> torch.Tensor:
    """Entropic OT plan (log domain) between uniform marginals a = 1/m and
    b = 1/n for a (m, n) similarity block; returns the plan scaled so its
    rows sum to 1 (a per-row distribution over the block's columns)."""
    m, n = sim.shape
    logK = sim / tau
    log_a = torch.full((m, 1), -float(np.log(m)), device=sim.device, dtype=sim.dtype)
    log_b = torch.full((1, n), -float(np.log(n)), device=sim.device, dtype=sim.dtype)
    u = torch.zeros((m, 1), device=sim.device, dtype=sim.dtype)
    v = torch.zeros((1, n), device=sim.device, dtype=sim.dtype)
    for _ in range(iters):
        u = log_a - torch.logsumexp(logK + v, dim=1, keepdim=True)
        v = log_b - torch.logsumexp(logK + u, dim=0, keepdim=True)
    plan = torch.exp(logK + u + v)
    return plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)


def sinkhorn_blockwise(
    b: Bundle,
    b1: np.ndarray,
    b2: np.ndarray,
    n_blocks: int,
    K: int,
    device: torch.device,
    tau: float = 0.05,
    iters: int = 100,
    max_block_elems: int = 33000 * 33000,
    metric: str = "cosine",
) -> tuple[SparseScores, dict]:
    """Per block: cosine similarity (m, n) -> Sinkhorn plan -> row top-K as
    scores. Blocks with m*n above max_block_elems fall back to plain cosine
    top-K (reported). Rows whose block has no KG2 columns get empty lists."""
    e1, e2 = _prep(metric, b.kg1_emb, b.kg2_emb, device)
    cols_out: list[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(b.n1)]
    vals_out: list[torch.Tensor] = [torch.empty(0) for _ in range(b.n1)]
    info = {"blocks": 0, "fallback_blocks": 0, "rows_without_block": int((b1 < 0).sum()),
            "max_block": [0, 0]}
    for blk in range(n_blocks):
        rows = np.nonzero(b1 == blk)[0]
        cols = np.nonzero(b2 == blk)[0]
        if rows.size == 0 or cols.size == 0:
            continue
        info["blocks"] += 1
        if rows.size * cols.size > info["max_block"][0] * info["max_block"][1]:
            info["max_block"] = [int(rows.size), int(cols.size)]
        rt = torch.from_numpy(rows).to(device)
        ct = torch.from_numpy(cols).to(device)
        sim = _sim_rows(metric, e1[rt], e2[ct])
        if rows.size * cols.size > max_block_elems:
            info["fallback_blocks"] += 1
            scores = sim
        else:
            scores = log_sinkhorn(sim, tau=tau, iters=iters)
        k = min(K, cols.size)
        top = scores.topk(k, dim=1)
        tv, ti = top.values.cpu(), ct[top.indices].cpu()
        for j, r in enumerate(rows.tolist()):
            cols_out[r] = ti[j]
            vals_out[r] = tv[j]
    return SparseScores(cols_out, vals_out), info


# ---------------------------------------------------------------------------
#  Fusion
# ---------------------------------------------------------------------------


def _minmax_shift(s: SparseScores) -> SparseScores:
    allv = torch.cat([v for v in s.vals if v.numel()]) if s.nnz() else torch.zeros(1)
    lo, hi = float(allv.min()), float(allv.max())
    span = hi - lo if hi > lo else 1.0
    return SparseScores(s.cols, [(v - lo) / span for v in s.vals])


def fuse(scores: list[SparseScores], how: str = "sum", rrf_k: int = 60) -> SparseScores:
    """sum: min-shift + min-max each variant, add over the union of entries.
    rrf: sum of 1/(rrf_k + rank) over each variant's per-row rank lists."""
    assert scores, "nothing to fuse"
    n_rows = scores[0].n_rows
    if how == "sum":
        parts = [_minmax_shift(s) for s in scores]
    elif how == "rrf":
        parts = []
        for s in scores:
            new_vals = []
            for v in s.vals:
                if v.numel() == 0:
                    new_vals.append(v)
                    continue
                rank = torch.argsort(torch.argsort(v, descending=True)) + 1
                new_vals.append(1.0 / (rrf_k + rank.float()))
            parts.append(SparseScores(s.cols, new_vals))
    else:
        raise ValueError(f"unknown fusion {how!r}")
    out_cols, out_vals = [], []
    for i in range(n_rows):
        acc: dict[int, float] = {}
        for p in parts:
            for c, v in zip(p.cols[i].tolist(), p.vals[i].tolist(), strict=False):
                acc[c] = acc.get(c, 0.0) + v
        if acc:
            cs = torch.tensor(list(acc.keys()), dtype=torch.long)
            vs = torch.tensor(list(acc.values()), dtype=torch.float32)
        else:
            cs, vs = torch.empty(0, dtype=torch.long), torch.empty(0)
        out_cols.append(cs)
        out_vals.append(vs)
    return SparseScores(out_cols, out_vals)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def rescore(
    b: Bundle,
    variants: list[str],
    device: torch.device,
    K: int = 50,
    csls_k: int = 10,
    partition_dir: str | None = None,
    kmeans_blocks: int = 0,
    tau: float = 0.05,
    sinkhorn_iters: int = 100,
    top_k: tuple[int, ...] = (1, 5, 10, 50),
    online_metrics: dict | None = None,
    max_block_elems: int = 33000 * 33000,
) -> dict:
    """Run the requested variants on one bundle; returns the rescore.json
    payload (per-variant metrics + runtime, parity result, block info)."""
    out: dict = {"n1": b.n1, "n2": b.n2, "n_test": len(b.link_pairs), "K": K,
                 "csls_k": csls_k, "device": str(device), "variants": {}}
    sparse_pool: dict[str, SparseScores] = {}

    def _put(name, metrics, t0, extra=None):
        d = dict(metrics)
        d["runtime_s"] = round(time.perf_counter() - t0, 2)
        if extra:
            d.update(extra)
        out["variants"][name] = d

    if "cosine_csls" in variants:
        t0 = time.perf_counter()
        m = dense_protocol_metrics(b, "cosine", True, csls_k, device, top_k)
        extra = {}
        if online_metrics is not None:
            extra["parity"] = parity_check(online_metrics, m, top_k)
        _put("cosine_csls", m, t0, extra)
    if "l1_csls" in variants:
        t0 = time.perf_counter()
        _put("l1_csls", dense_protocol_metrics(b, "l1", True, csls_k, device, top_k), t0)
    if "cosine" in variants:
        t0 = time.perf_counter()
        _put("cosine", dense_protocol_metrics(b, "cosine", False, csls_k, device, top_k), t0)
    if any(v in variants for v in ("bidir", "fuse_sum", "fuse_rrf")):
        t0 = time.perf_counter()
        rv, rc, cv, cr = chunked_topk_both("cosine", b.kg1_emb, b.kg2_emb, K, device, True, csls_k)
        sp = bidirectional_sparse(rv, rc, cv, cr, b.n1, b.n2, csls_k, apply_csls=True)
        sparse_pool["bidir"] = sp
        _put("bidir", sparse_rank_metrics(sp, b.link_pairs, top_k, k_cap=K), t0,
             {"nnz": sp.nnz()})
    if any(v in variants for v in ("sinkhorn", "fuse_sum", "fuse_rrf")):
        t0 = time.perf_counter()
        if partition_dir:
            b1, b2, nb = blocks_from_partition(b, partition_dir)
            src = f"partition:{partition_dir}"
        elif kmeans_blocks > 0:
            b1, b2, nb = blocks_from_kmeans(b, kmeans_blocks)
            src = f"kmeans:{kmeans_blocks}"
        else:
            b1 = np.zeros(b.n1, dtype=np.int64)
            b2 = np.zeros(b.n2, dtype=np.int64)
            nb = 1
            src = "single-block"
        rec = block_recall(b, b1, b2)
        sp, info = sinkhorn_blockwise(b, b1, b2, nb, K, device, tau, sinkhorn_iters,
                                      max_block_elems=max_block_elems)
        sparse_pool["sinkhorn"] = sp
        _put("sinkhorn", sparse_rank_metrics(sp, b.link_pairs, top_k, k_cap=K), t0,
             {"block_source": src, "block_recall": rec, "tau": tau, "iters": sinkhorn_iters} | info)
    for how in ("sum", "rrf"):
        name = f"fuse_{how}"
        if name in variants and len(sparse_pool) >= 2:
            t0 = time.perf_counter()
            fused = fuse([sparse_pool["bidir"], sparse_pool["sinkhorn"]], how)
            _put(name, sparse_rank_metrics(fused, b.link_pairs, top_k, k_cap=None), t0,
                 {"nnz": fused.nnz(), "inputs": ["bidir", "sinkhorn"]})
    return out
