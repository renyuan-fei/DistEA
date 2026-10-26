"""
Distributed entity alignment evaluation.

Each rank holds embeddings for its local_kg1 and local_kg2 entities.
We gather all embeddings across ranks, then rank by one of two similarity
metrics, selected via `metric`:

  - "l1" (default): NEGATIVE L1/cityblock distance on raw, non-normalised
    embeddings — matches ELsEA's GCN-Align get_hits() (DivEA/GCN_Align/
    metrics.py), which does `scipy.spatial.distance.cdist(Lvec, Rvec,
    metric='cityblock')`. use_csls defaults to False here, matching
    get_hits() having no CSLS at all.
  - "cosine": L2-normalise then rank by dot product — matches RREA/runner.py's
    CSLS_test() (`vec / np.linalg.norm(...)` then `np.matmul(embed1,
    embed2.T)` in RREA/CSLS.py's sim_handler). RREA's own CSLS_test()
    defaults csls=10, not off.

framework.py passes metric="cosine" (and use_csls=True, csls_k=10) by
default since RREA is DistEA's current active model; the "l1"/off defaults
here remain correct for direct/library callers and for the GCN-Align path.

Internally we work with NEGATIVE L1 distance ("higher = closer/better") so
the ranking/CSLS/argmax logic is a single, direction-agnostic implementation
shared with generate_pseudo_seeds — the sign flip is purely a bookkeeping
convenience, not a different metric.

All ranks participate in all_gather; only rank 0 logs the result.
"""

from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np
import torch
import torch.distributed as dist

try:
    from tqdm import trange
except ImportError:

    def trange(n, **kw):
        return range(n)


# ---------------------------------------------------------------------------
#  Similarity helpers
# ---------------------------------------------------------------------------


def _neg_l1_dist(
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Return (N1, N2) NEGATIVE L1/cityblock distance ("similarity": higher is
    closer/better) — matches ELsEA's get_hits(), which ranks by ascending
    `cdist(..., metric='cityblock')` on raw (non-normalised) embeddings.
    Negating lets every downstream consumer (CSLS, argmax, rank-by-descending)
    stay written in "higher is better" terms without a second code path.
    """
    n1, n2 = embed1.size(0), embed2.size(0)
    sim = torch.zeros(n1, n2, device=device)
    e1 = embed1.to(device)
    e2 = embed2.to(device)
    for i in range(0, n1, batch_size):
        # torch.cdist(p=1) == row-wise cityblock, without materializing
        # the (b, N2, D) broadcast intermediate an unsqueeze-subtract
        # version would allocate (~2.8 GB at N2=10.5K, D=128, b=512).
        # Identical values (exact same |a-b| sum).
        sim[i : i + batch_size] = -torch.cdist(e1[i : i + batch_size], e2, p=1)
    return sim


def _cosine_sim(
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Return (N1, N2) cosine similarity (higher = closer/better) — matches
    RREA/runner.py's CSLS_test(), which L2-normalises embeddings
    (`vec / np.linalg.norm(vec, axis=-1, keepdims=True)`) then ranks by
    `np.matmul(embed1, embed2.T)` (RREA/CSLS.py sim_handler) — i.e. cosine
    similarity via normalised dot product, not L1/cityblock distance.
    """
    e1 = torch.nn.functional.normalize(embed1.to(device), p=2, dim=1)
    e2 = torch.nn.functional.normalize(embed2.to(device), p=2, dim=1)
    n1 = e1.size(0)
    sim = torch.zeros(n1, e2.size(0), device=device)
    for i in range(0, n1, batch_size):
        sim[i : i + batch_size] = e1[i : i + batch_size] @ e2.t()
    return sim


def _compute_sim(
    metric: str,
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if metric == "cosine":
        return _cosine_sim(embed1, embed2, device)
    return _neg_l1_dist(embed1, embed2, device)


# ---------------------------------------------------------------------------
#  Chunked (never-materialize) evaluation path
# ---------------------------------------------------------------------------

# Above this many BYTES of dense (n1, n2) fp32 similarity, eval switches to
# the chunked two-pass path below. 100K-scale selective eval already needs
# ~19.6 GB dense (70000^2 x 4B); at 1M scale dense is infeasible.
# Override via env DISTEA_EVAL_DENSE_LIMIT_GB (e.g. "0" forces chunked
# always -- how the parity tests exercise it).
def _dense_limit_bytes() -> int:
    return int(float(os.environ.get("DISTEA_EVAL_DENSE_LIMIT_GB", "4")) * (1 << 30))


# Where the similarity work runs on the metrics-computing rank. Default
# is CPU (the gathered embeddings are CPU-staged under gloo). DISTEA_EVAL_DEVICE=cuda moves the
# chunked passes onto the rank's own GPU when it has one: the chunked path
# only ever holds O(chunk x N2) on the device, so at 1M scale (241K test
# pairs, 512 x 241K x 4 B = 0.5 GB per block) it fits beside the training
# residency, and the same ops that take ~4-6 min per eval at 100K on CPU
# (extrapolating ~12x to 1M) become seconds. Same arithmetic either way;
# the GPU parity check is tests/gpu_parity_eval_device.py.
def _sim_device(rank_device: torch.device) -> torch.device:
    want = os.environ.get("DISTEA_EVAL_DEVICE", "cpu").lower()
    if want == "cuda" and rank_device.type == "cuda" and torch.cuda.is_available():
        return rank_device
    return torch.device("cpu")


def _sim_rows(
    metric: str, e1_rows: torch.Tensor, e2: torch.Tensor
) -> torch.Tensor:
    """(B, N2) similarity block for a chunk of KG1 rows. For cosine the
    caller pre-normalises BOTH sides once (so this is a plain matmul);
    l1 matches _neg_l1_dist's negative-cityblock convention."""
    if metric == "cosine":
        return e1_rows @ e2.t()
    return -torch.cdist(e1_rows, e2, p=1)  # see _neg_l1_dist on why cdist



def _group_codes(groups, n: int, device):
    """groups: per-pair labels (list of str) aligned with the link list, or
    None. Returns (names, int64 code tensor on device) or (None, None)."""
    if groups is None:
        return None, None
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} labels for {n} link pairs")
    names = sorted(set(groups))
    lut = {nm: i for i, nm in enumerate(names)}
    return names, torch.tensor([lut[g] for g in groups], dtype=torch.long, device=device)


def _tally_groups(acc: dict, ranks: torch.Tensor, codes, names, top_k) -> None:
    """Add one batch of per-pair ranks to acc[name] = [n, rr_sum, {k: hits}]."""
    if codes is None:
        return
    for gi, nm in enumerate(names):
        mask = codes == gi
        cnt = int(mask.sum().item())
        if cnt == 0:
            continue
        r = ranks[mask]
        a = acc.setdefault(nm, [0, 0.0, dict.fromkeys(top_k, 0)])
        a[0] += cnt
        a[1] += (1.0 / r.float()).sum().item()
        for k in top_k:
            a[2][k] += int((r <= k).sum().item())


def _group_metrics(acc: dict, top_k) -> dict:
    """{"<name>/Hit@k", "<name>/MRR", "<name>/n"} for every tallied group."""
    out: dict = {}
    for nm in sorted(acc):
        n_g, rr, hits = acc[nm]
        out.update({f"{nm}/Hit@{k}": hits[k] / n_g for k in top_k})
        out[f"{nm}/MRR"] = rr / n_g
        out[f"{nm}/n"] = n_g
    return out


def _shard_range(n: int, chunk: int, rank: int, world_size: int) -> tuple[int, int]:
    """This rank's [lo, hi) of n rows for a sharded
    chunked pass, cut at multiples of `chunk` so every chunk a rank processes
    is exactly one the unsharded loop would process."""
    n_chunks = (n + chunk - 1) // chunk
    c0 = n_chunks * rank // world_size
    c1 = n_chunks * (rank + 1) // world_size
    return min(c0 * chunk, n), min(c1 * chunk, n)


def _collective_stage(device: torch.device) -> torch.device:
    """gloo collectives run on CPU tensors, nccl on the device (see
    _all_gather_embeddings)."""
    return torch.device("cpu") if dist.get_backend() == "gloo" else device


def _fit_chunk(chunk: int, n_cols: int, device: torch.device, ws: int = 1) -> int:
    """Rows per similarity block for the chunked eval / mutual-NN paths, so
    that ~4 (rows x n_cols) fp32 buffers (the block, its concat with the
    running column top-k, the CSLS-adjusted copy and topk's outputs) fit in
    half of the GPU memory still available (driver-free plus the caching
    allocator's unused reserve). Never larger than `chunk`; CPU: unchanged.
    The block size only sets how many rows are scored at once -- Hit@k, the
    CSLS thresholds and the mutual-NN pairs are the same for any value, MRR
    differs only in fp summation order (tests/test_eval_chunk_fit.py) --
    so this only avoids an OOM (e.g. mining pseudo-seeds over 1.35M x 1.35M
    candidates at FB_DBP2M scale with 512-row blocks next to ~72 GB of
    training state).
    With ws > 1 (sharded) every rank takes the minimum, because the row
    shards are chunk-aligned and must agree."""
    if device.type != "cuda" or n_cols <= 0:
        return chunk
    try:
        free = torch.cuda.mem_get_info(device)[0]
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    except Exception:  # noqa: BLE001 -- sizing must never fail an eval
        return chunk
    fit = int(0.5 * free / (4 * 4 * n_cols))
    c = max(16, min(chunk, (fit // 16) * 16))
    if ws > 1 and dist.is_available() and dist.is_initialized():
        t = torch.tensor([c], dtype=torch.long, device=_collective_stage(device))
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        c = int(t.item())
    if c < chunk:
        logging.info(f"[eval] row blocks of {c} instead of {chunk} rows: "
                     f"{free / 2**30:.1f} GiB of GPU memory left for {n_cols} columns")
    return c


def _merge_topk_across_ranks(running, k: int, n2: int, device) -> torch.Tensor:
    """Merge each rank's per-column running top-k ((<=k, n2), or None for an
    empty row slice) into the global per-column top-k, sorted like topk()'s
    own output -- the same values in the same order as one rank streaming
    every row, so the CSLS column threshold is unchanged."""
    stage = _collective_stage(device)
    part = torch.full((k, n2), float("-inf"), device=stage)
    if running is not None:
        part[: running.size(0)] = running.to(stage)
    parts = [torch.empty_like(part) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, part)
    return torch.cat(parts, dim=0).to(device).topk(k, dim=0).values


def _allreduce_tallies(hits: dict, rr_sum: float, g_acc: dict, g_names, top_k, device):
    """Sum Hit@k counts, the reciprocal-rank sum and the per-group tallies of
    a sharded pass B over all ranks (float64; counts stay exact)."""
    names = list(g_names or [])
    w = 2 + len(top_k)
    t = torch.zeros(1 + len(names), w, dtype=torch.float64)
    t[0, 1] = rr_sum
    for c, k in enumerate(top_k):
        t[0, 2 + c] = hits[k]
    for gi, nm in enumerate(names):
        a = g_acc.get(nm)
        if a:
            t[1 + gi, 0] = a[0]
            t[1 + gi, 1] = a[1]
            for c, k in enumerate(top_k):
                t[1 + gi, 2 + c] = a[2][k]
    t = t.to(_collective_stage(device))
    dist.all_reduce(t)
    t = t.cpu()
    hits = {k: int(round(t[0, 2 + c].item())) for c, k in enumerate(top_k)}
    out_acc = {}
    for gi, nm in enumerate(names):
        cnt = int(round(t[1 + gi, 0].item()))
        if cnt:
            out_acc[nm] = [cnt, float(t[1 + gi, 1].item()),
                           {k: int(round(t[1 + gi, 2 + c].item())) for c, k in enumerate(top_k)}]
    return hits, float(t[0, 1].item()), out_acc


def _chunked_metrics(
    metric: str,
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    test_links: list,
    device: torch.device,
    use_csls: bool = False,
    csls_k: int = 10,
    top_k: tuple[int, ...] = (1, 10),
    chunk: int = 512,
    groups=None,
    shard: tuple[int, int] | None = None,
) -> dict[str, float]:
    """
    Hit@k / MRR without ever materializing the (N1, N2) similarity matrix
    -- numerically identical to _compute_sim + _csls_thr/_apply_csls +
    _compute_metrics (same ops, same chunk arithmetic; covered by a parity
    test in tests/test_stage_v.py). Peak extra memory is O(chunk x N2).

    Pass A (CSLS only): stream all KG1 rows once, keeping a running
    (<=csls_k, N2) top-k buffer per column -> col_thr. Row thresholds
    don't need a pass of their own: they are recomputed from the same
    unadjusted row block pass B already computes.
    Pass B: for each chunk of TEST rows, build the row block, CSLS-adjust
    it in place, and rank each target column within its row.
    shard=(rank, world_size) (DISTEA_EVAL_SHARD=1): every
    rank holds the full embeddings (all_gather) and runs both passes on its
    own chunk-aligned slice of rows / test pairs; the per-column top-k and
    the counts are merged with one all_gather + one all_reduce. Hit@k and
    the CSLS thresholds are identical to the unsharded result; MRR differs
    only in float64 summation order. Every rank must call it.
    """
    if not test_links:
        return {f"Hit@{k}": 0.0 for k in top_k} | {"MRR": 0.0}
    rk, ws = shard if shard is not None else (0, 1)

    e1 = embed1.to(device)
    e2 = embed2.to(device)
    if metric == "cosine":
        e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
    n1 = e1.size(0)
    if ws > 1 and n1 < csls_k:
        ws = 1  # a merged top-k needs >= k real rows; never true at chunked scale
        if rk != 0:
            return {}
    chunk = _fit_chunk(chunk, e2.size(0), device, ws)

    col_thr = None
    if use_csls:
        running: torch.Tensor | None = None  # (<=csls_k, N2), best-so-far per column
        lo, hi = _shard_range(n1, chunk, rk, ws) if ws > 1 else (0, n1)
        for i in range(lo, hi, chunk):
            block = _sim_rows(metric, e1[i : min(i + chunk, hi)], e2)
            cat = block if running is None else torch.cat([running, block], dim=0)
            running = cat.topk(min(csls_k, cat.size(0)), dim=0).values
        if ws > 1:
            running = _merge_topk_across_ranks(running, csls_k, e2.size(0), device)
        col_thr = running.mean(dim=0)

    kg1_idx = torch.tensor([p[0] for p in test_links], device=device)
    kg2_idx = torch.tensor([p[1] for p in test_links], device=device)
    hits = dict.fromkeys(top_k, 0)
    rr_sum = 0.0
    n = len(test_links)
    g_names, g_codes = _group_codes(groups, n, device)
    g_acc: dict = {}

    lo, hi = _shard_range(n, chunk, rk, ws) if ws > 1 else (0, n)
    for i in range(lo, hi, chunk):
        j = min(i + chunk, hi)
        batch_kg1 = kg1_idx[i:j]
        batch_kg2 = kg2_idx[i:j]
        rows = _sim_rows(metric, e1[batch_kg1], e2)  # (B, N2), unadjusted
        if use_csls:
            row_thr = rows.topk(csls_k, dim=1).values.mean(dim=1)  # from raw sim,
            # exactly like the dense path (thr before _apply_csls)
            rows = 2 * rows - row_thr.unsqueeze(1) - col_thr.unsqueeze(0)
        targets = batch_kg2.unsqueeze(1)
        ranks = (rows > rows.gather(1, targets)).sum(dim=1) + 1
        for k in top_k:
            hits[k] += int((ranks <= k).sum().item())
        rr_sum += (1.0 / ranks.float()).sum().item()
        _tally_groups(g_acc, ranks, None if g_codes is None else g_codes[i:j],
                      g_names, top_k)
    if ws > 1:
        hits, rr_sum, g_acc = _allreduce_tallies(hits, rr_sum, g_acc, g_names, top_k, device)

    return ({f"Hit@{k}": hits[k] / n for k in top_k} | {"MRR": rr_sum / n}
            | _group_metrics(g_acc, top_k))


def _chunked_mutual_nn(
    metric: str,
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    device: torch.device,
    use_csls: bool = False,
    csls_k: int = 10,
    chunk: int = 512,
    return_row_best: bool = False,
    shard: tuple[int, int] | None = None,
):
    """
    (best_kg2, best_kg1) argmax pair for the mutual-NN pseudo-seed step
    without materializing (N1, N2) — needed for multi-iteration runs at 1M
    scale, since pseudo-seeds are mined at every iteration boundary.

    Same math as the dense path: row argmax over the CSLS-adjusted matrix
    (row_thr is constant within a row, so it never changes that row's
    argmax, but col_thr does and is applied); column argmax tracked as a
    running (best value, best row) pair per column across row chunks —
    row_thr shifts differ per row, so the adjusted values are compared.
    return_row_best: also return each row's
    best (CSLS-adjusted) score, the confidence --pseudo_seed_top_frac ranks
    mutual pairs by.
    shard=(rank, world_size) (DISTEA_EVAL_SHARD=1): each
    rank streams its chunk-aligned slice of rows; the column top-k (CSLS),
    the per-row argmax slices and the per-column (best value, best row) are
    merged across ranks -- ties go to the lowest rank, i.e. the earliest row,
    exactly as the sequential strict ">" update does. Every rank must call it
    and gets the full result.
    """
    e1 = embed1.to(device)
    e2 = embed2.to(device)
    if metric == "cosine":
        e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
    n1, n2 = e1.size(0), e2.size(0)
    rk, ws = shard if shard is not None else (0, 1)
    if ws > 1 and n1 < csls_k:
        ws, rk = 1, 0  # every rank computes the whole (tiny) problem itself
    chunk = _fit_chunk(chunk, n2, device, ws)
    lo, hi = _shard_range(n1, chunk, rk, ws) if ws > 1 else (0, n1)

    col_thr = None
    if use_csls:
        running: torch.Tensor | None = None
        for i in range(lo, hi, chunk):
            block = _sim_rows(metric, e1[i : min(i + chunk, hi)], e2)
            cat = block if running is None else torch.cat([running, block], dim=0)
            running = cat.topk(min(csls_k, cat.size(0)), dim=0).values
        if ws > 1:
            running = _merge_topk_across_ranks(running, csls_k, n2, device)
        col_thr = running.mean(dim=0)

    best_kg2 = torch.zeros(n1, dtype=torch.long, device=device)
    row_best = torch.zeros(n1, device=device) if return_row_best else None
    col_best_val = torch.full((n2,), float("-inf"), device=device)
    col_best_row = torch.zeros(n2, dtype=torch.long, device=device)
    for i in range(lo, hi, chunk):
        j = min(i + chunk, hi)
        rows = _sim_rows(metric, e1[i:j], e2)
        if use_csls:
            row_thr = rows.topk(csls_k, dim=1).values.mean(dim=1)
            rows = 2 * rows - row_thr.unsqueeze(1) - col_thr.unsqueeze(0)
        rb_arg = rows.argmax(dim=1)
        best_kg2[i:j] = rb_arg
        if return_row_best:
            row_best[i:j] = rows.gather(1, rb_arg.unsqueeze(1)).squeeze(1)
        chunk_max, chunk_arg = rows.max(dim=0)
        better = chunk_max > col_best_val
        col_best_val[better] = chunk_max[better]
        col_best_row[better] = chunk_arg[better] + i
    if ws > 1:
        stage = _collective_stage(device)
        spans = [_shard_range(n1, chunk, r, ws) for r in range(ws)]
        width = max(b - a for a, b in spans)

        def _gather_rows(t):
            part = torch.zeros(width, dtype=t.dtype, device=stage)
            part[: hi - lo] = t[lo:hi].to(stage)
            parts = [torch.empty_like(part) for _ in range(ws)]
            dist.all_gather(parts, part)
            return torch.cat([p[: b - a] for p, (a, b) in zip(parts, spans, strict=True)]).to(device)

        best_kg2 = _gather_rows(best_kg2)
        if return_row_best:
            row_best = _gather_rows(row_best)
        vals = [torch.empty(n2, device=stage) for _ in range(ws)]
        dist.all_gather(vals, col_best_val.to(stage))
        rows_ = [torch.empty(n2, dtype=torch.long, device=stage) for _ in range(ws)]
        dist.all_gather(rows_, col_best_row.to(stage))
        v = torch.stack(vals)  # (ws, n2); argmax picks the first max = lowest rank
        win = v.argmax(dim=0)
        col_best_row = torch.stack(rows_).gather(0, win.unsqueeze(0)).squeeze(0).to(device)
    if return_row_best:
        return best_kg2, col_best_row, row_best
    return best_kg2, col_best_row


def _eval_metrics_auto(
    metric: str,
    embed1: torch.Tensor,
    embed2: torch.Tensor,
    link_pairs: list,
    device: torch.device,
    use_csls: bool,
    csls_k: int,
    top_k: tuple[int, ...],
    groups=None,
    shard: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Dense path below the memory limit, chunked path above it. groups:
    optional per-pair labels; adds "<label>/Hit@k" etc. (see _group_metrics).
    shard: see _chunked_metrics (chunked path only; callers only pass it when
    the chunked path is taken, so every rank reaches the collectives)."""
    n1, n2 = embed1.size(0), embed2.size(0)
    if n1 * n2 * 4 > _dense_limit_bytes():
        logging.info(
            f"[eval] chunked path: dense sim would be "
            f"{n1 * n2 * 4 / (1 << 30):.1f} GB (> limit)"
            + (f", sharded over {shard[1]} ranks" if shard else "")
        )
        return _chunked_metrics(
            metric, embed1, embed2, link_pairs, device,
            use_csls=use_csls, csls_k=csls_k, top_k=top_k, groups=groups, shard=shard,
        )
    sim = _compute_sim(metric, embed1, embed2, device)
    if use_csls:
        row_thr = _csls_thr(sim, csls_k, device)
        col_thr = _csls_thr(sim.t(), csls_k, device)
        sim = _apply_csls(sim, row_thr, col_thr, device)
    return _compute_metrics(sim, link_pairs, device, top_k, groups=groups)


def _csls_thr(
    sim: torch.Tensor, k: int, device: torch.device, batch_size: int = 512
) -> torch.Tensor:
    """Compute per-row CSLS threshold: mean of top-k similarities."""
    n = sim.size(0)
    thr = torch.zeros(n, device=device)
    for i in range(0, n, batch_size):
        top = sim[i : i + batch_size].topk(k, dim=1).values
        thr[i : i + batch_size] = top.mean(dim=1)
    return thr


def _apply_csls(
    sim: torch.Tensor,
    row_thr: torch.Tensor,
    col_thr: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """CSLS-corrected similarity: csls_ij = 2*sim_ij - row_thr_i - col_thr_j.

    Transforms `sim` IN PLACE and returns it, rather than allocating a
    second (n1, n2) output tensor, which would double peak memory and does
    not fit at 100K-entity scale. Safe to mutate: both call sites (distributed_eval, selective_eval)
    already fully compute row_thr/col_thr from `sim` BEFORE calling this, and
    immediately rebind `sim = _apply_csls(sim, ...)` — nothing holds a
    reference to the pre-CSLS values afterward.
    """
    n1, n2 = sim.size()
    for i in range(0, n1, batch_size):
        sim[i : i + batch_size] = (
            2 * sim[i : i + batch_size]
            - row_thr[i : i + batch_size].unsqueeze(1)
            - col_thr.unsqueeze(0)
        )
    return sim


# ---------------------------------------------------------------------------
#  Metric computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    sim: torch.Tensor,
    test_links: list,
    device: torch.device,
    top_k: tuple[int, ...] = (1, 10),
    batch_size: int = 512,
    groups=None,
) -> dict[str, float]:
    """
    Compute Hit@k / MRR given full similarity matrix and test link index pairs.

    test_links: list of (kg1_row_idx, kg2_col_idx) in the sim matrix.
    """
    if not test_links:
        return {f"Hit@{k}": 0.0 for k in top_k} | {"MRR": 0.0}

    kg1_idx = torch.tensor([p[0] for p in test_links], device=device)
    kg2_idx = torch.tensor([p[1] for p in test_links], device=device)

    hits = dict.fromkeys(top_k, 0)
    rr_sum = 0.0
    n = len(test_links)
    g_names, g_codes = _group_codes(groups, n, device)
    g_acc: dict = {}

    for i in range(0, n, batch_size):
        batch_kg1 = kg1_idx[i : i + batch_size]
        batch_kg2 = kg2_idx[i : i + batch_size]
        rows = sim[batch_kg1]  # (B, N2)
        targets = batch_kg2.unsqueeze(1)  # (B, 1)

        # Rank of the target in descending similarity order
        ranks = (rows > rows.gather(1, targets)).sum(dim=1) + 1  # (B,)

        for k in top_k:
            hits[k] += int((ranks <= k).sum().item())
        rr_sum += (1.0 / ranks.float()).sum().item()
        _tally_groups(g_acc, ranks,
                      None if g_codes is None else g_codes[i : i + batch_size],
                      g_names, top_k)

    return ({f"Hit@{k}": hits[k] / n for k in top_k} | {"MRR": rr_sum / n}
            | _group_metrics(g_acc, top_k))


# ---------------------------------------------------------------------------
#  Embedding gather utilities
# ---------------------------------------------------------------------------


def _select_owned_rows(
    emb: torch.Tensor,
    all_global_ids: np.ndarray,
    wanted_gids,
    owned_local=None,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    (ids, rows) of `emb` for the global ids in `wanted_gids` that this rank
    should contribute to a gather. `all_global_ids` is the local node order
    [local_kg1 | halo_kg1 | local_kg2 | halo_kg2], so without `owned_local`
    a rank also hands in its HALO copies -- every entity that is a halo
    somewhere then arrives at rank 0 once per rank that sees it, computed
    from that rank's stale halo row and partial neighborhood. `owned_local`
    (local indices, or a bool mask over local nodes) restricts the
    contribution to the rank's own entities, which is what an evaluation
    of "the embedding of entity e" means; each entity has exactly one
    owner, so the gather is then duplicate-free by construction.
    """
    gid2local = {int(gid): i for i, gid in enumerate(all_global_ids)}
    if owned_local is not None:
        owned = torch.as_tensor(np.asarray(owned_local))
        if owned.dtype == torch.bool:
            owned = owned.nonzero(as_tuple=True)[0]
        owned_set = set(owned.tolist())
    else:
        owned_set = None
    ids, rows = [], []
    for gid in wanted_gids:
        li = gid2local.get(int(gid))
        if li is None or (owned_set is not None and li not in owned_set):
            continue
        ids.append(int(gid))
        rows.append(emb[li])
    D = emb.size(1)
    rows_t = torch.stack(rows) if rows else torch.zeros(0, D, device=emb.device)
    return np.array(ids, dtype=np.int64), rows_t


def _dedupe_gathered(
    all_emb: torch.Tensor, all_ids: np.ndarray, what: str
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Keep one row per global id after a gather (first occurrence, i.e. the
    lowest rank). Callers that pass `owned_local` never trigger this; it
    guards the un-owned path, where a `{gid: row}` map built over the
    gathered ids would silently pick the LAST copy (highest rank, usually
    a stale halo copy) as the target and leave every other copy in as a
    competitor -- the effect grows with world_size.
    """
    if all_ids.size == 0:
        return all_emb, all_ids
    _, first = np.unique(all_ids, return_index=True)
    if first.size == all_ids.size:
        return all_emb, all_ids
    first.sort()
    logging.warning(
        f"[eval] {what}: {all_ids.size - first.size} duplicate rows in the "
        f"gathered embeddings ({all_ids.size} -> {first.size}); keeping the "
        "lowest-rank copy. Pass owned_local= to gather owner rows only."
    )
    keep = torch.from_numpy(first).to(all_emb.device)
    return all_emb.index_select(0, keep), all_ids[first]


def _all_gather_embeddings(
    local_emb: torch.Tensor,
    local_ids: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Gather (emb, ids) from all ranks.  Returns (global_emb, global_ids).
    All ranks receive the full result.

    Stages the collective on CPU when the backend is gloo, regardless of
    `device`: every rank materializes world_size copies of the padded buffer
    (`dist.all_gather` requires a receive slot per rank), and when ranks are
    processes sharing one physical GPU (`--nproc_per_node=N` on one GPU)
    those copies pile up on the same card N times over, which can run the
    eval out of GPU memory at large world_size. gloo natively supports
    CPU-CPU collectives, and eval runs under torch.no_grad(), so there's no
    gradient-flow reason to keep this on GPU under gloo.

    Under nccl (multi-node, one GPU per rank) there is no CPU fallback to
    stage through: `dist.all_gather` on a CPU tensor raises `RuntimeError: No
    backend type associated with device type cpu`. So under nccl, stage on
    `device` instead — one GPU per rank means no multi-process-sharing-one-card
    duplication to worry about there.
    """
    stage = torch.device("cpu") if dist.get_backend() == "gloo" else device
    D = local_emb.size(1)
    emb_t = local_emb.to(stage)

    # Gather sizes
    local_n = torch.tensor([local_emb.size(0)], device=stage)
    all_n_list = [
        torch.zeros(1, dtype=torch.long, device=stage) for _ in range(world_size)
    ]
    dist.all_gather(all_n_list, local_n)
    all_n = [int(t.item()) for t in all_n_list]

    # Pad to max size for all_gather
    max_n = max(all_n)
    pad = max_n - local_emb.size(0)
    if pad > 0:
        emb_t = torch.cat([emb_t, torch.zeros(pad, D, device=stage)], dim=0)

    emb_list = [torch.zeros(max_n, D, device=stage) for _ in range(world_size)]
    dist.all_gather(emb_list, emb_t)

    # Gather IDs (as float for simplicity, then cast back)
    ids_t = torch.from_numpy(local_ids.astype(np.float32)).to(stage)
    if pad > 0:
        ids_t = torch.cat([ids_t, torch.zeros(pad, device=stage)])
    ids_list = [torch.zeros(max_n, device=stage) for _ in range(world_size)]
    dist.all_gather(ids_list, ids_t)

    # Trim padding and concatenate
    global_embs = torch.cat([emb_list[r][: all_n[r]] for r in range(world_size)], dim=0)
    global_ids = np.concatenate(
        [
            ids_list[r][: all_n[r]].cpu().numpy().astype(np.int64)
            for r in range(world_size)
        ]
    )
    return global_embs, global_ids


# ---------------------------------------------------------------------------
#  Public eval functions
# ---------------------------------------------------------------------------


def distributed_eval(
    model,
    g,
    h_init: torch.Tensor,
    all_global_ids: np.ndarray,
    local_kg1_idx: np.ndarray,
    local_kg2_idx: np.ndarray,
    test_links: list,
    rank: int,
    world_size: int,
    device: torch.device,
    use_csls: bool = False,
    csls_k: int = 10,
    top_k: tuple[int, ...] = (1, 10),
    metric: str = "l1",
) -> dict[str, float] | None:
    """
    Full all-gather evaluation (gathers all entity embeddings).

    metric="l1" (default) matches ELsEA's GCN-Align get_hits() (plain
    L1/cityblock ranking on raw, non-normalised embeddings, no CSLS).
    metric="cosine" matches RREA/runner.py's CSLS_test() (L2-normalise then
    rank by dot product) — pass this when evaluating an RREA model; framework.py
    does so by default since RREA is DistEA's current active model.
    use_csls defaults to False here (function-level, model-agnostic default);
    RREA's own CSLS_test() defaults to csls=10, which framework.py also
    matches by default. CSLS is otherwise a DistEA-only opt-in.

    Returns metrics dict on rank 0; None on other ranks.
    """
    model.eval()
    with torch.no_grad():
        emb = model(g, h_init)  # (N_local, D)

    kg1_emb = emb[local_kg1_idx]
    kg2_emb = emb[local_kg2_idx]
    kg1_ids = all_global_ids[local_kg1_idx]
    kg2_ids = all_global_ids[local_kg2_idx]

    all_kg1_emb, all_kg1_ids = _all_gather_embeddings(
        kg1_emb, kg1_ids, rank, world_size, device
    )
    all_kg2_emb, all_kg2_ids = _all_gather_embeddings(
        kg2_emb, kg2_ids, rank, world_size, device
    )

    if rank != 0:
        return None

    # Build id → row mappings
    kg1_id2row = {int(gid): i for i, gid in enumerate(all_kg1_ids)}
    kg2_id2row = {int(gid): i for i, gid in enumerate(all_kg2_ids)}

    # Convert test_links (global ids) to row indices
    link_pairs = []
    for e1, e2 in test_links:
        r1 = kg1_id2row.get(int(e1))
        r2 = kg2_id2row.get(int(e2))
        if r1 is not None and r2 is not None:
            link_pairs.append((r1, r2))

    # Similarity work stays on CPU unless DISTEA_EVAL_DEVICE=cuda — see
    # _all_gather_embeddings and _sim_device. Above the
    # dense-memory limit this switches to the chunked path that never
    # materializes (n1, n2).
    sim_dev = _sim_device(device)
    metrics = _eval_metrics_auto(
        metric, all_kg1_emb, all_kg2_emb, link_pairs, sim_dev,
        use_csls, csls_k, top_k,
    )
    logging.info(f"[eval] distributed_eval: {metrics}")
    return metrics


def selective_eval(
    model,
    g,
    h_init: torch.Tensor,
    all_global_ids: np.ndarray,
    test_links: list,
    rank: int,
    world_size: int,
    device: torch.device,
    use_csls: bool = False,
    csls_k: int = 10,
    top_k: tuple[int, ...] = (1, 10),
    save_path: str | None = None,
    metric: str = "l1",
    export_dir: str | None = None,
    owned_local=None,
    export_dtype: str = "fp32",
    export_meta: dict | None = None,
    link_groups: list | None = None,
) -> dict[str, float] | None:
    """
    Bandwidth-efficient evaluation: only gathers embeddings of entities in
    test_links (~15K) instead of the full entity set.  ~98.5% bandwidth saving.

    owned_local: local indices (or bool mask) of the entities this rank OWNS
    (local_kg1 + local_kg2). With it, halo copies are never gathered and the
    candidate set is exactly one row per entity; without it, see
    _select_owned_rows / _dedupe_gathered for what goes wrong.

    metric="l1" (default) matches ELsEA's GCN-Align get_hits() (plain
    L1/cityblock ranking on raw, non-normalised embeddings, no CSLS).
    metric="cosine" matches RREA/runner.py's CSLS_test() (L2-normalise then
    rank by dot product) — pass this when evaluating an RREA model; framework.py
    does so by default since RREA is DistEA's current active model.
    use_csls defaults to False here (function-level, model-agnostic default);
    RREA's own CSLS_test() defaults to csls=10, which framework.py also
    matches by default. CSLS is otherwise a DistEA-only opt-in.

    Returns metrics dict on rank 0; None on other ranks.
    """
    model.eval()
    with torch.no_grad():
        emb = model(g, h_init)  # (N_local, D)

    # Determine which test-link entities this rank contributes (owned only
    # when owned_local is given -- see _select_owned_rows)
    test_e1_set = {int(e1) for e1, e2 in test_links}
    test_e2_set = {int(e2) for e1, e2 in test_links}
    e1_ids, e1_emb_t = _select_owned_rows(emb, all_global_ids, test_e1_set, owned_local)
    e2_ids, e2_emb_t = _select_owned_rows(emb, all_global_ids, test_e2_set, owned_local)

    all_kg1_emb, all_kg1_ids = _all_gather_embeddings(
        e1_emb_t, e1_ids, rank, world_size, device
    )
    all_kg2_emb, all_kg2_ids = _all_gather_embeddings(
        e2_emb_t, e2_ids, rank, world_size, device
    )

    # DISTEA_EVAL_SHARD=1: every rank already holds the full gathered
    # embeddings, so the chunked passes are split across ranks instead of
    # rank 0 doing all of them while the rest wait (~27 s on DBP1M's
    # 241K x 241K test set). Off by default.
    shard_eval = world_size > 1 and os.environ.get("DISTEA_EVAL_SHARD", "0") == "1"
    if rank != 0 and not shard_eval:
        return None

    # One row per entity (a no-op when owned_local was given).
    all_kg1_emb, all_kg1_ids = _dedupe_gathered(all_kg1_emb, all_kg1_ids, "kg1")
    all_kg2_emb, all_kg2_ids = _dedupe_gathered(all_kg2_emb, all_kg2_ids, "kg2")

    kg1_id2row = {int(gid): i for i, gid in enumerate(all_kg1_ids)}
    kg2_id2row = {int(gid): i for i, gid in enumerate(all_kg2_ids)}

    link_pairs = []
    pair_groups = [] if link_groups is not None else None
    for li, (e1, e2) in enumerate(test_links):
        r1 = kg1_id2row.get(int(e1))
        r2 = kg2_id2row.get(int(e2))
        if r1 is not None and r2 is not None:
            link_pairs.append((r1, r2))
            if pair_groups is not None:
                pair_groups.append(link_groups[li])

    # Similarity work stays on CPU unless DISTEA_EVAL_DEVICE=cuda — see
    # _all_gather_embeddings and _sim_device.
    sim_dev = _sim_device(device)
    cpu = torch.device("cpu")  # the dense CSV branch below stays on CPU
    n1, n2 = all_kg1_emb.size(0), all_kg2_emb.size(0)
    chunked = n1 * n2 * 4 > _dense_limit_bytes()
    if shard_eval and not chunked and rank != 0:
        return None  # the dense path stays on rank 0
    if sim_dev.type == "cuda" and rank == 0:
        logging.info(f"[eval] similarity work on {sim_dev} (DISTEA_EVAL_DEVICE=cuda)")
    metrics = _eval_metrics_auto(
        metric, all_kg1_emb, all_kg2_emb, link_pairs, sim_dev,
        use_csls, csls_k, top_k, groups=pair_groups,
        shard=(rank, world_size) if (shard_eval and chunked) else None,
    )
    if rank != 0:
        return None
    logging.info(f"[eval] selective_eval: {metrics}")

    if save_path:
        if chunked:
            # The per-pair CSV needs full rows; at chunked scale it is
            # also impractically large. unified_eval works from the
            # export bundle instead.
            logging.warning(
                "[eval] save_path skipped: chunked eval path active "
                f"(dense sim would be {n1 * n2 * 4 / (1 << 30):.1f} GB)"
            )
        else:
            sim = _compute_sim(metric, all_kg1_emb, all_kg2_emb, cpu)
            if use_csls:
                row_thr = _csls_thr(sim, csls_k, cpu)
                col_thr = _csls_thr(sim.t(), csls_k, cpu)
                sim = _apply_csls(sim, row_thr, col_thr, cpu)
            _save_predictions_csv(
                sim, all_kg1_ids, all_kg2_ids, test_links,
                kg1_id2row, kg2_id2row, save_path,
            )

    if export_dir:
        # meta.json: what this bundle is + the online numbers it must
        # reproduce (scripts/rescore_embeddings.py's parity check), and fp16
        # as an exploration-only option (fp32 for parity).
        _meta = {"metric": metric, "use_csls": bool(use_csls), "csls_k": int(csls_k),
                 "sim_device": str(sim_dev), "world_size": int(world_size)}
        if export_meta:
            _meta.update(export_meta)
        _meta.update({k: float(v) for k, v in metrics.items()})
        _save_export_bundle(
            all_kg1_emb, all_kg1_ids, all_kg2_emb, all_kg2_ids, test_links, export_dir,
            dtype=export_dtype, meta=_meta,
        )

    return metrics


def _save_export_bundle(
    kg1_emb: torch.Tensor,
    kg1_ids: np.ndarray,
    kg2_emb: torch.Tensor,
    kg2_ids: np.ndarray,
    test_links: list,
    export_dir: str,
    dtype: str = "fp32",
    meta: dict | None = None,
) -> None:
    """Write the scripts/unified_eval.py export contract (rank 0 only):
    embeddings.npy/ids.npy/kg_labels.npy/test_links.json under export_dir,
    plus meta.json (dtype, n_kg1/n_kg2, width + whatever the caller passes).
    dtype="fp16" halves the bundle (1.22 GB instead of 2.45 GB at DBP1M) for
    exploration; the parity check needs fp32.
    Reuses selective_eval's already-gathered test-link-scoped embeddings —
    no extra gather cost. Raw (pre-CSLS, pre-normalisation) embeddings are
    saved so unified_eval.py can apply any {metric, use_csls} protocol
    after the fact: all reported numbers are computed by unified_eval.py, so
    the protocol must be swappable without rerunning training.
    """
    os.makedirs(export_dir, exist_ok=True)
    emb = torch.cat([kg1_emb.cpu(), kg2_emb.cpu()], dim=0)
    emb = (emb.half() if dtype == "fp16" else emb.float()).numpy()
    ids = np.concatenate([kg1_ids, kg2_ids]).astype(np.int64)
    info = {"dtype": dtype, "n_kg1": int(len(kg1_ids)), "n_kg2": int(len(kg2_ids)),
            "width": int(emb.shape[1]), "n_test_links": int(len(test_links))}
    if meta:
        info.update(meta)
    with open(os.path.join(export_dir, "meta.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    kg_labels = np.concatenate(
        [np.zeros(len(kg1_ids), dtype=np.int8), np.ones(len(kg2_ids), dtype=np.int8)]
    )
    np.save(os.path.join(export_dir, "embeddings.npy"), emb)
    np.save(os.path.join(export_dir, "ids.npy"), ids)
    np.save(os.path.join(export_dir, "kg_labels.npy"), kg_labels)
    with open(os.path.join(export_dir, "test_links.json"), "w") as f:
        json.dump([[int(e1), int(e2)] for e1, e2 in test_links], f)
    logging.info(f"[eval] export bundle saved → {export_dir}")


def _save_predictions_csv(
    sim: torch.Tensor,
    all_kg1_ids: np.ndarray,
    all_kg2_ids: np.ndarray,
    test_links: list,
    kg1_id2row: dict,
    kg2_id2row: dict,
    save_path: str,
) -> None:
    """Write per-pair alignment predictions to a CSV file (rank 0 only)."""
    best_col = sim.argmax(dim=1)
    best_score = sim.max(dim=1).values

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kg1_id", "predicted_kg2", "true_kg2", "rank", "score", "correct"])
        for e1, e2 in test_links:
            r1 = kg1_id2row.get(int(e1))
            r2 = kg2_id2row.get(int(e2))
            if r1 is None or r2 is None:
                continue
            pred_col = int(best_col[r1].item())
            rank_val = int((sim[r1] > sim[r1, r2]).sum().item()) + 1
            w.writerow([
                int(e1),
                int(all_kg2_ids[pred_col]),
                int(e2),
                rank_val,
                f"{best_score[r1].item():.4f}",
                pred_col == r2,
            ])
    logging.info(f"[eval] alignment predictions saved → {save_path}")


def generate_pseudo_seeds(
    model,
    g,
    h_init: torch.Tensor,
    all_global_ids: np.ndarray,
    local_kg1_idx: np.ndarray,
    local_kg2_idx: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
    threshold: float = 0.85,
    metric: str = "l1",
    use_csls: bool = False,
    csls_k: int = 10,
    candidate_kg1=None,
    candidate_kg2=None,
    sim_on_device: bool = False,
    top_frac: float = 1.0,
) -> list[tuple[int, int]]:
    """
    Generate pseudo alignment seeds for the next iteration by mutual nearest
    neighbour — matches RREA/runner.py's Runner.train()/iterative_training()
    exactly: a pair is accepted iff each side is the other's #1 nearest
    neighbour, no similarity/distance threshold on top of that. `threshold`
    is accepted but unused (kept for --seed_threshold compatibility; RREA
    has no equivalent knob).

    metric="l1" (default, GCN-Align convention) or "cosine" — RREA's own
    train()/iterative_training() L2-normalises embeddings before ranking
    (`Lvec = Lvec / np.linalg.norm(...)`), i.e. cosine, same as CSLS_test().
    framework.py passes metric="cosine" here to match, same as its eval calls.

    use_csls/csls_k: RREA/runner.py's real pseudo-seed step ranks by
    CSLS-corrected similarity, not raw cosine — `eval_alignment_by_sim_mat`
    is called with `csls=10` (RREA/CSLS.py:55, applied via `sim_handler`,
    CSLS.py:305-323), suppressing "hub" points that would otherwise look
    like the nearest neighbour of many others and produce spurious mutual-NN
    agreement. Default False (raw similarity); framework.py passes its
    --use_csls/--csls_k here to match its eval calls.

    candidate_kg1 / candidate_kg2 (framework's --pseudo_seed_policy
    unaligned|testside): global ids allowed on each side; the
    mutual-NN search then runs among those rows only, like RREA's rest_set
    (unaligned entities), instead of over every entity. None (default) keeps
    the full search. sim_on_device: run the search on DISTEA_EVAL_DEVICE
    (as selective_eval does) instead of always on CPU.
    top_frac < 1 (--pseudo_seed_top_frac): keep
    only that fraction of the mutual pairs, the most confident by their
    (CSLS-adjusted) similarity; 1.0 keeps every mutual pair (RREA).
    Returns list of (kg1_global_id, kg2_global_id) pairs. All ranks receive
    the same list (broadcast from rank 0).
    """
    model.eval()
    with torch.no_grad():
        emb = model(g, h_init)

    # Candidate rows are selected BEFORE the all_gather
    # (each rank keeps its own candidates, in local order) instead of after
    # it on rank 0: the gathered rows and their order are the same, but at
    # DBP1M scale the gather shrinks from every entity (1.68M x 768 floats,
    # ~5 GB per rank) to the test-side pool (~0.48M rows).
    lk1 = np.asarray(local_kg1_idx)
    lk2 = np.asarray(local_kg2_idx)
    if candidate_kg1 is not None:
        lk1 = lk1[np.isin(all_global_ids[lk1], np.asarray(candidate_kg1, dtype=np.int64))]
    if candidate_kg2 is not None:
        lk2 = lk2[np.isin(all_global_ids[lk2], np.asarray(candidate_kg2, dtype=np.int64))]
    kg1_emb = emb[torch.as_tensor(lk1, dtype=torch.long, device=emb.device)]
    kg2_emb = emb[torch.as_tensor(lk2, dtype=torch.long, device=emb.device)]
    kg1_ids = all_global_ids[lk1]
    kg2_ids = all_global_ids[lk2]

    all_kg1_emb, all_kg1_ids = _all_gather_embeddings(
        kg1_emb, kg1_ids, rank, world_size, device
    )
    all_kg2_emb, all_kg2_ids = _all_gather_embeddings(
        kg2_emb, kg2_ids, rank, world_size, device
    )

    # DISTEA_EVAL_SHARD=1: the chunked mutual-NN passes run
    # on every rank's slice (see _chunked_mutual_nn); rank 0 still does the
    # mutual check and broadcasts the seeds.
    shard_mine = world_size > 1 and os.environ.get("DISTEA_EVAL_SHARD", "0") == "1"
    seeds = []
    count_t = torch.tensor([0], device=device)
    if rank == 0 or shard_mine:
        # Similarity work stays on CPU — see _all_gather_embeddings.
        # Mutual nearest neighbours — matches RREA/runner.py's own pseudo-seed
        # step exactly (Runner.train()/iterative_training(): sorted mutual-NN
        # agreement between eval_alignment_by_sim_mat(Lvec,Rvec,...) and its
        # (Rvec,Lvec) mirror, no similarity/distance THRESHOLD anywhere).
        # `threshold` is accepted but unused — see main.py's --seed_threshold.
        # Above the dense-memory limit the argmax pair comes from the
        # chunked path (never materializes (n1, n2)), which is required at
        # 1M scale since this runs at every iteration boundary.
        cpu = _sim_device(device) if sim_on_device else torch.device("cpu")
        if candidate_kg1 is not None:
            m1 = np.isin(np.asarray(all_kg1_ids), np.asarray(candidate_kg1, dtype=np.int64))
            all_kg1_emb, all_kg1_ids = all_kg1_emb[torch.from_numpy(m1)], np.asarray(all_kg1_ids)[m1]
        if candidate_kg2 is not None:
            m2 = np.isin(np.asarray(all_kg2_ids), np.asarray(candidate_kg2, dtype=np.int64))
            all_kg2_emb, all_kg2_ids = all_kg2_emb[torch.from_numpy(m2)], np.asarray(all_kg2_ids)[m2]
        n1, n2 = all_kg1_emb.size(0), all_kg2_emb.size(0)
        row_best = None
        chunked = n1 > 0 and n2 > 0 and n1 * n2 * 4 > _dense_limit_bytes()
        if rank != 0 and not chunked:
            pass  # sharded mode, but only the chunked path is split; rank 0 does the rest
        elif n1 == 0 or n2 == 0:
            best_kg2 = torch.zeros(0, dtype=torch.long)
            best_kg1 = torch.zeros(0, dtype=torch.long)
        elif chunked:
            if rank == 0:
                logging.info(
                    f"[eval] pseudo-seeds via chunked mutual-NN: dense sim would "
                    f"be {n1 * n2 * 4 / (1 << 30):.1f} GB (> limit)"
                    + (f", sharded over {world_size} ranks" if shard_mine else "")
                )
            out = _chunked_mutual_nn(
                metric, all_kg1_emb, all_kg2_emb, cpu,
                use_csls=use_csls, csls_k=csls_k, return_row_best=top_frac < 1.0,
                shard=(rank, world_size) if shard_mine else None,
            )
            best_kg2, best_kg1 = out[0], out[1]
            row_best = out[2] if top_frac < 1.0 else None
        else:
            sim = _compute_sim(metric, all_kg1_emb, all_kg2_emb, cpu)
            if use_csls:
                row_thr = _csls_thr(sim, csls_k, cpu)
                col_thr = _csls_thr(sim.t(), csls_k, cpu)
                sim = _apply_csls(sim, row_thr, col_thr, cpu)
            best_kg2 = sim.argmax(dim=1)  # (N_kg1,) — best KG2 col per KG1 row
            best_kg1 = sim.argmax(dim=0)  # (N_kg2,) — best KG1 row per KG2 col
            if top_frac < 1.0:
                row_best = sim.gather(1, best_kg2.unsqueeze(1)).squeeze(1)
    if rank == 0:
        # Vectorised mutual check: pairs in KG1 row order.
        b2 = best_kg2.cpu().numpy()
        b1 = best_kg1.cpu().numpy()
        mutual = np.nonzero(b1[b2] == np.arange(b2.size))[0] if b2.size else np.zeros(0, dtype=np.int64)
        n_mutual = int(mutual.size)
        if top_frac < 1.0 and n_mutual > 0 and row_best is not None:
            conf = row_best.float().cpu().numpy()[mutual]
            n_keep = int(np.ceil(top_frac * n_mutual))
            mutual = np.sort(mutual[np.argsort(-conf, kind="stable")[:n_keep]])
        k1ids, k2ids = np.asarray(all_kg1_ids), np.asarray(all_kg2_ids)
        seeds = [(int(k1ids[i]), int(k2ids[b2[i]])) for i in mutual]
        logging.info(
            f"[eval] generate_pseudo_seeds: {len(seeds)} seeds"
            + (f" (most confident {top_frac:.0%} of {n_mutual} mutual pairs)" if top_frac < 1.0 else "")
        )
        count_t = torch.tensor([len(seeds)], device=device)

    dist.broadcast(count_t, src=0)
    n_seeds = int(count_t.item())

    if n_seeds == 0:
        return []

    # Broadcast seed pairs
    if rank == 0:
        seeds_t = torch.tensor(seeds, dtype=torch.long, device=device)  # (n_seeds, 2)
    else:
        seeds_t = torch.zeros(n_seeds, 2, dtype=torch.long, device=device)
    dist.broadcast(seeds_t, src=0)

    return [(int(r[0]), int(r[1])) for r in seeds_t.tolist()]
