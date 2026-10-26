"""
gcn.py — GCN-Align encoder for DistEA, ported faithfully from ELsEA
(DivEA/GCN_Align/{models,layers,inits}.py — TensorFlow) to PyTorch.

Implements GCN-Align: the simplest GNN-based entity alignment model.
  - Symmetric normalisation: D̂⁻¹/² Â D̂⁻¹/² where Â = A + I, D̂ = weighted
    row-sum of Â (NOT a plain edge count — see GCNLayer)
  - ReLU activation on hidden layers; identity on the output layer
  - Raw (NOT L2-normalised) output embeddings — ELsEA never normalises
    `model.outputs` anywhere in models.py/layers.py; GCN_Align_run.py's only
    L2-normalise happens after get_hits() has already run, purely to prep the
    saved .npz for external consumers. The actual align_loss/infoNce1/get_hits
    calls all operate on the raw GCN output. See eval.py for the matching
    L1-distance (not cosine) ranking.
  - Alignment loss uses L1 distance with KG-boundary-aware negatives

Node ordering within each rank's local graph:
    [local_kg1 | halo_kg1 | local_kg2 | halo_kg2]

kg1_size = len(local_kg1_idx) + len(halo_kg1_idx)  — KG1/KG2 boundary index
(halo_kg2 nodes are appended after local_kg2, so they fall on the KG2 side
of this boundary, which is correct — see dist_graph.py).

ELsEA's actual GCN_Align (models.py) has *no learnable weight matrix inside
the propagation stack at all*: its first GraphConvolution layer is built with
`featureless=True`, so that layer's "weight" variable (shape [N_entities,
se_dim]) is never multiplied against an input — it IS the per-entity
embedding table, used directly as the thing that gets propagated. Its second
layer is built with `transform=False` and `input_dim == output_dim`, which
GraphConvolution.__init__ special-cases into creating no weight variable at
all (pure `A @ x`, identity activation). DistEA's `h_init` (see
framework.py — a learnable (N, output_dim) nn.Parameter seeded per global
entity ID) already plays the role of that featureless table, so `GCNLayer`
below carries no `nn.Linear` of its own: every layer here is a pure
structural-propagation step, same as ELsEA. This is a deliberate departure
from vanilla GCN (Kipf & Welling), which does transform every layer — GCN-
Align's whole point is to let the *entity table itself* carry all the
capacity, not per-layer feature transforms.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    import dgl
    import dgl.function as fn

    _HAS_DGL = True
except (ImportError, FileNotFoundError):
    _HAS_DGL = False


# ---------------------------------------------------------------------------
#  Deterministic initial features  (LeapGNN infrastructure)
# ---------------------------------------------------------------------------


def _feature_rows(global_ids, dim: int, out: np.ndarray) -> None:
    """Fill out[i] with the ID-seeded draw of global_ids[i] (un-normalised)."""
    for i, gid in enumerate(global_ids):
        rng = np.random.default_rng(int(gid))
        out[i] = rng.standard_normal(dim).astype(np.float32)


def _feature_procs(n: int) -> int:
    """DISTEA_FEAT_PROCS=P (default 1): build_entity_features
    fills its rows in P forked worker processes. The per-ID generator loop is
    ~14 us per row in one Python thread (24 s for DBP1M's 1.68M rows at N=1,
    ~5 s per rank at N=8); each worker runs the
    same per-row code on a contiguous slice, so the table is bit-identical
    (tests/test_feature_procs.py). Capped by the CPUs this process may use and
    by n / 20000 (fork costs more than it saves on small tables). POSIX only;
    falls back to the serial loop on any fork / worker failure."""
    try:
        want = int(os.environ.get("DISTEA_FEAT_PROCS", "1") or 1)
    except ValueError:
        want = 1
    if want <= 1 or not hasattr(os, "fork"):
        return 1
    try:
        avail = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        avail = os.cpu_count() or 1
    return max(1, min(want, avail, n // 20000))


def _feature_rows_forked(global_ids, dim: int, out: np.ndarray, procs: int) -> bool:
    """_feature_rows over `procs` forked children writing disjoint row slices
    of an anonymous shared mapping; True when every child succeeded (rows
    copied into `out`), False otherwise (`out` untouched -- the caller then
    runs the serial loop). The children only run numpy code and leave via
    os._exit, so nothing the parent holds (CUDA context, process groups) is
    touched on their side."""
    import mmap

    n = len(global_ids)
    buf = mmap.mmap(-1, n * dim * 4)  # MAP_SHARED | MAP_ANONYMOUS
    shared = np.frombuffer(buf, dtype=np.float32).reshape(n, dim)
    bounds = np.linspace(0, n, procs + 1).astype(np.int64)
    pids: list[int] = []
    ok = True
    try:
        with warnings.catch_warnings():
            # Python 3.12 warns on fork() in a multi-threaded process; the
            # children never take a lock another thread could hold.
            warnings.simplefilter("ignore", DeprecationWarning)
            for p in range(procs):
                lo, hi = int(bounds[p]), int(bounds[p + 1])
                pid = os.fork()
                if pid == 0:  # child
                    code = 1
                    try:
                        _feature_rows(global_ids[lo:hi], dim, shared[lo:hi])
                        code = 0
                    finally:
                        os._exit(code)
                pids.append(pid)
    except OSError:
        ok = False
    for pid in pids:
        try:
            _, status = os.waitpid(pid, 0)
            ok = ok and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        except ChildProcessError:
            ok = False
    if ok:
        out[:] = shared
    del shared
    buf.close()
    return ok


def build_entity_features(
    global_ids: np.ndarray, dim: int, device: torch.device
) -> torch.Tensor:
    """
    Return (N, dim) float32 feature matrix for the given global entity IDs.

    Each row is seeded by its global entity ID so that:
      - The same global ID always produces the same feature vector.
      - Halo nodes on any rank have the same feature as the local copy.
      - No cross-rank communication is needed.
    """
    feats = np.zeros((len(global_ids), dim), dtype=np.float32)
    procs = _feature_procs(len(global_ids))
    if procs <= 1 or not _feature_rows_forked(global_ids, dim, feats, procs):
        _feature_rows(global_ids, dim, feats)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    feats /= norms
    return torch.from_numpy(feats).to(device)


# ---------------------------------------------------------------------------
#  GCN-Align layer
# ---------------------------------------------------------------------------


class GCNLayer(nn.Module):
    """
    Single GCN-Align propagation step (ELsEA style) — NO learnable weight.

    Aggregation: agg_v = Σ_u  norm(u, v) · W(u, v) · h_u
      where  norm(u, v) = 1 / sqrt(deg(u) · deg(v))
             deg(u)     = Σ_v W(u, v)   over  = adj + self-loop
                          (WEIGHTED row-sum, matching ELsEA's normalize_adj —
                          a plain edge count would silently ignore r2f/r2if)
             W(u, v)    = edge weight from r2f/r2if (relation importance)
             self-loops use W = 1
    No transform: the aggregated message IS the layer's output (activation
    applied by the parent GCN, not here) — see module docstring for why.

    Falls back to uniform W = 1 when the graph has no `edata["w"]` (tests).

    `featureless`: ELsEA's first GraphConvolution layer is built with
    `featureless=True` (layers.py) — dropout is applied to `x` (the "features"
    placeholder) BEFORE the featureless branch discards `x` entirely and uses
    `self.vars['weights_0']` (the entity table) directly instead, so that
    dropout has ZERO effect on the featureless layer. Only later (non-
    featureless) layers actually drop out their input. `featureless=True`
    here reproduces that exactly: dropout is skipped for this layer's `h`.
    """

    def __init__(self, dropout: float = 0.0, featureless: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.featureless = featureless

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        if not self.featureless:
            h = self.dropout(h)
        if _HAS_DGL and isinstance(g, dgl.DGLGraph):
            return self._dgl_forward(g, h)
        return self._sparse_forward(g, h)

    def _dgl_forward(self, g, h: torch.Tensor) -> torch.Tensor:
        # Rebuild graph with self-loops, preserving the r2f/r2if edge weights
        # set by dist_graph.py. We don't use dgl.add_self_loop directly because
        # it leaves edata for the new self-loop edges undefined, which would
        # break the combined-weight message function below.
        device = h.device
        g_clean = dgl.remove_self_loop(g)
        src, dst = g_clean.edges()

        if "w" in g_clean.edata:
            edge_w = g_clean.edata["w"]
        else:
            edge_w = torch.ones(g_clean.num_edges(), device=device)

        N = g_clean.num_nodes()
        sl_nodes = torch.arange(N, device=device)
        full_src = torch.cat([src, sl_nodes])
        full_dst = torch.cat([dst, sl_nodes])
        full_w = torch.cat([edge_w, torch.ones(N, device=device)])

        g_sl = dgl.graph((full_src, full_dst), num_nodes=N).to(device)
        g_sl.edata["w"] = full_w

        with g_sl.local_scope():
            # Weighted degree: sum of incident edge weights (incl. self-loop),
            # matching ELsEA's normalize_adj rowsum. dgl's in_degrees() would
            # only count edges and silently ignore "w" — not equivalent.
            g_sl.update_all(fn.copy_e("w", "m"), fn.sum("m", "deg"))
            deg = g_sl.ndata["deg"].clamp(min=1e-8)
            norm = deg.pow(-0.5)
            g_sl.ndata["n"] = norm
            g_sl.apply_edges(
                lambda e: {"w_combined": e.src["n"] * e.dst["n"] * e.data["w"]}
            )
            g_sl.ndata["h"] = h
            g_sl.update_all(fn.u_mul_e("h", "w_combined", "m"), fn.sum("m", "agg"))
            agg = g_sl.ndata["agg"]
        return agg

    def _sparse_forward(self, adj, h: torch.Tensor) -> torch.Tensor:
        # Build Â = adj + I
        N = h.size(0)
        device = h.device
        eye_idx = torch.arange(N, device=device).unsqueeze(0).expand(2, -1)
        eye = torch.sparse_coo_tensor(eye_idx, torch.ones(N, device=device), (N, N))
        a_hat = (adj + eye).coalesce()

        # Symmetric normalisation: D̂⁻¹/² Â D̂⁻¹/², D̂ = weighted row-sum of Â
        deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        idx = a_hat.indices()
        vals = a_hat.values() * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
        a_sym = torch.sparse_coo_tensor(idx, vals, a_hat.size(), device=device)

        return torch.sparse.mm(a_sym, h)


# ---------------------------------------------------------------------------
#  GCN model
# ---------------------------------------------------------------------------


class GCN(nn.Module):
    """
    Multi-layer GCN-Align encoder — pure structural propagation (no per-layer
    weight; see module docstring). `h` (DistEA's `h_init`) is the only
    learnable input; its width must equal `output_dim` since nothing in the
    stack changes dimensionality.

    Returns RAW (not L2-normalised) embeddings of shape (N_nodes, output_dim)
    — see module docstring for why. Callers that need unit vectors (e.g. an
    explicit cosine-similarity path) must normalise themselves.
    Architecture: [ReLU hidden layers] → [identity output layer], no norm.
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        # Layer 0 is featureless (ELsEA's first GraphConvolution) — dropout
        # is a no-op there; every later layer drops out its input (its
        # predecessor's output) — see GCNLayer's docstring.
        self.layers = nn.ModuleList(
            [GCNLayer(dropout=dropout, featureless=(i == 0)) for i in range(num_layers)]
        )

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def forward_blocks(self, blocks, h_init: torch.Tensor) -> torch.Tensor:
        """
        Mini-batch counterpart of forward() — additive, does not replace
        forward(). `blocks` is a list
        of minibatch_sampler.Block, outermost-hop-first, one per GCN layer
        (len(blocks) == len(self.layers)); `h_init` is the RAW initial
        feature tensor for `blocks[0].outer_ids` (the outermost sampled
        frontier) ONLY — every later layer's input is this method's own
        previous layer output, exactly like forward()'s `h` is reassigned
        each iteration, NOT a fresh gather of that layer's own initial
        embedding (multi-layer GNN semantics require layer i>0's input to
        be layer i-1's computed output).
        `blocks[i].inner_ids == blocks[i+1].outer_ids` by construction
        (minibatch_sampler.sample_multi_hop's layer semantics), so this
        threading lines up node-for-node between consecutive layers.

        Each layer's own weighted-symmetric-normalisation is already baked
        into `blocks[i].adj_t` (minibatch_sampler.build_block, using TRUE
        precomputed degrees, not degrees recomputed from the block's own
        partial edges) — so this reduces to a plain rectangular sparse
        matmul per layer instead of GCNLayer's square-adjacency one.
        Featureless-first-layer dropout skip (see GCNLayer.forward) is
        preserved: dropout is applied to every layer's input except layer 0.

        Correctness invariant (tests/test_stage_n.py): calling this with a
        single block per layer whose outer_ids/inner_ids together cover the
        WHOLE graph (i.e. "one batch, no sampling") must reproduce forward()
        numerically — if it doesn't, the block math itself is wrong,
        independent of whatever a real sampler does on top.
        """
        assert len(blocks) == len(self.layers), (
            f"forward_blocks needs one block per layer: got {len(blocks)} "
            f"blocks for {len(self.layers)} layers"
        )
        h = h_init
        for i, (block, layer) in enumerate(zip(blocks, self.layers, strict=False)):
            if not layer.featureless:
                h = layer.dropout(h)
            h = torch.sparse.mm(block.adj_t.t(), h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h


# ---------------------------------------------------------------------------
#  Alignment loss  (KG-boundary-aware, L1 distance)
# ---------------------------------------------------------------------------


def infonce_loss(
    emb: torch.Tensor,
    pairs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss (mirrors ELsEA's `infoNce1` in
    DivEA/GCN_Align/metrics.py).

    Steps:
      1. Look up KG1 / KG2 embeddings for each positive pair.
      2. z-score per dimension, then L2-normalize.
      3. Positive = cosine(e1, e2); negatives = cosine(e1, all other e2).
      4. Standard NCE softmax with temperature τ.
    """
    if len(pairs) == 0:
        return emb.new_zeros(1).squeeze()

    e1 = emb[pairs[:, 0]]                                  # (P, D)
    e2 = emb[pairs[:, 1]]                                  # (P, D)

    # z-score per dimension (matches ELsEA's z_score: tf.nn.moments returns
    # the BIASED/population variance — divide by N, not N-1 — so
    # unbiased=False here; negligible in practice at these batch sizes, but
    # this is meant to be an exact port, not an approximation)
    def _z(x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)
        return (x - mean) / std

    e1 = torch.nn.functional.normalize(_z(e1), p=2, dim=1)
    e2 = torch.nn.functional.normalize(_z(e2), p=2, dim=1)

    pos = (e1 * e2).sum(dim=1) / temperature               # (P,)
    sim = (e1 @ e2.T) / temperature                        # (P, P) — all pairs
    pos = torch.exp(pos)
    den = torch.exp(sim).sum(dim=1).clamp_min(1e-12)       # (P,)
    return -(torch.log(pos / den)).mean()


def consistency_reg_loss(
    h: torch.Tensor,
    idx: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared L2 distance between h_init's halo rows (`h[idx]`) and their
    last-known remote value (`target`, the same snapshot `_patch_halo_rows`
    already hard-overwrites those rows with once per iteration —
    framework.py). Adding `lambda * consistency_reg_loss(...)` to the
    training loss makes "stay close to the last-known remote value" part of
    the gradient signal continuously across every epoch, instead of only at
    iteration boundaries.

    Motivation: h_init is never all-reduced across ranks (unlike the
    model's own small parameters), and changing only *when*/*how often*
    the hard overwrite happens does not close the same-rank/cross-rank
    accuracy gap; this adds a gradient-level term instead.

    `target` must be detached (no grad) — it's a fixed anchor for this
    iteration, not something optimized itself.
    """
    if idx.numel() == 0:
        return h.new_zeros(())
    return ((h[idx] - target) ** 2).sum(dim=1).mean()


def nhsm_loss(
    emb: torch.Tensor,
    pairs: torch.Tensor,
    gamma: float = 1.0,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """
    Normalized Hard Sample Mining loss — Dual-AMN's alignment loss,
    ported from ClusterEA's own PyTorch port (ClusterEA/src/prev_models/
    duala/loss.py::align_loss, verified against the TF original in
    prev_models/rrea/dual_amn/duala.py, not reconstructed from the
    paper). This is the loss ClusterEA credits for
    compressing RREA's budget from ~1200 to ~50 epochs.

    Reference semantics, kept exactly:
      pos_dis        = ||e_l − e_r||²  (squared L2, per pair)
      neg_dis (side) = squared distance from that side's anchor to EVERY
                       row of `emb` (negatives = the whole table, both KGs —
                       consistent with RREA's own full-range sampling)
      loss_mat       = pos_dis − neg_dis + gamma, positive pair's own two
                       columns zeroed, then PER-ROW z-score with detached
                       mean/std (the "Normalized" part; reference uses
                       stop_gradient / .data assignment), then
                       logsumexp(λ·x + τ) with λ=30, τ=10 (reference
                       hardcodes both), mean over pairs and both sides.

    One deliberate implementation difference, numerics identical: the
    reference zeroes the positive columns by materializing TWO (P, N)
    one-hot matrices and multiplying by (1 − oh(l) − oh(r)); at DistEA's
    per-rank scale that is O(P·N) extra memory (≈6 GB at 100K/ws=4).
    scatter_-ing zeros into the same two entries per row is the same
    operation without the materialization (l and r can never collide:
    they index different KGs' rows). Covered by a unit test comparing
    against a literal one-hot reference implementation on a small case.

    Ignores neg_k / pair_weights / infonce by design — the reference loss
    has no such knobs; when --loss nhsm is active those flags are inert
    (dist_trainer.py logs this once).

    Memory: every intermediate is (P, N). At 100K/ws=1 that is
    20000 x 200000 fp32 = 14.9 GiB per tensor and the dense form needs
    several of them (more than a 40 GB A100 holds).
    Every per-row quantity (z-score statistics, logsumexp) depends only on
    its own row, so the loss is computed in row blocks of `chunk_rows`
    pairs under torch.utils.checkpoint — numerically identical to the dense
    form (tests/test_nhsm_chunked.py), peak memory bounded by one block.
    chunk_rows=None picks ~1 GB fp32 per (rows, N) temporary; small graphs
    (15K: P <= chunk_rows) take the plain dense path unchanged.
    """
    P = pairs.size(0)
    if chunk_rows is None:
        chunk_rows = max(256, (1 << 28) // max(emb.size(0), 1))
    if P <= chunk_rows:
        return _nhsm_rows_sum(emb, pairs, gamma) / P
    from torch.utils.checkpoint import checkpoint

    total = emb.new_zeros(())
    for start in range(0, P, chunk_rows):
        block = pairs[start:start + chunk_rows]
        total = total + checkpoint(_nhsm_rows_sum, emb, block, gamma, use_reentrant=False)
    return total / P


def _nhsm_rows_sum(emb: torch.Tensor, pairs: torch.Tensor, gamma: float) -> torch.Tensor:
    """Sum over `pairs` of the per-pair NHSM loss (both sides); see nhsm_loss."""
    left = pairs[:, 0]
    right = pairs[:, 1]
    l_emb = emb[left]
    r_emb = emb[right]

    pos_dis = torch.sum((l_emb - r_emb) ** 2, dim=-1, keepdim=True)  # (P, 1)

    def squared_dist(a: torch.Tensor) -> torch.Tensor:
        # ||a_i − emb_j||² via the reference's row-norm expansion.
        a_norm = torch.sum(a**2, dim=1, keepdim=True)          # (P, 1)
        b_norm = torch.sum(emb**2, dim=1).unsqueeze(0)          # (1, N)
        return a_norm + b_norm - 2.0 * (a @ emb.t())            # (P, N)

    l_neg_dis = squared_dist(l_emb)
    r_neg_dis = squared_dist(r_emb)

    row_idx = torch.arange(pairs.size(0), device=emb.device)

    def side_loss(neg_dis: torch.Tensor) -> torch.Tensor:
        loss = pos_dis - neg_dis + gamma                        # (P, N)
        # Zero the positive pair's own columns (reference: ×(1−oh(l)−oh(r))).
        loss = loss.index_put((row_idx, left), torch.zeros_like(row_idx, dtype=loss.dtype))
        loss = loss.index_put((row_idx, right), torch.zeros_like(row_idx, dtype=loss.dtype))
        # Per-row z-score with DETACHED statistics (reference stop_gradient).
        mean = loss.mean(dim=-1, keepdim=True).detach()
        std = loss.std(dim=-1, keepdim=True).detach()
        loss = (loss - mean) / std
        lamb, tau = 30.0, 10.0  # reference hardcodes both
        return torch.logsumexp(lamb * loss + tau, dim=-1)

    return torch.sum(side_loss(l_neg_dis) + side_loss(r_neg_dis))


def combined_align_loss(
    emb: torch.Tensor,
    pairs: torch.Tensor,
    kg1_size: int,
    weight: torch.Tensor,
    neg_k: int = 1,
    margin: float = 6.0,
    infonce_temp: float = 0.1,
    pair_weights: torch.Tensor | None = None,
    neg_kg1: torch.Tensor | None = None,
    neg_kg2: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ELsEA-style combined loss = w0 * InfoNCE + w1 * align_loss.

    `weight` is a learnable (2,) parameter; we softmax-normalize it inside,
    matching ELsEA's `weight_norm = softmax(weight) / sum(...)`.

    `pair_weights` (e.g. VBIP informativeness scores, see dist_vbip.py) is
    forwarded to the align_loss term only — InfoNCE's (P, P) softmax doesn't
    decompose per-pair as simply, so it stays unweighted. Likewise `neg_kg1`/
    `neg_kg2` (see alignment_loss) only affect the align_loss term — InfoNCE
    has no index-based negative sampling to override (its negatives are the
    other positives in the same batch).

    Defaults (neg_k=1, margin=6.0, temp=0.1) match ELsEA GCN_Align_run.py.
    """
    w = torch.softmax(weight, dim=0)
    w = w / w.sum()
    l_info = infonce_loss(emb, pairs, temperature=infonce_temp)
    l_algn = alignment_loss(
        emb, pairs, kg1_size, neg_k=neg_k, margin=margin, pair_weights=pair_weights,
        neg_kg1=neg_kg1, neg_kg2=neg_kg2,
    )
    return w[0] * l_info + w[1] * l_algn


def alignment_loss(
    emb: torch.Tensor,
    pairs: torch.Tensor,
    kg1_size: int,
    neg_k: int = 10,
    margin: float = 3.0,
    pair_weights: torch.Tensor | None = None,
    neg_kg1: torch.Tensor | None = None,
    neg_kg2: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    GCN-Align margin-based contrastive loss with KG-boundary-aware negatives.

    Uses L1 distance (as in the original GCN-Align paper).
    Negatives are sampled within each KG:
      - Corrupt e1: random KG1 entity  (indices [0, kg1_size))
      - Corrupt e2: random KG2 entity  (indices [kg1_size, N))

    Args:
        emb:          (N, D) raw (not L2-normalised) entity embeddings.
        pairs:        (P, 2) local index pairs — pairs[:, 0] = KG1, pairs[:, 1] = KG2.
        kg1_size:     number of KG1 nodes (local + halo) in the local graph.
        neg_k:        number of negatives per side per positive (ignored if
                      neg_kg1/neg_kg2 are given — their own shape wins).
        margin:       hinge margin.
        pair_weights: optional (P,) per-pair weight (e.g. VBIP score). None
                      reproduces the plain unweighted mean.
        neg_kg1:      optional pre-sampled (P, k) KG1 negative indices — pass
                      this (and neg_kg2) to reuse the same negatives across
                      several epochs, matching ELsEA's every-10-epochs
                      resampling (see DistTrainer's neg_resample_every) rather
                      than drawing fresh torch.randint negatives every call.
        neg_kg2:      optional pre-sampled (P, k) KG2 negative indices.

    Returns:
        Scalar loss.
    """
    if len(pairs) == 0:
        return emb.new_zeros(1).squeeze()

    N = emb.size(0)
    kg2_size = N - kg1_size
    if kg1_size <= 0 or kg2_size <= 0:
        return emb.new_zeros(1).squeeze()

    kg1_idx = pairs[:, 0]  # (P,)
    kg2_idx = pairs[:, 1]  # (P,)
    e1 = emb[kg1_idx]      # (P, D)
    e2 = emb[kg2_idx]      # (P, D)

    d_pos = (e1 - e2).abs().sum(dim=1)           # (P,)  L1
    P = len(pairs)

    # Corrupt KG1 side
    if neg_kg1 is None:
        neg_kg1 = torch.randint(0, kg1_size, (P, neg_k), device=emb.device)
    k1 = neg_kg1.size(1)
    d_pos_exp1 = d_pos.unsqueeze(1).expand(-1, k1)
    neg_e1 = emb[neg_kg1]                                    # (P, k1, D)
    e2_exp = e2.unsqueeze(1).expand(-1, k1, -1)
    d_neg_kg1 = (neg_e1 - e2_exp).abs().sum(dim=2)          # (P, k1)
    hinge_kg1 = torch.clamp(d_pos_exp1 - d_neg_kg1 + margin, min=0.0).mean(dim=1)

    # Corrupt KG2 side
    if neg_kg2 is None:
        neg_kg2 = torch.randint(kg1_size, N, (P, neg_k), device=emb.device)
    k2 = neg_kg2.size(1)
    d_pos_exp2 = d_pos.unsqueeze(1).expand(-1, k2)
    neg_e2 = emb[neg_kg2]                                    # (P, k2, D)
    e1_exp = e1.unsqueeze(1).expand(-1, k2, -1)
    d_neg_kg2 = (e1_exp - neg_e2).abs().sum(dim=2)          # (P, k2)
    hinge_kg2 = torch.clamp(d_pos_exp2 - d_neg_kg2 + margin, min=0.0).mean(dim=1)

    per_pair = hinge_kg1 + hinge_kg2  # (P,) — with pair_weights None the
    # result equals a plain .mean() over all hinges, since every pair has
    # the same neg_k count.
    if pair_weights is not None:
        w = pair_weights.to(per_pair.device)
        return (per_pair * w).sum() / w.sum().clamp_min(1e-12)
    return per_pair.mean()
