"""
Distributed VBIE/VBIP scorer for DistEA.

Two implementations, selected via `impl`/`--vbip_impl`:

  "simplified" (default): a loosely ELsEA-inspired diffusion — unweighted
  mean aggregation with a gamma-damped residual mix, single final sigmoid.
  This is NOT a faithful port of ELsEA's VBIP (see below); it is kept as the
  default so earlier MM results stay reproducible.

  "elsea": a faithful port of ELsEA's actual reference implementation
  (`divea/eviPassing.py::EviPassingLayer`/`EviPassingModel`/
  `PerfModel.compute_perf`, verified against the ELsEA/DivEA source) --
  weighted-sum
  aggregation over real (r2f/r2if-derived) directed edge weights plus
  self-loops, two explicit passes with a ratio-normalisation +
  2*sigmoid(gamma*x)-1 squash between them. Deliberately excludes ELsEA's
  leave-one-out candidate-benefit loop (`e[candi]=0` re-scoring) -- that
  mechanism selects candidate entities at partition-build time, a different
  pipeline stage from this call site (training-time alignment-loss pair
  weighting); porting it would mean restructuring partition.py/
  counterpart_disc.py, out of scope here.

Usage in DistEA (called from framework.py each iteration):
  1. compute_vbip_scores()  → local_scores tensor on CPU
  2. gather_vbip_scores()   → global {entity_id: score} dict on all ranks
  3. build_pair_weights()   → per-pair weight tensor passed to alignment_loss
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from comm_stats import CommStats

try:
    import dgl.function as fn

    _HAS_DGL = True
except (ImportError, FileNotFoundError):
    _HAS_DGL = False


# ---------------------------------------------------------------------------
#  VBIP GNN layers -- "simplified" (default) implementation
# ---------------------------------------------------------------------------


class _EviPassingLayer(nn.Module):
    """
    Single evidence-passing layer (no learnable parameters).
    Propagates anchor seed indicators through the graph.
    """

    def __init__(self):
        super().__init__()

    def forward(self, g, h: torch.Tensor, gamma: float = 0.9) -> torch.Tensor:
        """Diffuse features from neighbours, dampened by gamma."""
        if _HAS_DGL and hasattr(g, "ndata"):

            with g.local_scope():
                g.ndata["h"] = h
                g.update_all(fn.copy_u("h", "m"), fn.mean("m", "agg"))
                return gamma * g.ndata["agg"] + (1 - gamma) * h
        elif hasattr(g, "adj_self_norm"):
            # RREAGraph (models/rrea.py) — reuse its symmetric-normalised,
            # self-looped adjacency as the diffusion operator.
            agg = torch.sparse.mm(g.adj_self_norm, h)
            return gamma * agg + (1 - gamma) * h
        else:
            # Raw sparse adjacency fallback
            agg = torch.sparse.mm(g, h)
            return gamma * agg + (1 - gamma) * h


class _EviPassingModel(nn.Module):
    """
    Two-pass evidence propagation for VBIP scoring.
    Pass 1: anchor → proximity. Pass 2: re-diffuse → informativeness.
    """

    def __init__(self, num_layers: int = 2, gamma: float = 0.9):
        super().__init__()
        self.layers = nn.ModuleList([_EviPassingLayer() for _ in range(num_layers)])
        self.gamma = gamma

    def forward(self, g, anchor_mask: torch.Tensor) -> torch.Tensor:
        h = anchor_mask.float().unsqueeze(1)
        for layer in self.layers:
            h = layer(g, h, self.gamma)
        return h.squeeze(1)


# ---------------------------------------------------------------------------
#  VBIP GNN layers -- "elsea" (faithful) implementation
# ---------------------------------------------------------------------------


class _EviPassingLayerELsEA(nn.Module):
    """
    Faithful port of ELsEA's real EviPassingLayer (divea/eviPassing.py):
    weighted-sum aggregation over directed, weighted edges. No damping, no
    self-mixing -- unlike the simplified layer above, the real ELsEA layer
    is exactly `sum_u w(u,v) * h[u]`, nothing else.
    """

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        if _HAS_DGL and hasattr(g, "ndata"):

            with g.local_scope():
                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "o"))
                return g.ndata["o"]
        else:
            # g is a pre-weighted sparse adjacency (build_elsea_vbip_graph's
            # non-DGL fallback) -- weights are already baked into the values,
            # so this is a plain sparse matmul. torch.sparse.mm requires a
            # 2D dense operand; h here is the 1D per-entity score vector.
            return torch.sparse.mm(g, h.unsqueeze(1)).squeeze(1)


class _EviPassingModelELsEA(nn.Module):
    """
    Faithful port of ELsEA's real EviPassingModel: `layer_num` repeated
    weighted-sum propagations, no damping/residual mixing at any step (the
    simplified model's gamma*agg+(1-gamma)*h mixing does not exist in the
    real ELsEA layer). Real ELsEA also masks by an entity-existence vector
    `e` before each layer (`h = h*e`) for its leave-one-out candidate
    scoring -- omitted here since `e` is always all-ones at this call site
    (see module docstring for the scoping decision).
    """

    def __init__(self, layer_num: int = 2):
        super().__init__()
        self.layer_num = layer_num
        self.layer = _EviPassingLayerELsEA()

    def forward(self, g, h0: torch.Tensor) -> torch.Tensor:
        h = h0
        for _ in range(self.layer_num):
            h = self.layer(g, h)
        return h


def build_elsea_vbip_graph(data, global2local: dict, all_global_ids: np.ndarray, device: torch.device):
    """
    Build the ELsEA-style weighted adjacency for VBIP scoring -- a faithful
    port of divea/eviPassing.py's PerfModel.__init__ graph construction:
    forward edges (h->t), reverse edges (t->h), and self-loops, all weighted.

    Edge weight source: reuses dist_graph.py's already-validated r2f/r2if
    direction-dependent convention (w_fwd=max(r2if,0.3) for h->t,
    w_bwd=max(r2f,0.3) for t->h) rather than guessing at what raw
    relation-indexed array ELsEA's caller (not part of the released
    source) passes into PerfModel -- eviPassing.py alone applies the *same* scalar to both
    directions of a triple, but doesn't show where that scalar comes from.
    Reusing DistEA's own already-verified-against-ELsEA r2f/r2if scheme
    keeps one single source of truth for "real edge weight" across
    GCN-Align fidelity and VBIP, rather than inventing a second one.
    Self-loop weight = max(all edge weights) + 0.5, matching eviPassing.py
    literally (`max_weight = torch.max(edge_weights); max_weight + 0.5`).

    Returns a DGL graph (with edata["w"]) if DGL is available, else a
    coalesced torch.sparse COO tensor (N, N) with weights baked into the
    values, oriented so `torch.sparse.mm(g, h)` matches DGL's
    `update_all(fn.u_mul_e("h","w","m"), fn.sum("m","o"))` semantics (row
    index = destination, column index = source).
    """
    included = set(all_global_ids.tolist())
    src: list[int] = []
    dst: list[int] = []
    w: list[float] = []

    def add(h, r, t, edge_weights):
        if h in included and t in included:
            lh, lt = global2local[h], global2local[t]
            r2f, r2if = edge_weights.get(r, (1.0, 1.0))
            w_fwd = max(r2if, 0.3)  # head -> tail
            w_bwd = max(r2f, 0.3)  # tail -> head
            src.append(lh)
            dst.append(lt)
            w.append(w_fwd)
            src.append(lt)
            dst.append(lh)
            w.append(w_bwd)

    for h, r, t in data.kg1_triples:
        add(h, r, t, data.edge_weights1)
    for h, r, t in data.kg2_triples:
        add(h, r, t, data.edge_weights2)

    n = len(all_global_ids)
    self_w = (max(w) + 0.5) if w else 1.0

    idx = list(range(n))
    src_all = src + idx
    dst_all = dst + idx
    w_all = w + [self_w] * n

    if _HAS_DGL:
        import dgl

        src_t = torch.tensor(src_all, dtype=torch.long, device=device)
        dst_t = torch.tensor(dst_all, dtype=torch.long, device=device)
        g = dgl.graph((src_t, dst_t), num_nodes=n).to(device)
        g.edata["w"] = torch.tensor(w_all, dtype=torch.float32, device=device)
        return g
    else:
        indices = torch.tensor([dst_all, src_all], dtype=torch.long, device=device)
        values = torch.tensor(w_all, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()


def _compute_elsea_vbip_scores(
    g_weighted,
    anchor_local_indices: np.ndarray,
    num_nodes: int,
    device: torch.device,
    layer_num: int = 2,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Faithful ELsEA VBIP scoring (divea/eviPassing.py's
    PerfModel.compute_perf, minus the leave-one-out candidate loop):

      h0                    = anchor indicator (1 at train-linked entities)
      inanchorpath          = EviPassingModel(g, h0)              # pass 1
      ratio1                = inanchorpath / (inanchorpath + eps)
      weight_perf           = 2*sigmoid(gamma*ratio1) - 1
      inpath                = EviPassingModel(g, weight_perf)     # pass 2
      ratio2                = inpath / (inpath + eps)
      emb_perf              = 2*sigmoid(gamma*ratio2) - 1         # final score
    """
    model = _EviPassingModelELsEA(layer_num=layer_num).to(device)
    model.eval()

    h0 = torch.zeros(num_nodes, device=device)
    if len(anchor_local_indices) > 0:
        h0[torch.tensor(anchor_local_indices, dtype=torch.long, device=device)] = 1.0

    with torch.no_grad():
        inanchorpath = model(g_weighted, h0)
        ratio1 = inanchorpath / (inanchorpath + 1e-8)
        weight_perf = 2 / (1 + torch.exp(-gamma * ratio1)) - 1

        inpath = model(g_weighted, weight_perf)
        ratio2 = inpath / (inpath + 1e-8)
        emb_perf = 2 / (1 + torch.exp(-gamma * ratio2)) - 1

    return emb_perf.cpu()


# ---------------------------------------------------------------------------
#  VBIP computation functions
# ---------------------------------------------------------------------------


def compute_vbip_scores(
    g,
    anchor_local_indices: np.ndarray,
    num_nodes: int,
    device: torch.device,
    layer_num: int = 2,
    gamma: float | None = None,
    impl: str = "simplified",
) -> torch.Tensor:
    """
    Compute VBIP scores for all local nodes.

    Args:
        g:                    for impl="simplified": local DGL graph,
                              RREAGraph (models/rrea.py, uses
                              .adj_self_norm), or raw sparse adj.
                              for impl="elsea": a graph pre-built by
                              build_elsea_vbip_graph().
        anchor_local_indices: local indices of anchor (train-link) KG1 nodes
        num_nodes:            total number of nodes in local graph
        device:               compute device
        layer_num:            number of diffusion hops per pass
        gamma:                diffusion parameter; defaults to 0.9
                              (damping factor) for "simplified", 2.0
                              (sigmoid steepness, matching ELsEA's real
                              default) for "elsea"
        impl:                 "simplified" (default) or "elsea"

    Returns:
        (num_nodes,) float32 tensor on CPU, scores in [0, 1].
    """
    if impl == "elsea":
        return _compute_elsea_vbip_scores(
            g, anchor_local_indices, num_nodes, device, layer_num,
            gamma if gamma is not None else 2.0,
        )

    gamma = gamma if gamma is not None else 0.9
    model = _EviPassingModel(num_layers=layer_num, gamma=gamma).to(device)
    model.eval()

    anchor_mask = torch.zeros(num_nodes, device=device)
    if len(anchor_local_indices) > 0:
        anchor_mask[
            torch.tensor(anchor_local_indices, dtype=torch.long, device=device)
        ] = 1.0

    with torch.no_grad():
        scores = model(g, anchor_mask)

    scores = torch.sigmoid(scores)
    return scores.cpu()


def gather_vbip_scores(
    local_scores: torch.Tensor,
    all_global_ids: np.ndarray,
    local_kg1_idx: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
    comm_stats: CommStats | None = None,
) -> dict[int, float]:
    """
    All-gather KG1 VBIP scores across all ranks.

    Returns a global dict: {kg1_global_id: score} identical on all ranks.

    This is the compact, periodic cross-partition sync MM (Model Migration)
    relies on — only per-entity scalar scores cross the network, not raw
    features.
    """
    cs = comm_stats or CommStats()

    # Extract local KG1 scores
    local_kg1_scores = local_scores[local_kg1_idx]  # (N_local_kg1,)
    local_kg1_ids = all_global_ids[local_kg1_idx].astype(np.int64)  # (N_local_kg1,)

    # Pack as (id, score) pairs for gathering
    n = len(local_kg1_ids)
    packed = torch.zeros(n, 2, device=device)
    if n > 0:
        packed[:, 0] = torch.from_numpy(local_kg1_ids.astype(np.float32)).to(device)
        packed[:, 1] = local_kg1_scores.to(device)

    n_t = torch.tensor([n], device=device, dtype=torch.long)
    all_n_t = [
        torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)
    ]
    with cs.timed("mm_vbip_gather"):
        dist.all_gather(all_n_t, n_t)
    all_n = [int(x.item()) for x in all_n_t]
    max_n = max(all_n) if all_n else 1

    # Pad
    if n < max_n:
        packed = torch.cat([packed, torch.zeros(max_n - n, 2, device=device)], dim=0)

    gathered = [torch.zeros(max_n, 2, device=device) for _ in range(world_size)]
    with cs.timed("mm_vbip_gather"):
        dist.all_gather(gathered, packed)
    cs.record("mm_vbip_gather", num_bytes=max_n * 2 * 4 * world_size, calls=1)

    global_scores: dict[int, float] = {}
    for r in range(world_size):
        for j in range(all_n[r]):
            gid = int(gathered[r][j, 0].item())
            sc = float(gathered[r][j, 1].item())
            global_scores[gid] = sc

    return global_scores


def build_pair_weights(
    local_train_pairs: torch.Tensor,
    all_global_ids: np.ndarray,
    global_kg1_scores: dict[int, float],
    device: torch.device,
) -> torch.Tensor | None:
    """
    Build per-pair weight tensor from global KG1 scores.

    Args:
        local_train_pairs: (P, 2) local index pairs (kg1_local, kg2_local)
        all_global_ids:    global IDs in local node order
        global_kg1_scores: {global_kg1_id: vbip_score}
        device:            target device

    Returns:
        (P,) float32 weight tensor, or None if no pairs.
    """
    if local_train_pairs is None or len(local_train_pairs) == 0:
        return None

    kg1_local = (
        local_train_pairs[:, 0].cpu().numpy()
        if isinstance(local_train_pairs, torch.Tensor)
        else np.array(local_train_pairs)[:, 0]
    )

    weights = []
    for li in kg1_local:
        gid = int(all_global_ids[li])
        weights.append(global_kg1_scores.get(gid, 1.0))

    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    # Normalise to [0, 1] within this batch
    w_min, w_max = w_t.min(), w_t.max()
    if w_max > w_min:
        w_t = (w_t - w_min) / (w_max - w_min)
    return w_t


# ---------------------------------------------------------------------------
#  Combined convenience function
# ---------------------------------------------------------------------------


def compute_and_gather_vbip(
    g,
    train_links: list,
    global2local: dict,
    all_global_ids: np.ndarray,
    local_kg1_idx: np.ndarray,
    local_train_pairs,
    rank: int,
    world_size: int,
    device: torch.device,
    layer_num: int = 2,
    gamma: float | None = None,
    comm_stats: CommStats | None = None,
    impl: str = "simplified",
    data=None,
) -> tuple[torch.Tensor | None, dict[int, float]]:
    """
    Full VBIP pipeline: score → gather → pair weights.

    Falls back to (None, {}) on errors (framework uses uniform weights).

    Args:
        impl: "simplified" (default, loosely ELsEA-inspired diffusion) or
              "elsea" (faithful port of ELsEA's real VBIP math -- see module
              docstring). "elsea" requires `data` (the Data instance, for
              its kg1_triples/kg2_triples/edge_weights1/edge_weights2) to
              build the weighted graph; missing `data` falls back like any
              other error.

    Returns:
        (pair_weights_tensor_or_None, global_kg1_scores_dict)
    """
    try:
        # Find local anchor (train-link KG1) nodes
        anchor_locals = []
        for kg1_gid, _ in train_links:
            li = global2local.get(int(kg1_gid))
            if li is not None:
                anchor_locals.append(li)

        num_nodes = len(all_global_ids)

        if impl == "elsea":
            if data is None:
                raise ValueError("impl='elsea' requires `data` to build the weighted VBIP graph")
            g_for_scoring = build_elsea_vbip_graph(data, global2local, all_global_ids, device)
        else:
            g_for_scoring = g

        local_scores = compute_vbip_scores(
            g_for_scoring,
            np.array(anchor_locals, dtype=np.int64),
            num_nodes,
            device,
            layer_num,
            gamma,
            impl=impl,
        )
        global_scores = gather_vbip_scores(
            local_scores, all_global_ids, local_kg1_idx, rank, world_size, device,
            comm_stats=comm_stats,
        )
        pair_weights = build_pair_weights(
            local_train_pairs, all_global_ids, global_scores, device
        )
        return pair_weights, global_scores

    except Exception as e:
        logging.warning(
            f"[dist_vbip] VBIP failed ({e}), falling back to uniform weights"
        )
        return None, {}
