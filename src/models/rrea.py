"""
rrea.py — RREA (Relational Reflection Entity Alignment) encoder for DistEA,
ported from ELsEA's actual primary model
(ELsEA/DivEA/RREA/{layer,runner,utils}.py — TensorFlow/Keras) to PyTorch.

ELsEA's README (github.com/wx-qzhou/ELsEA) uses "rrea" for every example
`--ea_model` invocation — GCN-Align is a secondary, alternative backend in
the same DivEA framework. RREA is therefore DistEA's primary encoder;
models/gcn.py is kept as a tested secondary GCN-Align implementation.

Architecture (RREA/layer.py::NR_GraphAttention + RREA/runner.py::create_model):
  - Two learnable tables: entity embeddings h_init (N, D) — external, see
    framework.py, same role/halo-handling as models/gcn.py's h_init — and
    RELATION embeddings (2*rel_num, D) — internal to this module (`rel_emb`),
    kept in sync across ranks via ordinary all_reduce, same as any other
    model parameter (relations aren't partitioned, so no halo logic needed).
    Relations are directional: the reverse of relation r uses a SEPARATE
    learnable vector (id r + rel_num), not a shared/transposed one.
  - "avg" pre-aggregation (RREA/utils.py::get_matrix + RREA/runner.py::avg):
      ent_feature = adj_self_norm @ h_init          (self-loop, sym-normalised)
      rel_feature = rel_profile_norm @ rel_emb      (per-entity in/out
                                                      relation-count profile)
  - NR_GraphAttention, per depth/head: gather neighbour features via the RAW
    (unnormalised, non-self-looped) graph edges; look up each edge's own
    relation embedding (soft-averaged over co-occurring relations via
    edge_rel_val, matching RREA's r_val = 1/count for edges with >1
    relation); L2-normalise it; REFLECT the neighbour feature across the
    hyperplane defined by that unit relation vector — Householder reflection
    v - 2(v.n)n — this is the paper's "relational reflection" and is why
    RREA needs no per-relation weight MATRIX, only a per-relation VECTOR;
    attention-weight the reflected neighbours (GAT-style, from
    [self, reflected-neighbour, relation] via a learned kernel) and
    segment-sum per target node.
  - Every depth's output (including the un-propagated input) is concatenated
    — a dense/skip-connection pattern, not just the final layer's output.
  - The SAME encoder (shared weights) runs once on ent_feature and once on
    rel_feature; final embedding = concat of both, then dropout.
  - Loss: reuses models/gcn.py's alignment_loss/combined_align_loss/
    infonce_loss unchanged — RREA/runner.py::constrastive_align_loss is the
    same formula (softmax-(2,)-mixed InfoNCE + L1-margin hinge, gamma=6).
  - Negative sampling: RREA draws both corrupted endpoints from the FULL
    local entity range (np.random.randint(0, node_size, ...) — no KG-side
    restriction), not per-KG-boundary-restricted. See dist_trainer.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcn import build_entity_features  # deterministic, ID-seeded init
from rrea_minibatch import (
    RREAMinibatchBatch,
    RREAMinibatchIndex,
)


class RREAGraph:
    """
    Bundle of the tensors RREA's forward pass needs, built once per rank per
    iteration by dist_graph.py (see build_rrea_graph). Passed opaquely as the
    `g` argument everywhere models/gcn.py's `g` (a DGL graph / sparse adj)
    was passed — callers (dist_trainer.py, eval.py) never need to know its
    internal shape, only `model(g, h_init)`.

    adj_self_norm:  (N, N) sparse, symmetric-normalised WITH self-loop —
                    feeds the "avg" pre-aggregation for ent_feature.
    rel_profile_norm: (N, 2*rel_num) sparse, normalised in/out relation-count
                    profile per entity — feeds the "avg" pre-aggregation for
                    rel_feature.
    edge_index:     (E, 2) long — (src, dst) local indices, RAW graph (no
                    self-loop, no normalisation) — feeds NR_GraphAttention's
                    message passing.
    edge_rel:       (E,) long — relation id for each edge (directional: the
                    reverse-direction edge uses id + rel_num).
    edge_rel_val:   (E,) float — soft weight for edges whose (src,dst) pair
                    is connected by more than one relation (1/count, matches
                    RREA's r_val).
    rel_num:        total distinct forward relations (before doubling for
                    direction) — rel_emb table has width 2*rel_num.
    """

    __slots__ = (
        "adj_self_norm", "rel_profile_norm", "edge_index", "edge_rel",
        "edge_rel_val", "rel_num",
    )

    def __init__(self, adj_self_norm, rel_profile_norm, edge_index, edge_rel,
                 edge_rel_val, rel_num):
        self.adj_self_norm = adj_self_norm
        self.rel_profile_norm = rel_profile_norm
        self.edge_index = edge_index
        self.edge_rel = edge_rel
        self.edge_rel_val = edge_rel_val
        self.rel_num = rel_num


class NRGraphAttention(nn.Module):
    """
    Depth x attn_heads relational-reflection attention layers (RREA's
    NR_GraphAttention). No per-relation weight MATRIX anywhere — only the
    per-relation embedding VECTOR (looked up from whichever `rel_emb` the
    caller passes in) acts as a reflection plane. Same module instance is
    called on both ent_feature and rel_feature (RREA.forward), so its
    weights (attn_kernels, loss_weight) are shared between the two channels
    — matching layer.py, where `encoder` is a single object called twice.
    """

    def __init__(self, dim: int, depth: int = 2, attn_heads: int = 1,
                 dropout: float = 0.0, kernel_seed: int = 0, edge_chunk: int = 0,
                 softmax_axis: str = "dst", input_relu: bool = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.attn_heads = attn_heads
        self.dropout = nn.Dropout(dropout)
        # softmax_axis: which edge
        # endpoint groups the attention softmax. The original layer.py does
        # `sparse_softmax(att)` over the adjacency ROW = adj.indices[:, 0] =
        # the SELF node (the same index segment_sum aggregates into), i.e.
        # for node i: out_i = sum_j a_ij reflect(h_j) with sum_j a_ij = 1.
        # "dst" groups by the NEIGHBOUR while summing into `src` (self), so
        # the mass landing on a node is ~sum_j 1/deg(j) instead of 1 --
        # degree-dependent output scale. "src" reproduces the original;
        # "dst" is the default for backward compatibility.
        # input_relu: the original applies the activation to the input
        # features before the depth-0 slice is kept and before layer 1
        # (`features = self.activation(features); outputs.append(features)`);
        # False keeps the signed features (default).
        if softmax_axis not in ("dst", "src"):
            raise ValueError(f"softmax_axis must be 'dst' or 'src', got {softmax_axis!r}")
        self.softmax_axis = softmax_axis
        self.input_relu = bool(input_relu)
        # edge_chunk > 0: forward() processes the attention in
        # dst-grouped edge chunks of about this many edges (exact -- chunk
        # boundaries never split a softmax group), so the per-edge
        # intermediates ((E, 3*dim) att_in etc.) are bounded by the chunk
        # instead of the whole graph. Needed for the eval-time full-graph
        # forward on DBP1M/N=1 (16.4M edges: one att_in alone is 25 GB).
        # 0 = whole graph in one pass, bit-identical to the unchunked path.
        self.edge_chunk = int(edge_chunk)
        # attn_kernels[l][h]: (3*dim,) — scores cat([self, reflected_neigh, rel])
        #
        # Seeded via a LOCAL torch.Generator (not the global RNG stream)
        # so every rank builds the identical kernels without cross-rank
        # communication, matching how rel_emb/h_init are already made
        # deterministic (RREA.__init__ below, build_entity_features). Torch's
        # global RNG state differs by rank (each process consumes a
        # rank-dependent amount of prior randomness), so drawing from it
        # would start attn_kernels differently on every rank until the
        # first all_reduce(AVG).
        gen = torch.Generator().manual_seed(kernel_seed)
        self.attn_kernels = nn.ParameterList([
            nn.Parameter(_glorot((3 * dim, 1), generator=gen))
            for _ in range(depth * attn_heads)
        ])
        # NOTE: RREA/layer.py defines the (2,) combined-loss mixing weight
        # on the encoder object itself (`encoder.weight`). DistEA keeps that
        # parameter external instead (framework.py builds it fresh each
        # iteration and hands it to DistTrainer/combined_align_loss) — same
        # place models/gcn.py's GCN already keeps it, so both encoders share
        # one convention rather than each managing it differently.

    def _kernel(self, layer: int, h: int) -> torch.Tensor:
        return self.attn_kernels[layer * self.attn_heads + h]

    def forward(
        self,
        features: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel: torch.Tensor,
        edge_rel_val: torch.Tensor,
    ) -> torch.Tensor:
        """
        features:   (N, D) — ent_feature or rel_feature (the pre-aggregated
                    input for this channel).
        rel_emb:    (2*rel_num, D) — the shared relation embedding table.
        edge_index: (E, 2) long.
        edge_rel:   (E,) long.
        edge_rel_val: (E,) float.

        Returns (N, D*(depth+1)) — every depth's output concatenated,
        matching layer.py's `outputs = K.concatenate(outputs)`.
        """
        N = features.size(0)
        src, dst = edge_index[:, 0], edge_index[:, 1]
        chunks = None
        group_key = src if self.softmax_axis == "src" else dst
        if self.edge_chunk > 0 and dst.numel() > self.edge_chunk:
            # chunk boundaries must not split a softmax group -> chunk by
            # whichever endpoint the softmax is grouped by
            chunks = _dst_chunk_slices(group_key, self.edge_chunk)

        if self.input_relu:
            features = torch.relu(features)
        outputs = [features]
        h = features
        for layer in range(self.depth):
            head_outs = []
            for head in range(self.attn_heads):
                kernel = self._kernel(layer, head)
                if chunks is None:
                    new_features = self._head_messages(
                        h, rel_emb, src, dst, edge_rel, edge_rel_val, kernel, N, None)
                else:
                    new_features = None
                    for idx in chunks:
                        new_features = self._head_messages(
                            h, rel_emb, src[idx], dst[idx], edge_rel[idx],
                            edge_rel_val[idx], kernel, N, new_features)
                head_outs.append(new_features)

            h = torch.stack(head_outs, dim=0).mean(dim=0)  # attn_heads_reduction='average'
            h = torch.relu(h)
            outputs.append(h)

        return torch.cat(outputs, dim=1)

    def _head_messages(self, h, rel_emb, src, dst, edge_rel, edge_rel_val, kernel, N,
                       out: torch.Tensor | None) -> torch.Tensor:
        """One head's aggregated messages over the given edge set (the whole
        graph, or one dst-grouped chunk whose softmax groups are complete),
        accumulated into `out` (allocated when None). Same op sequence as the
        original single-pass forward."""
        # Per-edge relation vector: soft-average over co-occurring
        # relations (edge_rel_val), then L2-normalise — the
        # reflection plane's unit normal.
        # F.embedding == advanced indexing in the forward; its backward
        # reduces duplicate indices with parallel partial sums instead of
        # index_put_'s serial per-run walk, which matters when one relation
        # labels millions of edges or a hub has 85K in-edges (DBP1M; see
        # gas.gather_rows).
        if edge_rel_val.numel() == 0 or bool((edge_rel_val > 0).all()):
            # normalize(v * r) == normalize(r) for scalar v > 0: with the
            # all-positive 1/count weights the per-edge scale is a no-op
            # after the L2 normalisation, so normalise the (n_rel, dim)
            # table once and gather (identical values and gradients; saves
            # an (E, dim) multiply + norm + divide and their backward).
            rel_vec = F.embedding(edge_rel, F.normalize(rel_emb, p=2, dim=1))
        else:
            rel_vec = F.embedding(edge_rel, rel_emb) * edge_rel_val.unsqueeze(1)
            rel_vec = F.normalize(rel_vec, p=2, dim=1)

        neighs = F.embedding(dst, h)
        selfs = F.embedding(src, h)

        # Householder reflection of the neighbour feature across
        # the relation's hyperplane: v - 2(v.n)n, then the attention
        # logit kernel . [self, reflected neighbour, relation] -- see
        # edge_attention_core (no (E, 3*dim) concatenation).
        neighs_reflected, att_logits = edge_attention_core(selfs, neighs, rel_vec, kernel)
        att = _sparse_softmax_by_dst(att_logits, src if self.softmax_axis == "src" else dst, N)

        msg = neighs_reflected * att.unsqueeze(1)
        # dtype follows msg, not h: under bf16 autocast (--amp bf16)
        # the attention matmul upstream may emit a different dtype
        # than this layer's input, and index_add_ (autocast-
        # unmanaged) hard-errors on any mismatch. Identical to
        # h.dtype in fp32 mode.
        if out is None:
            out = torch.zeros(N, self.dim, device=h.device, dtype=msg.dtype)
        out.index_add_(0, src, msg)
        return out

    def forward_block_hop(
        self,
        layer: int,
        outer_feat: torch.Tensor,
        outer_ids,
        inner_ids,
        rel_emb: torch.Tensor,
        edges,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Mini-batch counterpart of one depth of forward()'s loop body
        (see rrea_minibatch.py) -- additive, does not replace forward(). `edges` is one
        rrea_minibatch.AttentionEdges tuple (edge_outer_idx, edge_inner_idx,
        edge_rel_id, edge_rel_val) from build_rrea_attention_block, where
        `inner_ids` MUST be a subset of `outer_ids` (rrea_minibatch's
        nested_active_sets guarantees this) -- `selfs` below is looked up
        by mapping each edge's inner-role entity back to its OWN row in
        `outer_feat`, exactly like full-graph forward()'s `selfs = h[src]`
        reads the same tensor `neighs` comes from, just restricted to this
        hop's sampled node set instead of the whole graph.

        Role mapping vs. forward()'s `src, dst = edge_index[:, 0], edge_index[:, 1]`:
        `dst` (softmax-group key, `neighs` source) -> outer/edge_outer_idx;
        `src` (index_add_ output target, `selfs` source) -> inner/edge_inner_idx
        -- see build_rrea_attention_block's docstring for why.

        Returns (len(inner_ids), dim) -- this hop's new feature for
        inner_ids only, NOT the whole graph (the memory-saving point of
        block-based mini-batch training).
        """
        e_outer, e_inner, e_rel, e_val = edges
        inner_size = len(inner_ids)
        # `edges` are Python lists (rrea_minibatch.build_rrea_attention_block)
        # or numpy arrays (minibatch_csr.build_rrea_attention_block_csr);
        # len()/as_tensor handle both, `not list` would not for arrays.
        if len(e_outer) == 0:
            return torch.zeros(inner_size, self.dim, device=device, dtype=outer_feat.dtype)

        outer_idx = torch.as_tensor(np.asarray(e_outer, dtype=np.int64), device=device)
        inner_idx = torch.as_tensor(np.asarray(e_inner, dtype=np.int64), device=device)
        rel_idx = torch.as_tensor(np.asarray(e_rel, dtype=np.int64), device=device)
        val = torch.as_tensor(np.asarray(e_val, dtype=np.float32), device=device)

        # inner -> outer row mapping via searchsorted instead of a per-call
        # Python dict + list comprehension (a hot-path cost this method
        # pays once per hop per channel per batch). Valid because
        # outer_ids is SORTED and inner_ids ⊆ outer_ids -- both guaranteed
        # by rrea_minibatch.nested_active_sets (np.array(sorted(...)),
        # monotonically nested); the array_equal assert fails loudly on a
        # contract violation (searchsorted alone would silently return a
        # wrong position).
        io = np.searchsorted(outer_ids, inner_ids)
        assert np.array_equal(np.asarray(outer_ids)[io], np.asarray(inner_ids)), (
            "inner_ids must be a subset of sorted outer_ids"
        )
        inner_to_outer = torch.from_numpy(io.astype(np.int64)).to(device)
        self_idx = inner_to_outer[inner_idx]

        head_outs = []
        for head in range(self.attn_heads):
            rel_vec = rel_emb[rel_idx] * val.unsqueeze(1)
            rel_vec = F.normalize(rel_vec, p=2, dim=1)

            neighs = outer_feat[outer_idx]
            selfs = outer_feat[self_idx]

            bias = (neighs * rel_vec).sum(dim=1, keepdim=True) * rel_vec
            neighs_reflected = neighs - 2 * bias

            att_in = torch.cat([selfs, neighs_reflected, rel_vec], dim=1)
            att_logits = (att_in @ self._kernel(layer, head)).squeeze(-1)
            # Softmax grouped by OUTER (dst-role) over len(outer_ids) groups
            # -- a purely local operation over group keys, correct to
            # restrict to this block's own outer_ids count instead of the
            # full graph's N, same as forward()'s own N is exactly "however
            # many nodes dst indexes into".
            att = _sparse_softmax_by_dst(
                att_logits, self_idx if self.softmax_axis == "src" else outer_idx, len(outer_ids))

            msg = neighs_reflected * att.unsqueeze(1)
            new_features = torch.zeros(inner_size, self.dim, device=device, dtype=outer_feat.dtype)
            new_features.index_add_(0, inner_idx, msg)
            head_outs.append(new_features)

        h = torch.stack(head_outs, dim=0).mean(dim=0)
        return torch.relu(h)


class RREA(nn.Module):
    """
    Full RREA encoder: dual-channel (entity, relation) through a SHARED
    NRGraphAttention, concatenated, dropout. `h_init` (external, see
    framework.py) is the entity table; `rel_emb` (internal) is the relation
    table — both learnable, both flow through the same ordinary
    model.parameters() / all_reduce sync path DistTrainer already uses.

    output_dim here matches RREA/runner.py's node_hidden=rel_hidden=128
    convention: the FINAL embedding width is output_dim * 2 * (depth+1)
    (x2 for the ent/rel dual channel), not output_dim — callers that need a
    fixed downstream width (e.g. halo_kg2_emb concatenation) must size
    against `.output_width`, not `output_dim`.
    """

    def __init__(
        self,
        rel_num: int,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        attn_heads: int = 1,
        rel_seed_offset: int = 0,
        edge_chunk: int = 0,
        rrea_dropout: float = 0.0,
        init_scale: float = 1.0,
        attn_softmax: str = "dst",
        input_relu: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.output_width = output_dim * 2 * (num_layers + 1)
        # The ORIGINAL RREA dropout (runner.py: Dropout(0.2)
        # on the final concatenated embedding, nothing inside the layers).
        # `dropout` above is NRGraphAttention's constructor argument, which
        # that module stores but never applies -- so this is the only
        # dropout that actually acts on the RREA path. 0 = identity (no op
        # is executed at all, so the default path is bit-identical).
        # GAS trains through gas.py, which applies the same dropout to the
        # loss rows only (never to final_hist / the history push).
        self.rrea_dropout = float(rrea_dropout)
        self.out_dropout = nn.Dropout(self.rrea_dropout)
        # kernel_seed fixed (not rank- or call-dependent) — see
        # NRGraphAttention's kernel_seed comment: every rank calls RREA()
        # with identical arguments (framework.py never passes anything
        # rank-specific here), so a fixed seed is enough to guarantee
        # identical attn_kernels everywhere, exactly like rel_emb below.
        self.encoder = NRGraphAttention(output_dim, depth=num_layers,
                                        attn_heads=attn_heads, dropout=dropout,
                                        kernel_seed=0, edge_chunk=edge_chunk,
                                        softmax_axis=attn_softmax, input_relu=input_relu)
        # Deterministic (ID-seeded) init so every rank builds the identical
        # rel_emb table without cross-rank communication — relations aren't
        # partitioned, every rank sees the full 2*rel_num table, and it's
        # kept in sync afterwards via ordinary all_reduce (it's an
        # nn.Parameter like any other, registered on this module).
        rel_ids = np.arange(rel_seed_offset, rel_seed_offset + 2 * rel_num)
        init = build_entity_features(rel_ids, output_dim, torch.device("cpu"))
        # --init_scale applies to the relation table too (the
        # original initialises both tables identically); 1.0 = unchanged.
        if float(init_scale) != 1.0:
            init = init * float(init_scale)
        self.init_scale = float(init_scale)
        self.rel_emb = nn.Parameter(init)

    def forward(self, g: RREAGraph, h_init: torch.Tensor) -> torch.Tensor:
        # torch.sparse.mm is on autocast's lower-precision list
        # (emits bf16 under --amp bf16) but SparseAddmmBackward has no
        # bf16 kernel -- backward hard-errors. Pin the two sparse
        # pre-aggregations to fp32; no-op when autocast is off.
        with torch.autocast(device_type=h_init.device.type, enabled=False):
            ent_feature = torch.sparse.mm(g.adj_self_norm, h_init)
            rel_feature = torch.sparse.mm(g.rel_profile_norm, self.rel_emb.to(h_init.device))

        out_ent = self.encoder(ent_feature, self.rel_emb.to(h_init.device),
                               g.edge_index, g.edge_rel, g.edge_rel_val)
        out_rel = self.encoder(rel_feature, self.rel_emb.to(h_init.device),
                               g.edge_index, g.edge_rel, g.edge_rel_val)
        out = torch.cat([out_ent, out_rel], dim=1)
        if self.rrea_dropout > 0.0:
            out = self.out_dropout(out)   # identity in eval mode
        return out

    def forward_blocks(
        self,
        batch: RREAMinibatchBatch,
        index: RREAMinibatchIndex,
        h_init_outer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mini-batch counterpart of forward() (see rrea_minibatch.py).
        Additive, does not replace forward().

        `batch` (rrea_minibatch.prepare_rrea_batch's output) and `index`
        (rrea_minibatch.RREAMinibatchIndex, built once per partition) carry
        everything needed except the actual learnable input: `h_init_outer`
        must be exactly `SparseEntityTable.gather(batch.outer_ids)` (or
        equivalent), matching GCN.forward_blocks' own `h_init` contract of
        "the raw initial feature tensor for the outermost sampled frontier
        ONLY".

        Two-stage pipeline per rrea_minibatch.py's module docstring:
        1. Pre-aggregation (one hop): `ent_feature = batch.pre_block.adj_t.T
           @ h_init_outer` (self.forward()'s `torch.sparse.mm(g.adj_self_norm,
           h_init)`, restricted to batch.active[1]); `rel_feature` computed
           directly per-entity from index's relation-profile counts (no
           neighbor expansion needed -- see compute_rel_profile_features).
        2. `num_layers` attention hops (NRGraphAttention.forward_block_hop),
           run once for the ent_feature-seeded channel and once for the
           rel_feature-seeded channel -- same SHARED encoder weights as
           forward() (`self.encoder` called twice there too).

        Every depth's output (both channels) is restricted to
        `batch.seed_ids`' rows before concatenation -- required because,
        unlike forward() where every depth's `h` stays whole-graph-sized
        (so `torch.cat(outputs, dim=1)` trivially aligns row-for-row), each
        block hop's output here is sized to that hop's (shrinking) inner_ids,
        a DIFFERENT row count per depth. `batch.active`'s nesting guarantees
        every seed id has a row at every depth, so this restriction is
        always well-defined; see NRGraphAttention.forward_block_hop's
        docstring for the outer/inner role mapping used at each hop.

        Returns (len(batch.seed_ids), output_width) -- rows in
        `batch.seed_ids` order (sorted global ids), matching
        GCN.forward_blocks' "output indexed by the innermost block's own
        id order" convention.
        """
        device = h_init_outer.device
        rel_emb = self.rel_emb.to(device)

        ent_feature = torch.sparse.mm(batch.pre_block.adj_t.t(), h_init_outer)
        # Backend-dispatched (python dict loops vs CSR) -- see
        # RREAMinibatchIndex.rel_profile_features.
        rel_feature = index.rel_profile_features(batch.active[1], rel_emb)

        out_ent = self._forward_blocks_channel(ent_feature, batch, device)
        out_rel = self._forward_blocks_channel(rel_feature, batch, device)
        return torch.cat([out_ent, out_rel], dim=1)

    def _forward_blocks_channel(
        self, features: torch.Tensor, batch: RREAMinibatchBatch, device: torch.device,
    ) -> torch.Tensor:
        """Runs one channel (ent or rel) through all `num_layers` attention
        hops, extracting `batch.seed_ids`' rows from every depth's output
        (including the un-propagated `features` input, depth 0) and
        concatenating them -- mirrors forward()'s `outputs = [features];
        ...; return torch.cat(outputs, dim=1)`, adapted for per-depth
        shrinking node sets (see forward_blocks' own docstring)."""
        rel_emb = self.rel_emb.to(device)
        seed_ids = np.asarray(batch.seed_ids)

        def seed_rows(h: torch.Tensor, h_ids) -> torch.Tensor:
            # Vectorized row extraction (instead of a per-call dict + per-seed
            # torch.stack of single rows -- a hot-path cost paid
            # once per depth per channel per batch). h_ids is sorted and
            # seed_ids ⊆ h_ids by nested_active_sets' construction; the
            # assert keeps contract violations loud (see forward_block_hop's
            # identical pattern for the full rationale).
            idx = np.searchsorted(h_ids, seed_ids)
            assert np.array_equal(np.asarray(h_ids)[idx], seed_ids), (
                "seed_ids must be a subset of sorted h_ids"
            )
            return h[torch.from_numpy(idx.astype(np.int64)).to(h.device)]

        outputs = [seed_rows(features, batch.active[1])]
        h = features
        h_ids = batch.active[1]
        for layer, edges in enumerate(batch.attention_edges):
            inner_ids = batch.active[layer + 2]
            h = self.encoder.forward_block_hop(layer, h, h_ids, inner_ids, rel_emb, edges, device)
            h_ids = inner_ids
            outputs.append(seed_rows(h, h_ids))

        return torch.cat(outputs, dim=1)


def _edge_attention_core_eager(selfs: torch.Tensor, neighs: torch.Tensor,
                               rel_vec: torch.Tensor, kernel: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-edge reflection + attention logit, the elementwise heart of one
    RREA head: returns (neighs_reflected, att_logits) for (E, dim) inputs.

    Same math as the original cat([selfs, neighs_reflected, rel_vec]) @
    kernel, written as three (E, dim) @ (dim, 1) products on the kernel's
    three slices so the (E, 3*dim) concatenation (one extra write + read
    of 3x the edge features per head, and its backward) never exists.
    Differs from the single matmul only in fp32 summation order."""
    bias = (neighs * rel_vec).sum(dim=1, keepdim=True) * rel_vec
    neighs_reflected = neighs - 2 * bias
    dim = rel_vec.size(1)
    # Multiply-and-row-sum instead of three (E, dim) @ (dim, 1) matmuls: the
    # same dot products, but as elementwise + reduction ops they fuse into
    # the compiled kernel (and its backward) instead of 3 extern gemv calls
    # per head call -- ~0.7 s of aten::mm per DBP1M epoch otherwise.
    k1, k2, k3 = kernel[:dim].reshape(1, dim), kernel[dim:2 * dim].reshape(1, dim), \
        kernel[2 * dim:].reshape(1, dim)
    att_logits = (selfs * k1).sum(dim=1) + (neighs_reflected * k2).sum(dim=1) + (rel_vec * k3).sum(dim=1)
    return neighs_reflected, att_logits


_EDGE_CORE_COMPILED = None  # resolved on first use when DISTEA_GAS_COMPILE_HEAD=1


def edge_attention_core(selfs: torch.Tensor, neighs: torch.Tensor,
                        rel_vec: torch.Tensor, kernel: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """_edge_attention_core_eager, or its torch.compile(dynamic=True)
    version when DISTEA_GAS_COMPILE_HEAD=1 (opt-in: Inductor fuses the
    elementwise chain and the row reductions into a few kernels instead of
    ~10 separate (E, dim) passes, forward and backward; the edge count
    changes per cluster, hence dynamic shapes). Falls back to eager, once,
    with a warning if compilation is unavailable."""
    global _EDGE_CORE_COMPILED
    import os

    if os.environ.get("DISTEA_GAS_COMPILE_HEAD", "0").lower() not in ("1", "true", "yes"):
        return _edge_attention_core_eager(selfs, neighs, rel_vec, kernel)
    if _EDGE_CORE_COMPILED is None:
        try:
            _EDGE_CORE_COMPILED = torch.compile(_edge_attention_core_eager, dynamic=True)
        except Exception as exc:  # pragma: no cover - depends on the torch build
            import logging

            logging.warning(f"DISTEA_GAS_COMPILE_HEAD: torch.compile unavailable ({exc}); eager")
            _EDGE_CORE_COMPILED = _edge_attention_core_eager
    try:
        return _EDGE_CORE_COMPILED(selfs, neighs, rel_vec, kernel)
    except Exception as exc:  # pragma: no cover
        import logging

        logging.warning(f"DISTEA_GAS_COMPILE_HEAD: compiled core failed ({exc}); eager from now on")
        _EDGE_CORE_COMPILED = _edge_attention_core_eager
        return _edge_attention_core_eager(selfs, neighs, rel_vec, kernel)


def _glorot(shape, generator: torch.Generator | None = None) -> torch.Tensor:
    """Glorot/Xavier uniform init, matching inits.py::glorot used by
    ELsEA's attn_kernel_initializer default. `generator` (optional): draw
    from a specific torch.Generator instead of the global RNG stream, so
    the result is reproducible independent of whatever the global stream's
    state happens to be (see NRGraphAttention's kernel_seed)."""
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    limit = (6.0 / (fan_in + fan_out)) ** 0.5
    if generator is not None:
        return (torch.rand(shape, generator=generator) * 2 - 1) * limit
    return (torch.rand(shape) * 2 - 1) * limit


def _dst_chunk_slices(dst: torch.Tensor, chunk: int) -> list[torch.Tensor]:
    """Index slices covering every edge exactly once, each holding complete
    dst groups (edges sorted by dst, boundaries only at group starts) and
    about `chunk` edges -- a single group longer than `chunk` forms a chunk
    of its own (a DBP1M hub has 85K in-edges)."""
    order = torch.argsort(dst)
    d_sorted = dst[order]
    E = int(d_sorted.numel())
    change = torch.nonzero(d_sorted[1:] != d_sorted[:-1]).flatten() + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dst.device), change]).cpu().numpy()
    bounds: list[tuple[int, int]] = []
    a = 0
    while a < E:
        target = a + chunk
        if target >= E:
            bounds.append((a, E))
            break
        j = int(np.searchsorted(starts, target, side="right")) - 1  # last group start <= target
        b = int(starts[j])
        if b <= a:  # the group starting at/before `a` alone exceeds `chunk`: take it whole
            k = int(np.searchsorted(starts, a, side="right"))
            b = int(starts[k]) if k < len(starts) else E
        bounds.append((a, b))
        a = b
    return [order[s:e] for s, e in bounds]


def _sparse_softmax_by_dst(logits: torch.Tensor, dst: torch.Tensor, N: int) -> torch.Tensor:
    """Softmax over edges grouped by destination node (segment softmax) —
    matches tf.sparse.sparse_softmax over the adjacency's row (dst) index."""
    max_per_dst = torch.full((N,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_per_dst.index_reduce_(0, dst, logits, "amax", include_self=True)
    # Per-edge gathers via F.embedding instead of advanced indexing: the
    # values are identical, but `x[idx]`'s backward is index_put_(accumulate)
    # whose CUDA kernel serialises on repeated indices (a DBP1M hub has ~85K
    # edges in one group: ~0.9 s/epoch in indexing_backward_kernel on these
    # two gathers under --attn_softmax src), while embedding_dense_backward
    # reduces duplicates with segmented partial sums.
    shifted = logits - F.embedding(dst, max_per_dst.unsqueeze(1)).squeeze(1)
    exp = shifted.exp()
    # dtype follows exp, not logits: under CUDA bf16 autocast (--amp bf16),
    # exp() is on autocast's fp32-required list, so `exp` comes back fp32
    # even from bf16 logits -- index_add_ (autocast-unmanaged) hard-errors
    # on the mismatch (CPU autocast leaves exp() uncast, so only CUDA is
    # affected). Identical in fp32 mode.
    sum_per_dst = torch.zeros(N, device=logits.device, dtype=exp.dtype)
    sum_per_dst.index_add_(0, dst, exp)
    return exp / F.embedding(dst, sum_per_dst.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
